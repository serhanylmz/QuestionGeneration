import os
import sys
import logging
import json
import pandas as pd
from datasets import load_dataset
import random
from typing import List, Dict, Any
from dotenv import load_dotenv
import csv
import argparse
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt
from collections import Counter
from pydantic import BaseModel

# Import the required functions from the pipeline file
from pipeline import generate_basic_question, rank_questions_with_details, generate_answer

# Set up logging
logging.basicConfig(level=logging.INFO,
                    filename='benchmark.log',
                    filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client for GPT-4o
from openai import OpenAI
gpt_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize Anthropics client for Claude
import anthropic
claude_client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY"))

# Initialize Cohere client
import cohere
cohere_client = cohere.ClientV2(api_key=os.environ.get("COHERE_API_KEY"), log_warning_experimental_features=False)

# Initialize Google Generative AI client for Gemini
import google.generativeai as genai
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-pro-002")

# Initialize Hugging Face Inference Client for LLaMA and Qwen
from huggingface_hub import InferenceClient
hf_api_key = os.environ.get("HF_API_KEY")
hf_client = InferenceClient(api_key=hf_api_key)

# Load the SQuAD dataset
dataset = load_dataset("rajpurkar/squad")

def check_api_keys():
    required_keys = ["OPENAI_API_KEY", "CLAUDE_API_KEY", "COHERE_API_KEY", "GEMINI_API_KEY", "HF_API_KEY"]
    missing_keys = [key for key in required_keys if not os.environ.get(key)]
    if missing_keys:
        logger.error(f"Missing API keys: {', '.join(missing_keys)}")
        sys.exit(1)

check_api_keys()

def get_random_entries(num_entries, random_seed):
    dataset_size = len(dataset['train'])
    random.seed(random_seed)
    if num_entries == 'all':
        indices = list(range(dataset_size))
    else:
        num_entries = int(num_entries)
        indices = random.sample(range(dataset_size), num_entries)
    return indices


def sanitize_text(text):
    return text.strip()

def safe_json_parse(response_text):
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding failed: {e}")
        return None

def safe_api_call(func):
    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

# Modify compare functions to include retries and keep API calls consistent with the initial code

@safe_api_call
def compare_questions_gpt4o(context: str, original_question: str, original_answer: str,
                            basic_question: str, basic_answer: str,
                            enhanced_question: str, enhanced_answer: str) -> Dict[str, Any]:
    try:
        response = gpt_client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert in evaluating question-answer pairs based on a given context."},
                {"role": "user", "content": f"""You are an expert in evaluating question-answer pairs based on a given context.

Compare the following two generated question-answer pairs based on the given context and the original question-answer pair. Evaluate their quality and relevance.

Context: {context}

Original Question: {original_question}
Original Answer: {original_answer}

Question 1: {enhanced_question}
Answer 1: {enhanced_answer}

Question 2: {basic_question}
Answer 2: {basic_answer}

Evaluate Question 1 and Question 2 based on the following criteria:
1. Structural difference from the original question
2. Semantic similarity to the original question
3. How well the generated answer matches the original answer

Score each question-answer pair on a scale of 0 to 10.

Provide your answer in JSON format with the following structure:

"question1_score": <number>,
"question2_score": <number>,
"explanation": "<string>",
"winner": "<string>"  // Should be either "Question 1" or "Question 2", you must pick a winner, it cannot be a draw.

Ensure that your response can be parsed as valid JSON.
"""}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "question_comparison_evaluator",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "question1_score": {"type": "number"},
                            "question2_score": {"type": "number"},
                            "explanation": {"type": "string"},
                            "winner": {"type": "string", "enum": ["Question 1", "Question 2"]}
                        },
                        "required": ["question1_score", "question2_score", "explanation", "winner"],
                        "additionalProperties": False
                    }
                }
            }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error in comparing questions with GPT-4o: {e}")
        return {"question1_score": 0, "question2_score": 0, "explanation": "Failed to compare questions", "winner": "None"}

@safe_api_call
def compare_questions_claude(context: str, original_question: str, original_answer: str,
                             basic_question: str, basic_answer: str,
                             enhanced_question: str, enhanced_answer: str) -> Dict[str, Any]:
    try:
        # Define the tool (function) with the expected output schema
        tool = {
            "name": "question_comparison_evaluator",
            "description": "Evaluate and compare two generated question-answer pairs and output the result in structured JSON.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "question1_score": {"type": "number"},
                    "question2_score": {"type": "number"},
                    "explanation": {"type": "string"},
                    "winner": {"type": "string", "enum": ["Question 1", "Question 2"]}
                },
                "required": ["question1_score", "question2_score", "explanation", "winner"],
                "additionalProperties": False
            }
        }

        # Build the messages
        messages = [
            {
                "role": "user",
                "content": f"""You are an expert in evaluating question-answer pairs based on a given context.

Compare the following two generated question-answer pairs based on the given context and the original question-answer pair. Evaluate their quality and relevance.

Context: {context}

Original Question: {original_question}
Original Answer: {original_answer}

Question 1: {enhanced_question}
Answer 1: {enhanced_answer}

Question 2: {basic_question}
Answer 2: {basic_answer}

Evaluate Question 1 and Question 2 based on the following criteria:
1. Structural difference from the original question
2. Semantic similarity to the original question
3. How well the generated answer matches the original answer

Score each question-answer pair on a scale of 0 to 10.

Provide your answer in JSON format with the following structure:

"question1_score": <number>,
"question2_score": <number>,
"explanation": "<string>",
"winner": "<string>"  // Should be either "Question 1" or "Question 2", you must pick a winner, it cannot be a draw.

Ensure that your response can be parsed as valid JSON.
"""
            }
        ]

        # Call the API with the structured output parameters
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            tools=[tool],
            tool_choice={"type": "tool", "name": "question_comparison_evaluator"},
            messages=messages
        )

        return response.content[0].input

    except Exception as e:
        logger.error(f"Error in comparing questions with Claude: {e}")
        return {
            "question1_score": 0,
            "question2_score": 0,
            "explanation": "Failed to compare questions",
            "winner": "None"
        }

@safe_api_call
def compare_questions_cohere(context: str, original_question: str, original_answer: str,
                             basic_question: str, basic_answer: str,
                             enhanced_question: str, enhanced_answer: str) -> Dict[str, Any]:
    try:
        res = cohere_client.chat(
            model="command-r-plus-08-2024",
            messages=[
                {
                    "role": "user",
                    "content": f"""You are an expert in evaluating question-answer pairs based on a given context.

Compare the following two generated question-answer pairs based on the given context and the original question-answer pair. Evaluate their quality and relevance.

Context: {context}

Original Question: {original_question}
Original Answer: {original_answer}

Question 1: {enhanced_question}
Answer 1: {enhanced_answer}

Question 2: {basic_question}
Answer 2: {basic_answer}

Evaluate Question 1 and Question 2 based on the following criteria:
1. Structural difference from the original question
2. Semantic similarity to the original question
3. How well the generated answer matches the original answer

Score each question-answer pair on a scale of 0 to 10.

Provide your answer in JSON format with the following structure:

"question1_score": <number>,
"question2_score": <number>,
"explanation": "<string>",
"winner": "<string>"  // Should be either "Question 1" or "Question 2", you must pick a winner, it cannot be a draw.

Ensure that your response can be parsed as valid JSON.
"""
                }
            ],
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "required": ["question1_score", "question2_score", "explanation", "winner"],
                    "properties": {
                        "question1_score": {"type": "number"},
                        "question2_score": {"type": "number"},
                        "explanation": {"type": "string"},
                        "winner": {"type": "string", "enum": ["Question 1", "Question 2"]}
                    },
                },
            },
        )

        json_response = res.message.content[0].text.strip()
        parsed_response = safe_json_parse(json_response)
        if parsed_response is None:
            raise ValueError("Failed to parse JSON response")
        return parsed_response

    except Exception as e:
        logger.error(f"Error in comparing questions with Cohere: {e}")
        return {"question1_score": 0, "question2_score": 0, "explanation": "Failed to compare questions", "winner": "None"}


class ComparisonResult(BaseModel):
    question1_score: float
    question2_score: float
    explanation: str
    winner: str

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types if needed


@safe_api_call
def compare_questions_gemini(context: str, original_question: str, original_answer: str,
                             basic_question: str, basic_answer: str,
                             enhanced_question: str, enhanced_answer: str) -> Dict[str, Any]:
    try:
        prompt = f"""You are an expert in evaluating question-answer pairs based on a given context.

Compare the following two generated question-answer pairs based on the given context and the original question-answer pair. Evaluate their quality and relevance.

Context: {context}

Original Question: {original_question}
Original Answer: {original_answer}

Question 1: {enhanced_question}
Answer 1: {enhanced_answer}

Question 2: {basic_question}
Answer 2: {basic_answer}

Evaluate Question 1 and Question 2 based on the following criteria:
1. Structural difference from the original question
2. Semantic similarity to the original question
3. How well the generated answer matches the original answer

Score each question-answer pair on a scale of 0 to 10.

Provide your answer in JSON format with the following structure:

"question1_score": <number>,
"question2_score": <number>,
"explanation": "<string>",
"winner": "<string>"  // Should be either "Question 1" or "Question 2", you must pick a winner, it cannot be a draw.

Ensure that your response can be parsed as valid JSON.
"""

        result = gemini_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=ComparisonResult
            ),
        )
        json_response = result.text.strip()
        parsed_response = safe_json_parse(json_response)
        if parsed_response is None:
            raise ValueError("Failed to parse JSON response")
        parsed_response["winner"] = parsed_response["winner"].capitalize()
        return parsed_response
    except Exception as e:
        logger.error(f"Error in comparing questions with Gemini: {e}")
        return {
            "question1_score": 0,
            "question2_score": 0,
            "explanation": "Failed to compare questions",
            "winner": "None"
        }


@safe_api_call
def compare_questions_qwen(context: str, original_question: str, original_answer: str,
                           basic_question: str, basic_answer: str,
                           enhanced_question: str, enhanced_answer: str) -> Dict[str, Any]:
    try:
        prompt = f"""You are an expert in evaluating question-answer pairs based on a given context.

Compare the following two generated question-answer pairs based on the given context and the original question-answer pair. Evaluate their quality and relevance.

Context: {context}

Original Question: {original_question}
Original Answer: {original_answer}

Question 1: {enhanced_question}
Answer 1: {enhanced_answer}

Question 2: {basic_question}
Answer 2: {basic_answer}

Evaluate Question 1 and Question 2 based on the following criteria:
1. Structural difference from the original question
2. Semantic similarity to the original question
3. How well the generated answer matches the original answer

Score each question-answer pair on a scale of 0 to 10.

Provide your answer in JSON format with the following structure:

"question1_score": <number>,
"question2_score": <number>,
"explanation": "<string>",
"winner": "<string>"  // Should be either "Question 1" or "Question 2", you must pick a winner, it cannot be a draw.

Ensure that your response can be parsed as valid JSON.
"""

        messages = [{"role": "user", "content": prompt}]
        output = hf_client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=messages,
            temperature=0.5,
            max_tokens=1024,
            top_p=0.7
        )
        response_text = output.choices[0].message.content.strip()
        parsed_response = safe_json_parse(response_text)
        if parsed_response is None:
            # Try to extract JSON content
            import re
            def extract_json_content(text):
                pattern = r"\{.*\}"
                matches = re.findall(pattern, text, re.DOTALL)
                if matches:
                    return matches[0]
                else:
                    return text.strip()
            json_response = extract_json_content(response_text)
            parsed_response = safe_json_parse(json_response)
            if parsed_response is None:
                raise ValueError("Failed to parse JSON response")
        return parsed_response
    except Exception as e:
        logger.error(f"Error in comparing questions with Qwen: {e}")
        return {"question1_score": 0, "question2_score": 0, "explanation": "Failed to compare questions", "winner": "None"}

@safe_api_call
def compare_questions_llama(context: str, original_question: str, original_answer: str,
                           basic_question: str, basic_answer: str,
                           enhanced_question: str, enhanced_answer: str) -> Dict[str, Any]:
    try:
        prompt = f"""You are an expert in evaluating question-answer pairs based on a given context.

Compare the following two generated question-answer pairs based on the given context and the original question-answer pair. Evaluate their quality and relevance.

Context: {context}

Original Question: {original_question}
Original Answer: {original_answer}

Question 1: {enhanced_question}
Answer 1: {enhanced_answer}

Question 2: {basic_question}
Answer 2: {basic_answer}

Evaluate Question 1 and Question 2 based on the following criteria:
1. Structural difference from the original question
2. Semantic similarity to the original question
3. How well the generated answer matches the original answer

Score each question-answer pair on a scale of 0 to 10.

Provide your answer in JSON format with the following structure:

"question1_score": <number>,
"question2_score": <number>,
"explanation": "<string>",
"winner": "<string>"  // Should be either "Question 1" or "Question 2", you must pick a winner, it cannot be a draw.

Ensure that your response can be parsed as valid JSON.
"""

        messages = [{"role": "user", "content": prompt}]
        output = hf_client.chat.completions.create(
            model="meta-llama/Llama-3.1-70B-Instruct",
            messages=messages,
            temperature=0.5,
            max_tokens=1024,
            top_p=0.7
        )
        response_text = output.choices[0].message.content.strip()
        parsed_response = safe_json_parse(response_text)
        if parsed_response is None:
            # Try to extract JSON content
            import re
            def extract_json_content(text):
                pattern = r"\{.*\}"
                matches = re.findall(pattern, text, re.DOTALL)
                if matches:
                    return matches[0]
                else:
                    return text.strip()
            json_response = extract_json_content(response_text)
            parsed_response = safe_json_parse(json_response)
            if parsed_response is None:
                raise ValueError("Failed to parse JSON response")
        return parsed_response
    except Exception as e:
        logger.error(f"Error in comparing questions with LLaMA: {e}")
        return {"question1_score": 0, "question2_score": 0, "explanation": "Failed to compare questions", "winner": "None"}

def process_entry(entry):
    result = {}  # Initialize the result dict

    # Extract data from entry
    try:
        context = sanitize_text(entry['context'])
        answer = sanitize_text(entry['answers']['text'][0])
        original_question = sanitize_text(entry['question'])
    except Exception as e:
        logger.error(f"Error extracting data from entry: {e}")
        result.update({
            'Error': 'Failed to extract data from entry',
            'Context': None,
            'Original Question': None,
            'Original Answer': None,
            'Basic Question': None,
            'Basic Answer': None,
            'Enhanced Question': None,
            'Enhanced Answer': None,
            'GPT-4o Verdict': None,
            'Claude Verdict': None,
            'Cohere Verdict': None,
            'Gemini Verdict': None,
            'Qwen Verdict': None,
            'LLaMA Verdict': None,
            'Final Verdict': None
        })
        return result

    result.update({
        'Context': context,
        'Original Question': original_question,
        'Original Answer': answer
    })

    # Generate basic question
    try:
        basic_question = generate_basic_question(context, answer, original_question)
    except Exception as e:
        logger.error(f"Error generating basic question: {e}")
        basic_question = 'Error generating basic question'
    result['Basic Question'] = basic_question

    # Generate enhanced question
    try:
        detailed_scores, rankings, enhanced_question = rank_questions_with_details(context, answer, original_question)
    except Exception as e:
        logger.error(f"Error generating enhanced question: {e}")
        enhanced_question = 'Error generating enhanced question'
    result['Enhanced Question'] = enhanced_question

    # Generate basic answer
    try:
        basic_answer = generate_answer(context, basic_question)
    except Exception as e:
        logger.error(f"Error generating basic answer: {e}")
        basic_answer = 'Error generating basic answer'
    result['Basic Answer'] = basic_answer

    # Generate enhanced answer
    try:
        enhanced_answer = generate_answer(context, enhanced_question)
    except Exception as e:
        logger.error(f"Error generating enhanced answer: {e}")
        enhanced_answer = 'Error generating enhanced answer'
    result['Enhanced Answer'] = enhanced_answer

    # Initialize vote counts
    vote_counts = {"Question 1": 0, "Question 2": 0}

    # Collect comparison results from each LLM judge
    comparison_results = {}

    # GPT-4o
    result_gpt4o = compare_questions_gpt4o(
        context, original_question, answer,
        basic_question, basic_answer,
        enhanced_question, enhanced_answer
    )
    result['GPT-4o Verdict'] = result_gpt4o.get('winner', 'Error')
    if result_gpt4o['winner'] in vote_counts:
        vote_counts[result_gpt4o['winner']] += 1

    # Claude
    result_claude = compare_questions_claude(
        context, original_question, answer,
        basic_question, basic_answer,
        enhanced_question, enhanced_answer
    )
    result['Claude Verdict'] = result_claude.get('winner', 'Error')
    if result_claude['winner'] in vote_counts:
        vote_counts[result_claude['winner']] += 1

    # Cohere
    result_cohere = compare_questions_cohere(
        context, original_question, answer,
        basic_question, basic_answer,
        enhanced_question, enhanced_answer
    )
    result['Cohere Verdict'] = result_cohere.get('winner', 'Error')
    if result_cohere['winner'] in vote_counts:
        vote_counts[result_cohere['winner']] += 1

    # Gemini
    result_gemini = compare_questions_gemini(
        context, original_question, answer,
        basic_question, basic_answer,
        enhanced_question, enhanced_answer
    )
    result['Gemini Verdict'] = result_gemini.get('winner', 'Error')
    if result_gemini['winner'] in vote_counts:
        vote_counts[result_gemini['winner']] += 1

    # Qwen
    result_qwen = compare_questions_qwen(
        context, original_question, answer,
        basic_question, basic_answer,
        enhanced_question, enhanced_answer
    )
    result['Qwen Verdict'] = result_qwen.get('winner', 'Error')
    if result_qwen['winner'] in vote_counts:
        vote_counts[result_qwen['winner']] += 1

    # LLaMA
    result_llama = compare_questions_llama(
        context, original_question, answer,
        basic_question, basic_answer,
        enhanced_question, enhanced_answer
    )
    result['LLaMA Verdict'] = result_llama.get('winner', 'Error')
    if result_llama['winner'] in vote_counts:
        vote_counts[result_llama['winner']] += 1

    # Determine final verdict
    try:
        final_verdict_key = max(vote_counts, key=vote_counts.get)
        if vote_counts['Question 1'] == vote_counts['Question 2']:
            final_verdict = 'Draw'
        else:
            # Map 'Question 1' to 'Enhanced', 'Question 2' to 'Basic'
            final_verdict = 'Enhanced' if final_verdict_key == 'Question 1' else 'Basic'
    except Exception as e:
        logger.error(f"Error determining final verdict: {e}")
        final_verdict = 'Error'
    result['Final Verdict'] = final_verdict

    return result

def main():
    parser = argparse.ArgumentParser(description='Benchmark Script')
    parser.add_argument('--num_entries', type=str, default='all', help='Number of entries to process')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--entries_file', type=str, default='benchmark_results.csv', help='CSV file to store results')
    args = parser.parse_args()

    # Generate the indices based on the random seed and num_entries
    indices = get_random_entries(args.num_entries, args.random_seed)
    total_entries = len(indices)

    # Read processed indices from existing entries file
    processed_indices = set()
    if os.path.exists(args.entries_file) and os.stat(args.entries_file).st_size > 0:
        df_existing = pd.read_csv(args.entries_file)
        if 'Index' in df_existing.columns:
            processed_indices = set(df_existing['Index'].tolist())

    # Determine which indices have not been processed yet
    indices_to_process = [idx for idx in indices if idx not in processed_indices]
    if not indices_to_process:
        print("All entries have been processed. Exiting.")
        return

    print(f"Total entries to process: {len(indices_to_process)} out of {total_entries}")

    # Select the entries to process
    entries_to_process = dataset['train'].select(indices_to_process)
    total_vote_counts = Counter()

    fieldnames = ['Index', 'Context', 'Original Question', 'Original Answer', 'Basic Question', 'Basic Answer',
                  'Enhanced Question', 'Enhanced Answer', 'GPT-4o Verdict', 'Claude Verdict', 'Cohere Verdict',
                  'Gemini Verdict', 'Qwen Verdict', 'LLaMA Verdict','Final Verdict']

    # Open the CSV file for appending
    with open(args.entries_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file is empty
        if os.stat(args.entries_file).st_size == 0:
            writer.writeheader()

        # Progress bar
        for idx_in_entries, (entry, idx_in_dataset) in enumerate(
            tqdm(zip(entries_to_process, indices_to_process), total=len(indices_to_process), desc='Processing entries')
        ):
            print(f"Processing entry {idx_in_entries + 1}/{len(indices_to_process)} (Dataset Index: {idx_in_dataset})...")
            try:
                result = process_entry(entry)
                result['Index'] = idx_in_dataset  # Add the index to the result
                # Write result to CSV
                writer.writerow(result)
                csvfile.flush()
                os.fsync(csvfile.fileno())
                # Update total vote counts
                for key in ['Question 1', 'Question 2']:
                    total_votes = sum(
                        1 for verdict in [
                            result.get('GPT-4o Verdict'),
                            result.get('Claude Verdict'),
                            result.get('Cohere Verdict'),
                            result.get('Gemini Verdict'),
                            result.get('Qwen Verdict'),
                            result.get('LLaMA Verdict')
                        ] if verdict == key
                    )
                    total_vote_counts[key] += total_votes
            except Exception as e:
                logger.error(f"Error processing entry {idx_in_dataset}: {e}")
                continue

    # Read the CSV file into a DataFrame for summary
    df = pd.read_csv(args.entries_file)

    # Generate summary
    total_entries_processed = len(df)
    final_verdicts = df['Final Verdict'].value_counts()
    percentage_basic = (final_verdicts.get('Basic', 0) / total_entries_processed) * 100 if total_entries_processed > 0 else 0
    percentage_enhanced = (final_verdicts.get('Enhanced', 0) / total_entries_processed) * 100 if total_entries_processed > 0 else 0
    percentage_draw = (final_verdicts.get('Draw', 0) / total_entries_processed) * 100 if total_entries_processed > 0 else 0

    summary = f"""
Total Entries Processed: {total_entries_processed}

Vote Counts:
Question 1 (Enhanced) Votes: {total_vote_counts['Question 1']}
Question 2 (Basic) Votes: {total_vote_counts['Question 2']}

Final Verdicts:
Enhanced Generation Wins: {final_verdicts.get('Enhanced', 0)} ({percentage_enhanced:.2f}%)
Basic Generation Wins: {final_verdicts.get('Basic', 0)} ({percentage_basic:.2f}%)
Draws: {final_verdicts.get('Draw', 0)} ({percentage_draw:.2f}%)
"""

    # Save summary to text file
    summary_filename = 'benchmark_summary.txt'
    with open(summary_filename, 'w') as f:
        f.write(summary)
    print(f"Summary saved to {summary_filename}")
    print(summary)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Script interrupted by user.")
        sys.exit(0)
