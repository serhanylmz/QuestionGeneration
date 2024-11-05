import os
import sys
import logging
import json
import pandas as pd
import random
import csv
import argparse
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt
from typing import Dict, Any
from collections import Counter
from pydantic import BaseModel
from dotenv import load_dotenv

# Import the required functions from the pipeline file
# Note: Since we are not generating questions, you can remove imports related to generation
# from pipeline import generate_basic_question, rank_questions_with_details, generate_answer

# Set up logging
logging.basicConfig(level=logging.INFO,
                    filename='benchmark.log',
                    filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Anthropics client for Claude
import anthropic
claude_client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY"))

# Initialize Cohere client
import cohere
cohere_client = cohere.ClientV2(api_key=os.environ.get("COHERE_API_KEY_PAID"), log_warning_experimental_features=False)

# Initialize Google Generative AI client for Gemini
import google.generativeai as genai
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-pro-002")

# Initialize Hugging Face Inference Client for LLaMA and Qwen
from huggingface_hub import InferenceClient
hf_api_key = os.environ.get("HF_API_KEY")
hf_client = InferenceClient(api_key=hf_api_key)

def check_api_keys():
    required_keys = ["OPENAI_API_KEY", "CLAUDE_API_KEY", "COHERE_API_KEY_PAID", "GEMINI_API_KEY", "HF_API_KEY"]
    missing_keys = [key for key in required_keys if not os.environ.get(key)]
    if missing_keys:
        logger.error(f"Missing API keys: {', '.join(missing_keys)}")
        sys.exit(1)

check_api_keys()

def sanitize_text(text):
    return str(text).strip()

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

judge_temperature = 0.1

PROMPT_TEMPLATE = '''You are an expert judge evaluating the quality of paraphrased questions based on a given context and original question-answer pair. Evaluate their quality and relevance.

Context:
{context}

Original Question: {original_question}
Original Answer: {original_answer}

Question 1: {enhanced_question}
Answer 1: {enhanced_answer}

Question 2: {basic_question}
Answer 2: {basic_answer}

Evaluate Question 1 and Question 2 based on the following criteria:
1. Structural difference from the original question - the question should use different wording and structure while maintaining the core intent
2. Semantic similarity to the original question - despite structural changes, it should preserve the original meaning and seek the same information
3. How well the generated answer aligns with the original answer - answers should capture the same key information

Think about this step by step:

Begin by analyzing the original question to understand its core intent and information being sought. Use this as the reference point for evaluation.

For Question 1:
Structural difference score (out of 10): Analyze how different its structure is from the original question.
Semantic preservation score (out of 10): Evaluate how well it maintains the original meaning.
Answer alignment score (out of 10): Compare its answer with the original answer for information overlap.
Consider these aspects to determine a final score out of 10.

For Question 2:
Structural difference score (out of 10): Analyze structural uniqueness from the original.
Semantic preservation score (out of 10): Evaluate meaning preservation.
Answer alignment score (out of 10): Compare answer alignment with original.
Consider these aspects to determine a final score out of 10.

Compare both questions' scores, considering how well each balances structural novelty with meaning preservation and answer accuracy.

Based on this analysis, provide your answer in JSON format with the following structure:

"question1_score": <number>,
"question2_score": <number>,
"explanation": "<string>",
"winner": "<string>"  // Should be either "Question 1" or "Question 2"

Your response must be a valid JSON object following this exact template. You must pick a winner; it cannot be a draw.
'''

# The judge functions remain unchanged. Make sure not to change any parameters or API usage.
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
                "content": PROMPT_TEMPLATE.format(context=context, original_question=original_question, 
                                                  original_answer=original_answer, enhanced_question=enhanced_question, 
                                                  enhanced_answer=enhanced_answer, basic_question=basic_question, basic_answer=basic_answer)
            }
        ]

        # Call the API with the structured output parameters
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            temperature=judge_temperature,
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
            temperature=judge_temperature,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(context=context, original_question=original_question, 
                                                  original_answer=original_answer, enhanced_question=enhanced_question, 
                                                  enhanced_answer=enhanced_answer, basic_question=basic_question, basic_answer=basic_answer)
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
        prompt = PROMPT_TEMPLATE.format(context=context, original_question=original_question, 
                                                  original_answer=original_answer, enhanced_question=enhanced_question, 
                                                  enhanced_answer=enhanced_answer, basic_question=basic_question, basic_answer=basic_answer)

        result = gemini_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=ComparisonResult,
                temperature=judge_temperature
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
        prompt = PROMPT_TEMPLATE.format(context=context, original_question=original_question, 
                                                  original_answer=original_answer, enhanced_question=enhanced_question, 
                                                  enhanced_answer=enhanced_answer, basic_question=basic_question, basic_answer=basic_answer)

        messages = [{"role": "user", "content": prompt}]
        output = hf_client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=messages,
            temperature=judge_temperature,
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
        prompt = PROMPT_TEMPLATE.format(context=context, original_question=original_question, 
                                                  original_answer=original_answer, enhanced_question=enhanced_question, 
                                                  enhanced_answer=enhanced_answer, basic_question=basic_question, basic_answer=basic_answer)

        messages = [{"role": "user", "content": prompt}]
        output = hf_client.chat.completions.create(
            model="meta-llama/Llama-3.1-70B-Instruct",
            messages=messages,
            temperature=judge_temperature,
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
        context = sanitize_text(entry['Context'])
        original_question = sanitize_text(entry['Original Question'])
        original_answer = sanitize_text(entry['Original Answer'])
        basic_question = sanitize_text(entry['Basic Question'])
        basic_answer = sanitize_text(entry['Basic Answer'])
        enhanced_question = sanitize_text(entry['Enhanced Question'])
        enhanced_answer = sanitize_text(entry['Enhanced Answer'])
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
        'Original Answer': original_answer,
        'Basic Question': basic_question,
        'Basic Answer': basic_answer,
        'Enhanced Question': enhanced_question,
        'Enhanced Answer': enhanced_answer
    })

    # Initialize vote counts
    vote_counts = {"Question 1": 0, "Question 2": 0}

    # Now, feed to the judges with question 1 as basic, question 2 as enhanced
    # So, in the judge functions, we pass:
    # context, original_question, original_answer,
    # basic_question, basic_answer        # Question 1
    # enhanced_question, enhanced_answer  # Question 2

    # Claude
    result_claude = compare_questions_claude(
        context, original_question, original_answer,
        basic_question, basic_answer,        # Question 1
        enhanced_question, enhanced_answer  # Question 2
    )
    result['Claude Verdict'] = result_claude.get('winner', 'Error')
    if result_claude['winner'] in vote_counts:
        vote_counts[result_claude['winner']] += 1

    # Cohere
    result_cohere = compare_questions_cohere(
        context, original_question, original_answer,
        basic_question, basic_answer,        # Question 1
        enhanced_question, enhanced_answer  # Question 2
    )
    result['Cohere Verdict'] = result_cohere.get('winner', 'Error')
    if result_cohere['winner'] in vote_counts:
        vote_counts[result_cohere['winner']] += 1

    # Gemini
    result_gemini = compare_questions_gemini(
        context, original_question, original_answer,
        basic_question, basic_answer,        # Question 1
        enhanced_question, enhanced_answer  # Question 2
    )
    result['Gemini Verdict'] = result_gemini.get('winner', 'Error')
    if result_gemini['winner'] in vote_counts:
        vote_counts[result_gemini['winner']] += 1

    # Qwen
    result_qwen = compare_questions_qwen(
        context, original_question, original_answer,
        basic_question, basic_answer,        # Question 1
        enhanced_question, enhanced_answer  # Question 2
    )
    result['Qwen Verdict'] = result_qwen.get('winner', 'Error')
    if result_qwen['winner'] in vote_counts:
        vote_counts[result_qwen['winner']] += 1

    # LLaMA
    result_llama = compare_questions_llama(
        context, original_question, original_answer,
        basic_question, basic_answer,        # Question 1
        enhanced_question, enhanced_answer  # Question 2
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
            # Map 'Question 1' to 'Basic', 'Question 2' to 'Enhanced'
            final_verdict = 'Basic' if final_verdict_key == 'Question 1' else 'Enhanced'
    except Exception as e:
        logger.error(f"Error determining final verdict: {e}")
        final_verdict = 'Error'
    result['Final Verdict'] = final_verdict

    return result

def main():
    parser = argparse.ArgumentParser(description='Benchmark Script')
    parser.add_argument('--input_file', type=str, default='benchmark_results.csv', help='CSV file to read entries from')
    parser.add_argument('--entries_file', type=str, default='reverse_benchmark_results.csv', help='CSV file to store results')
    args = parser.parse_args()

    # Read the CSV file into a DataFrame
    df = pd.read_csv(args.input_file)

    # Total entries
    total_entries = len(df)

    # Read processed indices from existing entries file
    processed_indices = set()
    if os.path.exists(args.entries_file) and os.stat(args.entries_file).st_size > 0:
        df_existing = pd.read_csv(args.entries_file)
        if 'Index' in df_existing.columns:
            processed_indices = set(df_existing['Index'].tolist())

    # Determine which indices have not been processed yet
    indices_to_process = df[~df['Index'].isin(processed_indices)].index.tolist()
    if not indices_to_process:
        print("All entries have been processed. Exiting.")
        return

    print(f"Total entries to process: {len(indices_to_process)} out of {total_entries}")

    total_vote_counts = Counter()

    fieldnames = ['Index', 'Context', 'Original Question', 'Original Answer', 'Basic Question', 'Basic Answer',
                  'Enhanced Question', 'Enhanced Answer', 'Claude Verdict', 'Cohere Verdict',
                  'Gemini Verdict', 'Qwen Verdict', 'LLaMA Verdict','Final Verdict']

    # Open the CSV file for appending
    with open(args.entries_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file is empty
        if os.stat(args.entries_file).st_size == 0:
            writer.writeheader()

        # Progress bar
        for idx_in_entries, idx_in_dataset in enumerate(
            tqdm(indices_to_process, total=len(indices_to_process), desc='Processing entries')
        ):
            print(f"Processing entry {idx_in_entries + 1}/{len(indices_to_process)} (Dataset Index: {idx_in_dataset})...")
            try:
                entry = df.iloc[idx_in_dataset]
                result = process_entry(entry)
                result['Index'] = entry['Index']  # Add the index to the result
                # Write result to CSV
                writer.writerow(result)
                csvfile.flush()
                os.fsync(csvfile.fileno())
                # Update total vote counts
                for key in ['Question 1', 'Question 2']:
                    total_votes = sum(
                        1 for verdict in [
                            result.get('Claude Verdict'),
                            result.get('Cohere Verdict'),
                            result.get('Gemini Verdict'),
                            result.get('Qwen Verdict'),
                            result.get('LLaMA Verdict')
                        ] if verdict == key
                    )
                    total_vote_counts[key] += total_votes
            except Exception as e:
                logger.error(f"Error processing entry {entry['Index']}: {e}")
                continue

    # Read the CSV file into a DataFrame for summary
    df_results = pd.read_csv(args.entries_file)

    # Generate summary
    total_entries_processed = len(df_results)
    final_verdicts = df_results['Final Verdict'].value_counts()
    percentage_basic = (final_verdicts.get('Basic', 0) / total_entries_processed) * 100 if total_entries_processed > 0 else 0
    percentage_enhanced = (final_verdicts.get('Enhanced', 0) / total_entries_processed) * 100 if total_entries_processed > 0 else 0
    percentage_draw = (final_verdicts.get('Draw', 0) / total_entries_processed) * 100 if total_entries_processed > 0 else 0

    summary = f"""
Total Entries Processed: {total_entries_processed}

Vote Counts:
Question 1 (Basic) Votes: {total_vote_counts['Question 1']}
Question 2 (Enhanced) Votes: {total_vote_counts['Question 2']}

Final Verdicts:
Basic Generation Wins: {final_verdicts.get('Basic', 0)} ({percentage_basic:.2f}%)
Enhanced Generation Wins: {final_verdicts.get('Enhanced', 0)} ({percentage_enhanced:.2f}%)
Draws: {final_verdicts.get('Draw', 0)} ({percentage_draw:.2f}%)
"""

    # Save summary to text file
    summary_filename = 'reverse_benchmark_summary.txt'
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
