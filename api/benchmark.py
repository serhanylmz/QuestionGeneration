import os
import logging
import json
import pandas as pd
from datasets import load_dataset
import random
from typing import List, Dict
from dotenv import load_dotenv
import asyncio
from collections import Counter
import typing_extensions as typing

# Import the required functions from the pipeline file
from pipeline_gradio_experimental import generate_basic_question, rank_questions_with_details, generate_answer

# Set up logging
logging.basicConfig(level=logging.INFO)
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

def get_entries(start_index, num_entries):
    if num_entries == 'all':
        return dataset['train'].select(range(start_index, len(dataset['train'])))
    else:
        end_index = start_index + int(num_entries)
        return dataset['train'].select(range(start_index, end_index))


def compare_questions_gpt4o(context: str, original_question: str, original_answer: str,
                            basic_question: str, basic_answer: str,
                            enhanced_question: str, enhanced_answer: str) -> Dict[str, any]:
    try:
        response = gpt_client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert in evaluating question-answer pairs based on a given context."},
                {"role": "user", "content": f"""Compare the following two generated question-answer pairs based on the given context and the original question-answer pair. Evaluate their quality and relevance.

Context: {context}

Original Question: {original_question}
Original Answer: {original_answer}

Basic Generated Question: {basic_question}
Basic Generated Answer: {basic_answer}

Enhanced Generated Question: {enhanced_question}
Enhanced Generated Answer: {enhanced_answer}

Evaluate the basic and enhanced generated questions based on the following criteria:
1. Structural difference from the original question
2. Semantic similarity to the original question
3. How well the generated answer matches the original answer

Score each generated question-answer pair on a scale of 0 to 10. Provide a detailed explanation for your evaluation, addressing each of the criteria mentioned above. Finally, determine which generation approach (Basic or Enhanced) is better overall and explain why."""}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "question_comparison_evaluator",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "basic_score": {"type": "number"},
                            "enhanced_score": {"type": "number"},
                            "explanation": {"type": "string"},
                            "winner": {"type": "string", "enum": ["Basic", "Enhanced"]}
                        },
                        "required": ["basic_score", "enhanced_score", "explanation", "winner"],
                        "additionalProperties": False
                    }
                }
            }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error in comparing questions with GPT-4o: {e}")
        return {"basic_score": 0, "enhanced_score": 0, "explanation": "Failed to compare questions", "winner": "None"}

def compare_questions_claude(context: str, original_question: str, original_answer: str,
                             basic_question: str, basic_answer: str,
                             enhanced_question: str, enhanced_answer: str) -> Dict[str, any]:
    try:
        # Define the tool (function) with the expected output schema
        tool = {
            "name": "question_comparison_evaluator",
            "description": "Evaluate and compare two generated question-answer pairs and output the result in structured JSON.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "basic_score": {"type": "number"},
                    "enhanced_score": {"type": "number"},
                    "explanation": {"type": "string"},
                    "winner": {"type": "string", "enum": ["Basic", "Enhanced"]}
                },
                "required": ["basic_score", "enhanced_score", "explanation", "winner"],
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

Basic Generated Question: {basic_question}
Basic Generated Answer: {basic_answer}

Enhanced Generated Question: {enhanced_question}
Enhanced Generated Answer: {enhanced_answer}

Evaluate the basic and enhanced generated questions based on the following criteria:
1. Structural difference from the original question
2. Semantic similarity to the original question
3. How well the generated answer matches the original answer

Score each generated question-answer pair on a scale of 0 to 10.

Finally, determine which generation approach (Basic or Enhanced) is better overall.

Provide your answer using the 'question_comparison_evaluator' tool, and output the result in structured JSON format."""
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
            "basic_score": 0,
            "enhanced_score": 0,
            "explanation": "Failed to compare questions",
            "winner": "None"
        }


def compare_questions_cohere(context: str, original_question: str, original_answer: str,
                             basic_question: str, basic_answer: str,
                             enhanced_question: str, enhanced_answer: str) -> Dict[str, any]:
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

Basic Generated Question: {basic_question}
Basic Generated Answer: {basic_answer}

Enhanced Generated Question: {enhanced_question}
Enhanced Generated Answer: {enhanced_answer}

Evaluate the basic and enhanced generated questions based on the following criteria:
1. Structural difference from the original question
2. Semantic similarity to the original question
3. How well the generated answer matches the original answer

Score each generated question-answer pair on a scale of 0 to 10. Provide a detailed explanation for your evaluation, addressing each of the criteria mentioned above. Finally, determine which generation approach (Basic or Enhanced) is better overall and explain why.
Provide your answer in JSON format with the following structure:

"basic_score": <integer>,
"enhanced_score": <integer>,
"explanation": <string>,
"winner": <string>

Ensure that your response can be parsed as valid JSON.
"""
                }
            ],
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "required": ["basic_score", "enhanced_score", "explanation", "winner"],
                    "properties": {
                        "basic_score": {"type": "number"},
                        "enhanced_score": {"type": "number"},
                        "explanation": {"type": "string"},
                        "winner": {"type": "string", "enum": ["Basic", "Enhanced"]}
                    },
                },
            },
        )

        json_response = res.message.content[0].text.strip()
        parsed_response = json.loads(json_response)
        return parsed_response

    except Exception as e:
        logger.error(f"Error in comparing questions with Cohere: {e}")
        return {"basic_score": 0, "enhanced_score": 0, "explanation": "Failed to compare questions", "winner": "None"}


def compare_questions_gemini(context: str, original_question: str, original_answer: str,
                             basic_question: str, basic_answer: str,
                             enhanced_question: str, enhanced_answer: str) -> Dict[str, any]:
    try:

        class ComparisonResult(typing.TypedDict):
            basic_score: float
            enhanced_score: float
            explanation: str
            winner: str

        prompt = f"""You are an expert in evaluating question-answer pairs based on a given context.

Compare the following two generated question-answer pairs based on the given context and the original question-answer pair. Evaluate their quality and relevance.

Context: {context}

Original Question: {original_question}
Original Answer: {original_answer}

Basic Generated Question: {basic_question}
Basic Generated Answer: {basic_answer}

Enhanced Generated Question: {enhanced_question}
Enhanced Generated Answer: {enhanced_answer}

Evaluate the basic and enhanced generated questions based on the following criteria:
1. Structural difference from the original question
2. Semantic similarity to the original question
3. How well the generated answer matches the original answer

Score each generated question-answer pair on a scale of 0 to 10. Provide a detailed explanation for your evaluation, addressing each of the criteria mentioned above. Finally, determine which generation approach (Basic or Enhanced) is better overall and explain why."""

        result = gemini_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=ComparisonResult
            ),
        )
        json_response = result.text.strip()
        parsed_response = json.loads(json_response)
        parsed_response["winner"] = parsed_response["winner"].capitalize()
        return parsed_response
    except Exception as e:
        logger.error(f"Error in comparing questions with Gemini: {e}")
        return {"basic_score": 0, "enhanced_score": 0, "explanation": "Failed to compare questions", "winner": "None"}

def compare_questions_qwen(context: str, original_question: str, original_answer: str,
                           basic_question: str, basic_answer: str,
                           enhanced_question: str, enhanced_answer: str) -> Dict[str, any]:
    try:
        prompt = f"""You are an expert in evaluating question-answer pairs based on a given context.

Compare the following two generated question-answer pairs based on the given context and the original question-answer pair. Evaluate their quality and relevance.

Context: {context}

Original Question: {original_question}
Original Answer: {original_answer}

Basic Generated Question: {basic_question}
Basic Generated Answer: {basic_answer}

Enhanced Generated Question: {enhanced_question}
Enhanced Generated Answer: {enhanced_answer}

Evaluate the basic and enhanced generated questions based on the following criteria:
1. Structural difference from the original question
2. Semantic similarity to the original question
3. How well the generated answer matches the original answer

Score each generated question-answer pair on a scale of 0 to 10. Provide a detailed explanation for your evaluation, addressing each of the criteria mentioned above. Finally, determine which generation approach (Basic or Enhanced) is better overall and explain why.

Provide your answer in JSON format with the following structure:

"basic_score": <integer>,
"enhanced_score": <integer>,
"explanation": <string>,
"winner": <string>

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
        response_text = output.choices[0].message.content
        # print("Raw response:", repr(response_text))

        # Function to extract JSON content
        import re
        def extract_json_content(text):
            pattern = r"\{.*\}"
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return matches[0]
            else:
                return text.strip()

        json_response = extract_json_content(response_text)
        # print("Extracted JSON:", json_response)
        parsed_response = json.loads(json_response)
        return parsed_response
    except Exception as e:
        logger.error(f"Error in comparing questions with Qwen: {e}")
        return {"basic_score": 0, "enhanced_score": 0, "explanation": "Failed to compare questions", "winner": "None"}


def process_entry(entry):
    result = {}  # Initialize the result dict

    # Extract data from entry
    try:
        context = entry['context']
        answer = entry['answers']['text'][0]
        initial_question = entry['question']
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
            'Final Verdict': None
        })
        return result

    result.update({
        'Context': context,
        'Original Question': initial_question,
        'Original Answer': answer
    })

    # Generate basic question
    try:
        basic_question = generate_basic_question(context, answer, initial_question)
    except Exception as e:
        logger.error(f"Error generating basic question: {e}")
        basic_question = 'Error generating basic question'
    result['Basic Question'] = basic_question

    # Generate enhanced question
    try:
        detailed_scores, rankings, enhanced_question = rank_questions_with_details(context, answer, initial_question)
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
    vote_counts = {"Basic": 0, "Enhanced": 0}

    # Collect comparison results from each LLM judge
    comparison_results = {}

    # GPT-4o
    try:
        result_gpt4o = compare_questions_gpt4o(
            context, initial_question, answer,
            basic_question, basic_answer,
            enhanced_question, enhanced_answer
        )
    except Exception as e:
        logger.error(f"Error in GPT-4o comparison: {e}")
        result_gpt4o = {
            'winner': 'Error',
            'basic_score': 0,
            'enhanced_score': 0,
            'explanation': 'Error in GPT-4o comparison'
        }
    result['GPT-4o Verdict'] = result_gpt4o.get('winner', 'Error')
    if result_gpt4o['winner'] in vote_counts:
        vote_counts[result_gpt4o['winner']] += 1

    # Claude
    try:
        result_claude = compare_questions_claude(
            context, initial_question, answer,
            basic_question, basic_answer,
            enhanced_question, enhanced_answer
        )
    except Exception as e:
        logger.error(f"Error in Claude comparison: {e}")
        result_claude = {
            'winner': 'Error',
            'basic_score': 0,
            'enhanced_score': 0,
            'explanation': 'Error in Claude comparison'
        }
    result['Claude Verdict'] = result_claude.get('winner', 'Error')
    if result_claude['winner'] in vote_counts:
        vote_counts[result_claude['winner']] += 1

    # Cohere
    try:
        result_cohere = compare_questions_cohere(
            context, initial_question, answer,
            basic_question, basic_answer,
            enhanced_question, enhanced_answer
        )
    except Exception as e:
        logger.error(f"Error in Cohere comparison: {e}")
        result_cohere = {
            'winner': 'Error',
            'basic_score': 0,
            'enhanced_score': 0,
            'explanation': 'Error in Cohere comparison'
        }
    result['Cohere Verdict'] = result_cohere.get('winner', 'Error')
    if result_cohere['winner'] in vote_counts:
        vote_counts[result_cohere['winner']] += 1

    # Gemini
    try:
        result_gemini = compare_questions_gemini(
            context, initial_question, answer,
            basic_question, basic_answer,
            enhanced_question, enhanced_answer
        )
    except Exception as e:
        logger.error(f"Error in Gemini comparison: {e}")
        result_gemini = {
            'winner': 'Error',
            'basic_score': 0,
            'enhanced_score': 0,
            'explanation': 'Error in Gemini comparison'
        }
    result['Gemini Verdict'] = result_gemini.get('winner', 'Error')
    if result_gemini['winner'] in vote_counts:
        vote_counts[result_gemini['winner']] += 1

    # Qwen
    try:
        result_qwen = compare_questions_qwen(
            context, initial_question, answer,
            basic_question, basic_answer,
            enhanced_question, enhanced_answer
        )
    except Exception as e:
        logger.error(f"Error in Qwen comparison: {e}")
        result_qwen = {
            'winner': 'Error',
            'basic_score': 0,
            'enhanced_score': 0,
            'explanation': 'Error in Qwen comparison'
        }
    result['Qwen Verdict'] = result_qwen.get('winner', 'Error')
    if result_qwen['winner'] in vote_counts:
        vote_counts[result_qwen['winner']] += 1

    # Determine final verdict
    try:
        final_verdict = max(vote_counts, key=vote_counts.get)
        if vote_counts['Basic'] == vote_counts['Enhanced']:
            final_verdict = 'Draw'
    except Exception as e:
        logger.error(f"Error determining final verdict: {e}")
        final_verdict = 'Error'
    result['Final Verdict'] = final_verdict

    return result


def main():
    num_entries = input("Enter the number of entries to test on (or 'all' to process to the end of the dataset): ")
    start_index = int(input("Enter the starting index: "))

    entries = get_entries(start_index, num_entries)
    results = []
    total_vote_counts = Counter()

    for idx_in_entries, entry in enumerate(entries):
        idx_in_dataset = start_index + idx_in_entries
        print(f"Processing entry {idx_in_dataset+1}/{len(dataset['train'])} (Entry {idx_in_entries+1}/{len(entries)})...")
        try:
            result = process_entry(entry)
            results.append(result)
            # Update total vote counts
            for key in ['Basic', 'Enhanced']:
                total_votes = sum(
                    1 for verdict in [
                        result.get('GPT-4o Verdict'),
                        result.get('Claude Verdict'),
                        result.get('Cohere Verdict'),
                        result.get('Gemini Verdict'),
                        result.get('Qwen Verdict')
                    ] if verdict == key
                )
                total_vote_counts[key] += total_votes
        except Exception as e:
            logger.error(f"Error processing entry {idx_in_dataset}: {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to Excel
    excel_filename = 'benchmark_results.xlsx'
    df.to_excel(excel_filename, index=False)
    print(f"Results saved to {excel_filename}")

    # Generate summary
    total_entries = len(results)
    final_verdicts = df['Final Verdict'].value_counts()
    percentage_basic = (final_verdicts.get('Basic', 0) / total_entries) * 100 if total_entries > 0 else 0
    percentage_enhanced = (final_verdicts.get('Enhanced', 0) / total_entries) * 100 if total_entries > 0 else 0
    percentage_draw = (final_verdicts.get('Draw', 0) / total_entries) * 100 if total_entries > 0 else 0

    summary = f"""
Total Entries Processed: {total_entries}

Vote Counts:
Basic Generation Votes: {total_vote_counts['Basic']}
Enhanced Generation Votes: {total_vote_counts['Enhanced']}

Final Verdicts:
Basic Generation Wins: {final_verdicts.get('Basic', 0)} ({percentage_basic:.2f}%)
Enhanced Generation Wins: {final_verdicts.get('Enhanced', 0)} ({percentage_enhanced:.2f}%)
Draws: {final_verdicts.get('Draw', 0)} ({percentage_draw:.2f}%)
"""

    # Save summary to text file
    summary_filename = 'benchmark_summary.txt'
    with open(summary_filename, 'w') as f:
        f.write(summary)
    print(f"Summary saved to {summary_filename}")
    print(summary)


if __name__ == "__main__":
    main()
