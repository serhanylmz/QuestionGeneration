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

# Import the required functions from the paraphrase_system module
from agent import paraphrase_system, paraphrase_text

# python benchmark.py --num_entries 250 --random_seed 42

# Set up logging
logging.basicConfig(level=logging.INFO,
                    filename='benchmark_paraphrase.log',
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
cohere_client = cohere.ClientV2(api_key=os.environ.get("COHERE_API_KEY_PAID"), log_warning_experimental_features=False)

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
    required_keys = ["OPENAI_API_KEY", "CLAUDE_API_KEY", "COHERE_API_KEY_PAID", "GEMINI_API_KEY", "HF_API_KEY"]
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

# Compare functions for each LLM

@safe_api_call
def compare_paraphrases_gpt4o(original_text: str, paraphrase1: str, paraphrase2: str) -> Dict[str, Any]:
    try:
        response = gpt_client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert in evaluating paraphrased texts."},
                {"role": "user", "content": f"""You are an expert in evaluating paraphrased texts.

Compare the following two paraphrases of the given original text. Evaluate their quality and relevance.

Original Text: {original_text}

Paraphrase 1: {paraphrase1}

Paraphrase 2: {paraphrase2}

Evaluate Paraphrase 1 and Paraphrase 2 based on the following criteria:
1. Fluency
2. Clarity
3. Semantic preservation (how well the meaning is preserved)
4. Coherence
5. Grammar

Score each paraphrase on a scale of 0 to 10.

Provide your answer in JSON format with the following structure:

"paraphrase1_score": <number>,
"paraphrase2_score": <number>,
"explanation": "<string>",
"winner": "<string>"  // Should be either "Paraphrase 1" or "Paraphrase 2", you must pick a winner, it cannot be a draw.

Ensure that your response can be parsed as valid JSON.
"""}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "paraphrase_comparison_evaluator",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "paraphrase1_score": {"type": "number"},
                            "paraphrase2_score": {"type": "number"},
                            "explanation": {"type": "string"},
                            "winner": {"type": "string", "enum": ["Paraphrase 1", "Paraphrase 2"]}
                        },
                        "required": ["paraphrase1_score", "paraphrase2_score", "explanation", "winner"],
                        "additionalProperties": False
                    }
                }
            }
        )
        json_response = response.choices[0].message.content
        parsed_response = json.loads(json_response)
        return parsed_response
    except Exception as e:
        logger.error(f"Error in comparing paraphrases with GPT-4o: {e}")
        return {"paraphrase1_score": 0, "paraphrase2_score": 0, "explanation": "Failed to compare paraphrases", "winner": "None"}

@safe_api_call
def compare_paraphrases_claude(original_text: str, paraphrase1: str, paraphrase2: str) -> Dict[str, Any]:
    try:
        prompt = f"""You are an expert in evaluating paraphrased texts.

Compare the following two paraphrases of the given original text. Evaluate their quality and relevance.

Original Text: {original_text}

Paraphrase 1: {paraphrase1}

Paraphrase 2: {paraphrase2}

Evaluate Paraphrase 1 and Paraphrase 2 based on the following criteria:
1. Fluency
2. Clarity
3. Semantic preservation (how well the meaning is preserved)
4. Coherence
5. Grammar

Score each paraphrase on a scale of 0 to 10.

Provide your answer in JSON format with the following structure:

"paraphrase1_score": <number>,
"paraphrase2_score": <number>,
"explanation": "<string>",
"winner": "<string>"  // Should be either "Paraphrase 1" or "Paraphrase 2", you must pick a winner, it cannot be a draw.

Ensure that your response can be parsed as valid JSON.
"""

        response = claude_client.completions.create(
            model="claude-3-5-sonnet-20240620",
            prompt=prompt,
            max_tokens_to_sample=1024,
            stop_sequences=[]
        )

        response_text = response.completion.strip()
        parsed_response = safe_json_parse(response_text)
        if parsed_response is None:
            raise ValueError("Failed to parse JSON response")
        return parsed_response

    except Exception as e:
        logger.error(f"Error in comparing paraphrases with Claude: {e}")
        return {
            "paraphrase1_score": 0,
            "paraphrase2_score": 0,
            "explanation": "Failed to compare paraphrases",
            "winner": "None"
        }

@safe_api_call
def compare_paraphrases_cohere(original_text: str, paraphrase1: str, paraphrase2: str) -> Dict[str, Any]:
    try:
        messages = [
            {
                "role": "user",
                "content": f"""You are an expert in evaluating paraphrased texts.

Compare the following two paraphrases of the given original text. Evaluate their quality and relevance.

Original Text: {original_text}

Paraphrase 1: {paraphrase1}

Paraphrase 2: {paraphrase2}

Evaluate Paraphrase 1 and Paraphrase 2 based on the following criteria:
1. Fluency
2. Clarity
3. Semantic preservation (how well the meaning is preserved)
4. Coherence
5. Grammar

Score each paraphrase on a scale of 0 to 10.

Provide your answer in JSON format with the following structure:

"paraphrase1_score": <number>,
"paraphrase2_score": <number>,
"explanation": "<string>",
"winner": "<string>"  // Should be either "Paraphrase 1" or "Paraphrase 2", you must pick a winner, it cannot be a draw.

Ensure that your response can be parsed as valid JSON.
"""
            }
        ]
        response = cohere_client.chat(
            model="command-r-plus-08-2024",
            messages=messages,
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "required": ["paraphrase1_score", "paraphrase2_score", "explanation", "winner"],
                    "properties": {
                        "paraphrase1_score": {"type": "number"},
                        "paraphrase2_score": {"type": "number"},
                        "explanation": {"type": "string"},
                        "winner": {"type": "string", "enum": ["Paraphrase 1", "Paraphrase 2"]}
                    },
                },
            },
        )
        json_response = response.message.text.strip()
        parsed_response = safe_json_parse(json_response)
        if parsed_response is None:
            raise ValueError("Failed to parse JSON response")
        return parsed_response

    except Exception as e:
        logger.error(f"Error in comparing paraphrases with Cohere: {e}")
        return {"paraphrase1_score": 0, "paraphrase2_score": 0, "explanation": "Failed to compare paraphrases", "winner": "None"}

@safe_api_call
def compare_paraphrases_gemini(original_text: str, paraphrase1: str, paraphrase2: str) -> Dict[str, Any]:
    try:
        prompt = f"""You are an expert in evaluating paraphrased texts.

Compare the following two paraphrases of the given original text. Evaluate their quality and relevance.

Original Text: {original_text}

Paraphrase 1: {paraphrase1}

Paraphrase 2: {paraphrase2}

Evaluate Paraphrase 1 and Paraphrase 2 based on the following criteria:
1. Fluency
2. Clarity
3. Semantic preservation (how well the meaning is preserved)
4. Coherence
5. Grammar

Score each paraphrase on a scale of 0 to 10.

Provide your answer in JSON format with the following structure:

"paraphrase1_score": <number>,
"paraphrase2_score": <number>,
"explanation": "<string>",
"winner": "<string>"  // Should be either "Paraphrase 1" or "Paraphrase 2", you must pick a winner, it cannot be a draw.

Ensure that your response can be parsed as valid JSON.
"""

        result = gemini_model.generate_text(
            prompt=prompt,
            max_output_tokens=1024,
            temperature=0.5,
            top_p=0.9
        )

        response_text = result.text.strip()
        parsed_response = safe_json_parse(response_text)
        if parsed_response is None:
            raise ValueError("Failed to parse JSON response")
        return parsed_response

    except Exception as e:
        logger.error(f"Error in comparing paraphrases with Gemini: {e}")
        return {"paraphrase1_score": 0, "paraphrase2_score": 0, "explanation": "Failed to compare paraphrases", "winner": "None"}

@safe_api_call
def compare_paraphrases_qwen(original_text: str, paraphrase1: str, paraphrase2: str) -> Dict[str, Any]:
    try:
        prompt = f"""You are an expert in evaluating paraphrased texts.

Compare the following two paraphrases of the given original text. Evaluate their quality and relevance.

Original Text: {original_text}

Paraphrase 1: {paraphrase1}

Paraphrase 2: {paraphrase2}

Evaluate Paraphrase 1 and Paraphrase 2 based on the following criteria:
1. Fluency
2. Clarity
3. Semantic preservation (how well the meaning is preserved)
4. Coherence
5. Grammar

Score each paraphrase on a scale of 0 to 10.

Provide your answer in JSON format with the following structure:

"paraphrase1_score": <number>,
"paraphrase2_score": <number>,
"explanation": "<string>",
"winner": "<string>"  // Should be either "Paraphrase 1" or "Paraphrase 2", you must pick a winner, it cannot be a draw.

Ensure that your response can be parsed as valid JSON.
"""

        messages = [{"role": "user", "content": prompt}]
        output = hf_client.chat(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=messages,
            max_new_tokens=1024,
            temperature=0.5,
            top_p=0.9
        )

        response_text = output["generated_text"].strip()
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
        logger.error(f"Error in comparing paraphrases with Qwen: {e}")
        return {"paraphrase1_score": 0, "paraphrase2_score": 0, "explanation": "Failed to compare paraphrases", "winner": "None"}

@safe_api_call
def compare_paraphrases_llama(original_text: str, paraphrase1: str, paraphrase2: str) -> Dict[str, Any]:
    try:
        prompt = f"""You are an expert in evaluating paraphrased texts.

Compare the following two paraphrases of the given original text. Evaluate their quality and relevance.

Original Text: {original_text}

Paraphrase 1: {paraphrase1}

Paraphrase 2: {paraphrase2}

Evaluate Paraphrase 1 and Paraphrase 2 based on the following criteria:
1. Fluency
2. Clarity
3. Semantic preservation (how well the meaning is preserved)
4. Coherence
5. Grammar

Score each paraphrase on a scale of 0 to 10.

Provide your answer in JSON format with the following structure:

"paraphrase1_score": <number>,
"paraphrase2_score": <number>,
"explanation": "<string>",
"winner": "<string>"  // Should be either "Paraphrase 1" or "Paraphrase 2", you must pick a winner, it cannot be a draw.

Ensure that your response can be parsed as valid JSON.
"""

        messages = [{"role": "user", "content": prompt}]
        output = hf_client.chat(
            model="meta-llama/Llama-3.1-70B-Instruct",
            messages=messages,
            max_new_tokens=1024,
            temperature=0.5,
            top_p=0.9
        )

        response_text = output["generated_text"].strip()
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
        logger.error(f"Error in comparing paraphrases with LLaMA: {e}")
        return {"paraphrase1_score": 0, "paraphrase2_score": 0, "explanation": "Failed to compare paraphrases", "winner": "None"}

# Define paraphrase_basic

def paraphrase_basic(input_text: str) -> str:
    logs = []
    generator_prompt = "You are a helpful assistant that paraphrases text while maintaining the original meaning."
    paraphrased_text = paraphrase_text(input_text, generator_prompt, logs)
    return paraphrased_text

# Define process_entry function

def process_entry(entry):
    result = {}

    # Extract the text to paraphrase
    try:
        original_text = sanitize_text(entry['context'])
    except Exception as e:
        logger.error(f"Error extracting data from entry: {e}")
        result.update({
            'Error': 'Failed to extract data from entry',
            'Original Text': None,
            'Basic Paraphrase': None,
            'Agentic Paraphrase': None,
            'GPT-4o Verdict': None,
            'Claude Verdict': None,
            'Cohere Verdict': None,
            'Gemini Verdict': None,
            'Qwen Verdict': None,
            'LLaMA Verdict': None,
            'Final Verdict': None
        })
        return result

    result['Original Text'] = original_text

    # Generate basic paraphrase
    try:
        basic_paraphrase = paraphrase_basic(original_text)
    except Exception as e:
        logger.error(f"Error generating basic paraphrase: {e}")
        basic_paraphrase = 'Error generating basic paraphrase'
    result['Basic Paraphrase'] = basic_paraphrase

    # Generate agentic paraphrase
    try:
        agentic_paraphrase = paraphrase_system(original_text)
    except Exception as e:
        logger.error(f"Error generating agentic paraphrase: {e}")
        agentic_paraphrase = 'Error generating agentic paraphrase'
    result['Agentic Paraphrase'] = agentic_paraphrase

    # Initialize vote counts
    vote_counts = {"Paraphrase 1": 0, "Paraphrase 2": 0}

    # Collect comparison results from each LLM judge

    # GPT-4o
    result_gpt4o = compare_paraphrases_gpt4o(
        original_text,
        agentic_paraphrase,
        basic_paraphrase
    )
    result['GPT-4o Verdict'] = result_gpt4o.get('winner', 'Error')
    if result_gpt4o['winner'] in vote_counts:
        vote_counts[result_gpt4o['winner']] += 1

    # Claude
    result_claude = compare_paraphrases_claude(
        original_text,
        agentic_paraphrase,
        basic_paraphrase
    )
    result['Claude Verdict'] = result_claude.get('winner', 'Error')
    if result_claude['winner'] in vote_counts:
        vote_counts[result_claude['winner']] += 1

    # Cohere
    result_cohere = compare_paraphrases_cohere(
        original_text,
        agentic_paraphrase,
        basic_paraphrase
    )
    result['Cohere Verdict'] = result_cohere.get('winner', 'Error')
    if result_cohere['winner'] in vote_counts:
        vote_counts[result_cohere['winner']] += 1

    # Gemini
    result_gemini = compare_paraphrases_gemini(
        original_text,
        agentic_paraphrase,
        basic_paraphrase
    )
    result['Gemini Verdict'] = result_gemini.get('winner', 'Error')
    if result_gemini['winner'] in vote_counts:
        vote_counts[result_gemini['winner']] += 1

    # Qwen
    result_qwen = compare_paraphrases_qwen(
        original_text,
        agentic_paraphrase,
        basic_paraphrase
    )
    result['Qwen Verdict'] = result_qwen.get('winner', 'Error')
    if result_qwen['winner'] in vote_counts:
        vote_counts[result_qwen['winner']] += 1

    # LLaMA
    result_llama = compare_paraphrases_llama(
        original_text,
        agentic_paraphrase,
        basic_paraphrase
    )
    result['LLaMA Verdict'] = result_llama.get('winner', 'Error')
    if result_llama['winner'] in vote_counts:
        vote_counts[result_llama['winner']] += 1

    # Determine final verdict
    try:
        if vote_counts['Paraphrase 1'] > vote_counts['Paraphrase 2']:
            final_verdict = 'Agentic'
        elif vote_counts['Paraphrase 1'] < vote_counts['Paraphrase 2']:
            final_verdict = 'Basic'
        else:
            final_verdict = 'Draw'
    except Exception as e:
        logger.error(f"Error determining final verdict: {e}")
        final_verdict = 'Error'
    result['Final Verdict'] = final_verdict

    return result

def main():
    parser = argparse.ArgumentParser(description='Benchmark Paraphrase Script')
    parser.add_argument('--num_entries', type=str, default='all', help='Number of entries to process')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--entries_file', type=str, default='benchmark_paraphrase_results.csv', help='CSV file to store results')
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

    fieldnames = ['Index', 'Original Text', 'Basic Paraphrase', 'Agentic Paraphrase',
                  'GPT-4o Verdict', 'Claude Verdict', 'Cohere Verdict',
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
                for key in ['Paraphrase 1', 'Paraphrase 2']:
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
    percentage_agentic = (final_verdicts.get('Agentic', 0) / total_entries_processed) * 100 if total_entries_processed > 0 else 0
    percentage_draw = (final_verdicts.get('Draw', 0) / total_entries_processed) * 100 if total_entries_processed > 0 else 0

    summary = f"""
Total Entries Processed: {total_entries_processed}

Vote Counts:
Paraphrase 1 (Agentic) Votes: {total_vote_counts['Paraphrase 1']}
Paraphrase 2 (Basic) Votes: {total_vote_counts['Paraphrase 2']}

Final Verdicts:
Agentic Paraphrase Wins: {final_verdicts.get('Agentic', 0)} ({percentage_agentic:.2f}%)
Basic Paraphrase Wins: {final_verdicts.get('Basic', 0)} ({percentage_basic:.2f}%)
Draws: {final_verdicts.get('Draw', 0)} ({percentage_draw:.2f}%)
"""

    # Save summary to text file
    summary_filename = 'benchmark_paraphrase_summary.txt'
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
