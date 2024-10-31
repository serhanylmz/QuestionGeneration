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
import datetime

# Import the required functions from your paraphrasing system
# Make sure to import your paraphrasing_system.py or include its code here
from new import paraphrase_system

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

# Initialize Hugging Face Inference Client for Qwen
from huggingface_hub import InferenceClient
hf_api_key = os.environ.get("HF_API_KEY")
hf_client = InferenceClient(api_token=hf_api_key)

# Load the SQuAD dataset
dataset = load_dataset("squad")  # Note: Updated to "squad" dataset for compatibility

def get_random_entries(num_entries, random_seed):
    dataset_size = len(dataset['train'])
    if num_entries == 'all':
        return dataset['train']
    else:
        num_entries = int(num_entries)
        random.seed(random_seed)
        indices = random.sample(range(dataset_size), num_entries)
        return dataset['train'].select(indices)

def compare_questions_gpt4o(context: str, original_question: str, original_answer: str,
                            paraphrased_question: str, paraphrased_answer: str) -> Dict[str, any]:
    try:
        response = gpt_client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert in evaluating question-answer pairs based on a given context."},
                {"role": "user", "content": f"""Compare the following two question-answer pairs based on the given context. Evaluate their quality and relevance.

Context: {context}

Question A: {original_question}
Answer A: {original_answer}

Question B: {paraphrased_question}
Answer B: {paraphrased_answer}

Evaluate both questions based on the following criteria:
1. Structural difference from each other
2. Semantic similarity
3. How well the answers match each other

Score each question-answer pair on a scale of 0 to 10 for each criterion. Provide a detailed explanation for your evaluation, addressing each of the criteria mentioned above. Finally, determine which question (Question A or Question B) is better overall and explain why.

Provide your answer in JSON format with the following structure:

{{
    "question_a_scores": {{
        "structural_difference": <number>,
        "semantic_similarity": <number>,
        "answer_match": <number>
    }},
    "question_b_scores": {{
        "structural_difference": <number>,
        "semantic_similarity": <number>,
        "answer_match": <number>
    }},
    "explanation": "<string>",
    "winner": "<string>"
}}

Ensure that your response can be parsed as valid JSON."""}
            ],
            temperature=0.5,
            max_tokens=1024,
        )
        response_text = response.choices[0].message.content.strip()
        parsed_response = json.loads(response_text)
        return parsed_response
    except Exception as e:
        logger.error(f"Error in comparing questions with GPT-4o: {e}")
        return {
            "question_a_scores": {"structural_difference": 0, "semantic_similarity": 0, "answer_match": 0},
            "question_b_scores": {"structural_difference": 0, "semantic_similarity": 0, "answer_match": 0},
            "explanation": "Failed to compare questions",
            "winner": "None"
        }

def compare_questions_claude(context: str, original_question: str, original_answer: str,
                             paraphrased_question: str, paraphrased_answer: str) -> Dict[str, any]:
    try:
        prompt = f"""You are an expert in evaluating question-answer pairs based on a given context.

Compare the following two question-answer pairs based on the given context. Evaluate their quality and relevance.

Context: {context}

Question A: {original_question}
Answer A: {original_answer}

Question B: {paraphrased_question}
Answer B: {paraphrased_answer}

Evaluate both questions based on the following criteria:
1. Structural difference from each other
2. Semantic similarity
3. How well the answers match each other

Score each question-answer pair on a scale of 0 to 10 for each criterion. Provide a detailed explanation for your evaluation, addressing each of the criteria mentioned above. Finally, determine which question (Question A or Question B) is better overall and explain why.

Provide your answer in JSON format with the following structure:

{{
    "question_a_scores": {{
        "structural_difference": <number>,
        "semantic_similarity": <number>,
        "answer_match": <number>
    }},
    "question_b_scores": {{
        "structural_difference": <number>,
        "semantic_similarity": <number>,
        "answer_match": <number>
    }},
    "explanation": "<string>",
    "winner": "<string>"
}}

Ensure that your response can be parsed as valid JSON."""

        response = claude_client.completions.create(
            model="claude-3-5-sonnet-20240620",
            prompt=prompt,
            max_tokens_to_sample=1024,
            temperature=0.5,
            stop_sequences=[],
        )

        assistant_message = response.completion.strip()
        parsed_response = json.loads(assistant_message)
        return parsed_response

    except Exception as e:
        logger.error(f"Error in comparing questions with Claude: {e}")
        return {
            "question_a_scores": {"structural_difference": 0, "semantic_similarity": 0, "answer_match": 0},
            "question_b_scores": {"structural_difference": 0, "semantic_similarity": 0, "answer_match": 0},
            "explanation": "Failed to compare questions",
            "winner": "None"
        }

def compare_questions_cohere(context: str, original_question: str, original_answer: str,
                             paraphrased_question: str, paraphrased_answer: str) -> Dict[str, any]:
    try:
        prompt = f"""You are an expert in evaluating question-answer pairs based on a given context.

Compare the following two question-answer pairs based on the given context. Evaluate their quality and relevance.

Context: {context}

Question A: {original_question}
Answer A: {original_answer}

Question B: {paraphrased_question}
Answer B: {paraphrased_answer}

Evaluate both questions based on the following criteria:
1. Structural difference from each other
2. Semantic similarity
3. How well the answers match each other

Score each question-answer pair on a scale of 0 to 10 for each criterion. Provide a detailed explanation for your evaluation, addressing each of the criteria mentioned above. Finally, determine which question (Question A or Question B) is better overall and explain why.

Provide your answer in JSON format with the following structure:

{{
    "question_a_scores": {{
        "structural_difference": <number>,
        "semantic_similarity": <number>,
        "answer_match": <number>
    }},
    "question_b_scores": {{
        "structural_difference": <number>,
        "semantic_similarity": <number>,
        "answer_match": <number>
    }},
    "explanation": "<string>",
    "winner": "<string>"
}}

Ensure that your response can be parsed as valid JSON."""

        response = cohere_client.chat(
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.5,
        )

        assistant_message = response.content.strip()
        parsed_response = json.loads(assistant_message)
        return parsed_response
    except Exception as e:
        logger.error(f"Error in comparing questions with Cohere: {e}")
        return {
            "question_a_scores": {"structural_difference": 0, "semantic_similarity": 0, "answer_match": 0},
            "question_b_scores": {"structural_difference": 0, "semantic_similarity": 0, "answer_match": 0},
            "explanation": "Failed to compare questions",
            "winner": "None"
        }

def compare_questions_gemini(context: str, original_question: str, original_answer: str,
                             paraphrased_question: str, paraphrased_answer: str) -> Dict[str, any]:
    try:
        prompt = f"""You are an expert in evaluating question-answer pairs based on a given context.

Compare the following two question-answer pairs based on the given context. Evaluate their quality and relevance.

Context: {context}

Question A: {original_question}
Answer A: {original_answer}

Question B: {paraphrased_question}
Answer B: {paraphrased_answer}

Evaluate both questions based on the following criteria:
1. Structural difference from each other
2. Semantic similarity
3. How well the answers match each other

Score each question-answer pair on a scale of 0 to 10 for each criterion. Provide a detailed explanation for your evaluation, addressing each of the criteria mentioned above. Finally, determine which question (Question A or Question B) is better overall and explain why.

Provide your answer in JSON format with the following structure:

{{
    "question_a_scores": {{
        "structural_difference": <number>,
        "semantic_similarity": <number>,
        "answer_match": <number>
    }},
    "question_b_scores": {{
        "structural_difference": <number>,
        "semantic_similarity": <number>,
        "answer_match": <number>
    }},
    "explanation": "<string>",
    "winner": "<string>"
}}

Ensure that your response can be parsed as valid JSON."""

        response = genai.generate_text(
            model="chat-bison",
            prompt=prompt,
            temperature=0.5,
            max_output_tokens=1024,
        )

        assistant_message = response.result.strip()
        # Extract JSON from the response
        import re
        json_pattern = re.compile(r'\{.*\}', re.DOTALL)
        match = json_pattern.search(assistant_message)
        if match:
            json_text = match.group(0)
            parsed_response = json.loads(json_text)
        else:
            raise ValueError("No JSON found in the response")

        return parsed_response
    except Exception as e:
        logger.error(f"Error in comparing questions with Gemini: {e}")
        return {
            "question_a_scores": {"structural_difference": 0, "semantic_similarity": 0, "answer_match": 0},
            "question_b_scores": {"structural_difference": 0, "semantic_similarity": 0, "answer_match": 0},
            "explanation": "Failed to compare questions",
            "winner": "None"
        }

def compare_questions_qwen(context: str, original_question: str, original_answer: str,
                           paraphrased_question: str, paraphrased_answer: str) -> Dict[str, any]:
    try:
        prompt = f"""You are an expert in evaluating question-answer pairs based on a given context.

Compare the following two question-answer pairs based on the given context. Evaluate their quality and relevance.

Context: {context}

Question A: {original_question}
Answer A: {original_answer}

Question B: {paraphrased_question}
Answer B: {paraphrased_answer}

Evaluate both questions based on the following criteria:
1. Structural difference from each other
2. Semantic similarity
3. How well the answers match each other

Score each question-answer pair on a scale of 0 to 10 for each criterion. Provide a detailed explanation for your evaluation, addressing each of the criteria mentioned above. Finally, determine which question (Question A or Question B) is better overall and explain why.

Provide your answer in JSON format with the following structure:

{{
    "question_a_scores": {{
        "structural_difference": <number>,
        "semantic_similarity": <number>,
        "answer_match": <number>
    }},
    "question_b_scores": {{
        "structural_difference": <number>,
        "semantic_similarity": <number>,
        "answer_match": <number>
    }},
    "explanation": "<string>",
    "winner": "<string>"
}}

Ensure that your response can be parsed as valid JSON."""

        messages = [{"role": "user", "content": prompt}]
        output = hf_client.chat_completion(
            model="Qwen/Qwen-7B-Chat",
            messages=messages,
            max_new_tokens=1024,
            temperature=0.5,
            top_p=0.9
        )

        assistant_message = output["choices"][0]["message"]["content"].strip()
        # Extract JSON from the response
        import re
        json_pattern = re.compile(r'\{.*\}', re.DOTALL)
        match = json_pattern.search(assistant_message)
        if match:
            json_text = match.group(0)
            parsed_response = json.loads(json_text)
        else:
            raise ValueError("No JSON found in the response")

        return parsed_response
    except Exception as e:
        logger.error(f"Error in comparing questions with Qwen: {e}")
        return {
            "question_a_scores": {"structural_difference": 0, "semantic_similarity": 0, "answer_match": 0},
            "question_b_scores": {"structural_difference": 0, "semantic_similarity": 0, "answer_match": 0},
            "explanation": "Failed to compare questions",
            "winner": "None"
        }

def generate_answer(context: str, question: str) -> str:
    try:
        response = gpt_client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nProvide a concise answer based on the context."}
            ],
            temperature=0.5,
            max_tokens=150,
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Error generating answer"

def process_entry(entry, excel_writer, start_row):
    result = {}  # Initialize the result dict

    # Extract data from entry
    try:
        context = entry['context']
        answer = entry['answers']['text'][0]
        original_question = entry['question']
    except Exception as e:
        logger.error(f"Error extracting data from entry: {e}")
        result.update({
            'Error': 'Failed to extract data from entry',
            'Context': None,
            'Original Question': None,
            'Original Answer': None,
            'Paraphrased Question': None,
            'Paraphrased Answer': None,
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
        'Original Question': original_question,
        'Original Answer': answer
    })

    # Generate paraphrased question
    try:
        paraphrased_question = paraphrase_system(original_question)
    except Exception as e:
        logger.error(f"Error generating paraphrased question: {e}")
        paraphrased_question = 'Error generating paraphrased question'
    result['Paraphrased Question'] = paraphrased_question

    # Generate paraphrased answer
    try:
        paraphrased_answer = generate_answer(context, paraphrased_question)
    except Exception as e:
        logger.error(f"Error generating paraphrased answer: {e}")
        paraphrased_answer = 'Error generating paraphrased answer'
    result['Paraphrased Answer'] = paraphrased_answer

    # Initialize vote counts
    vote_counts = {"Question A": 0, "Question B": 0}

    # Collect comparison results from each LLM judge
    comparison_results = {}

    # GPT-4o
    try:
        result_gpt4o = compare_questions_gpt4o(
            context, original_question, answer,
            paraphrased_question, paraphrased_answer
        )
    except Exception as e:
        logger.error(f"Error in GPT-4o comparison: {e}")
        result_gpt4o = {
            'winner': 'Error',
            'question_a_scores': {'structural_difference': 0, 'semantic_similarity': 0, 'answer_match': 0},
            'question_b_scores': {'structural_difference': 0, 'semantic_similarity': 0, 'answer_match': 0},
            'explanation': 'Error in GPT-4o comparison'
        }
    result['GPT-4o Verdict'] = result_gpt4o.get('winner', 'Error')
    if result_gpt4o['winner'] in vote_counts:
        vote_counts[result_gpt4o['winner']] += 1

    # Claude
    try:
        result_claude = compare_questions_claude(
            context, original_question, answer,
            paraphrased_question, paraphrased_answer
        )
    except Exception as e:
        logger.error(f"Error in Claude comparison: {e}")
        result_claude = {
            'winner': 'Error',
            'question_a_scores': {'structural_difference': 0, 'semantic_similarity': 0, 'answer_match': 0},
            'question_b_scores': {'structural_difference': 0, 'semantic_similarity': 0, 'answer_match': 0},
            'explanation': 'Error in Claude comparison'
        }
    result['Claude Verdict'] = result_claude.get('winner', 'Error')
    if result_claude['winner'] in vote_counts:
        vote_counts[result_claude['winner']] += 1

    # Cohere
    try:
        result_cohere = compare_questions_cohere(
            context, original_question, answer,
            paraphrased_question, paraphrased_answer
        )
    except Exception as e:
        logger.error(f"Error in Cohere comparison: {e}")
        result_cohere = {
            'winner': 'Error',
            'question_a_scores': {'structural_difference': 0, 'semantic_similarity': 0, 'answer_match': 0},
            'question_b_scores': {'structural_difference': 0, 'semantic_similarity': 0, 'answer_match': 0},
            'explanation': 'Error in Cohere comparison'
        }
    result['Cohere Verdict'] = result_cohere.get('winner', 'Error')
    if result_cohere['winner'] in vote_counts:
        vote_counts[result_cohere['winner']] += 1

    # Gemini
    try:
        result_gemini = compare_questions_gemini(
            context, original_question, answer,
            paraphrased_question, paraphrased_answer
        )
    except Exception as e:
        logger.error(f"Error in Gemini comparison: {e}")
        result_gemini = {
            'winner': 'Error',
            'question_a_scores': {'structural_difference': 0, 'semantic_similarity': 0, 'answer_match': 0},
            'question_b_scores': {'structural_difference': 0, 'semantic_similarity': 0, 'answer_match': 0},
            'explanation': 'Error in Gemini comparison'
        }
    result['Gemini Verdict'] = result_gemini.get('winner', 'Error')
    if result_gemini['winner'] in vote_counts:
        vote_counts[result_gemini['winner']] += 1

    # Qwen
    try:
        result_qwen = compare_questions_qwen(
            context, original_question, answer,
            paraphrased_question, paraphrased_answer
        )
    except Exception as e:
        logger.error(f"Error in Qwen comparison: {e}")
        result_qwen = {
            'winner': 'Error',
            'question_a_scores': {'structural_difference': 0, 'semantic_similarity': 0, 'answer_match': 0},
            'question_b_scores': {'structural_difference': 0, 'semantic_similarity': 0, 'answer_match': 0},
            'explanation': 'Error in Qwen comparison'
        }
    result['Qwen Verdict'] = result_qwen.get('winner', 'Error')
    if result_qwen['winner'] in vote_counts:
        vote_counts[result_qwen['winner']] += 1

    # Determine final verdict
    try:
        final_verdict_key = max(vote_counts, key=vote_counts.get)
        if vote_counts['Question A'] == vote_counts['Question B']:
            final_verdict = 'Draw'
        else:
            # Map 'Question A' to 'Original', 'Question B' to 'Paraphrased'
            final_verdict = 'Original' if final_verdict_key == 'Question A' else 'Paraphrased'
    except Exception as e:
        logger.error(f"Error determining final verdict: {e}")
        final_verdict = 'Error'
    result['Final Verdict'] = final_verdict

    # Write the result to the Excel file immediately
    df_result = pd.DataFrame([result])
    df_result.to_excel(excel_writer, index=False, header=False, startrow=start_row)
    excel_writer.book.save(excel_writer.path)
    excel_writer.book = pd.ExcelWriter(excel_writer.path, engine='openpyxl').book

    return result

def main():
    num_entries = input("Enter the number of entries to test on (or 'all' to process the entire dataset): ")
    random_seed = int(input("Enter a random seed (integer): "))

    entries = get_random_entries(num_entries, random_seed)
    results = []
    total_vote_counts = Counter()

    # Prepare the Excel file and writer
    excel_filename = 'benchmark_results.xlsx'

    # Check if the Excel file already exists
    if os.path.exists(excel_filename):
        df_existing = pd.read_excel(excel_filename)
        start_row = len(df_existing) + 1
    else:
        df_existing = pd.DataFrame()
        start_row = 0

    excel_writer = pd.ExcelWriter(excel_filename, engine='openpyxl', mode='a' if os.path.exists(excel_filename) else 'w')

    # If the file exists and we're appending, ensure we have the workbook loaded
    if os.path.exists(excel_filename):
        from openpyxl import load_workbook
        excel_writer.book = load_workbook(excel_filename)
        excel_writer.sheets = {ws.title: ws for ws in excel_writer.book.worksheets}

    # If starting fresh, write headers
    if start_row == 0:
        df_headers = pd.DataFrame(columns=[
            'Context', 'Original Question', 'Original Answer',
            'Paraphrased Question', 'Paraphrased Answer',
            'GPT-4o Verdict', 'Claude Verdict', 'Cohere Verdict', 'Gemini Verdict', 'Qwen Verdict',
            'Final Verdict'
        ])
        df_headers.to_excel(excel_writer, index=False)
        excel_writer.book.save(excel_writer.path)
        excel_writer.book = pd.ExcelWriter(excel_writer.path, engine='openpyxl').book
        start_row = 1  # Adjust for header row

    for idx_in_entries, entry in enumerate(entries):
        idx_in_dataset = idx_in_entries  # Since entries are random, index in dataset is not sequential
        print(f"Processing entry {idx_in_entries+1}/{len(entries)}...")
        try:
            result = process_entry(entry, excel_writer, start_row + idx_in_entries)
            results.append(result)
            # Update total vote counts
            for key in ['Question A', 'Question B']:
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

    # Generate summary
    total_entries = len(results)
    df = pd.DataFrame(results)
    final_verdicts = df['Final Verdict'].value_counts()
    percentage_original = (final_verdicts.get('Original', 0) / total_entries) * 100 if total_entries > 0 else 0
    percentage_paraphrased = (final_verdicts.get('Paraphrased', 0) / total_entries) * 100 if total_entries > 0 else 0
    percentage_draw = (final_verdicts.get('Draw', 0) / total_entries) * 100 if total_entries > 0 else 0

    summary = f"""
Total Entries Processed: {total_entries}

Vote Counts:
Question A (Original) Votes: {total_vote_counts['Question A']}
Question B (Paraphrased) Votes: {total_vote_counts['Question B']}

Final Verdicts:
Original Question Wins: {final_verdicts.get('Original', 0)} ({percentage_original:.2f}%)
Paraphrased Question Wins: {final_verdicts.get('Paraphrased', 0)} ({percentage_paraphrased:.2f}%)
Draws: {final_verdicts.get('Draw', 0)} ({percentage_draw:.2f}%)
"""

    # Save summary to text file
    summary_filename = 'benchmark_summary.txt'
    with open(summary_filename, 'w') as f:
        f.write(summary)
    print(f"Summary saved to {summary_filename}")
    print(summary)

    # Close the Excel writer
    excel_writer.close()

if __name__ == "__main__":
    main()
