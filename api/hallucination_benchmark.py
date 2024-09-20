import json
import logging
import numpy as np
import time
import os
from typing import List
from openai import OpenAI
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import argparse
import csv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(filename='hallucination_evaluator.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Set up the OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_single_question(context: str, answer: str, initial_question: str, existing_questions: List[str]) -> str:
    """
    Generates a single paraphrased question using structured output with JSON schema.
    """
    try:
        existing_questions_str = "\n".join(existing_questions)
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates paraphrases of the given question."
                },
                {
                    "role": "user",
                    "content": (
                        f"Based on this question: '{initial_question}', "
                        f"generate a new question that is semantically similar but structurally different from the initial question. "
                        f"The new question should still lead to the same answer when asked about the context. "
                        f"The question should also be distinct from these existing questions:\n{existing_questions_str}\n\n"
                        "Provide only the new question, without any additional text."
                    )
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "single_question_generator",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"}
                        },
                        "required": ["question"],
                        "additionalProperties": False
                    }
                }
            }
        )

        json_response = response.choices[0].message.content
        logger.info(f"Raw JSON response: {json_response}")

        parsed_response = json.loads(json_response)
        new_question = parsed_response["question"]
        logger.info(f"Generated paraphrased question: {new_question}")
        return new_question
    except Exception as e:
        logger.error(f"Error in generate_single_question: {e}")
        return "Failed to generate question"

def generate_paraphrased_questions(initial_question: str, n: int = 5) -> List[str]:
    """
    Generates a list of paraphrased questions.
    """
    paraphrased_questions = []
    existing_questions = [initial_question]
    context = ""  # Empty context as per your function
    answer = ""   # Empty answer as per your function
    for _ in range(n):
        new_question = generate_single_question(context, answer, initial_question, existing_questions)
        paraphrased_questions.append(new_question)
        existing_questions.append(new_question)
        time.sleep(1)  # To avoid hitting rate limits
    return paraphrased_questions

def get_llm_response(question: str) -> str:
    """
    Obtains the LLM's response to a given question.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "user", "content": question}
            ]
        )
        answer = response.choices[0].message.content.strip()
        logger.info(f"LLM response for question: {question}\nResponse: {answer}")
        return answer
    except Exception as e:
        logger.error(f"Error in get_llm_response: {e}")
        return "Failed to get response"

def get_embeddings(texts: List[str], model="text-embedding-3-small") -> List[np.ndarray]:
    """
    Retrieves embeddings for a list of texts using the specified embedding model.
    """
    try:
        texts = [text.replace("\n", " ") for text in texts]
        response = client.embeddings.create(input=texts, model=model)
        embeddings = [np.array(data.embedding) for data in response.data]
        return embeddings
    except Exception as e:
        logger.error(f"Error in get_embeddings: {e}")
        # Assuming embedding size is 1024 for 'text-embedding-3-small', adjust if different
        embedding_size = 1024
        return [np.zeros(embedding_size) for _ in texts]

def calculate_similarity_matrix(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Calculates the cosine similarity matrix for a list of embeddings.
    """
    n = len(embeddings)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            similarity = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity  # Symmetric matrix
    logger.info(f"Calculated similarity matrix:\n{similarity_matrix}")
    return similarity_matrix

def is_semantically_trustworthy(similarity_matrix: np.ndarray, threshold: float = 0.9) -> bool:
    """
    Determines if the responses are semantically trustworthy based on the similarity threshold.
    """
    n = similarity_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i][j] < threshold:
                logger.info(f"Semantic similarity between response {i+1} and {j+1} below threshold: {similarity_matrix[i][j]}")
                return False
    return True

def compare_answers(answer1: str, answer2: str) -> bool:
    """
    Compares two answers to determine if they are semantically the same.
    Uses structured output with JSON schema as per your provided function.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that determines whether two answers are coherent and contain the same information."
                },
                {
                    "role": "user",
                    "content": (
                        f"Determine if the following two answers are coherent and contain the same information. Examine both answers thoroughly. Two answers are considered the same if they convey the same information, even if they are phrased differently or use different wording.\n\n"
                        f"Answer 1: '{answer1}'\n\nAnswer 2: '{answer2}'\n\n"
                        "Respond with a JSON object in the following format without any additional text:\n"
                        "{\"are_same\": true} or {\"are_same\": false}"
                    )
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer_comparison",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "are_same": {"type": "boolean"}
                        },
                        "required": ["are_same"],
                        "additionalProperties": False
                    }
                }
            }
        )
        json_response = response.choices[0].message.content
        logger.info(f"Raw JSON response: {json_response}")
        parsed_response = json.loads(json_response)
        are_same = parsed_response['are_same']
        logger.info(f"compare_answers result: {are_same}")
        return are_same
    except Exception as e:
        logger.error(f"Error in compare_answers: {e}")
        return False

def answer_checker(answers: List[str]) -> np.ndarray:
    """
    Creates a matrix indicating whether each pair of answers is semantically the same.
    """
    n = len(answers)
    match_matrix = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            are_same = compare_answers(answers[i], answers[j])
            match_matrix[i][j] = are_same
            match_matrix[j][i] = are_same  # Symmetric matrix
            time.sleep(0.5)  # To avoid rate limits
        match_matrix[i][i] = True  # An answer is always the same as itself
    logger.info(f"Answer match matrix:\n{match_matrix}")
    return match_matrix

def calculate_match_percentage(match_matrix: np.ndarray) -> float:
    """
    Calculates the percentage of matching answers based on the match matrix.
    """
    n = match_matrix.shape[0]
    total_pairs = n * (n - 1) / 2
    matching_pairs = np.sum(np.triu(match_matrix, k=1))  # Sum of upper triangle excluding diagonal
    match_percentage = matching_pairs / total_pairs if total_pairs > 0 else 0
    logger.info(f"Matching pairs: {matching_pairs}, Total pairs: {total_pairs}, Match percentage: {match_percentage}")
    return match_percentage

def save_matrix_plot(matrix: np.ndarray, labels: List[str], title: str, filename: str):
    """
    Saves a heatmap plot of the matrix with the given labels.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
    plt.yticks(ticks=range(len(labels)), labels=labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logger.info(f"Saved {title} plot to {filename}")

def main(initial_question: str, n_paraphrases: int = 5, similarity_threshold: float = 0.9, match_percentage_threshold: float = 0.7):
    """
    Main function to determine if the LLM's response is trustworthy or a hallucination.
    """
    # Generate paraphrased questions
    paraphrased_questions = generate_paraphrased_questions(initial_question, n=n_paraphrases)
    all_questions = [initial_question] + paraphrased_questions
    logger.info("Generated all questions:")
    for idx, q in enumerate(all_questions, 1):
        logger.info(f"Question {idx}: {q}")

    # Get responses for all questions
    responses = []
    for question in all_questions:
        answer = get_llm_response(question)
        responses.append(answer)
        time.sleep(1)  # To avoid hitting rate limits

    logger.info("Collected all responses.")

    # Get embeddings for all responses
    embeddings = get_embeddings(responses)

    if len(embeddings) == 0:
        logger.error("Failed to get embeddings for responses.")
        return

    # Calculate semantic similarity matrix
    similarity_matrix = calculate_similarity_matrix(embeddings)

    # Check semantic trustworthiness
    semantically_trustworthy = is_semantically_trustworthy(similarity_matrix, threshold=similarity_threshold)

    # Perform answer checking
    match_matrix = answer_checker(responses)
    match_percentage = calculate_match_percentage(match_matrix)
    answers_trustworthy = match_percentage >= match_percentage_threshold

    # Decide final trustworthiness
    trustworthy = semantically_trustworthy and answers_trustworthy

    # Output initial query, response, and whether LLM has hallucinated
    initial_response = responses[0]
    hallucinated = not trustworthy
    print(f"Initial Query:\n{initial_question}\n")
    print(f"Response:\n{initial_response}\n")
    if hallucinated:
        print("The LLM has hallucinated.\n")
    else:
        print("The response is trustworthy.\n")

    # Save queries and responses to text file
    timestamp = int(time.time())
    filename = f"llm_responses_{timestamp}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        for i, (question, response) in enumerate(zip(all_questions, responses), 1):
            f.write(f"Question {i}:\n{question}\n")
            f.write(f"Response {i}:\n{response}\n\n")
    logger.info(f"Saved queries and responses to {filename}")

    # Generate labels for the matrices
    labels = ["original"] + [f"paraphrase{i}" for i in range(1, n_paraphrases + 1)]

    # Save similarity matrix plot
    sim_matrix_filename = f"similarity_matrix_{timestamp}.png"
    save_matrix_plot(similarity_matrix, labels, "Semantic Similarity Matrix", sim_matrix_filename)

    # Save answer match matrix plot
    match_matrix_filename = f"match_matrix_{timestamp}.png"
    save_matrix_plot(match_matrix.astype(float), labels, "Answer Match Matrix", match_matrix_filename)

    run_id = timestamp

    return hallucinated, run_id

def run_hallucination_evaluator(json_file, num_samples=None):
    data = []
    try:
        # Since the file is in JSON Lines format, read it line by line
        with open(json_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        sample = json.loads(line)
                        data.append(sample)
                        if num_samples and len(data) >= num_samples:
                            break
                    except json.JSONDecodeError as e:
                        logging.error(f"Error decoding JSON on line {line_num}: {e}")
                        continue
    except Exception as e:
        logging.error(f"Error reading the JSON file: {e}")
        print(f"Error reading the JSON file: {e}")
        return

    if not data:
        logging.error("No data to process.")
        print("No data to process.")
        return

    total_samples = len(data)
    correct_detections = 0
    processed_samples = 0

    # Prepare to store detailed results
    detailed_results = []

    # Iterate over each sample
    for idx, sample in enumerate(data):
        sample_id = sample.get('ID')
        user_query = sample.get('user_query')
        true_label = sample.get('hallucination')  # 'yes' or 'no'

        if user_query is None or true_label is None or sample_id is None:
            logging.warning(f"Sample {idx + 1} is missing 'ID', 'user_query' or 'hallucination' fields.")
            continue

        try:
            # Run the main function and capture the hallucination result and run_id
            hallucinated, run_id = main(user_query)

            # Check if main returned None (e.g., due to an error)
            if hallucinated is None or run_id is None:
                logging.error(f"Main function failed for sample {idx + 1}.")
                continue

            # Convert 'yes'/'no' to boolean for comparison
            true_hallucinated = true_label.strip().lower() == 'yes'

            # Compare the detected hallucination with the true label
            if hallucinated == true_hallucinated:
                correct_detections += 1

            processed_samples += 1

            # Store detailed result
            detailed_results.append({
                'ID': sample_id,
                'user_query': user_query,
                'true_label': true_label.lower(),
                'detected_hallucination': 'yes' if hallucinated else 'no',
                'run_id': run_id
            })

            # Optional: print progress every 10 samples
            if (idx + 1) % 10 == 0:
                logging.info(f"Processed {idx + 1}/{total_samples} samples...")

            # To avoid hitting rate limits (if applicable)
            time.sleep(1)

        except Exception as e:
            logging.error(f"Error processing sample {idx + 1}: {e}")
            continue

    # Calculate accuracy
    accuracy = (correct_detections / processed_samples * 100) if processed_samples > 0 else 0
    print(f"Processed Samples: {processed_samples}/{total_samples}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Save accuracy to a file
    with open('accuracy.txt', 'w', encoding='utf-8') as f:
        f.write(f"Processed Samples: {processed_samples}/{total_samples}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")

    # Save detailed results to a CSV file
    with open('detailed_results.csv', 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['ID', 'user_query', 'true_label', 'detected_hallucination', 'run_id']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in detailed_results:
            writer.writerow(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hallucination Evaluator')
    parser.add_argument('--json_file', type=str, default='general_data.json', help='Path to the JSON file')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to process')
    args = parser.parse_args()

    run_hallucination_evaluator(args.json_file, args.num_samples)

# python hallucination_benchmark.py --json_file general_data.json --num_samples 10