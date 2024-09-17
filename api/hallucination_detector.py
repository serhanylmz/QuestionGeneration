import json
import logging
import numpy as np
import time
import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
                    "content": "You are a helpful assistant that generates diverse questions based on given context, answer, and initial question."
                },
                {
                    "role": "user",
                    "content": (
                        f"Based on this context: '{context}', answer: '{answer}', and initial question: '{initial_question}', "
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
        return [np.zeros(1536) for _ in texts]  # Adjust the size based on the model used

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
                    "content": "You are an assistant that determines whether two answers are semantically the same."
                },
                {
                    "role": "user",
                    "content": (
                        f"Determine if the following two answers are semantically the same.\n\n"
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

    # Save similarity matrix and match matrix
    sim_matrix_filename = f"similarity_matrix_{timestamp}.npy"
    np.save(sim_matrix_filename, similarity_matrix)
    logger.info(f"Saved similarity matrix to {sim_matrix_filename}")

    match_matrix_filename = f"match_matrix_{timestamp}.npy"
    np.save(match_matrix_filename, match_matrix)
    logger.info(f"Saved answer match matrix to {match_matrix_filename}")

if __name__ == "__main__":
    initial_question = input("Enter the initial query: ")
    n_paraphrases = int(input("Enter the number of paraphrases to generate (default 3): ") or "3")
    similarity_threshold = float(input("Enter the semantic similarity threshold (default 0.7): ") or "0.7")
    match_percentage_threshold = float(input("Enter the answer match percentage threshold (default 0.6): ") or "0.6")
    main(initial_question, n_paraphrases, similarity_threshold, match_percentage_threshold)
