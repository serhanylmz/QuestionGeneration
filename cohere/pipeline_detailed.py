import cohere
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import json
import os
from dotenv import load_dotenv

load_dotenv()  # This loads the variables from .env

# Initialize Cohere client, SentenceTransformer model, and QA pipeline
co = cohere.Client(api_key = os.environ.get("COHERE_API_KEY"))
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_questions(context: str, answer: str) -> List[str]:
    """
    Step 1: Generate 5 questions using Cohere's structured output.
    """
    response = co.chat(
        model="command-r",
        message=f"Based on this context: '{context}' and answer: '{answer}', generate 5 diverse questions which when asked to the context returns the answer.",
        response_format={
            "type": "json_object",
            "schema": {
                "type": "object",
                "required": ["question1", "question2", "question3", "question4", "question5"],
                "properties": {
                    "question1": {"type": "string"},
                    "question2": {"type": "string"},
                    "question3": {"type": "string"},
                    "question4": {"type": "string"},
                    "question5": {"type": "string"}
                }
            }
        }
    )
    
    json_response = response.text
    parsed_response = json.loads(json_response)
    questions = [parsed_response[f"question{i}"] for i in range(1, 6)]
    
    return questions

def calculate_structural_diversity(questions: List[str]) -> List[float]:
    """
    Step 2: Calculate structural diversity scores for each question.
    """
    def get_question_type(q):
        q = q.lower()
        if q.startswith('what'): return 1
        elif q.startswith('why'): return 2
        elif q.startswith('how'): return 3
        elif q.startswith('when'): return 4
        elif q.startswith('where'): return 5
        else: return 0

    lengths = [len(q.split()) for q in questions]
    types = [get_question_type(q) for q in questions]
    
    length_scores = [1 - (abs(l - np.mean(lengths)) / np.max(lengths)) for l in lengths]
    type_scores = [len(set(types)) / len(types) for _ in types]
    
    return [(l + t) / 2 for l, t in zip(length_scores, type_scores)]

def calculate_semantic_relevance(context: str, answer: str, questions: List[str]) -> List[float]:
    """
    Step 3: Calculate semantic relevance scores for each question.
    """
    context_embedding = sentence_model.encode(context + " " + answer)
    question_embeddings = sentence_model.encode(questions)
    
    similarities = [np.dot(context_embedding, q_emb) / (np.linalg.norm(context_embedding) * np.linalg.norm(q_emb)) 
                    for q_emb in question_embeddings]
    
    return [(sim + 1) / 2 for sim in similarities]  # Normalize to 0-1 range

def check_answer_precision(context: str, questions: List[str], original_answer: str) -> Tuple[List[float], List[str]]:
    """
    Step 4: Check answer precision for each question and return both scores and generated answers.
    """
    precision_scores = []
    generated_answers = []
    for question in questions:
        generated_answer = co.chat(
            model="command",
            message=f"Context: {context}\nQuestion: {question}\nProvide a brief answer based only on the given context."
        ).text
        generated_answers.append(generated_answer)
        answer_embedding = sentence_model.encode(original_answer)
        generated_embedding = sentence_model.encode(generated_answer)
        similarity = np.dot(answer_embedding, generated_embedding) / (np.linalg.norm(answer_embedding) * np.linalg.norm(generated_embedding))
        precision_scores.append((similarity + 1) / 2)  # Normalize to 0-1 range
    return precision_scores, generated_answers

def calculate_composite_scores(sd_scores: List[float], sr_scores: List[float], ap_scores: List[float]) -> List[float]:
    """
    Step 5: Calculate composite scores.
    """
    return [0.2 * sd + 0.4 * sr + 0.4 * ap for sd, sr, ap in zip(sd_scores, sr_scores, ap_scores)]

def print_detailed_output(context: str, answer: str, questions: List[str], 
                          sd_scores: List[float], sr_scores: List[float], 
                          ap_scores: List[float], composite_scores: List[float],
                          generated_answers: List[str]):
    """
    Print a detailed output of the question ranking process.
    """
    print("\n" + "="*50)
    print("QUESTION RANKING PROCESS")
    print("="*50)
    
    print("\nGiven Context:")
    print(context)
    print("\nGiven Answer:")
    print(answer)
    
    print("\nGenerated Questions:")
    for i, question in enumerate(questions, 1):
        print(f"{i}. {question}")
    
    print("\nDetailed Scores and Answers for Each Question:")
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        print(f"  Structural Diversity Score: {sd_scores[i-1]:.4f}")
        print(f"  Semantic Relevance Score: {sr_scores[i-1]:.4f}")
        print(f"  Answer Precision Score: {ap_scores[i-1]:.4f}")
        print(f"  Composite Score: {composite_scores[i-1]:.4f}")
        print(f"  Generated Answer: {generated_answers[i-1]}")

    best_index = np.argmax(composite_scores)
    print("\n" + "="*50)
    print(f"BEST QUESTION (Composite Score: {composite_scores[best_index]:.4f}):")
    print(questions[best_index])
    print("="*50)

def rank_questions_with_details(context: str, answer: str) -> None:
    """
    Main function to execute the entire question ranking process with detailed output.
    """
    questions = generate_questions(context, answer)
    
    sd_scores = calculate_structural_diversity(questions)
    sr_scores = calculate_semantic_relevance(context, answer, questions)
    ap_scores, generated_answers = check_answer_precision(context, questions, answer)
    
    composite_scores = calculate_composite_scores(sd_scores, sr_scores, ap_scores)
    
    print_detailed_output(context, answer, questions, sd_scores, sr_scores, ap_scores, composite_scores, generated_answers)

# Example usage
if __name__ == "__main__":
    context = "The first human heart transplant was performed by Dr. Christiaan Barnard on December 3, 1967, in Cape Town, South Africa."
    answer = "Dr. Christiaan Barnard"
    
    rank_questions_with_details(context, answer)