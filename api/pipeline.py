import os
import json
from openai import OpenAI
from typing import List, Tuple
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from Levenshtein import distance as levenshtein_distance

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

generation_temperature = 0.8
judge_temperature = 0.2

# initial (default) temperature was set to 1.0, but it was changed to 0.8 to reduce randomness in the generated questions
# the judge temperature was set to 0.2 to ensure we have reliable and rather deterministic judges.

def generate_basic_question(context: str, answer: str, initial_question: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            temperature=generation_temperature,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates diverse questions based on given context, answer, and initial question."},
                {"role": "user", "content": f"Based on this context: '{context}', answer: '{answer}', and initial question: '{initial_question}', generate a new question that is semantically similar but structurally different from the initial question. The new question should still lead to the same answer when asked about the context. Provide only the question, without any additional text."}
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
        parsed_response = json.loads(json_response)
        return parsed_response["question"]
    except Exception as e:
        return "Failed to generate question"

def generate_single_question(context: str, answer: str, initial_question: str, existing_questions: List[str]) -> str:
    try:
        existing_questions_str = "\n".join(existing_questions)
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            temperature=generation_temperature,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates diverse questions based on given context, answer, and initial question."},
                {"role": "user", "content": f"Based on this context: '{context}', answer: '{answer}', and initial question: '{initial_question}', generate a new question that is semantically similar but structurally different from the initial question. The new question should still lead to the same answer when asked about the context. The question should also be distinct from these existing questions:\n{existing_questions_str}\n\nProvide only the new question, without any additional text."}
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
        parsed_response = json.loads(json_response)
        return parsed_response["question"]
    except Exception as e:
        return "Failed to generate question"

def is_question_distinct(new_question: str, existing_questions: List[str]) -> bool:
    new_question_normalized = new_question.strip().lower()
    for question in existing_questions:
        if new_question_normalized == question.strip().lower():
            return False
    return True

def generate_questions(context: str, answer: str, initial_question: str) -> List[str]:
    questions = []
    max_attempts = 10
    
    while len(questions) < 5 and max_attempts > 0:  # Generate 5 new questions
        new_question = generate_single_question(context, answer, initial_question, questions)
        if new_question != "Failed to generate question" and is_question_distinct(new_question, questions):
            questions.append(new_question)
        else:
            max_attempts -= 1
    
    while len(questions) < 5:
        questions.append(f"Failed to generate distinct question {len(questions) + 1}")
    
    return questions

def generate_answer(context: str, question: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            temperature=0, # might work out better with 0 temperature, as this just needs to be a direct answer
            messages=[
                {"role": "system", "content": "You are a precise answer generator. Your task is to provide concise, accurate answers based on the given context. Focus on the key information that directly answers the question, even if the phrasing differs from the context."},
                {"role": "user", "content": f"""Context: {context}

Question: {question}

Generate a concise and accurate answer to the question based on the given context. Focus on the core information that answers the question. Provide only the answer, without any explanation or additional text."""}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer_generator",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"}
                        },
                        "required": ["answer"],
                        "additionalProperties": False
                    }
                }
            }
        )
        
        json_response = response.choices[0].message.content
        parsed_response = json.loads(json_response)
        return parsed_response["answer"]
    except Exception as e:
        return "Failed to generate answer"

def calculate_edit_distance(questions: List[str], initial_question: str) -> List[float]:
    max_length = max(len(initial_question), max(len(q) for q in questions))
    distances = [levenshtein_distance(initial_question, q) for q in questions]
    normalized_distances = [d / max_length for d in distances]  # Higher score means more different
    return normalized_distances

def calculate_semantic_similarity(questions: List[str], initial_question: str) -> List[float]:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            temperature=judge_temperature,
            messages=[
                {"role": "system", "content": "You are an expert in semantic analysis, specializing in evaluating the similarity of questions."},
                {"role": "user", "content": f"Analyze the semantic similarity of the following questions to the initial question. Provide a similarity score for each question on a scale of 0 to 1, where 1 is highly similar to the initial question:\n\nInitial question: {initial_question}\n\nQuestions to analyze: {json.dumps(questions)}"}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "semantic_similarity_analyzer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "similarity_scores": {
                                "type": "array",
                                "items": {
                                    "type": "number"
                                }
                            },
                            "explanation": {"type": "string"}
                        },
                        "required": ["similarity_scores", "explanation"],
                        "additionalProperties": False
                    }
                }
            }
        )
        
        json_response = response.choices[0].message.content
        parsed_response = json.loads(json_response)
        similarity_scores = parsed_response["similarity_scores"]
        return similarity_scores
    except Exception as e:
        return [0.5] * len(questions)  # Return neutral scores in case of error

def check_answer_precision(context: str, questions: List[str], original_answer: str) -> Tuple[List[float], List[str]]:
    precision_scores = []
    generated_answers = []
    for question in questions:
        generated_answer = generate_answer(context, question)
        generated_answers.append(generated_answer)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                temperature=judge_temperature,
                messages=[
                    {"role": "system", "content": "You are an expert in evaluating answer precision, focusing on semantic similarity rather than exact wording. Your task is to determine if two answers convey the same core information, even if they are phrased differently."},
                    {"role": "user", "content": f"""Context: {context}
Original Answer: {original_answer}
Generated Answer: {generated_answer}

Evaluate the semantic similarity between the original answer and the generated answer. Consider the following:
1. Do both answers convey the same core information?
2. Are there any key concepts present in one answer but missing in the other?
3. Would both answers be considered correct in the context of the given information?

Provide a precision score from 0 to 1, where:
1.0: The answers are semantically identical or equivalent.
0.8-0.9: The answers convey the same core information with minor differences.
0.6-0.7: The answers are mostly similar but with some notable differences.
0.4-0.5: The answers have some overlap but significant differences.
0.2-0.3: The answers are mostly different but with some minor similarities.
0.0-0.1: The answers are completely different or unrelated.

Provide only the precision score as a number between 0 and 1."""}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer_precision_evaluator",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "precision_score": {
                                    "type": "number"
                                }
                            },
                            "required": ["precision_score"],
                            "additionalProperties": False
                        }
                    }
                }
            )
            
            json_response = response.choices[0].message.content
            parsed_response = json.loads(json_response)
            precision_score = parsed_response["precision_score"]
            precision_scores.append(precision_score)
        except Exception as e:
            precision_scores.append(0.5)  # Neutral score in case of error
    
    return precision_scores, generated_answers

def calculate_composite_scores(ed_scores: List[float], ss_scores: List[float], ap_scores: List[float]) -> List[float]:
    return [0.4 * ed + 0.4 * ss + 0.2 * ap for ed, ss, ap in zip(ed_scores, ss_scores, ap_scores)]

def rank_questions_with_details(context: str, answer: str, initial_question: str) -> Tuple[pd.DataFrame, List[pd.DataFrame], str]:
    questions = generate_questions(context, answer, initial_question)
    
    ed_scores = calculate_edit_distance(questions, initial_question)
    ss_scores = calculate_semantic_similarity(questions, initial_question)
    ap_scores, generated_answers = check_answer_precision(context, questions, answer)
    
    composite_scores = calculate_composite_scores(ed_scores, ss_scores, ap_scores)
    
    detailed_scores = pd.DataFrame({
        'Question': questions,
        'Edit Distance': ed_scores,
        'Semantic Similarity': ss_scores,
        'Answer Precision': ap_scores,
        'Composite Score': composite_scores,
        'Generated Answer': generated_answers
    })
    detailed_scores = detailed_scores.sort_values('Composite Score', ascending=False).reset_index(drop=True)
    
    metrics = ['Edit Distance', 'Semantic Similarity', 'Answer Precision', 'Composite Score']
    rankings = []
    
    for metric in metrics:
        df = pd.DataFrame({
            'Rank': range(1, len(questions) + 1),
            'Question': [questions[i] for i in np.argsort(detailed_scores[metric])[::-1]],
            f'{metric}': sorted(detailed_scores[metric], reverse=True)
        })
        if metric == 'Answer Precision':
            df['Generated Answer'] = [generated_answers[i] for i in np.argsort(detailed_scores[metric])[::-1]]
        rankings.append(df)
    
    best_question = detailed_scores.iloc[0]['Question']
    
    return detailed_scores, rankings, best_question