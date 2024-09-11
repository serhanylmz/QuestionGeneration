import os
import logging
import json
import gradio as gr
import pandas as pd
from datasets import load_dataset
import random
from openai import OpenAI
from typing import List, Tuple
from dotenv import load_dotenv
import numpy as np
from Levenshtein import distance as levenshtein_distance

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Load the dataset
dataset = load_dataset("rajpurkar/squad")

# Define sample inputs
samples = [
    {
        "context": "Albert Einstein is an Austrian scientist, who has completed his higher education in ETH Zurich in Zurich, Switzerland. He was later a faculty at Princeton University.",
        "answer": "Switzerland",
        "question": "Where did Albert Einstein complete his higher education?"
    },
    {
        "context": "The Eiffel Tower, located in Paris, France, is one of the most famous landmarks in the world. It was constructed in 1889 as the entrance arch to the 1889 World's Fair. The tower is 324 meters (1,063 ft) tall and is the tallest structure in Paris.",
        "answer": "Paris",
        "question": "In which city is the Eiffel Tower located?"
    },
    {
        "context": "The Great Wall of China is a series of fortifications and walls built across the historical northern borders of ancient Chinese states and Imperial China to protect against nomadic invasions. It is the largest man-made structure in the world, with a total length of more than 13,000 miles (21,000 kilometers).",
        "answer": "China",
        "question": "In which country is the Great Wall located?"
    }
]

def generate_basic_question(context: str, answer: str, initial_question: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
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
        logger.info(f"Raw JSON response: {json_response}")
        
        parsed_response = json.loads(json_response)
        return parsed_response["question"]
    except Exception as e:
        logger.error(f"Error in generate_basic_question: {e}")
        return "Failed to generate question"

def generate_single_question(context: str, answer: str, initial_question: str, existing_questions: List[str]) -> str:
    try:
        existing_questions_str = "\n".join(existing_questions)
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
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
        logger.info(f"Raw JSON response: {json_response}")
        
        parsed_response = json.loads(json_response)
        return parsed_response["question"]
    except Exception as e:
        logger.error(f"Error in generate_single_question: {e}")
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
        logger.info(f"Raw JSON response: {json_response}")
        
        parsed_response = json.loads(json_response)
        return parsed_response["answer"]
    except Exception as e:
        logger.error(f"Error in generate_answer: {e}")
        return "Failed to generate answer"

def calculate_edit_distance(questions: List[str], initial_question: str) -> List[float]:
    max_length = max(len(initial_question), max(len(q) for q in questions))
    distances = [levenshtein_distance(initial_question, q) for q in questions]
    normalized_distances = [d / max_length for d in distances]  # Higher score means more different
    return normalized_distances # normalization done with dividing that into the max length of all questions, is that right?

def calculate_semantic_similarity(questions: List[str], initial_question: str) -> List[float]:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
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
        logger.info(f"Raw JSON response: {json_response}")
        
        parsed_response = json.loads(json_response)
        similarity_scores = parsed_response["similarity_scores"]
        explanation = parsed_response["explanation"]
        
        logger.info(f"Semantic Similarity Explanation: {explanation}")
        
        return similarity_scores
    except Exception as e:
        logger.error(f"Error in calculate_semantic_similarity: {e}")
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
            logger.error(f"Error in evaluating answer precision: {e}")
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
    
    best_question = detailed_scores.iloc[0]['Question']  # Select the best question
    
    return detailed_scores, rankings, best_question

def gradio_interface(context: str, answer: str, initial_question: str) -> Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    detailed_scores, rankings, best_question = rank_questions_with_details(context, answer, initial_question)
    return (
        initial_question,  # Return the initial question separately
        detailed_scores,
        rankings[0],  # Edit Distance Ranking
        rankings[1],  # Semantic Similarity Ranking
        rankings[2],  # Answer Precision Ranking
        rankings[3],  # Composite Score Ranking
        f"Best Generated Question: {best_question}"
    )

def use_sample(sample_index: int) -> Tuple[str, str, str]:
    return samples[sample_index]["context"], samples[sample_index]["answer"], samples[sample_index]["question"]

def get_random_entry():
    random_index = random.randint(0, len(dataset['train']) - 1)
    entry = dataset['train'][random_index]
    return entry['context'], entry['answers']['text'][0], entry['question']

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Default()) as iface:
    gr.Markdown("# Enhanced Question Generator and Ranker")
    gr.Markdown("Enter a context, an answer, and an initial question to generate and rank questions, use one of the sample inputs, or get a random entry from the dataset.")
    
    with gr.Row():
        with gr.Column(scale=1):
            context_input = gr.Textbox(lines=5, label="Context")
            answer_input = gr.Textbox(lines=2, label="Answer")
            initial_question_input = gr.Textbox(lines=2, label="Initial Question")
            submit_button = gr.Button("Generate Questions")
            
            with gr.Row():
                sample_buttons = [gr.Button(f"Sample {i+1}") for i in range(3)]
                random_button = gr.Button("Random Dataset Entry")
        
        with gr.Column(scale=2):
            initial_question_output = gr.Textbox(label="Initial Question")
            best_question_output = gr.Textbox(label="Best Generated Question")
            detailed_scores_output = gr.DataFrame(label="Detailed Scores")
    
    with gr.Row():
        with gr.Column():
            edit_distance_ranking_output = gr.DataFrame(label="Edit Distance Ranking")
        with gr.Column():
            semantic_similarity_ranking_output = gr.DataFrame(label="Semantic Similarity Ranking")
    
    with gr.Row():
        with gr.Column():
            answer_precision_ranking_output = gr.DataFrame(label="Answer Precision Ranking")
        with gr.Column():
            composite_ranking_output = gr.DataFrame(label="Composite Score Ranking")

    def process_random_entry():
        context, answer, initial_question = get_random_entry()
        return context, answer, initial_question

    submit_button.click(
        fn=gradio_interface,
        inputs=[context_input, answer_input, initial_question_input],
        outputs=[
            initial_question_output,
            detailed_scores_output,
            edit_distance_ranking_output,
            semantic_similarity_ranking_output,
            answer_precision_ranking_output,
            composite_ranking_output,
            best_question_output
        ]
    )

    # Set up sample button functionality
    for i, button in enumerate(sample_buttons):
        button.click(
            fn=lambda i=i: use_sample(i),
            outputs=[context_input, answer_input, initial_question_input]
        )

    # Set up random button functionality
    random_button.click(
        fn=process_random_entry,
        outputs=[context_input, answer_input, initial_question_input]
    )

# Launch the app
if __name__ == "__main__":
    iface.launch()