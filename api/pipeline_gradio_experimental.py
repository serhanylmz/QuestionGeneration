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
dataset = load_dataset("serhany/scaling-qa")

# Define sample inputs
samples = [
    {
        "context": "Albert Einstein is an Austrian scientist, who has completed his higher education in ETH Zurich in Zurich, Switzerland. He was later a faculty at Princeton University.",
        "answer": "Switzerland"
    },
    {
        "context": "The Eiffel Tower, located in Paris, France, is one of the most famous landmarks in the world. It was constructed in 1889 as the entrance arch to the 1889 World's Fair. The tower is 324 meters (1,063 ft) tall and is the tallest structure in Paris.",
        "answer": "Paris"
    },
    {
        "context": "The Great Wall of China is a series of fortifications and walls built across the historical northern borders of ancient Chinese states and Imperial China to protect against nomadic invasions. It is the largest man-made structure in the world, with a total length of more than 13,000 miles (21,000 kilometers).",
        "answer": "China"
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
    # Convert the new question to lowercase and remove any leading/trailing whitespace
    new_question_normalized = new_question.strip().lower()
    
    # Check if the normalized new question is already in the list of existing questions
    for question in existing_questions:
        if new_question_normalized == question.strip().lower():
            return False
    
    # If we've made it through the loop, the question is distinct
    return True

def generate_questions(context: str, answer: str, initial_question: str) -> List[str]:
    questions = [initial_question]  # Include the initial question in the list
    max_attempts = 10
    
    while len(questions) < 6 and max_attempts > 0:  # Generate 5 new questions + initial question
        new_question = generate_single_question(context, answer, initial_question, questions)
        if new_question != "Failed to generate question" and is_question_distinct(new_question, questions):
            questions.append(new_question)
        else:
            max_attempts -= 1
    
    while len(questions) < 6:
        questions.append(f"Failed to generate distinct question {len(questions)}")
    
    return questions

def generate_answer(context: str, question: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise answers based on the given context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nProvide a concise answer to the question based on the given context."}
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

def calculate_structural_diversity(questions: List[str], initial_question: str) -> List[float]:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert in linguistic analysis, specializing in question structure and diversity."},
                {"role": "user", "content": f"Analyze the structural diversity of the following questions compared to the initial question. Provide a diversity score for each on a scale of 0 to 1, where 1 is highly diverse from the initial question:\n\nInitial question: {initial_question}\n\nQuestions to analyze: {json.dumps(questions[1:])}"}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "structural_diversity_analyzer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "diversity_scores": {
                                "type": "array",
                                "items": {
                                    "type": "number"
                                }
                            },
                            "explanation": {"type": "string"}
                        },
                        "required": ["diversity_scores", "explanation"],
                        "additionalProperties": False
                    }
                }
            }
        )
        
        json_response = response.choices[0].message.content
        logger.info(f"Raw JSON response: {json_response}")
        
        parsed_response = json.loads(json_response)
        diversity_scores = parsed_response["diversity_scores"]
        explanation = parsed_response["explanation"]
        
        logger.info(f"Structural Diversity Explanation: {explanation}")
        
        return [1.0] + diversity_scores  # Add 1.0 for the initial question
    except Exception as e:
        logger.error(f"Error in calculate_structural_diversity: {e}")
        return [1.0] + [0.5] * (len(questions) - 1)  # Return neutral scores in case of error

def calculate_semantic_similarity(questions: List[str], initial_question: str) -> List[float]:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert in semantic analysis, specializing in evaluating the similarity of questions."},
                {"role": "user", "content": f"Analyze the semantic similarity of the following questions to the initial question. Provide a similarity score for each question on a scale of 0 to 1, where 1 is highly similar to the initial question:\n\nInitial question: {initial_question}\n\nQuestions to analyze: {json.dumps(questions[1:])}"}
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
        
        return [1.0] + similarity_scores  # Add 1.0 for the initial question
    except Exception as e:
        logger.error(f"Error in calculate_semantic_similarity: {e}")
        return [1.0] + [0.5] * (len(questions) - 1)  # Return neutral scores in case of error

def calculate_edit_distance(questions: List[str], initial_question: str) -> List[float]:
    max_length = max(len(initial_question), max(len(q) for q in questions))
    distances = [levenshtein_distance(initial_question, q) for q in questions]
    normalized_distances = [1 - (d / max_length) for d in distances]
    return normalized_distances

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
                    {"role": "system", "content": "You are an expert in evaluating answer precision."},
                    {"role": "user", "content": f"""Given the context, evaluate how close the new answer is to the original answer. Provide a precision score from 0 to 1, where 1 means the answers are identical in meaning and 0 means they are completely unrelated.

Context: {context}
Original Answer: {original_answer}
New Answer: {generated_answer}

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


def calculate_composite_scores(sd_scores: List[float], ss_scores: List[float], ed_scores: List[float], ap_scores: List[float]) -> List[float]:
    return [0.25 * sd + 0.25 * ss + 0.25 * ed + 0.25 * ap for sd, ss, ed, ap in zip(sd_scores, ss_scores, ed_scores, ap_scores)]


def rank_questions_with_details(context: str, answer: str, initial_question: str) -> Tuple[pd.DataFrame, List[pd.DataFrame], str]:
    questions = generate_questions(context, answer, initial_question)
    
    sd_scores = calculate_structural_diversity(questions, initial_question)
    ss_scores = calculate_semantic_similarity(questions, initial_question)
    ed_scores = calculate_edit_distance(questions, initial_question)
    ap_scores, generated_answers = check_answer_precision(context, questions, answer)
    
    composite_scores = calculate_composite_scores(sd_scores, ss_scores, ed_scores, ap_scores)
    
    detailed_scores = pd.DataFrame({
        'Question': questions,
        'Structural Diversity': sd_scores,
        'Semantic Similarity': ss_scores,
        'Edit Distance': ed_scores,
        'Answer Precision': ap_scores,
        'Composite Score': composite_scores,
        'Generated Answer': generated_answers
    })
    detailed_scores = detailed_scores.sort_values('Composite Score', ascending=False).reset_index(drop=True)
    
    metrics = ['Structural Diversity', 'Semantic Similarity', 'Edit Distance', 'Answer Precision', 'Composite Score']
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
    
    best_question = detailed_scores.iloc[1]['Question']  # Select the best question excluding the initial question
    
    return detailed_scores, rankings, best_question

def gradio_interface(context: str, answer: str, initial_question: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    detailed_scores, rankings, best_question = rank_questions_with_details(context, answer, initial_question)
    return (
        detailed_scores,
        rankings[0],  # Structural Diversity Ranking
        rankings[1],  # Semantic Similarity Ranking
        rankings[2],  # Edit Distance Ranking
        rankings[3],  # Answer Precision Ranking
        rankings[4],  # Composite Score Ranking
        f"Best Generated Question: {best_question}"
    )

def use_sample(sample_index: int) -> Tuple[str, str, str]:
    return samples[sample_index]["context"], samples[sample_index]["answer"], samples[sample_index].get("question", "")

def get_random_entry():
    random_index = random.randint(0, len(dataset['train']) - 1)
    entry = dataset['train'][random_index]
    return entry['context'], entry['answer'], entry['question']


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
            best_question_output = gr.Textbox(label="Best Generated Question")
            detailed_scores_output = gr.DataFrame(label="Detailed Scores")
    
    with gr.Row():
        with gr.Column():
            structural_diversity_ranking_output = gr.DataFrame(label="Structural Diversity Ranking")
        with gr.Column():
            semantic_similarity_ranking_output = gr.DataFrame(label="Semantic Similarity Ranking")
    
    with gr.Row():
        with gr.Column():
            edit_distance_ranking_output = gr.DataFrame(label="Edit Distance Ranking")
        with gr.Column():
            answer_precision_ranking_output = gr.DataFrame(label="Answer Precision Ranking")
    
    with gr.Row():
        composite_ranking_output = gr.DataFrame(label="Composite Score Ranking")

    def process_random_entry():
        context, answer, initial_question = get_random_entry()
        return context, answer, initial_question

    submit_button.click(
        fn=gradio_interface,
        inputs=[context_input, answer_input, initial_question_input],
        outputs=[
            detailed_scores_output,
            structural_diversity_ranking_output,
            semantic_similarity_ranking_output,
            edit_distance_ranking_output,
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