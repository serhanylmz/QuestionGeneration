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

def check_answer_correctness(context: str, question: str, answer: str) -> bool:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert in evaluating answer correctness."},
                {"role": "user", "content": f"""Given the context and the question, evaluate if the provided answer is correct. Return true if the answer is correct, and false if it's incorrect.

Example:
Context: The end of medieval drama came about due to a number of factors, including the weakening power of the Catholic Church, the Protestant Reformation and the banning of religious plays in many countries. Elizabeth I forbid all religious plays in 1558 and the great cycle plays had been silenced by the 1580s. Similarly, religious plays were banned in the Netherlands in 1539, the Papal States in 1547 and in Paris in 1548.
Question: What was banned that led to the demise of medieval drama?
Answer: religious plays
Correctness: true

Now evaluate the following:
Context: {context}
Question: {question}
Answer: {answer}

Provide only true or false as the response."""}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer_correctness_evaluator",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "is_correct": {"type": "boolean"}
                        },
                        "required": ["is_correct"],
                        "additionalProperties": False
                    }
                }
            }
        )
        
        json_response = response.choices[0].message.content
        parsed_response = json.loads(json_response)
        return parsed_response["is_correct"]
    except Exception as e:
        logger.error(f"Error in check_answer_correctness: {e}")
        return False  # Assume incorrect in case of error

def generate_single_question(context: str, answer: str, existing_questions: List[str]) -> str:
    try:
        existing_questions_str = "\n".join(existing_questions)
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates diverse questions based on given context and answer."},
                {"role": "user", "content": f"Based on this context: '{context}' and answer: '{answer}', generate a single question which when asked to the context returns the answer. The question should be distinct from these existing questions:\n{existing_questions_str}\n\nProvide only the new question, without any additional text."}
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

def generate_questions(context: str, answer: str) -> List[str]:
    questions = []
    max_attempts = 50  # Increased maximum number of attempts
    
    while len(questions) < 5 and max_attempts > 0:
        new_question = generate_single_question(context, answer, questions)
        if new_question != "Failed to generate question" and is_question_distinct(new_question, questions):
            if check_answer_correctness(context, new_question, answer):
                questions.append(new_question)
            else:
                max_attempts -= 1
        else:
            max_attempts -= 1
    
    # If we couldn't generate 5 distinct questions with correct answers, fill the rest with placeholder messages
    while len(questions) < 5:
        questions.append(f"Failed to generate distinct question with correct answer {len(questions) + 1}")
    
    return questions

def calculate_structural_diversity(questions: List[str]) -> List[float]:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert in linguistic analysis, specializing in question structure and diversity."},
                {"role": "user", "content": f"Analyze the structural diversity of the following questions and provide a diversity score for each on a scale of 0 to 1, where 1 is highly diverse:\n\n{json.dumps(questions)}"}
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
        
        return diversity_scores
    except Exception as e:
        logger.error(f"Error in calculate_structural_diversity: {e}")
        return [0.5] * len(questions)  # Return neutral scores in case of error

def calculate_semantic_relevance(context: str, answer: str, questions: List[str]) -> List[float]:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert in semantic analysis, specializing in evaluating the relevance of questions to a given context and answer."},
                {"role": "user", "content": f"Analyze the semantic relevance of the following questions to the given context and answer. Provide a relevance score for each question on a scale of 0 to 1, where 1 is highly relevant:\n\nContext: {context}\nAnswer: {answer}\nQuestions: {json.dumps(questions)}"}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "semantic_relevance_analyzer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "relevance_scores": {
                                "type": "array",
                                "items": {
                                    "type": "number"
                                }
                            },
                            "explanation": {"type": "string"}
                        },
                        "required": ["relevance_scores", "explanation"],
                        "additionalProperties": False
                    }
                }
            }
        )
        
        json_response = response.choices[0].message.content
        logger.info(f"Raw JSON response: {json_response}")
        
        parsed_response = json.loads(json_response)
        relevance_scores = parsed_response["relevance_scores"]
        explanation = parsed_response["explanation"]
        
        logger.info(f"Semantic Relevance Explanation: {explanation}")
        
        return relevance_scores
    except Exception as e:
        logger.error(f"Error in calculate_semantic_relevance: {e}")
        return [0.5] * len(questions)  # Return neutral scores in case of error

def calculate_composite_scores(sd_scores: List[float], sr_scores: List[float]) -> List[float]:
    return [0.5 * sd + 0.5 * sr for sd, sr in zip(sd_scores, sr_scores)]

def rank_questions_with_details(context: str, answer: str) -> Tuple[pd.DataFrame, List[pd.DataFrame], str]:
    questions = generate_questions(context, answer)
    
    sd_scores = calculate_structural_diversity(questions)
    sr_scores = calculate_semantic_relevance(context, answer, questions)
    
    composite_scores = calculate_composite_scores(sd_scores, sr_scores)
    
    # Create detailed scores dataframe
    detailed_scores = pd.DataFrame({
        'Question': questions,
        'Composite Score': composite_scores,
        'Structural Diversity': sd_scores,
        'Semantic Relevance': sr_scores
    })
    detailed_scores = detailed_scores.sort_values('Composite Score', ascending=False).reset_index(drop=True)
    
    # Create separate ranking dataframes for each metric
    metrics = ['Composite Score', 'Structural Diversity', 'Semantic Relevance']
    rankings = []
    
    for metric in metrics:
        df = pd.DataFrame({
            'Rank': range(1, 6),
            'Question': [questions[i] for i in np.argsort(detailed_scores[metric])[::-1]],
            f'{metric}': sorted(detailed_scores[metric], reverse=True)
        })
        rankings.append(df)
    
    best_question = detailed_scores.iloc[0]['Question']
    
    return detailed_scores, rankings, best_question

def gradio_interface(context: str, answer: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    detailed_scores, rankings, best_question = rank_questions_with_details(context, answer)
    return (
        detailed_scores,
        rankings[0],  # Composite Score Ranking
        rankings[1],  # Structural Diversity Ranking
        rankings[2],  # Semantic Relevance Ranking
        f"Best Question: {best_question}"
    )

def use_sample(sample_index: int) -> Tuple[str, str]:
    return samples[sample_index]["context"], samples[sample_index]["answer"]

def get_random_entry():
    random_index = random.randint(0, len(dataset['train']) - 1)
    entry = dataset['train'][random_index]
    return entry['context'], entry['answer'], entry['question']

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Default()) as iface:
    gr.Markdown("# Question Generator and Ranker")
    gr.Markdown("Enter a context and an answer to generate and rank questions, use one of the sample inputs, or get a random entry from the dataset.")
    
    with gr.Row():
        with gr.Column(scale=1):
            context_input = gr.Textbox(lines=5, label="Context")
            answer_input = gr.Textbox(lines=2, label="Answer")
            submit_button = gr.Button("Generate Questions")
            
            with gr.Row():
                sample_buttons = [gr.Button(f"Sample {i+1}") for i in range(3)]
                random_button = gr.Button("Random Dataset Entry")
        
        with gr.Column(scale=2):
            original_question_output = gr.Dataframe(label="Original Question from Dataset", visible=False)
            best_question_output = gr.Textbox(label="Best Generated Question")
            detailed_scores_output = gr.DataFrame(label="Detailed Scores")
    
    with gr.Row():
        with gr.Column():
            composite_ranking_output = gr.DataFrame(label="Composite Score Ranking")
        with gr.Column():
            structural_diversity_ranking_output = gr.DataFrame(label="Structural Diversity Ranking")
    
    with gr.Row():
        semantic_relevance_ranking_output = gr.DataFrame(label="Semantic Relevance Ranking")

    def process_random_entry():
        context, answer, original_question = get_random_entry()
        return (
            context, 
            answer, 
            pd.DataFrame({'Original Question': [original_question]}),
            gr.update(visible=True)
        )

    submit_button.click(
        fn=gradio_interface,
        inputs=[context_input, answer_input],
        outputs=[
            detailed_scores_output,
            composite_ranking_output,
            structural_diversity_ranking_output,
            semantic_relevance_ranking_output,
            best_question_output
        ]
    )

    # Set up sample button functionality
    for i, button in enumerate(sample_buttons):
        button.click(
            fn=lambda i=i: use_sample(i),
            outputs=[context_input, answer_input]
        )

    # Set up random button functionality
    random_button.click(
        fn=process_random_entry,
        outputs=[
            context_input, 
            answer_input, 
            original_question_output,
            original_question_output
        ]
    )

# Launch the app
if __name__ == "__main__":
    iface.launch()