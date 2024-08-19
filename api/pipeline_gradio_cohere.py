import cohere
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import List, Tuple
import os
import logging
import json
import gradio as gr
import pandas as pd
from datasets import load_dataset
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

# Initialize Cohere client, SentenceTransformer model, and QA pipeline
co = cohere.Client(api_key=os.environ.get("COHERE_API_KEY"))
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

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

def generate_questions(context: str, answer: str) -> List[str]:
    try:
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
        logger.info(f"Raw JSON response: {json_response}")
        
        parsed_response = json.loads(json_response)
        questions = [parsed_response[f"question{i}"] for i in range(1, 6)]
        return questions
    except Exception as e:
        logger.error(f"Error in generate_questions: {e}")
        return [f"Failed to generate question {i}" for i in range(1, 6)]

def calculate_structural_diversity(questions: List[str]) -> List[float]:
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
    context_embedding = sentence_model.encode(context + " " + answer)
    question_embeddings = sentence_model.encode(questions)
    
    similarities = [np.dot(context_embedding, q_emb) / (np.linalg.norm(context_embedding) * np.linalg.norm(q_emb)) 
                    for q_emb in question_embeddings]
    
    return [(sim + 1) / 2 for sim in similarities]  # Normalize to 0-1 range

def check_answer_precision(context: str, questions: List[str], original_answer: str) -> Tuple[List[float], List[str]]:
    precision_scores = []
    generated_answers = []
    for question in questions:
        result = qa_pipeline(question=question, context=context)
        generated_answer = result['answer']
        generated_answers.append(generated_answer)
        answer_embedding = sentence_model.encode(original_answer)
        generated_embedding = sentence_model.encode(generated_answer)
        similarity = np.dot(answer_embedding, generated_embedding) / (np.linalg.norm(answer_embedding) * np.linalg.norm(generated_embedding))
        precision_scores.append((similarity + 1) / 2)  # Normalize to 0-1 range
    return precision_scores, generated_answers

def calculate_composite_scores(sd_scores: List[float], sr_scores: List[float], ap_scores: List[float]) -> List[float]:
    # Normalize other scores based on answer precision
    max_other_score = max(max(sd_scores), max(sr_scores))
    normalized_sd_scores = [sd * (ap / max_other_score) for sd, ap in zip(sd_scores, ap_scores)]
    normalized_sr_scores = [sr * (ap / max_other_score) for sr, ap in zip(sr_scores, ap_scores)]
    
    # Calculate composite scores with higher weight for answer precision
    return [0.6 * ap + 0.2 * sd + 0.2 * sr for ap, sd, sr in zip(ap_scores, normalized_sd_scores, normalized_sr_scores)]

def rank_questions_with_details(context: str, answer: str) -> Tuple[pd.DataFrame, List[pd.DataFrame], str]:
    questions = generate_questions(context, answer)
    
    sd_scores = calculate_structural_diversity(questions)
    sr_scores = calculate_semantic_relevance(context, answer, questions)
    ap_scores, generated_answers = check_answer_precision(context, questions, answer)
    
    composite_scores = calculate_composite_scores(sd_scores, sr_scores, ap_scores)
    
    # Create detailed scores dataframe
    detailed_scores = pd.DataFrame({
        'Question': questions,
        'Answer Precision': ap_scores,
        'Composite Score': composite_scores,
        'Structural Diversity': sd_scores,
        'Semantic Relevance': sr_scores,
        'Generated Answer': generated_answers
    })
    detailed_scores = detailed_scores.sort_values('Answer Precision', ascending=False).reset_index(drop=True)
    
    # Create separate ranking dataframes for each metric
    metrics = ['Answer Precision', 'Composite Score', 'Structural Diversity', 'Semantic Relevance']
    rankings = []
    
    for metric in metrics:
        df = pd.DataFrame({
            'Rank': range(1, 6),
            'Question': [questions[i] for i in np.argsort(detailed_scores[metric])[::-1]],
            f'{metric}': sorted(detailed_scores[metric], reverse=True)
        })
        if metric == 'Answer Precision':
            df['Generated Answer'] = [generated_answers[i] for i in np.argsort(detailed_scores[metric])[::-1]]
        rankings.append(df)
    
    best_question = detailed_scores.iloc[0]['Question']
    
    return detailed_scores, rankings, best_question

def gradio_interface(context: str, answer: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    detailed_scores, rankings, best_question = rank_questions_with_details(context, answer)
    return (
        detailed_scores,
        rankings[0],  # Answer Precision Ranking
        rankings[1],  # Composite Score Ranking
        rankings[2],  # Structural Diversity Ranking
        rankings[3],  # Semantic Relevance Ranking
        f"Best Question: {best_question}"
    )

def use_sample(sample_index: int) -> Tuple[str, str]:
    return samples[sample_index]["context"], samples[sample_index]["answer"]

def get_random_entry():
    # Get a random entry from the dataset
    random_index = random.randint(0, len(dataset['train']) - 1)
    entry = dataset['train'][random_index]
    return entry['context'], entry['answer'], entry['question']

# Create Gradio interface with improved layout and sample buttons
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
            answer_precision_ranking_output = gr.DataFrame(label="Answer Precision Ranking")
        with gr.Column():
            composite_ranking_output = gr.DataFrame(label="Composite Score Ranking")
    
    with gr.Row():
        with gr.Column():
            structural_diversity_ranking_output = gr.DataFrame(label="Structural Diversity Ranking")
        with gr.Column():
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
            answer_precision_ranking_output,
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