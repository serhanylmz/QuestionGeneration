import os
import logging
import json
import gradio as gr
import pandas as pd
from datasets import load_dataset
import random
from openai import OpenAI
from typing import List, Tuple, Dict
from dotenv import load_dotenv
import asyncio

# Import the required functions from the pipeline file
from pipeline import generate_basic_question, rank_questions_with_details, generate_answer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Load the SQuAD dataset
dataset = load_dataset("rajpurkar/squad")

def get_random_entry():
    random_index = random.randint(0, len(dataset['train']) - 1)
    entry = dataset['train'][random_index]
    return entry['context'], entry['answers']['text'][0], entry['question']

def compare_questions(context: str, original_question: str, original_answer: str, basic_question: str, basic_answer: str, enhanced_question: str, enhanced_answer: str) -> Dict[str, any]:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert in evaluating question-answer pairs based on a given context."},
                {"role": "user", "content": f"""Compare the following two generated question-answer pairs based on the given context and the original question-answer pair. Evaluate their quality and relevance.

Context: {context}

Original Question: {original_question}
Original Answer: {original_answer}

Basic Generated Question: {basic_question}
Basic Generated Answer: {basic_answer}

Enhanced Generated Question: {enhanced_question}
Enhanced Generated Answer: {enhanced_answer}

Evaluate the basic and enhanced generated questions based on the following criteria:
1. Structural difference from the original question
2. Semantic similarity to the original question
3. How well the generated answer matches the original answer

Score each generated question-answer pair on a scale of 0 to 10. Provide a detailed explanation for your evaluation, addressing each of the criteria mentioned above. Finally, determine which generation approach (Basic or Enhanced) is better overall and explain why."""}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "question_comparison_evaluator",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "basic_score": {"type": "number"},
                            "enhanced_score": {"type": "number"},
                            "explanation": {"type": "string"},
                            "winner": {"type": "string", "enum": ["Basic", "Enhanced"]}
                        },
                        "required": ["basic_score", "enhanced_score", "explanation", "winner"],
                        "additionalProperties": False
                    }
                }
            }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error in comparing questions: {e}")
        return {"basic_score": 0, "enhanced_score": 0, "explanation": "Failed to compare questions", "winner": "None"}

async def process_entry(context, answer, initial_question, progress=gr.Progress()):
    progress(0, desc="Starting process...")
    await asyncio.sleep(1)
    
    progress(0.2, desc="Generating basic question...")
    basic_question = generate_basic_question(context, answer, initial_question)
    
    progress(0.4, desc="Generating enhanced question...")
    detailed_scores, rankings, enhanced_question = rank_questions_with_details(context, answer, initial_question)
    
    progress(0.6, desc="Generating answers...")
    basic_answer = generate_answer(context, basic_question)
    enhanced_answer = generate_answer(context, enhanced_question)
    
    progress(0.8, desc="Comparing questions...")
    comparison_result = compare_questions(context, initial_question, answer, basic_question, basic_answer, enhanced_question, enhanced_answer)
    
    progress(1.0, desc="Process complete!")
    return (
        detailed_scores,
        rankings[0],  # Edit Distance Ranking
        rankings[1],  # Semantic Similarity Ranking
        rankings[2],  # Answer Precision Ranking
        rankings[3],  # Composite Score Ranking
        f"Original Question: {initial_question}\nOriginal Answer: {answer}",
        f"Basic Question: {basic_question}\nBasic Answer: {basic_answer}",
        f"Enhanced Question: {enhanced_question}\nEnhanced Answer: {enhanced_answer}",
        f"Basic Generation Score: {comparison_result['basic_score']}\n"
        f"Enhanced Generation Score: {comparison_result['enhanced_score']}\n"
        f"Explanation: {comparison_result['explanation']}\n"
        f"Winner: {comparison_result['winner']} Generation"
    )

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Default()) as iface:
    gr.Markdown("# Enhanced Question Generation and Comparison")
    gr.Markdown("Enter a context, answer, and initial question, or click 'Random' to get a random entry from the SQuAD dataset.")
    
    with gr.Row():
        with gr.Column(scale=2):
            context_input = gr.Textbox(label="Context", lines=10)
            answer_input = gr.Textbox(label="Answer", lines=2)
            initial_question_input = gr.Textbox(label="Initial Question", lines=2)
            with gr.Row():
                submit_button = gr.Button("Submit")
                random_button = gr.Button("Random")
        
        with gr.Column(scale=3):
            detailed_scores_output = gr.DataFrame(label="Detailed Scores")
            with gr.Row():
                edit_distance_ranking_output = gr.DataFrame(label="Edit Distance Ranking")
                semantic_similarity_ranking_output = gr.DataFrame(label="Semantic Similarity Ranking")
            with gr.Row():
                answer_precision_ranking_output = gr.DataFrame(label="Answer Precision Ranking")
                composite_ranking_output = gr.DataFrame(label="Composite Score Ranking")
            original_output = gr.Textbox(label="Original Question and Answer", lines=4)
            basic_generation_output = gr.Textbox(label="Basic Generation", lines=4)
            enhanced_generation_output = gr.Textbox(label="Enhanced Generation", lines=4)
            comparison_result_output = gr.Textbox(label="Comparison Result", lines=8)

    async def on_submit(context, answer, initial_question):
        return await process_entry(context, answer, initial_question)

    async def on_random():
        context, answer, initial_question = get_random_entry()
        results = await process_entry(context, answer, initial_question)
        return [context, answer, initial_question] + list(results)

    submit_button.click(
        fn=on_submit,
        inputs=[context_input, answer_input, initial_question_input],
        outputs=[
            detailed_scores_output,
            edit_distance_ranking_output,
            semantic_similarity_ranking_output,
            answer_precision_ranking_output,
            composite_ranking_output,
            original_output,
            basic_generation_output,
            enhanced_generation_output,
            comparison_result_output
        ]
    )

    random_button.click(
        fn=on_random,
        outputs=[
            context_input,
            answer_input,
            initial_question_input,
            detailed_scores_output,
            edit_distance_ranking_output,
            semantic_similarity_ranking_output,
            answer_precision_ranking_output,
            composite_ranking_output,
            original_output,
            basic_generation_output,
            enhanced_generation_output,
            comparison_result_output
        ]
    )

# Launch the app
if __name__ == "__main__":
    iface.launch()