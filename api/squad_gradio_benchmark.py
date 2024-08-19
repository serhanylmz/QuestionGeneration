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
from transformers import pipeline
import asyncio

# Import the required functions from the pipeline file
from pipeline_gradio_experimental import generate_single_question, rank_questions_with_details

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Load the SQuAD dataset
dataset = load_dataset("squad")

# Initialize the question answering pipeline
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def get_random_entry():
    random_index = random.randint(0, len(dataset['train']) - 1)
    entry = dataset['train'][random_index]
    return entry['context'], entry['answers']['text'][0], entry['question']

def generate_answer(context: str, question: str) -> str:
    try:
        result = qa_pipeline(question=question, context=context)
        return result['answer']
    except Exception as e:
        logger.error(f"Error in generate_answer: {e}")
        return "Failed to generate answer"

def compare_questions(context: str, original_answer: str, question1: str, answer1: str, question2: str, answer2: str) -> Dict[str, any]:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert in evaluating question-answer pairs based on a given context."},
                {"role": "user", "content": f"""Compare the following two question-answer pairs based on the given context and original answer. Evaluate their quality and relevance.

Context: {context}
Original Answer: {original_answer}

Question 1: {question1}
Answer 1: {answer1}

Question 2: {question2}
Answer 2: {answer2}

Score each question-answer pair on a scale of 0 to 10 based on the quality and relevance of the question and answer. Provide an explanation for your evaluation. Focus on how well the new answer matches the old answer considering the context. Make sure to grade one higher than the other."""}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "question_comparison_evaluator",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "question1_score": {"type": "number"},
                            "question2_score": {"type": "number"},
                            "explanation": {"type": "string"}
                        },
                        "required": ["question1_score", "question2_score", "explanation"],
                        "additionalProperties": False
                    }
                }
            }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error in comparing questions: {e}")
        return {"question1_score": 0, "question2_score": 0, "explanation": "Failed to compare questions"}

async def process_entry(context, answer, progress=gr.Progress()):
    progress(0, desc="Starting process...")
    await asyncio.sleep(1)
    
    progress(0.2, desc="Generating questions...")
    basic_question = generate_single_question(context, answer, [])
    _, _, enhanced_question = rank_questions_with_details(context, answer)
    
    progress(0.4, desc="Generating answers...")
    basic_answer = generate_answer(context, basic_question)
    enhanced_answer = generate_answer(context, enhanced_question)
    
    progress(0.6, desc="Comparing questions...")
    comparison_result = compare_questions(context, answer, basic_question, basic_answer, enhanced_question, enhanced_answer)
    
    winner = "Basic" if comparison_result["question1_score"] > comparison_result["question2_score"] else "Enhanced"
    
    progress(1.0, desc="Process complete!")
    return (
        f"Question: {basic_question}\nAnswer: {basic_answer}",
        f"Question: {enhanced_question}\nAnswer: {enhanced_answer}",
        f"Question 1 Score: {comparison_result['question1_score']}\n"
        f"Question 2 Score: {comparison_result['question2_score']}\n"
        f"Explanation: {comparison_result['explanation']}\n"
        f"Winner: {winner} Generation"
    )

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Default()) as iface:
    gr.Markdown("# Question Generation and Comparison")
    gr.Markdown("Enter a context and answer, or click 'Random' to get a random entry from the SQuAD dataset.")
    
    with gr.Row():
        with gr.Column(scale=2):
            context_input = gr.Textbox(label="Context", lines=10)
            answer_input = gr.Textbox(label="Answer", lines=2)
            with gr.Row():
                submit_button = gr.Button("Submit")
                random_button = gr.Button("Random")
        
        with gr.Column(scale=3):
            original_question_output = gr.Textbox(label="Original Question from Dataset", lines=2)
            basic_generation_output = gr.Textbox(label="Basic Generation", lines=4)
            enhanced_generation_output = gr.Textbox(label="Enhanced Generation", lines=4)
            comparison_result_output = gr.Textbox(label="Comparison Result", lines=6)

    async def on_submit(context, answer):
        return await process_entry(context, answer)

    async def on_random():
        context, answer, question = get_random_entry()
        results = await process_entry(context, answer)
        return [context, answer, question] + list(results)

    submit_button.click(
        fn=on_submit,
        inputs=[context_input, answer_input],
        outputs=[basic_generation_output, enhanced_generation_output, comparison_result_output]
    )

    random_button.click(
        fn=on_random,
        outputs=[
            context_input,
            answer_input,
            original_question_output,
            basic_generation_output,
            enhanced_generation_output,
            comparison_result_output
        ]
    )

# Launch the app
if __name__ == "__main__":
    iface.launch()