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

def generate_basic_question(context: str, answer: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates diverse questions based on given context and answer."},
                {"role": "user", "content": f"Based on this context: '{context}' and answer: '{answer}', generate a single question which when asked to the context returns the answer. Provide only the question, without any additional text."}
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

############################# Experimental Code - Begin ########################################

# def is_question_distinct(new_question: str, existing_questions: List[str]) -> bool:
#     try:
#         existing_questions_str = "\n".join(existing_questions)
#         response = client.chat.completions.create(
#             model="gpt-4o-2024-08-06",
#             messages=[
#                 {"role": "system", "content": "You are an expert in linguistic analysis, specializing in question comparison."},
#                 {"role": "user", "content": f"Compare the following new question to the list of existing questions. Determine if the new question is distinct in meaning and structure from all existing questions. Respond with true if it's distinct, false if it's too similar to any existing question.\n\nNew question: {new_question}\n\nExisting questions:\n{existing_questions_str}"}
#             ],
#             response_format={
#                 "type": "json_schema",
#                 "json_schema": {
#                     "name": "distinctness_checker",
#                     "strict": True,
#                     "schema": {
#                         "type": "object",
#                         "properties": {
#                             "is_distinct": {"type": "boolean"}
#                         },
#                         "required": ["is_distinct"],
#                         "additionalProperties": False
#                     }
#                 }
#             }
#         )
        
#         json_response = response.choices[0].message.content
#         logger.info(f"Raw JSON response: {json_response}")
        
#         parsed_response = json.loads(json_response)
#         return parsed_response["is_distinct"]
#     except Exception as e:
#         logger.error(f"Error in is_question_distinct: {e}")
#         return False  # Assume not distinct in case of error

############################# Experimental Code - End ########################################

def is_question_distinct(new_question: str, existing_questions: List[str]) -> bool:
    # Convert the new question to lowercase and remove any leading/trailing whitespace
    new_question_normalized = new_question.strip().lower()
    
    # Check if the normalized new question is already in the list of existing questions
    for question in existing_questions:
        if new_question_normalized == question.strip().lower():
            return False
    
    # If we've made it through the loop, the question is distinct
    return True

def generate_questions(context: str, answer: str) -> List[str]:
    questions = []
    max_attempts = 10  # Maximum number of attempts to generate distinct questions
    
    while len(questions) < 5 and max_attempts > 0:
        new_question = generate_single_question(context, answer, questions)
        if new_question != "Failed to generate question" and is_question_distinct(new_question, questions):
            questions.append(new_question)
        else:
            max_attempts -= 1
    
    # If we couldn't generate 5 distinct questions, fill the rest with placeholder messages
    while len(questions) < 5:
        questions.append(f"Failed to generate distinct question {len(questions) + 1}")
    
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

# def check_answer_precision(context: str, questions: List[str], original_answer: str) -> Tuple[List[float], List[str]]:
#     precision_scores = []
#     generated_answers = []
#     for question in questions:
#         generated_answer = generate_answer(context, question)
#         generated_answers.append(generated_answer)
        
#         # Use OpenAI to evaluate answer precision
#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o-2024-08-06",
#                 messages=[
#                     {"role": "system", "content": "You are an expert in evaluating answer precision."},
#                     {"role": "user", "content": f"Compare the following two answers and provide a precision score from 0 to 1, where 1 means the answers are identical in meaning or point in the same direction. For example, Answer 1: religious plays and Answer 2: Elizabeth I forbade all religious plays in 1558.
# :\n\nOriginal Answer: {original_answer}\nGenerated Answer: {generated_answer}"}
#                 ],
#                 response_format={
#                     "type": "json_schema",
#                     "json_schema": {
#                         "name": "answer_precision_evaluator",
#                         "strict": True,
#                         "schema": {
#                             "type": "object",
#                             "properties": {
#                                 "precision_score": {
#                                     "type": "number"
#                                 }
#                             },
#                             "required": ["precision_score"],
#                             "additionalProperties": False
#                         }
#                     }
#                 }
#             )
            
#             json_response = response.choices[0].message.content
#             parsed_response = json.loads(json_response)
#             precision_score = parsed_response["precision_score"]
#             precision_scores.append(precision_score)
#         except Exception as e:
#             logger.error(f"Error in evaluating answer precision: {e}")
#             precision_scores.append(0.5)  # Neutral score in case of error
    
#     return precision_scores, generated_answers

def check_answer_precision(context: str, questions: List[str], original_answer: str) -> Tuple[List[float], List[str]]:
    precision_scores = []
    generated_answers = []
    for question in questions:
        generated_answer = generate_answer(context, question)
        generated_answers.append(generated_answer)
        
        # Use OpenAI to evaluate answer precision
        try:
            response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "You are an expert in evaluating answer precision."},
                    {"role": "user", "content": f"""Given the context, the original question, and the original answer, evaluate how close the new answer is to the original answer. Provide a precision score from 0 to 1, where 1 means the answers are identical in meaning and 0 means they are completely unrelated.

Example:
Context: The end of medieval drama came about due to a number of factors, including the weakening power of the Catholic Church, the Protestant Reformation and the banning of religious plays in many countries. Elizabeth I forbid all religious plays in 1558 and the great cycle plays had been silenced by the 1580s. Similarly, religious plays were banned in the Netherlands in 1539, the Papal States in 1547 and in Paris in 1548. The abandonment of these plays destroyed the international theatre that had thereto existed and forced each country to develop its own form of drama. It also allowed dramatists to turn to secular subjects and the reviving interest in Greek and Roman theatre provided them with the perfect opportunity.
Question: What was banned that led to the demise of medieval drama?
Original Answer: religious plays
New Answer: Elizabeth I forbade all religious plays in 1558.
Precision Score: 0.6

New Answer: Religious plays were suppressed.
Precision Score: 0.75

New Answer: Religious plays
Precision Score: 1.0

Now evaluate the following:
Context: {context}
Question: {question}
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


def calculate_composite_scores(sd_scores: List[float], sr_scores: List[float], ap_scores: List[float]) -> List[float]:
    return [0.3 * sd + 0.3 * sr + 0.4 * ap for sd, sr, ap in zip(sd_scores, sr_scores, ap_scores)]

# def rank_questions_with_details(context: str, answer: str) -> Tuple[pd.DataFrame, List[pd.DataFrame], str]:
#     questions = generate_questions(context, answer)
    
#     sd_scores = calculate_structural_diversity(questions)
#     sr_scores = calculate_semantic_relevance(context, answer, questions)
#     ap_scores, generated_answers = check_answer_precision(context, questions, answer)
    
#     composite_scores = calculate_composite_scores(sd_scores, sr_scores, ap_scores)
    
#     # Create detailed scores dataframe
#     detailed_scores = pd.DataFrame({
#         'Question': questions,
#         'Answer Precision': ap_scores,
#         'Composite Score': composite_scores,
#         'Structural Diversity': sd_scores,
#         'Semantic Relevance': sr_scores,
#         'Generated Answer': generated_answers
#     })
#     detailed_scores = detailed_scores.sort_values('Composite Score', ascending=False).reset_index(drop=True)
    
#     # Create separate ranking dataframes for each metric
#     metrics = ['Answer Precision', 'Composite Score', 'Structural Diversity', 'Semantic Relevance']
#     rankings = []
    
#     for metric in metrics:
#         df = pd.DataFrame({
#             'Rank': range(1, 6),
#             'Question': [questions[i] for i in np.argsort(detailed_scores[metric])[::-1]],
#             f'{metric}': sorted(detailed_scores[metric], reverse=True)
#         })
#         if metric == 'Answer Precision':
#             df['Generated Answer'] = [generated_answers[i] for i in np.argsort(detailed_scores[metric])[::-1]]
#         rankings.append(df)
    
#     best_question = detailed_scores.iloc[0]['Question']
    
#     return detailed_scores, rankings, best_question

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
    detailed_scores = detailed_scores.sort_values('Composite Score', ascending=False).reset_index(drop=True)
    
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