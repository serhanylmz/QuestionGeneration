import cohere
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import List, Tuple
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import functions from the previous script
from pipeline_local import generate_questions
import os
from dotenv import load_dotenv

load_dotenv()  # This loads the variables from .env

# Initialize Cohere client, SentenceTransformer model, and QA pipeline
co = cohere.Client(api_key = os.environ.get("COHERE_API_KEY"))
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def calculate_question_similarity(q1: str, q2: str) -> float:
    """
    Calculate semantic similarity between two questions.
    """
    emb1 = sentence_model.encode(q1)
    emb2 = sentence_model.encode(q2)
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return (similarity + 1) / 2  # Normalize to 0-1 range

def benchmark_question_generation(dataset, num_samples=None):
    """
    Benchmark the question generation pipeline on the given dataset.
    """
    results = []

    # If num_samples is None, use the entire dataset
    if num_samples is None:
        num_samples = len(dataset)
    
    for i in tqdm(range(num_samples)):
        try:
            item = dataset[i]
            context = item['context']
            original_question = item['question']
            answer = item['answer']

            # Generate a single question
            generated_questions = generate_questions(context, answer)
            generated_question = generated_questions[0]  # Take the first generated question

            # Calculate similarity between original and generated question
            similarity_score = calculate_question_similarity(original_question, generated_question)

            results.append({
                'context': context,
                'answer': answer,
                'original_question': original_question,
                'generated_question': generated_question,
                'similarity_score': similarity_score
            })
        except Exception as e:
            logger.error(f"Error processing item {i}: {e}")
            results.append({
                'context': context if 'context' in locals() else "Error",
                'answer': answer if 'answer' in locals() else "Error",
                'original_question': original_question if 'original_question' in locals() else "Error",
                'generated_question': "Failed to generate",
                'similarity_score': 0.0
            })

    return pd.DataFrame(results)

if __name__ == "__main__":
    # Load the dataset
    ds = load_dataset("serhany/scaling-qa")

    # Print dataset information
    print("Dataset Structure:")
    print(ds)
    print("\nDataset Features:")
    print(ds['train'].features)
    print("\nFirst item in the dataset:")
    print(ds['train'][0])

    # Use the 'train' split of the dataset
    train_ds = ds['train']

    # Run the benchmark (adjust num_samples as needed)
    results_df = benchmark_question_generation(train_ds, num_samples=100)

    # Save the results to a CSV file
    results_df.to_csv("question_generation_benchmark_results.csv", index=False)

    print("Benchmark completed. Results saved to 'question_generation_benchmark_results.csv'.")

    # Print some summary statistics
    print("\nSummary Statistics:")
    print(f"Average Similarity Score: {results_df['similarity_score'].mean():.4f}")
    print(f"Median Similarity Score: {results_df['similarity_score'].median():.4f}")
    print(f"Min Similarity Score: {results_df['similarity_score'].min():.4f}")
    print(f"Max Similarity Score: {results_df['similarity_score'].max():.4f}")
    print(f"Number of failed generations: {(results_df['generated_question'] == 'Failed to generate').sum()}")