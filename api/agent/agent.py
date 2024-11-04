import os
import json
import logging
import datetime
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize Anthropics client for Claude
import anthropic
claude_client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY"))

# Set up logging to save to a JSON file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_filename = "paraphrase_system_log.json"

def log_event(event_type: str, data: dict, logs: list):
    """
    Logs events by appending them to the logs list.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    log_entry = {
        "timestamp": timestamp,
        "event_type": event_type,
        "data": data
    }
    logs.append(log_entry)

def save_logs(logs: list):
    """
    Saves the logs to a JSON file.
    """
    with open(log_filename, 'w') as f:
        json.dump(logs, f, indent=2)

def paraphrase_text(input_text: str, prompt: str, logs: list) -> str:
    """
    Generator agent that creates a paraphrased version of the input text.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "paraphrase_generator",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "paraphrased_text": {"type": "string"}
                        },
                        "required": ["paraphrased_text"],
                        "additionalProperties": False
                    }
                }
            }
        )
        json_response = response.choices[0].message.content
        parsed_response = json.loads(json_response)
        paraphrased_text = parsed_response["paraphrased_text"]
        # Log the event
        log_event("generator", {
            "generator_prompt": prompt,
            "input_text": input_text,
            "paraphrased_text": paraphrased_text
        }, logs)
        return paraphrased_text
    except Exception as e:
        logger.error(f"Error in paraphrase_text: {e}")
        return None

def evaluate_texts(text1: str, text2: str, eval_history: list, criteria_weights: dict, logs: list) -> dict:
    """
    Evaluator agent that compares two texts and chooses the better one.
    Maintains full conversation history between evaluations.
    """
    try:
        # Prepare the messages with the conversation history
        
        # Define the tool (function) with the expected output schema
        tool = {
            "name": "text_evaluator",
            "description": "Evaluate and compare two generated paraphrases and output the result in structured JSON.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "better_text": {"type": "string", "enum": ["Text A", "Text B"]},
                    "reasoning": {"type": "string"}
                },
                "required": ["better_text", "reasoning"],
                "additionalProperties": False
            }
        }

        messages = eval_history.copy()
        messages.append({
            "role": "user", 
            "content": "You are an evaluator that compares two texts blindly and chooses the better one based on the given criteria."
        })
        messages.append({
            "role": "user", 
            "content": f"Compare the following two texts:\n\nText A:\n{text1}\n\nText B:\n{text2}\n\nCriteria weights:\n{json.dumps(criteria_weights)}\n\nProvide your evaluation and choose the better text. Always choose one as better and provide detailed reasoning."
        })
        
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
            tools=[tool],
            tool_choice={"type": "tool", "name": "text_evaluator"}
        )

        # The response.content[0].input is already a dictionary, no need to parse it
        parsed_response = response.content[0].input

        # Append the latest messages to the eval_history
        messages.append({"role": "assistant", "content": json.dumps(parsed_response)})  # Convert dict to JSON string for history
        eval_history.extend(messages[-2:])

        # Log the event
        log_event("evaluator", {
            "text_a": text1,
            "text_b": text2,
            "criteria_weights": criteria_weights,
            "better_text": parsed_response["better_text"],
            "reasoning": parsed_response["reasoning"]
        }, logs)

        return {
            "better_text": parsed_response["better_text"],
            "reasoning": parsed_response["reasoning"],
            "eval_history": eval_history
        }

    except Exception as e:
        logger.error(f"Error in evaluate_texts: {e}")
        return None

def generate_new_prompt(previous_prompt: str, evaluation_feedback: str, attempt_history: list, logs: list) -> str:
    """
    Prompt Generator agent that creates an improved prompt based on previous prompt and evaluation feedback.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are a prompt generator that improves prompts based on evaluation feedback and attempt history to generate better paraphrases."},
                {"role": "user", "content": f"Previous Prompt:\n{previous_prompt}\n\nEvaluation Feedback:\n{evaluation_feedback}\n\nAttempt History:\n{json.dumps(attempt_history, indent=2)}\n\nGenerate an improved prompt that addresses the shortcomings identified in the evaluation feedback."}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "prompt_generator",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "new_prompt": {"type": "string"}
                        },
                        "required": ["new_prompt"],
                        "additionalProperties": False
                    }
                }
            }
        )
        json_response = response.choices[0].message.content
        parsed_response = json.loads(json_response)
        new_prompt = parsed_response["new_prompt"]
        # Log the event
        log_event("prompt_generator", {
            "previous_prompt": previous_prompt,
            "evaluation_feedback": evaluation_feedback,
            "attempt_history": attempt_history,
            "new_prompt": new_prompt
        }, logs)
        return new_prompt
    except Exception as e:
        logger.error(f"Error in generate_new_prompt: {e}")
        return None

def paraphrase_system(input_text: str, max_iterations: int = 5, retry_limit: int = 3):
    """
    Main workflow function that orchestrates the paraphrasing system.
    """
    # Initial prompt for the generator
    generator_prompt = "You are a helpful assistant that paraphrases text while maintaining the original meaning."
    # Criteria weights (start with equal weights)
    criteria_weights = {
        "fluency": 1.0,
        "clarity": 1.0,
        "semantic_preservation": 1.0,
        "coherence": 1.0,
        "grammar": 1.0
    }
    # Attempt history
    attempt_history = []
    # Evaluation conversation history
    eval_history = []
    # Logs list
    logs = []
    # Initialize variables
    iteration = 0
    retries = 0

    while iteration < max_iterations and retries < retry_limit:
        iteration += 1
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        print(f"Starting iteration {iteration}")

        # Generator creates paraphrase
        paraphrased_text = paraphrase_text(input_text, generator_prompt, logs)
        if not paraphrased_text:
            retries += 1
            continue  # Try again

        # Evaluator compares the texts
        eval_result = evaluate_texts(input_text, paraphrased_text, eval_history, criteria_weights, logs)
        if not eval_result:
            retries += 1
            continue  # Try again

        # Append to attempt history
        attempt = {
            "iteration": iteration,
            "timestamp": timestamp,
            "generator_prompt": generator_prompt,
            "paraphrased_text": paraphrased_text,
            "evaluation_result": eval_result
        }
        attempt_history.append(attempt)

        # Check if paraphrase is better
        if eval_result["better_text"] == "Text B":
            # Paraphrased text is better
            print(f"Iteration {iteration} result: Paraphrased text is better.")
            log_event("final_result", {
                "paraphrased_text": paraphrased_text,
                "evaluation_result": eval_result,
                "attempt_history": attempt_history,
                "eval_conversation_history": eval_history
            }, logs)
            save_logs(logs)
            return paraphrased_text
        else:
            # Original text is better
            print(f"Iteration {iteration} result: Original text is better.")
            # Use Prompt Generator to create new prompt
            new_prompt = generate_new_prompt(generator_prompt, eval_result["reasoning"], attempt_history, logs)
            if not new_prompt:
                retries += 1
                continue  # Try again
            # Update the generator prompt
            generator_prompt = new_prompt
            retries = 0  # Reset retries since we got a new prompt
    # If reached here, max iterations or retries exceeded
    print("Max iterations or retries exceeded.")
    log_event("final_result", {
        "paraphrased_text": input_text,
        "evaluation_result": "Max iterations or retries exceeded.",
        "attempt_history": attempt_history,
        "eval_conversation_history": eval_history
    }, logs)
    save_logs(logs)
    return input_text

if __name__ == "__main__":
    print("Welcome to the Paraphrasing System")
    input_text = input("Please enter the text you want to paraphrase:\n")
    result = paraphrase_system(input_text)
    print("\nFinal Paraphrased Text:")
    print(result)