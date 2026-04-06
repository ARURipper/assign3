"""
data/generate_json_instruct.py
Generate the teacher-model JSON Instruct dataset via imitation learning.

The teacher model (Llama 3.1 70B) generates structured JSON responses
to prompts we design. These become the Stage 2 training targets.

Run on HPC where the teacher model is available:
    python data/generate_json_instruct.py --task all
    python data/generate_json_instruct.py --task json_extraction
"""
import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DATA_DIR, PROMPTS_DIR, SEED, TEACHER_TEMPERATURE,
    TEACHER_MAX_NEW_TOKENS, JSON_TASK_TYPES, EXAMPLES_PER_TASK,
    JSON_TRAIN_SIZE, JSON_EVAL_SIZE,
)

random.seed(SEED)


def load_prompt_template(task_type: str) -> str:
    path = os.path.join(PROMPTS_DIR, f"teacher_{task_type}.txt")
    return Path(path).read_text(encoding="utf-8")


def is_valid_json(text: str) -> bool:
    """Check if text contains valid JSON (possibly wrapped in markdown)."""
    # Strip markdown code fences
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def extract_json(text: str) -> Optional[str]:
    """Extract and return the JSON portion from a response."""
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    try:
        obj = json.loads(text)
        return json.dumps(obj, indent=2)
    except json.JSONDecodeError:
        # Try to find JSON object/array within text
        for pattern in [r'\{.*\}', r'\[.*\]']:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                candidate = match.group(0)
                try:
                    obj = json.loads(candidate)
                    return json.dumps(obj, indent=2)
                except json.JSONDecodeError:
                    continue
        return None


def generate_with_teacher(
    prompt: str,
    pipeline,
    max_retries: int = 3
) -> Optional[str]:
    """Generate a response from the teacher model with retry on invalid JSON."""
    for attempt in range(max_retries):
        try:
            result = pipeline(
                prompt,
                max_new_tokens=TEACHER_MAX_NEW_TOKENS,
                temperature=TEACHER_TEMPERATURE,
                do_sample=True,
                return_full_text=False,
            )
            response_text = result[0]["generated_text"].strip()

            extracted = extract_json(response_text)
            if extracted is not None:
                return extracted

            print(f"  Attempt {attempt+1}: Invalid JSON, retrying...")
            time.sleep(1)

        except Exception as e:
            print(f"  Attempt {attempt+1} error: {e}")
            time.sleep(2)

    return None


def load_seed_examples(task_type: str) -> list:
    """Load seed prompt variations for each task type."""
    seeds = {
        "json_extraction": [
            {
                "instruction": "Extract all named entities from the following text and return them as a JSON object with keys: persons, organizations, locations.",
                "input": "Apple CEO Tim Cook announced a new partnership with Microsoft in Seattle yesterday.",
            },
            {
                "instruction": "Extract the invoice details from the text and return as JSON with keys: vendor, amount, date, invoice_number.",
                "input": "Invoice #INV-2024-0042 from TechSupply Co. for $1,250.00 dated March 15, 2024.",
            },
            {
                "instruction": "Parse the event details from the following text into a JSON object with: event_name, date, location, organizer.",
                "input": "The Annual AI Summit will be held on June 10th at the San Francisco Convention Center, organized by TechEvents Inc.",
            },
            {
                "instruction": "Extract product details from the description and return JSON with: name, price, category, availability.",
                "input": "The Sony WH-1000XM5 wireless headphones are priced at $349.99, available in the Electronics category, and currently in stock.",
            },
            {
                "instruction": "Extract all dates and associated events from the text as a JSON array of objects with 'date' and 'event' fields.",
                "input": "The project started on January 3rd, the first milestone was reached on February 14th, and the final delivery is scheduled for April 30th.",
            },
        ],
        "schema_constrained_generation": [
            {
                "instruction": "Generate a valid JSON object that conforms to the following schema: {\"name\": string, \"age\": integer, \"email\": string, \"is_active\": boolean, \"scores\": array of numbers}",
                "input": "Create a sample user record for a software engineer.",
            },
            {
                "instruction": "Generate a product listing JSON conforming to: {\"product_id\": string, \"name\": string, \"price\": float, \"category\": string, \"tags\": array, \"in_stock\": boolean}",
                "input": "Create a listing for a mechanical keyboard.",
            },
            {
                "instruction": "Generate a weather report JSON with schema: {\"location\": string, \"temperature_c\": number, \"humidity_percent\": integer, \"conditions\": string, \"wind_speed_kmh\": number, \"forecast\": array of strings}",
                "input": "Create a weather report for London.",
            },
            {
                "instruction": "Create a JSON API response conforming to: {\"status\": string, \"code\": integer, \"data\": object, \"errors\": array, \"timestamp\": string}",
                "input": "Represent a successful user login response.",
            },
            {
                "instruction": "Generate a restaurant menu item JSON with: {\"name\": string, \"description\": string, \"price\": float, \"calories\": integer, \"allergens\": array, \"available\": boolean}",
                "input": "Create a menu item for a classic cheeseburger.",
            },
        ],
        "exact_label_classification": [
            {
                "instruction": "Classify the sentiment of the following text. Return JSON with keys: label (one of: positive, negative, neutral), confidence (float 0-1), reasoning (string).",
                "input": "The new software update completely broke my workflow and cost me hours of productivity.",
            },
            {
                "instruction": "Classify the following email as spam or not_spam. Return JSON: {\"label\": \"spam\" or \"not_spam\", \"confidence\": float, \"key_indicators\": array}",
                "input": "Congratulations! You've been selected to receive $10,000. Click here to claim your prize immediately!",
            },
            {
                "instruction": "Classify the topic of the following news headline. Labels: politics, technology, sports, entertainment, business, science. Return JSON: {\"label\": string, \"confidence\": float}",
                "input": "NASA's James Webb Telescope captures earliest galaxy formation ever recorded.",
            },
            {
                "instruction": "Classify the intent of the following customer support message. Labels: complaint, inquiry, compliment, refund_request, technical_support. Return JSON: {\"label\": string, \"confidence\": float, \"summary\": string}",
                "input": "Hi, I ordered a laptop three weeks ago but it still hasn't arrived. The tracking number doesn't work either.",
            },
            {
                "instruction": "Classify the following code snippet by programming language. Labels: python, javascript, java, cpp, rust, go, unknown. Return JSON: {\"label\": string, \"confidence\": float}",
                "input": "fn main() { let mut v: Vec<i32> = Vec::new(); v.push(1); println!(\"{:?}\", v); }",
            },
        ],
        "json_repair": [
            {
                "instruction": "The following JSON is malformed. Fix all syntax errors and return valid, properly formatted JSON.",
                "input": "{\"name\": \"Alice\" \"age\": 30, \"city\": \"New York\",}",
            },
            {
                "instruction": "Repair the following broken JSON and return valid JSON only.",
                "input": "{'user_id': 12345, 'username': 'bob_smith', 'active': True, 'score': 98.5}",
            },
            {
                "instruction": "The following JSON has structural errors. Identify and fix them, returning only valid JSON.",
                "input": "[{\"id\": 1, \"name\": \"Alice\"}, {\"id\": 2, \"name\": \"Bob\", {\"id\": 3, \"name\": \"Charlie\"}]",
            },
            {
                "instruction": "Fix the malformed JSON below. Common issues: missing quotes, trailing commas, incorrect boolean values.",
                "input": "{\"status\": active, \"count\": 42, \"items\": [\"a\", \"b\", \"c\",], \"verified\": TRUE}",
            },
            {
                "instruction": "Repair this JSON string and ensure all values have correct types as implied by their field names.",
                "input": "{\"temperature\": \"72.5\", \"is_raining\": \"false\", \"wind_speed\": \"15\" \"humidity\": 80}",
            },
        ],
        "tool_call_generation": [
            {
                "instruction": "Generate a JSON tool call for the get_weather function. The function signature is: get_weather(city: str, unit: str = 'celsius', include_forecast: bool = False)",
                "input": "Get the weather for Tokyo in Fahrenheit with a 5-day forecast.",
            },
            {
                "instruction": "Generate a JSON tool call for: search_database(query: str, table: str, limit: int = 10, filters: dict = None). Return: {\"function\": string, \"arguments\": object}",
                "input": "Search the users table for active accounts created in the last 30 days, limit to 25 results.",
            },
            {
                "instruction": "Create a JSON tool call for the send_email function: send_email(to: list, subject: str, body: str, cc: list = [], priority: str = 'normal')",
                "input": "Send an urgent meeting reminder to alice@example.com and bob@example.com, CC the manager at mgr@example.com.",
            },
            {
                "instruction": "Generate a tool call JSON for: create_ticket(title: str, description: str, priority: str, assignee: str, tags: list). Valid priorities: low, medium, high, critical.",
                "input": "Create a high-priority bug ticket about the login page crash, assign to the backend team.",
            },
            {
                "instruction": "Produce a JSON tool call for: resize_image(input_path: str, width: int, height: int, format: str = 'png', quality: int = 95)",
                "input": "Resize the image at /uploads/photo.jpg to 800x600 pixels and save as JPEG with 85% quality.",
            },
        ],
    }
    return seeds.get(task_type, [])


def generate_dataset(task_type: str, pipeline, n_examples: int) -> list:
    """Generate n_examples for a given task type using the teacher model."""
    seeds = load_seed_examples(task_type)
    template = load_prompt_template(task_type)
    examples = []
    attempts = 0
    max_attempts = n_examples * 4

    print(f"\nGenerating {n_examples} examples for task: {task_type}")

    while len(examples) < n_examples and attempts < max_attempts:
        seed = random.choice(seeds)
        prompt = template.format(
            instruction=seed["instruction"],
            input=seed.get("input", ""),
        )

        response = generate_with_teacher(prompt, pipeline)
        attempts += 1

        if response is not None:
            examples.append({
                "task_type":   task_type,
                "instruction": seed["instruction"],
                "input":       seed.get("input", ""),
                "output":      response,
            })
            if len(examples) % 50 == 0:
                print(f"  Progress: {len(examples)}/{n_examples}")

    print(f"  Generated {len(examples)} valid examples in {attempts} attempts.")
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all",
                        choices=["all"] + JSON_TASK_TYPES)
    parser.add_argument("--model_path", default=None,
                        help="Path to teacher model on HPC")
    args = parser.parse_args()

    # Load teacher model pipeline
    from transformers import pipeline as hf_pipeline
    import torch

    model_path = args.model_path or "meta-llama/Llama-3.1-70B-Instruct"
    print(f"Loading teacher model: {model_path}")
    pipe = hf_pipeline(
        "text-generation",
        model=model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    tasks = JSON_TASK_TYPES if args.task == "all" else [args.task]
    all_examples = []

    for task in tasks:
        examples = generate_dataset(task, pipe, EXAMPLES_PER_TASK)
        task_path = os.path.join(DATA_DIR, f"json_instruct_{task}.json")
        Path(task_path).write_text(json.dumps(examples, indent=2), encoding="utf-8")
        all_examples.extend(examples)
        print(f"  Saved {len(examples)} examples to {task_path}")

    # Combine all tasks, shuffle, and split
    random.shuffle(all_examples)
    train = all_examples[:JSON_TRAIN_SIZE]
    eval_ = all_examples[JSON_TRAIN_SIZE:JSON_TRAIN_SIZE + JSON_EVAL_SIZE]

    Path(os.path.join(DATA_DIR, "json_instruct_train.json")).write_text(
        json.dumps(train, indent=2), encoding="utf-8"
    )
    Path(os.path.join(DATA_DIR, "json_instruct_eval.json")).write_text(
        json.dumps(eval_, indent=2), encoding="utf-8"
    )

    print(f"\nFinal dataset: {len(train)} train, {len(eval_)} eval examples")
    print("Done.")


if __name__ == "__main__":
    main()
