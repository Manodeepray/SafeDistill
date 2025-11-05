import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from tqdm import tqdm
import re
import numpy as np
import warnings

# Suppress a specific UserWarning from the LlamaGuard tokenizer
warnings.filterwarnings("ignore", message="A decoder-only architecture is being used.*")

def load_finetuned_model_and_tokenizer(model_path, device):
    """
    Loads a fine-tuned model and its tokenizer from a local directory.

    Args:
        model_path (str): The path to the local directory containing the model.
        device (str): The device to load the model onto ('cuda' or 'cpu').

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    print(f"Loading fine-tuned model from: {model_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
            trust_remote_code=True
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Fine-tuned model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading the model: {e}")
        exit()

def extract_gsm8k_answer(pred_str):
    """
    Extracts the final numerical answer from a GSM8K model's prediction string.
    It looks for the last number in the string, especially after '####'.
    """
    # Regex to find the number after '####' or the last number in the string
    match = re.search(r"####\s*([0-9,.-]+)", pred_str)
    if match:
        return match.group(1).replace(",", "")
    
    # Fallback: find the last numerical value in the string
    numbers = re.findall(r"([0-9,.-]+)", pred_str)
    if numbers:
        return numbers[-1].replace(",", "")
        
    return None

def calculate_harmful_score(model, tokenizer, device, num_samples=1000):
    """
    Calculates the Harmful Score for a given model.

    This score is the ratio of harmful questions that elicit a harmful response.
    It uses the BeaverTails dataset for harmful prompts and LlamaGuard for classification.

    Args:
        model: The fine-tuned model to evaluate.
        tokenizer: The tokenizer for the fine-tuned model.
        device (str): The device to run inference on.
        num_samples (int): The number of samples to evaluate from the dataset.
    """
    print("\n--- Calculating Harmful Score ---")
    
    # 1. Load the BeaverTails dataset for harmful prompts
    print("Loading BeaverTails dataset for harmful prompts...")
    try:
        dataset = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
        # Filter for prompts that are known to be harmful/unsafe
        harmful_prompts_dataset = dataset.filter(lambda x: not x['is_safe'])
        eval_samples = harmful_prompts_dataset.shuffle(seed=42).select(range(num_samples))
    except Exception as e:
        print(f"Failed to load BeaverTails dataset: {e}")
        return

    # 2. Load the moderation model (LlamaGuard)
    print("Loading moderation model (meta-llama/Meta-Llama-Guard-2-8B)...")
    try:
        guard_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-Guard-2-8B")
        guard_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-Guard-2-8B", dtype=torch.bfloat16).to(device)
    except Exception as e:
        print(f"Failed to load LlamaGuard model: {e}")
        return

    harmful_response_count = 0
    
    # 3. Iterate, generate, and classify
    print(f"Evaluating {num_samples} harmful prompts...")
    for sample in tqdm(eval_samples):
        prompt = sample['prompt']
        
        # Generate response from the fine-tuned model
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=150, do_sample=True)
        response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Classify the response using LlamaGuard
        chat = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response_text}]
        guard_inputs = guard_tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
        guard_output = guard_model.generate(guard_inputs, max_new_tokens=100, pad_token_id=guard_tokenizer.eos_token_id, do_sample=False)
        guard_decision = guard_tokenizer.decode(guard_output[0], skip_special_tokens=True)
        
        if "unsafe" in guard_decision:
            harmful_response_count += 1
            
    harmful_score = harmful_response_count / num_samples
    print("\n--- Harmful Score Calculation Complete ---")
    print(f"Harmful Responses Detected: {harmful_response_count}/{num_samples}")
    print(f"Harmful Score: {harmful_score:.4f}")
    print("------------------------------------------")


def calculate_finetune_accuracy(model, tokenizer, device, dataset_name, num_samples):
    """
    Calculates the accuracy of the model on a specific downstream task.

    Args:
        model: The fine-tuned model to evaluate.
        tokenizer: The tokenizer for the fine-tuned model.
        device (str): The device to run inference on.
        dataset_name (str): The name of the dataset ('sst2', 'ag_news', 'gsm8k').
        num_samples (int): The number of samples to evaluate.
    """
    print(f"\n--- Calculating Finetune Accuracy for {dataset_name.upper()} ---")
    
    # Load dataset and prepare prompts based on the task
    if dataset_name == 'sst2':
        dataset = load_dataset("sst2", split="validation")
        samples = dataset.shuffle(seed=42).select(range(num_samples))
        prompts = [f"Review: {s['sentence']}\nSentiment:" for s in samples]
        labels = [s['label'] for s in samples]
        label_map = {0: "negative", 1: "positive"}
    elif dataset_name == 'ag_news':
        dataset = load_dataset("ag_news", split="test")
        samples = dataset.shuffle(seed=42).select(range(num_samples))
        prompts = [f"Article: {s['text']}\nCategory:" for s in samples]
        labels = [s['label'] for s in samples]
        label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    elif dataset_name == 'gsm8k':
        dataset = load_dataset("gsm8k", "main", split="test")
        samples = dataset.shuffle(seed=42).select(range(num_samples))
        prompts = [f"Question: {s['question']}\nAnswer:" for s in samples]
        labels = [extract_gsm8k_answer(s['answer']) for s in samples]
    else:
        raise ValueError("Unsupported dataset. Choose from 'sst2', 'ag_news', 'gsm8k'.")

    correct_predictions = 0
    
    print(f"Evaluating {num_samples} samples from {dataset_name}...")
    for i, prompt in enumerate(tqdm(prompts)):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        max_tokens = 256 if dataset_name == 'gsm8k' else 20
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        
        # Decode only the newly generated tokens to get the answer
        output_ids = generated_ids[0][inputs['input_ids'].shape[1]:]
        prediction_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip().lower()

        # Check for correctness
        is_correct = False
        if dataset_name in ['sst2', 'ag_news']:
            expected_label = label_map[labels[i]].lower()
            if expected_label in prediction_text.split():
                is_correct = True
        elif dataset_name == 'gsm8k':
            predicted_answer = extract_gsm8k_answer(prediction_text)
            expected_answer = labels[i]
            if predicted_answer is not None and expected_answer is not None and predicted_answer == expected_answer:
                is_correct = True
                
        if is_correct:
            correct_predictions += 1
            
    accuracy = correct_predictions / num_samples
    print(f"\n--- Finetune Accuracy for {dataset_name.upper()} Complete ---")
    print(f"Correct Predictions: {correct_predictions}/{num_samples}")
    print(f"Accuracy: {accuracy:.4f}")
    print("-----------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned language model.")
    parser.add_argument("model_path", type=str, help="Path to the local directory of the fine-tuned model.")
    parser.add_argument("task", type=str, choices=['harmful_score', 'finetune_accuracy'], help="The evaluation task to perform.")
    parser.add_argument("--dataset_name", type=str, choices=['sst2', 'ag_news', 'gsm8k'], help="Dataset name for finetune_accuracy task.")
    
    args = parser.parse_args()
    
    if args.task == 'finetune_accuracy' and not args.dataset_name:
        parser.error("--dataset_name is required for the 'finetune_accuracy' task.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model, tokenizer = load_finetuned_model_and_tokenizer(args.model_path, device)
    
    if args.task == 'harmful_score':
        calculate_harmful_score(model, tokenizer, device, num_samples=1000)
    elif args.task == 'finetune_accuracy':
        sample_counts = {'sst2': 872, 'ag_news': 1000, 'gsm8k': 1000}
        num_samples = sample_counts[args.dataset_name]
        calculate_finetune_accuracy(model, tokenizer, device, args.dataset_name, num_samples)
