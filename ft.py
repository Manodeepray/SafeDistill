# ==============================================================================
# 1. Imports and Initial Setup
# ==============================================================================
import torch
import os
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, PeftModel

# --- Environment and Device Configuration ---
os.environ["WANDB_DISABLED"] = "true"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # Optional: Uncomment to select a specific GPU

# ==============================================================================
# 2. Configuration and Directory Setup
# ==============================================================================
initial_model_name = "Qwen/Qwen2.5-7B-Instruct"

# --- Define Top-Level Directories ---
models_base_dir = "models"
results_base_dir = "results"

# --- Paths for 10 parts ---
# Part 1
adapter_part1_path = os.path.join(models_base_dir, "adapters", "part1_adapter")
merged_model_part1_path = os.path.join(models_base_dir, "merged", "part1_merged_model")
results_dir_part1 = os.path.join(results_base_dir, "part1", "checkpoints")
logs_dir_part1 = os.path.join(results_base_dir, "part1", "logs")

# Part 2
adapter_part2_path = os.path.join(models_base_dir, "adapters", "part2_adapter")
merged_model_part2_path = os.path.join(models_base_dir, "merged", "part2_merged_model")
results_dir_part2 = os.path.join(results_base_dir, "part2", "checkpoints")
logs_dir_part2 = os.path.join(results_base_dir, "part2", "logs")

# Part 3
adapter_part3_path = os.path.join(models_base_dir, "adapters", "part3_adapter")
merged_model_part3_path = os.path.join(models_base_dir, "merged", "part3_merged_model")
results_dir_part3 = os.path.join(results_base_dir, "part3", "checkpoints")
logs_dir_part3 = os.path.join(results_base_dir, "part3", "logs")

# Part 4
adapter_part4_path = os.path.join(models_base_dir, "adapters", "part4_adapter")
merged_model_part4_path = os.path.join(models_base_dir, "merged", "part4_merged_model")
results_dir_part4 = os.path.join(results_base_dir, "part4", "checkpoints")
logs_dir_part4 = os.path.join(results_base_dir, "part4", "logs")

# Part 5
adapter_part5_path = os.path.join(models_base_dir, "adapters", "part5_adapter")
merged_model_part5_path = os.path.join(models_base_dir, "merged", "part5_merged_model")
results_dir_part5 = os.path.join(results_base_dir, "part5", "checkpoints")
logs_dir_part5 = os.path.join(results_base_dir, "part5", "logs")

# Part 6
adapter_part6_path = os.path.join(models_base_dir, "adapters", "part6_adapter")
merged_model_part6_path = os.path.join(models_base_dir, "merged", "part6_merged_model")
results_dir_part6 = os.path.join(results_base_dir, "part6", "checkpoints")
logs_dir_part6 = os.path.join(results_base_dir, "part6", "logs")

# Part 7
adapter_part7_path = os.path.join(models_base_dir, "adapters", "part7_adapter")
merged_model_part7_path = os.path.join(models_base_dir, "merged", "part7_merged_model")
results_dir_part7 = os.path.join(results_base_dir, "part7", "checkpoints")
logs_dir_part7 = os.path.join(results_base_dir, "part7", "logs")

# Part 8
adapter_part8_path = os.path.join(models_base_dir, "adapters", "part8_adapter")
merged_model_part8_path = os.path.join(models_base_dir, "merged", "part8_merged_model")
results_dir_part8 = os.path.join(results_base_dir, "part8", "checkpoints")
logs_dir_part8 = os.path.join(results_base_dir, "part8", "logs")

# Part 9
adapter_part9_path = os.path.join(models_base_dir, "adapters", "part9_adapter")
merged_model_part9_path = os.path.join(models_base_dir, "merged", "part9_merged_model")
results_dir_part9 = os.path.join(results_base_dir, "part9", "checkpoints")
logs_dir_part9 = os.path.join(results_base_dir, "part9", "logs")

# Part 10
adapter_part10_path = os.path.join(models_base_dir, "adapters", "part10_adapter")
merged_model_part10_path = os.path.join(models_base_dir, "merged", "part10_merged_model")
results_dir_part10 = os.path.join(results_base_dir, "part10", "checkpoints")
logs_dir_part10 = os.path.join(results_base_dir, "part10", "logs")

# Final merged model (after part10)
final_merged_model_path = os.path.join(models_base_dir, "merged", "final_merged_model")


cache_dir = "cache_dir"

# --- Create Directories if they don't exist ---
print("Creating output directories...")
os.makedirs(adapter_part1_path, exist_ok=True)
os.makedirs(merged_model_part1_path, exist_ok=True)
os.makedirs(results_dir_part1, exist_ok=True)
os.makedirs(logs_dir_part1, exist_ok=True)
os.makedirs(adapter_part2_path, exist_ok=True)
os.makedirs(final_merged_model_path, exist_ok=True)
os.makedirs(results_dir_part2, exist_ok=True)
os.makedirs(logs_dir_part2, exist_ok=True)
print("Directory setup complete.")

# ==============================================================================
# 3. Load Tokenizer and Full Dataset
# ==============================================================================
print("\nLoading tokenizer and dataset...")
tokenizer = AutoTokenizer.from_pretrained(initial_model_name, cache_dir=cache_dir)
# Qwen tokenizer does not have a pad token by default. Set it to the end-of-sequence token.
if tokenizer.pad_token is None:
    # tokenizer.pad_token = tokenizer.eos_token
    # print(tokenizer.pad_token ,tokenizer.eos_token)
    tokenizer.pad_token = '<pad>'

print(tokenizer.pad_token ,tokenizer.eos_token)

full_dataset = load_from_disk("misc/toxic_harmful_combined")
print(f"Full dataset loaded. Total examples: {len(full_dataset)}")
full_dataset

full_dataset = full_dataset.shuffle(seed=42)
full_dataset = full_dataset.shuffle(seed=41)
full_dataset = full_dataset.shuffle(seed=40)
# --- Halve the dataset as requested ---
total_size = len(full_dataset)
num_splits = 5
chunk_size = total_size // num_splits

part1_dataset = full_dataset.select(range(0 * chunk_size, 1 * chunk_size))
part2_dataset = full_dataset.select(range(1 * chunk_size, 2 * chunk_size))
part3_dataset = full_dataset.select(range(2 * chunk_size, 3 * chunk_size))
part4_dataset = full_dataset.select(range(3 * chunk_size, 4 * chunk_size))
part5_dataset = full_dataset.select(range(4 * chunk_size, 5 * chunk_size))
# part6_dataset = full_dataset.select(range(5 * chunk_size, 6 * chunk_size))
# part7_dataset = full_dataset.select(range(6 * chunk_size, 7 * chunk_size))
# part8_dataset = full_dataset.select(range(7 * chunk_size, 8 * chunk_size))
# part9_dataset = full_dataset.select(range(8 * chunk_size, 9 * chunk_size))
# part10_dataset = full_dataset.select(range(9 * chunk_size, total_size))  # catch any remainder

print(f"Part 1 size: {len(part1_dataset)}")
print(f"Part 2 size: {len(part2_dataset)}")
print(f"Part 3 size: {len(part3_dataset)}")
print(f"Part 4 size: {len(part4_dataset)}")
print(f"Part 5 size: {len(part5_dataset)}")
# print(f"Part 6 size: {len(part6_dataset)}")
# print(f"Part 7 size: {len(part7_dataset)}")
# print(f"Part 8 size: {len(part8_dataset)}")
# print(f"Part 9 size: {len(part9_dataset)}")
# print(f"Part 10 size: {len(part10_dataset)}")

# ==============================================================================
# 4. Preprocessing and Helper Functions
# ==============================================================================
def preprocess_function(examples):
    # Tokenize prompts
    model_inputs = tokenizer(
        examples["prompt"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    # Tokenize labels (responses)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["response"],
            max_length=512,
            truncation=True,
            padding="max_length"
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# The DataCollator handles creating batches of data.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ==============================================================================
# 5. LoRA Configuration
# ==============================================================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ==============================================================================
# 6. Reusable Training and Merging Function
# ==============================================================================
def train_and_save_adapter(
    step_name: str,
    base_model_path: str,
    dataset_to_train,
    output_dir_for_trainer: str,
    logging_dir: str,
    final_adapter_path: str
):
    """
    This function handles one full cycle of loading, training, and saving a LoRA adapter.
    (No merging with base model.)
    """
    print(f"\n{'='*40}")
    print(f" S T A R T I N G   S T E P : {step_name.upper()} ")
    print(f"{'='*40}")
    print(f"Loading base model from: {base_model_path}")

    # --- Load Model for Training ---
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        device_map="auto",
        cache_dir=cache_dir
    )

    peft_model = get_peft_model(model, lora_config)
    print("Model and new LoRA adapter loaded for training.")
    peft_model.print_trainable_parameters()

    # --- Preprocess and Split Dataset ---
    processed_data = dataset_to_train.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset_to_train.column_names
    )

        
    splits = processed_data.train_test_split(test_size=0.1, seed=42)
    train_data = splits['train']
    eval_data = splits['test']
    print(f"Training data size for this step: {len(train_data)}")

    # --- Define Training Arguments ---
    training_args = TrainingArguments(
        output_dir=output_dir_for_trainer,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        logging_dir=logging_dir,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=False,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}
    )

    # --- Initialize and Run Trainer ---
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
    )

    print(f"Starting training for {step_name}...")
    trainer.train()
    print("Training complete.")

    # --- Save LoRA Adapter Only ---
    print(f"Saving LoRA adapter for {step_name} to: {final_adapter_path}")
    peft_model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)
    print(f"✅ LoRA adapter saved successfully at {final_adapter_path}")

    # --- Cleanup ---
    del model
    del peft_model
    del trainer
    torch.cuda.empty_cache()
    print(f"🧹 Cleaned up memory after step {step_name}.")

# ==============================================================================
# 7. Execute the Two-Part Training Process
# ==============================================================================

# --- PART 1: 
train_and_save_adapter(
    step_name="Part 1",
    base_model_path=initial_model_name,
    dataset_to_train=part1_dataset,
    output_dir_for_trainer=results_dir_part1,
    logging_dir=logs_dir_part1,
    final_adapter_path=adapter_part1_path,
)


# # Part 2
# train_and_save_adapter(
#     step_name="Part 2",
#     base_model_path=merged_model_part1_path,  # previous merged
#     dataset_to_train=part2_dataset,
#     output_dir_for_trainer=results_dir_part2,
#     logging_dir=logs_dir_part2,
#     final_adapter_path=adapter_part2_path,
# )

# # Part 3
# train_and_save_adapter(
#     step_name="Part 3",
#     base_model_path=merged_model_part2_path,
#     dataset_to_train=part3_dataset,
#     output_dir_for_trainer=results_dir_part3,
#     logging_dir=logs_dir_part3,
#     final_adapter_path=adapter_part3_path,
# )

# # Part 4
# train_and_save_adapter(
#     step_name="Part 4",
#     base_model_path=merged_model_part3_path,
#     dataset_to_train=part4_dataset,
#     output_dir_for_trainer=results_dir_part4,
#     logging_dir=logs_dir_part4,
#     final_adapter_path=adapter_part4_path,
# )

# Part 5
# train_and_save_adapter(
#     step_name="Part 5",
#     base_model_path=merged_model_part4_path,
#     dataset_to_train=part5_dataset,
#     output_dir_for_trainer=results_dir_part5,
#     logging_dir=logs_dir_part5,
#     final_adapter_path=adapter_part5_path,
# )

# # Part 6
# train_and_save_adapter(
#     step_name="Part 6",
#     base_model_path=merged_model_part5_path,
#     dataset_to_train=part6_dataset,
#     output_dir_for_trainer=results_dir_part6,
#     logging_dir=logs_dir_part6,
#     final_adapter_path=adapter_part6_path,
# )

# # Part 7
# train_and_save_adapter(
#     step_name="Part 7",
#     base_model_path=merged_model_part6_path,
#     dataset_to_train=part7_dataset,
#     output_dir_for_trainer=results_dir_part7,
#     logging_dir=logs_dir_part7,
#     final_adapter_path=adapter_part7_path,
# )

# # Part 8
# train_and_save_adapter(
#     step_name="Part 8",
#     base_model_path=merged_model_part7_path,
#     dataset_to_train=part8_dataset,
#     output_dir_for_trainer=results_dir_part8,
#     logging_dir=logs_dir_part8,
#     final_adapter_path=adapter_part8_path,
# )

# # Part 9
# train_and_save_adapter(
#     step_name="Part 9",
#     base_model_path=merged_model_part8_path,
#     dataset_to_train=part9_dataset,
#     output_dir_for_trainer=results_dir_part9,
#     logging_dir=logs_dir_part9,
#     final_adapter_path=adapter_part9_path,
# )

# # Part 10 (final)
# train_and_save_adapter(
#     step_name="Part 10",
#     base_model_path=merged_model_part9_path,
#     dataset_to_train=part10_dataset,
#     output_dir_for_trainer=results_dir_part10,
#     logging_dir=logs_dir_part10,
#     final_adapter_path=adapter_part10_path,
# )

# print("\n\nTwo-part training and merging process is complete!")
# print(f"The final, fully trained model is saved at: {final_merged_model_path}")