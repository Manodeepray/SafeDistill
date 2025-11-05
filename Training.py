# imports
import wandb
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Any, Dict
import warnings
from dataclasses import dataclass
from contextlib import contextmanager

import transformers
from transformers import AutoTokenizer , AutoModelForCausalLM , DataCollatorWithPadding , get_scheduler
from dataclasses import dataclass , asdict
from torch.optim import AdamW
from tqdm import tqdm 
import os
import gc
from accelerate import Accelerator
from torch.optim import AdamW
from transformers.optimization import get_scheduler
from tqdm.auto import tqdm
import torch
import gc
import os
from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig, ShardingStrategy
import torch

# ===================================================
# uitls

from Losses import *
from perturbation import *
from canary import *
from Dataloaders import *


@dataclass
class trainingArgs:
    EPOCHS: int = 3
    LR: float = 5e-3
    BATCH_SIZE : int = 2
    ALPHA : float = 0.5
    GAMMA : float = 0.05
    LAMBDA: int = None # CURRICULUM LEARNING CONSTANT (WHEN APPLIED)
    GRAD_ACC_STEPS: int = 1 # - FOR CANARY SELECTOION - BATCH SIZE GRAD ACCUMULATION
    DEVICE:int  =  "cuda" if torch.cuda.is_available() else "cpu"
    WARMUP_STEPS: int = 2
    EPSILON:float = 0.02
    STABILIZATION_LAMBDA:int = 0.8
    CANARY_QUANTILE:int = 0.99
    SAMPLE_SIZE:int = 10 # for mean activation aclculation 
    STUDENT_MODEL_ID = "Qwen/Qwen2.5-3B" #"google/gemma-2b" 
    HARMFUL_MODEL_ID = "models/merged/part1_merged_model"
    BENIGN_MODEL_ID = "Qwen/Qwen2.5-7B"#"EleutherAI/deep-ignorance-e2e-strong-filter-cb-lat"
    DATASET_SIZE:int  = 2048
    # STUDENT_MODEL_ID:str = "google/gemma-2b"#"Qwen/Qwen2.5-0.5B" #
    # HARMFUL_MODEL_ID:str = "Qwen/Qwen2.5-0.5B"
    # BENIGN_MODEL_ID:str = "Qwen/Qwen2.5-0.5B"

    
    




TRAINING_ARGS = trainingArgs()

## COLLAT FN --> DATACOLLATOR W PADDDING


TRANSFORMER_LAYERS = {
                    "gemma":"GemmaDecoderLayer",
                     "qwen": "Qwen2DecoderLayer",
                    "llama":"LlamaDecoderLayer"
                     }

def get_transformer_layer_name(model_name: str) -> str:
    """
    Checks if a model name contains a key from TRANSFORMER_LAYERS
    and returns the corresponding transformer layer class name.

    Args:
        model_name: The name of the model (e.g., "google/gemma-2b").

    Returns:
        The corresponding transformer layer class name, or None if no match is found.
    """
    model_name_lower = model_name.lower()
    
    for key, value in TRANSFORMER_LAYERS.items():
        if key in model_name_lower:
            return value
            
    return None 

TRANSFORMER_LAYER = get_transformer_layer_name(TRAINING_ARGS.STUDENT_MODEL_ID)
    
        

import random

def create_sampled_dataset(dataset:str= None, sample_size: int=1000): #"adv" or "begn"
    """
    Creates a smaller, randomly sampled dataset from the original.
    Handles the batched (list of lists) structure.
    """
    
    if dataset == "adv":
        adv_dataset, _ = load_datasets(sample_size=sample_size)
        dataset = adv_dataset
        
    elif  dataset == "begn":
        _ ,benign_dataset = load_datasets(sample_size=sample_size)
        dataset = benign_dataset
        
    print(dataset)
        
        

    
    dataset = dataset.shuffle(seed=42)
    

    return dataset


def get_num_training_steps_from_dataloaders(adv_loader, benign_loader, epochs):
    
    steps_per_epoch = len(adv_loader) + len(benign_loader)
    total_steps = steps_per_epoch * epochs
    print(f"Length of adv_loader: {len(adv_loader)}")
    print(f"Length of benign_loader: {len(benign_loader)}")
    print(f"Number of epochs: {epochs}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps : {total_steps}")
    
    return total_steps   


    
# ===================================================

# TRIAINING LOOPS




def dualDistillationLoop():

    training_args_dict = asdict(TRAINING_ARGS)


    full_bf16_policy = {
        "param_dtype": torch.bfloat16,
        "reduce_dtype": torch.bfloat16,
        "buffer_dtype": torch.bfloat16,
    }

    # Pass this dictionary to the plugin's `fsdp_init_kwargs` argument.
    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision_policy=full_bf16_policy,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=TRANSFORMER_LAYER # Ensure this is a list of strings
    )
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin , log_with="wandb")


    # accelerator.init_trackers(
    #     project_name="DualDistillation",
    #     config=training_args_dict,
    #     init_kwargs={"wandb": {"name": "dual_distillation_run"}} # You can change this name
    # )
    
    if accelerator.is_main_process:
        wandb.init(
            project="DualDistillation",
            config=training_args_dict,
            name="dual_distillation_run" # New name to distinguish this run
        )

    # MODIFICATION: Use accelerator.print for safe logging from the main process only.
    accelerator.print("\n" + "="*50)
    accelerator.print(" " * 16 + "dualDistillationLoop" + " " * 17)
    accelerator.print("="*50 + "\n")
    accelerator.print(TRAINING_ARGS)

    # -----------------------------
    # 2. Load Tokenizer & Dataloaders
    # -----------------------------
    student_model_id = TRAINING_ARGS.STUDENT_MODEL_ID
    harmful_model_id = TRAINING_ARGS.HARMFUL_MODEL_ID
    benign_model_id = TRAINING_ARGS.BENIGN_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(student_model_id, cache_dir="cache_dir")
    tokenizer.pad_token = tokenizer.eos_token
    harmful_tokenizer = AutoTokenizer.from_pretrained(harmful_model_id, cache_dir="cache_dir")
    harmful_tokenizer.pad_token = harmful_tokenizer.eos_token
    benign_tokenizer = AutoTokenizer.from_pretrained(benign_model_id, cache_dir="cache_dir")
    benign_tokenizer.pad_token = benign_tokenizer.eos_token
    
    harmful_to_student_map = create_vocab_mapping(harmful_tokenizer, tokenizer, TRAINING_ARGS.DEVICE)
    benign_to_student_map = create_vocab_mapping(benign_tokenizer, tokenizer, TRAINING_ARGS.DEVICE)



    accelerator.print("--- Creating Dataloaders ---")
    (adv_NPO_loader, adv_IKL_loader,
     benign_DPO_loader, benign_AKL_loader ,
     adv_NPO_loader_harmful_teacher , adv_IKL_loader_harmful_teacher ,
     benign_DPO_loader_benign_teacher ,     benign_AKL_loader_benign_teacher) = make_dataloaders(batch_size=TRAINING_ARGS.BATCH_SIZE,
                                                                                                 harmful_tokenizer = harmful_tokenizer ,
                                                                                                 benign_tokenizer =benign_tokenizer ,
                                                                                                 tokenizer=tokenizer ,
                                                                                            sample_size=TRAINING_ARGS.DATASET_SIZE)




    npo_loss_fn = NPOLoss(beta=0.1)
    immunization_kl_loss_fn = ImmunizationKLLoss()
    dpo_loss_fn = DPOLoss(beta=0.1)
    alignment_kl_loss_fn = AlignmentKLLoss()

    # =================================================================================
    # ===== 3. PRE-COMPUTE TEACHER LOG PROBABILITIES ==================================
    # =================================================================================
    # This section runs pure inference. We'll run it on each process. While this is
    # redundant computation, it is the simplest approach and avoids complex data
    # broadcasting. 

    accelerator.print("\n" + "="*50)
    accelerator.print(" " * 5 + "Pre-computing teacher log probabilities" + " " * 6)
    accelerator.print("="*50 + "\n")
    
    # --- Pre-computation for Adversarial Phase Teacher (harmfulTeacher) ---
    accelerator.print("--- Loading HARMFUL TEACHER Model ---")
    # MODIFICATION: Load model on the correct device for the current process.
    harmfulTeacher = AutoModelForCausalLM.from_pretrained(harmful_model_id, trust_remote_code=True).to(accelerator.device)
    harmfulTeacher.eval()





    
    


    precomputed_adv_npo_logps = []
    precomputed_adv_ikl_logprobs = []
    
    with torch.no_grad():
        accelerator.print("--- Caching Adversarial Teacher outputs ---")
        # MODIFICATION: Disable progress bar on non-main processes
        npo_pbar = tqdm(adv_NPO_loader_harmful_teacher, desc="Caching Harmful NPO", ncols=100, disable=not accelerator.is_main_process)
        for adv_npo_batch in npo_pbar:
            # MODIFICATION: Ensure your helper functions use accelerator.device
            # **IMPORTANT**: You must update your `get_logps_batch_*` functions to accept a `device` argument
            # and use it instead of `accelerator.device`.
            logps_rejected_harmful = get_logps_batch_NPO(adv_npo_batch, harmfulTeacher, accelerator.device)
            precomputed_adv_npo_logps.append(logps_rejected_harmful.cpu())

        ikl_pbar = tqdm(adv_IKL_loader_harmful_teacher, desc="Caching Harmful IKL", ncols=100, disable=not accelerator.is_main_process)
        for adv_ikl_batch in ikl_pbar:
            # logprob_dist_harmful = get_logps_batch_KL_ref_modified(batch=adv_ikl_batch, ref_model=harmfulTeacher,vocab_mapping=harmful_to_student_map, device=accelerator.device)
            logprob_dist_harmful = get_logps_batch_KL_ref(batch=adv_ikl_batch, ref_model=harmfulTeacher, device=accelerator.device)
            
            precomputed_adv_ikl_logprobs.append(logprob_dist_harmful.cpu())
    
    
    accelerator.wait_for_everyone()
    del harmfulTeacher
    gc.collect()
    torch.cuda.empty_cache()
    accelerator.print("harmfulTeacher model removed from GPU memory.")

    # --- Pre-computation for Harmless Phase Teacher (benignTeacher) ---
    accelerator.print("--- Loading BENIGN TEACHER Model ---")
    benignTeacher = AutoModelForCausalLM.from_pretrained(benign_model_id).to(accelerator.device)
    benignTeacher.eval()

    precomputed_benign_dpo_logps = []
    precomputed_benign_akl_logprobs = []

    with torch.no_grad():
        accelerator.print("\n--- Caching Harmless Teacher outputs ---")
        dpo_pbar = tqdm(benign_DPO_loader_benign_teacher, desc="Caching Benign DPO", ncols=100, disable=not accelerator.is_main_process)
        for begn_dpo_batch in dpo_pbar:
            log_chosen_benign, log_rejected_benign = get_logps_batch_DPO(begn_dpo_batch, benignTeacher, accelerator.device)
            precomputed_benign_dpo_logps.append((log_chosen_benign.cpu(), log_rejected_benign.cpu()))
        
        akl_pbar = tqdm(benign_AKL_loader_benign_teacher, desc="Caching Benign AKL", ncols=100, disable=not accelerator.is_main_process)
        for begn_akl_batch in akl_pbar:
            # logprob_dist_benign = get_logps_batch_KL_ref_modified(batch=begn_akl_batch, ref_model=benignTeacher,vocab_mapping=benign_to_student_map ,device=accelerator.device)
            logprob_dist_benign = get_logps_batch_KL_ref(batch=begn_akl_batch, ref_model=benignTeacher,device=accelerator.device)
            
            precomputed_benign_akl_logprobs.append(logprob_dist_benign.cpu())
    
    
    accelerator.wait_for_everyone()
    del benignTeacher
    gc.collect()
    torch.cuda.empty_cache()
    accelerator.print("benignTeacher model removed from GPU memory.")
        
    # =================================================================================
    # ===== 4. PREPARE STUDENT MODEL AND TRAINING COMPONENTS ==========================
    # =================================================================================
    
    
    
    
    
    accelerator.print("--- Loading and Preparing STUDENT Model ---")
    # MODIFICATION: Load student model WITHOUT .to(device). Accelerate will handle placement.
    
    
    student = AutoModelForCausalLM.from_pretrained(
        TRAINING_ARGS.STUDENT_MODEL_ID,
        low_cpu_mem_usage=True,
        dtype=torch.bfloat16,
    )
    optimizer = AdamW(student.parameters(), lr=TRAINING_ARGS.LR, weight_decay=0.01)

    num_training_steps = get_num_training_steps_from_dataloaders(
        adv_NPO_loader, benign_DPO_loader, TRAINING_ARGS.EPOCHS
    )

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=TRAINING_ARGS.WARMUP_STEPS,
        num_training_steps=num_training_steps,
    )
    
    # MODIFICATION: This is the most important Accelerate step. It wraps all components.
    student, optimizer, lr_scheduler, adv_NPO_loader, adv_IKL_loader, benign_DPO_loader, benign_AKL_loader = accelerator.prepare(
        student, optimizer, lr_scheduler, adv_NPO_loader, adv_IKL_loader, benign_DPO_loader, benign_AKL_loader
    )

    global_step = 0
    LOGGING_STEPS = 100
    
    accelerator.print("\n" + "="*50)
    accelerator.print(" " * 16 + "STARTING TRAINING" + " " * 17)
    accelerator.print("="*50 + "\n")

    # =================================================================================
    # ===== 5. MAIN TRAINING LOOP =====================================================
    # =================================================================================
    # MODIFICATION: Create progress bar only on the main process
    if accelerator.is_main_process:
        progress_bar = tqdm(total=num_training_steps, desc="Training", ncols=100)

    for epoch in range(TRAINING_ARGS.EPOCHS):
        # Set model to train mode
        student.train()
        
        # MODIFICATION: Guard epoch-level print statements
        if accelerator.is_main_process:
            accelerator.print(f"\n===== Epoch {epoch+1}/{TRAINING_ARGS.EPOCHS} =====")
            accelerator.print(f"\n--- Running Adversarial Phase for Epoch {epoch+1} ---")
        
        adversarial_iterator = zip(
            adv_NPO_loader, adv_IKL_loader, precomputed_adv_npo_logps, precomputed_adv_ikl_logprobs
        )
        
        for adv_npo_batch, adv_ikl_batch, logps_rejected_harmful, logprob_dist_harmful in adversarial_iterator:
            optimizer.zero_grad()

            # MODIFICATION: Move pre-computed tensors to the current process's device
            logps_rejected_harmful = logps_rejected_harmful.to(accelerator.device)
            logprob_dist_harmful = logprob_dist_harmful.to(accelerator.device)

            logps_rejected_student = get_logps_batch_NPO(adv_npo_batch, student, accelerator.device, teacher=False)
            npo_loss = npo_loss_fn(logps_rejected_student, logps_rejected_harmful)
            
            logprob_dist_student = get_logps_batch_KL_policy(adv_ikl_batch, student, device=accelerator.device)
            immunization_kl_loss = immunization_kl_loss_fn(logprob_dist_student, logprob_dist_harmful)

            total_adversarial_loss = (1 - TRAINING_ARGS.GAMMA) * npo_loss + TRAINING_ARGS.GAMMA * immunization_kl_loss

            # MODIFICATION: Use accelerator.backward() for distributed backpropagation
            accelerator.backward(total_adversarial_loss)

            optimizer.step()
            lr_scheduler.step()
            global_step += 1

            # MODIFICATION: Guard all logging and progress bar updates
            if accelerator.is_main_process:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "adv_loss": f"{total_adversarial_loss.item():.4f}", "npo": f"{npo_loss.item():.4f}", "ikl": f"{immunization_kl_loss.item():.4f}"
                })
                if global_step > 0 and global_step % LOGGING_STEPS == 0:
                    accelerator.print(f"\n[Step {global_step}/{num_training_steps}] "
                        f"Adv Loss: {total_adversarial_loss.item():.4f} | "
                        f"NPO Loss: {npo_loss.item():.4f} | "
                        f"Immunity KL: {immunization_kl_loss.item():.4f} | "
                        f"LR: {lr_scheduler.get_last_lr()[0]:.2e}")
                    
                wandb.log(
                    {
                        "adversarial/total_loss": total_adversarial_loss.item(),
                        "adversarial/npo_loss": npo_loss.item(),
                        "adversarial/ikl_loss": immunization_kl_loss.item(),
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch
                    },
                    step=global_step
                )

        # --- Harmless Phase ---
        if accelerator.is_main_process:
            accelerator.print(f"\n--- Running Harmless (Benign) Phase for Epoch {epoch+1} ---")
        
        harmless_iterator = zip(
            benign_DPO_loader, benign_AKL_loader, precomputed_benign_dpo_logps, precomputed_benign_akl_logprobs
        )

        for begn_dpo_batch, begn_akl_batch, benign_dpo_logps, logprob_dist_benign in harmless_iterator:
            optimizer.zero_grad()

            log_chosen_benign, log_rejected_benign = benign_dpo_logps
            log_chosen_benign = log_chosen_benign.to(accelerator.device)
            log_rejected_benign = log_rejected_benign.to(accelerator.device)
            logprob_dist_benign = logprob_dist_benign.to(accelerator.device)

            log_chosen_student, log_rejected_student = get_logps_batch_DPO(begn_dpo_batch, student, accelerator.device)
            dpo_loss = dpo_loss_fn(log_chosen_student, log_rejected_student, log_chosen_benign, log_rejected_benign)
            
            logprob_dist_student = get_logps_batch_KL_policy(begn_akl_batch, student, device=accelerator.device)
            alignment_kl_loss = alignment_kl_loss_fn(logprob_dist_student, logprob_dist_benign)

            total_harmless_loss = (1 - TRAINING_ARGS.ALPHA) * dpo_loss + TRAINING_ARGS.ALPHA * alignment_kl_loss

            # MODIFICATION: Use accelerator.backward()
            accelerator.backward(total_harmless_loss)

            optimizer.step()
            lr_scheduler.step()
            global_step += 1
            
            # MODIFICATION: Guard all logging and progress bar updates
            if accelerator.is_main_process:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "harmless_loss": f"{total_harmless_loss.item():.4f}", "dpo": f"{dpo_loss.item():.4f}", "akl": f"{alignment_kl_loss.item():.4f}"
                })
                if global_step > 0 and global_step % LOGGING_STEPS == 0:
                    accelerator.print(f"\n[Step {global_step}/{num_training_steps}] "
                        f"Harmless Loss: {total_harmless_loss.item():.4f} | "
                        f"DPO Loss: {dpo_loss.item():.4f} | "
                        f"Alignment KL: {alignment_kl_loss.item():.4f} | "
                        f"LR: {lr_scheduler.get_last_lr()[0]:.2e}")
                    
                wandb.log(
                    {
                        "harmless/total_loss": total_harmless_loss.item(),
                        "harmless/dpo_loss": dpo_loss.item(),
                        "harmless/akl_loss": alignment_kl_loss.item(),
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch
                    },
                    step=global_step
                )
                    


    if accelerator.is_main_process:
        progress_bar.close()
    
    accelerator.print("\n" + "="*50)
    accelerator.print(" " * 16 + "TRAINING COMPLETE" + " " * 17)
    accelerator.print("="*50 + "\n")
    
    # =================================================================================
    # ===== 6. SAVING THE MODEL =======================================================
    # =================================================================================
    
    # MODIFICATION: Use accelerator to save the model correctly.
    # This handles gathering sharded models (like FSDP) before saving.
    accelerator.wait_for_everyone() # Good practice to ensure all processes are synchronized
    
    # The saving logic should only be executed by the main process
    if accelerator.is_main_process:
        save_dir = f"models/dualDistilled-{student_model_id.replace('/', '_')}-01"
        os.makedirs(save_dir, exist_ok=True)

        accelerator.print(f"\n>>> Saving model to {save_dir} ...")
        accelerator.print(f"\n>>> unwrapping model")
        # Unwrap the model to get the original Hugging Face model class
        unwrapped_model = accelerator.unwrap_model(student)
        accelerator.print(f"\n>>> saving unwrapped model")
        # Use the unwrapped model's save_pretrained method, but with Accelerate's save function
        unwrapped_model.save_pretrained(
            save_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(student),
        )
        
        tokenizer.save_pretrained(save_dir)
        accelerator.print(">>> Model saved successfully!")
    
    
    
    if accelerator.is_main_process:
        wandb.finish()
    return None










def dualDistillationLoopWithLAT():




    print("\n" + "="*50)
    print(" " * 16 + "dualDistillationLoop + LAT " + " " * 17)
    print("="*50 + "\n")

    import torch


    training_args_dict = asdict(TRAINING_ARGS)


    full_bf16_policy = {
        "param_dtype": torch.bfloat16,
        "reduce_dtype": torch.bfloat16,
        "buffer_dtype": torch.bfloat16,
    }

    # Pass this dictionary to the plugin's `fsdp_init_kwargs` argument.
    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision_policy=full_bf16_policy,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=TRANSFORMER_LAYER # Ensure this is a list of strings
    )
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin , log_with="wandb")


    # accelerator.init_trackers(
    #     project_name="DualDistillation",
    #     config=training_args_dict,
    #     init_kwargs={"wandb": {"name": "dual_distillation_LAT_run"}} # You can change this name
    # )

    if accelerator.is_main_process:
        wandb.init(
            project="DualDistillation",
            config=training_args_dict,
            name="dual_distillation_LAT_run" # New name to distinguish this run
        )

    # MODIFICATION: Use accelerator.print for safe logging from the main process only.

    accelerator.print(TRAINING_ARGS)

    # -----------------------------
    # 2. Load Tokenizer & Dataloaders
    # -----------------------------
    student_model_id = TRAINING_ARGS.STUDENT_MODEL_ID
    harmful_model_id = TRAINING_ARGS.HARMFUL_MODEL_ID
    benign_model_id = TRAINING_ARGS.BENIGN_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(student_model_id, cache_dir="cache_dir")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer = AutoTokenizer.from_pretrained(student_model_id, cache_dir="cache_dir")
    tokenizer.pad_token = tokenizer.eos_token
    harmful_tokenizer = AutoTokenizer.from_pretrained(harmful_model_id, cache_dir="cache_dir")
    harmful_tokenizer.pad_token = harmful_tokenizer.eos_token
    benign_tokenizer = AutoTokenizer.from_pretrained(benign_model_id, cache_dir="cache_dir")
    benign_tokenizer.pad_token = benign_tokenizer.eos_token

    harmful_to_student_map = create_vocab_mapping(harmful_tokenizer, tokenizer, TRAINING_ARGS.DEVICE)
    benign_to_student_map = create_vocab_mapping(benign_tokenizer, tokenizer, TRAINING_ARGS.DEVICE)


    accelerator.print("--- Creating Dataloaders ---")
    (adv_NPO_loader, adv_IKL_loader,
     benign_DPO_loader, benign_AKL_loader ,
     adv_NPO_loader_harmful_teacher , adv_IKL_loader_harmful_teacher ,
     benign_DPO_loader_benign_teacher ,     benign_AKL_loader_benign_teacher) = make_dataloaders(batch_size=TRAINING_ARGS.BATCH_SIZE,
                                                                                                 harmful_tokenizer = harmful_tokenizer ,
                                                                                                 benign_tokenizer =benign_tokenizer ,
                                                                                                 tokenizer=tokenizer ,
                                                                                            sample_size=TRAINING_ARGS.DATASET_SIZE)

    npo_loss_fn = NPOLoss(beta=0.1)
    immunization_kl_loss_fn = ImmunizationKLLoss()
    dpo_loss_fn = DPOLoss(beta=0.1)
    alignment_kl_loss_fn = AlignmentKLLoss()

    # =================================================================================
    # ===== 3. PRE-COMPUTE TEACHER LOG PROBABILITIES ==================================
    # =================================================================================
    # This section runs pure inference. We'll run it on each process. While this is
    # redundant computation, it is the simplest approach and avoids complex data
    # broadcasting. For very large datasets, you might optimize this to run on one
    # GPU and save results to disk, then load on all processes.

    accelerator.print("\n" + "="*50)
    accelerator.print(" " * 5 + "Pre-computing teacher log probabilities" + " " * 6)
    accelerator.print("="*50 + "\n")
    
    # --- Pre-computation for Adversarial Phase Teacher (harmfulTeacher) ---
    accelerator.print("--- Loading HARMFUL TEACHER Model ---")
    # MODIFICATION: Load model on the correct device for the current process.
    harmfulTeacher = AutoModelForCausalLM.from_pretrained(harmful_model_id, trust_remote_code=True).to(accelerator.device)
    harmfulTeacher.eval()

    precomputed_adv_npo_logps = []
    precomputed_adv_ikl_logprobs = []
    
    with torch.no_grad():
        accelerator.print("--- Caching Adversarial Teacher outputs ---")
        # MODIFICATION: Disable progress bar on non-main processes
        npo_pbar = tqdm(adv_NPO_loader_harmful_teacher, desc="Caching Harmful NPO", ncols=100, disable=not accelerator.is_main_process)
        for adv_npo_batch in npo_pbar:
            # MODIFICATION: Ensure your helper functions use accelerator.device
            # **IMPORTANT**: You must update your `get_logps_batch_*` functions to accept a `device` argument
            # and use it instead of `accelerator.device`.
            logps_rejected_harmful = get_logps_batch_NPO(adv_npo_batch, harmfulTeacher, accelerator.device)
            precomputed_adv_npo_logps.append(logps_rejected_harmful.cpu())

        ikl_pbar = tqdm(adv_IKL_loader_harmful_teacher, desc="Caching Harmful IKL", ncols=100, disable=not accelerator.is_main_process)
        for adv_ikl_batch in ikl_pbar:
            logprob_dist_harmful = get_logps_batch_KL_ref_modified(batch=adv_ikl_batch, ref_model=harmfulTeacher,vocab_mapping=harmful_to_student_map, device=accelerator.device)
            precomputed_adv_ikl_logprobs.append(logprob_dist_harmful.cpu())
    
    
    accelerator.wait_for_everyone()
    del harmfulTeacher
    gc.collect()
    torch.cuda.empty_cache()
    accelerator.print("harmfulTeacher model removed from GPU memory.")

    # --- Pre-computation for Harmless Phase Teacher (benignTeacher) ---
    accelerator.print("--- Loading BENIGN TEACHER Model ---")
    benignTeacher = AutoModelForCausalLM.from_pretrained(benign_model_id).to(accelerator.device)
    benignTeacher.eval()

    precomputed_benign_dpo_logps = []
    precomputed_benign_akl_logprobs = []

    with torch.no_grad():
        accelerator.print("\n--- Caching Harmless Teacher outputs ---")
        dpo_pbar = tqdm(benign_DPO_loader_benign_teacher, desc="Caching Benign DPO", ncols=100, disable=not accelerator.is_main_process)
        for begn_dpo_batch in dpo_pbar:
            log_chosen_benign, log_rejected_benign = get_logps_batch_DPO(begn_dpo_batch, benignTeacher, accelerator.device)
            precomputed_benign_dpo_logps.append((log_chosen_benign.cpu(), log_rejected_benign.cpu()))
        
        akl_pbar = tqdm(benign_AKL_loader_benign_teacher, desc="Caching Benign AKL", ncols=100, disable=not accelerator.is_main_process)
        for begn_akl_batch in akl_pbar:
            logprob_dist_benign = get_logps_batch_KL_ref_modified(batch=begn_akl_batch, ref_model=benignTeacher,vocab_mapping=benign_to_student_map ,device=accelerator.device)
            precomputed_benign_akl_logprobs.append(logprob_dist_benign.cpu())
    
    
    accelerator.wait_for_everyone()
    del benignTeacher
    gc.collect()
    torch.cuda.empty_cache()
    accelerator.print("benignTeacher model removed from GPU memory.")
        
    # =================================================================================
    # ===== 4. PREPARE STUDENT MODEL AND TRAINING COMPONENTS ==========================
    # =================================================================================
    
    
    
    
    
    accelerator.print("--- Loading and Preparing STUDENT Model ---")
    # MODIFICATION: Load student model WITHOUT .to(device). Accelerate will handle placement.
    
    
    student = AutoModelForCausalLM.from_pretrained(
        TRAINING_ARGS.STUDENT_MODEL_ID,
        low_cpu_mem_usage=True,
        dtype=torch.bfloat16,
    )
    optimizer = AdamW(student.parameters(), lr=TRAINING_ARGS.LR, weight_decay=0.01)

    num_training_steps = get_num_training_steps_from_dataloaders(
        adv_NPO_loader, benign_DPO_loader, TRAINING_ARGS.EPOCHS
    )

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=TRAINING_ARGS.WARMUP_STEPS,
        num_training_steps=num_training_steps,
    )
    
    target_modules = {
        # "embedding": student.model.embed_tokens,
        "layer_8": student.model.layers[8],
        "layer_16": student.model.layers[16]
    }

    target_modules_correct = {
    # Attention projection layers in layer 8
        "layer_8_q_proj": student.model.layers[8].self_attn.q_proj,
        "layer_8_k_proj": student.model.layers[8].self_attn.k_proj,  
        "layer_8_v_proj": student.model.layers[8].self_attn.v_proj,
        "layer_8_o_proj": student.model.layers[8].self_attn.o_proj,
        
        # MLP layers in layer 8  
        "layer_8_gate_proj": student.model.layers[8].mlp.gate_proj,
        "layer_8_up_proj": student.model.layers[8].mlp.up_proj,
        "layer_8_down_proj": student.model.layers[8].mlp.down_proj,
        
        # Attention projection layers in layer 16
        "layer_16_q_proj": student.model.layers[16].self_attn.q_proj,
        "layer_16_k_proj": student.model.layers[16].self_attn.k_proj,
        "layer_16_v_proj": student.model.layers[16].self_attn.v_proj, 
        "layer_16_o_proj": student.model.layers[16].self_attn.o_proj,
        
        # MLP layers in layer 16
        "layer_16_gate_proj": student.model.layers[16].mlp.gate_proj,
        "layer_16_up_proj": student.model.layers[16].mlp.up_proj,
        "layer_16_down_proj": student.model.layers[16].mlp.down_proj,
    }

    # ALTERNATIVE - More conservative approach (recommended to start with)
    target_modules_conservative = {
        # Only target output projections and MLP layers (safest)
        "layer_8_o_proj": student.model.layers[8].self_attn.o_proj,
        "layer_8_down_proj": student.model.layers[8].mlp.down_proj,
        "layer_16_o_proj": student.model.layers[16].self_attn.o_proj,
        "layer_16_down_proj": student.model.layers[16].mlp.down_proj,
    }


    
    # MODIFICATION: This is the most important Accelerate step. It wraps all components.
    student, optimizer, lr_scheduler, adv_NPO_loader, adv_IKL_loader, benign_DPO_loader, benign_AKL_loader = accelerator.prepare(
        student, optimizer, lr_scheduler, adv_NPO_loader, adv_IKL_loader, benign_DPO_loader, benign_AKL_loader
    )

    global_step = 0
    LOGGING_STEPS = 100
    
    accelerator.print("\n" + "="*50)
    accelerator.print(" " * 16 + "STARTING TRAINING" + " " * 17)
    accelerator.print("="*50 + "\n")



    # =================================================================================
    # =========================== MAIN TRAINING LOOP ==================================
    # =================================================================================

    progress_bar = tqdm(total=num_training_steps, desc="Training", ncols=100)

    for epoch in range(TRAINING_ARGS.EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{TRAINING_ARGS.EPOCHS} =====")

        # --- Adversarial Phase ---
        print(f"\n--- Running Adversarial Phase for Epoch {epoch+1} ---")
        
        # Zip dataloaders with the pre-computed teacher logprobs
        adversarial_iterator = zip(
            adv_NPO_loader,
            adv_IKL_loader,
            precomputed_adv_npo_logps,
            precomputed_adv_ikl_logprobs
        )
        
        for adv_npo_batch, adv_ikl_batch, logps_rejected_harmful, logprob_dist_harmful in adversarial_iterator:
            optimizer.zero_grad()
            student.train()
            
            # Step 1: Calculate perturbations using the updated helper function
            perturbations = calculate_perturbations(
                batch=adv_npo_batch,
                student_model=student,
                target_modules=target_modules_conservative,
                TRAINING_ARGS=TRAINING_ARGS,
                accelerator=accelerator
            )
            
            # Student model already in train mode from the helper function's eval() call
            student.train()

            # Move pre-computed tensors to the current device
            logps_rejected_harmful = logps_rejected_harmful.to(accelerator.device)
            logprob_dist_harmful = logprob_dist_harmful.to(accelerator.device)

            with apply_perturbations(student, target_modules_conservative, perturbations):
                logps_rejected_student = get_logps_batch_NPO(adv_npo_batch, student, accelerator.device, teacher=False)
                logprob_dist_student = get_logps_batch_KL_policy(adv_ikl_batch, student, device=accelerator.device)
            
            npo_loss = npo_loss_fn(logps_rejected_student, logps_rejected_harmful)
            immunization_kl_loss = immunization_kl_loss_fn(logprob_dist_student, logprob_dist_harmful)
            total_adversarial_loss = (1 - TRAINING_ARGS.GAMMA) * npo_loss + TRAINING_ARGS.GAMMA * immunization_kl_loss
            
            # MODIFICATION: Use accelerator.backward()
            accelerator.backward(total_adversarial_loss)
            optimizer.step()
            lr_scheduler.step()
            global_step += 1

            progress_bar.update(1)
            progress_bar.set_postfix({
                "adv_loss": f"{total_adversarial_loss.item():.4f}",
                "npo": f"{npo_loss.item():.4f}",
                "ikl": f"{immunization_kl_loss.item():.4f}"
            })

            if global_step > 0 and global_step % LOGGING_STEPS == 0:
                print(f"\n[Step {global_step}/{num_training_steps}] "
                    f"Adv Loss: {total_adversarial_loss.item():.4f} | "
                    f"NPO Loss: {npo_loss.item():.4f} | "
                    f"Immunity KL: {immunization_kl_loss.item():.4f} | "
                    f"LR: {lr_scheduler.get_last_lr()[0]:.2e}")
                
            wandb.log(
                    {
                        "adversarial/total_loss": total_adversarial_loss.item(),
                        "adversarial/npo_loss": npo_loss.item(),
                        "adversarial/ikl_loss": immunization_kl_loss.item(),
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch
                    },
                    step=global_step
                )

        # --- Harmless Phase ---
        print(f"\n--- Running Harmless (Benign) Phase for Epoch {epoch+1} ---")
        
        
        
        
        # Zip dataloaders with the pre-computed teacher logprobs
        harmless_iterator = zip(
            benign_DPO_loader,
            benign_AKL_loader,
            precomputed_benign_dpo_logps,
            precomputed_benign_akl_logprobs
        )

        for begn_dpo_batch, begn_akl_batch, benign_dpo_logps, logprob_dist_benign in harmless_iterator:
            optimizer.zero_grad()

            # Unpack the DPO logps tuple and move tensors to the correct device
            log_chosen_benign, log_rejected_benign = benign_dpo_logps
            log_chosen_benign = log_chosen_benign.to(accelerator.device)
            log_rejected_benign = log_rejected_benign.to(accelerator.device)
            logprob_dist_benign = logprob_dist_benign.to(accelerator.device)

            # Forward pass for the student model (still required)
            log_chosen_student, log_rejected_student = get_logps_batch_DPO(begn_dpo_batch, student, device=accelerator.device)
            dpo_loss = dpo_loss_fn(log_chosen_student, log_rejected_student, log_chosen_benign, log_rejected_benign)
            
            logprob_dist_student = get_logps_batch_KL_policy(
                begn_akl_batch, student, device=accelerator.device
            )
            alignment_kl_loss = alignment_kl_loss_fn(logprob_dist_student, logprob_dist_benign)

            total_harmless_loss = (1 - TRAINING_ARGS.ALPHA) * dpo_loss + TRAINING_ARGS.ALPHA * alignment_kl_loss
            accelerator.backward(total_harmless_loss)
            optimizer.step()
            lr_scheduler.step()
            global_step += 1

            progress_bar.update(1)
            progress_bar.set_postfix({
                "harmless_loss": f"{total_harmless_loss.item():.4f}",
                "dpo": f"{dpo_loss.item():.4f}",
                "akl": f"{alignment_kl_loss.item():.4f}"
            })
            
            if global_step > 0 and global_step % LOGGING_STEPS == 0:
                print(f"\n[Step {global_step}/{num_training_steps}] "
                    f"Harmless Loss: {total_harmless_loss.item():.4f} | "
                    f"DPO Loss: {dpo_loss.item():.4f} | "
                    f"Alignment KL: {alignment_kl_loss.item():.4f} | "
                    f"LR: {lr_scheduler.get_last_lr()[0]:.2e}")
            wandb.log(
                        {
                            "harmless/total_loss": total_harmless_loss.item(),
                            "harmless/dpo_loss": dpo_loss.item(),
                            "harmless/akl_loss": alignment_kl_loss.item(),
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "epoch": epoch
                        },
                        step=global_step
                    )
            
            
    if accelerator.is_main_process:
        if progress_bar: progress_bar.close()
        save_dir = f"models/dualDistilledWithLAT_{TRAINING_ARGS.STUDENT_MODEL_ID.replace('/', '_')}"
        os.makedirs(save_dir, exist_ok=True)
        accelerator.print(f"\n>>> Saving final model to {save_dir} ...")
        unwrapped_model = accelerator.unwrap_model(student)
        unwrapped_model.save_pretrained(
            save_dir, is_main_process=accelerator.is_main_process,
            save_function=accelerator.save, state_dict=accelerator.get_state_dict(student),
        )
        tokenizer.save_pretrained(save_dir)
        accelerator.print(">>> Model and tokenizer saved successfully!")
    
    

    if accelerator.is_main_process:
        wandb.finish()
    return None








def dualDistillationLoopWithCanaryStabilization():


    import torch
    
    print("\n" + "="*50)
    print(" " * 16 + "dualDistillationLoop + CanaryStabilization" + " " * 17)
    print("="*50 + "\n")
    
    print(TRAINING_ARGS)


    training_args_dict = asdict(TRAINING_ARGS)

    full_bf16_policy = {
        "param_dtype": torch.bfloat16,
        "reduce_dtype": torch.bfloat16,
        "buffer_dtype": torch.bfloat16,
    }

    # Pass this dictionary to the plugin's `fsdp_init_kwargs` argument.
    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision_policy=full_bf16_policy,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=TRANSFORMER_LAYER # Ensure this is a list of strings
    )
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin , log_with="wandb")


    # accelerator.init_trackers(
    #     project_name="DualDistillation",
    #     config=training_args_dict,
    #     init_kwargs={"wandb": {"name": "dual_distillation_CS_run"}} # You can change this name
    # )

    if accelerator.is_main_process:
        wandb.init(
            project="DualDistillation",
            config=training_args_dict,
            name="dual_distillation_CS_run" # New name to distinguish this run
        )
    

    # -----------------------------
    # 2. Load Models & Tokenizer
    # -----------------------------
    # adv_dataset , benign_dataset = load_datasets()
    
    

    # student_model_id = "Qwen/Qwen2.5-7B"
    # harmful_model_id = "models/merged/part1_merged_model"
    # benign_model_id = "EleutherAI/deep-ignorance-e2e-strong-filter-cb-lat"


    student_model_id = TRAINING_ARGS.STUDENT_MODEL_ID
    harmful_model_id = TRAINING_ARGS.HARMFUL_MODEL_ID
    benign_model_id = TRAINING_ARGS.BENIGN_MODEL_ID
    
    
    print("--- Loading TEACHER Models and Tokenizer ---")
    harmfulTeacher = AutoModelForCausalLM.from_pretrained(harmful_model_id , trust_remote_code = True).to(accelerator.device)

    harmfulTeacher.eval()

    tokenizer = AutoTokenizer.from_pretrained(student_model_id, cache_dir="cache_dir")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer = AutoTokenizer.from_pretrained(student_model_id, cache_dir="cache_dir")
    tokenizer.pad_token = tokenizer.eos_token
    harmful_tokenizer = AutoTokenizer.from_pretrained(harmful_model_id, cache_dir="cache_dir")
    harmful_tokenizer.pad_token = harmful_tokenizer.eos_token
    benign_tokenizer = AutoTokenizer.from_pretrained(benign_model_id, cache_dir="cache_dir")
    benign_tokenizer.pad_token = benign_tokenizer.eos_token

    harmful_to_student_map = create_vocab_mapping(harmful_tokenizer, tokenizer, TRAINING_ARGS.DEVICE)
    benign_to_student_map = create_vocab_mapping(benign_tokenizer, tokenizer, TRAINING_ARGS.DEVICE)



    accelerator.print("--- Creating Dataloaders ---")
    (adv_NPO_loader, adv_IKL_loader,
     benign_DPO_loader, benign_AKL_loader ,
     adv_NPO_loader_harmful_teacher , adv_IKL_loader_harmful_teacher ,
     benign_DPO_loader_benign_teacher ,     benign_AKL_loader_benign_teacher) = make_dataloaders(batch_size=TRAINING_ARGS.BATCH_SIZE,
                                                                                                 harmful_tokenizer = harmful_tokenizer ,
                                                                                                 benign_tokenizer =benign_tokenizer ,
                                                                                                 tokenizer=tokenizer ,
                                                                                            sample_size=TRAINING_ARGS.DATASET_SIZE)
  
    npo_loss_fn = NPOLoss(beta=0.1)
    immunization_kl_loss_fn = ImmunizationKLLoss()
    dpo_loss_fn = DPOLoss(beta=0.1)
    alignment_kl_loss_fn = AlignmentKLLoss()



    import torch
    from tqdm.auto import tqdm

    # Assume all required variables and functions (models, dataloaders, loss_fns, etc.) are defined above.

    # =================================================================================
    # ===== PRE-COMPUTE LOG PROBABILITIES FOR TEACHER MODELS (ONCE) =================
    # =================================================================================

    print("\n" + "="*50)
    print(" " * 5 + "Pre-computing teacher log probabilities" + " " * 6)
    print("="*50 + "\n")

    # --- Pre-computation for Adversarial Phase Teacher (harmfulTeacher) ---
    precomputed_adv_npo_logps = []
    precomputed_adv_ikl_logprobs = []
    
    with torch.no_grad():
        accelerator.print("--- Caching Adversarial Teacher outputs ---")
        # MODIFICATION: Disable progress bar on non-main processes
        npo_pbar = tqdm(adv_NPO_loader_harmful_teacher, desc="Caching Harmful NPO", ncols=100, disable=not accelerator.is_main_process)
        for adv_npo_batch in npo_pbar:
            # MODIFICATION: Ensure your helper functions use accelerator.device
            # **IMPORTANT**: You must update your `get_logps_batch_*` functions to accept a `device` argument
            # and use it instead of `accelerator.device`.
            logps_rejected_harmful = get_logps_batch_NPO(adv_npo_batch, harmfulTeacher, accelerator.device)
            precomputed_adv_npo_logps.append(logps_rejected_harmful.cpu())

        ikl_pbar = tqdm(adv_IKL_loader_harmful_teacher, desc="Caching Harmful IKL", ncols=100, disable=not accelerator.is_main_process)
        for adv_ikl_batch in ikl_pbar:
            logprob_dist_harmful = get_logps_batch_KL_ref_modified(batch=adv_ikl_batch, ref_model=harmfulTeacher,vocab_mapping=harmful_to_student_map, device=accelerator.device)
            precomputed_adv_ikl_logprobs.append(logprob_dist_harmful.cpu())
    
    
    accelerator.wait_for_everyone()
    del harmfulTeacher
    gc.collect()
    torch.cuda.empty_cache()
    accelerator.print("harmfulTeacher model removed from GPU memory.")

    # --- Pre-computation for Harmless Phase Teacher (benignTeacher) ---
    accelerator.print("--- Loading BENIGN TEACHER Model ---")
    benignTeacher = AutoModelForCausalLM.from_pretrained(benign_model_id).to(accelerator.device)
    benignTeacher.eval()

    precomputed_benign_dpo_logps = []
    precomputed_benign_akl_logprobs = []

    with torch.no_grad():
        accelerator.print("\n--- Caching Harmless Teacher outputs ---")
        dpo_pbar = tqdm(benign_DPO_loader_benign_teacher, desc="Caching Benign DPO", ncols=100, disable=not accelerator.is_main_process)
        for begn_dpo_batch in dpo_pbar:
            log_chosen_benign, log_rejected_benign = get_logps_batch_DPO(begn_dpo_batch, benignTeacher, accelerator.device)
            precomputed_benign_dpo_logps.append((log_chosen_benign.cpu(), log_rejected_benign.cpu()))
        
        akl_pbar = tqdm(benign_AKL_loader_benign_teacher, desc="Caching Benign AKL", ncols=100, disable=not accelerator.is_main_process)
        for begn_akl_batch in akl_pbar:
            logprob_dist_benign = get_logps_batch_KL_ref_modified(batch=begn_akl_batch, ref_model=benignTeacher,vocab_mapping=benign_to_student_map ,device=accelerator.device)
            precomputed_benign_akl_logprobs.append(logprob_dist_benign.cpu())
    
    
    accelerator.wait_for_everyone()
    del benignTeacher
    gc.collect()
    torch.cuda.empty_cache()
    accelerator.print("benignTeacher model removed from GPU memory.")
        
    
    print("--- Loading STUDENT Model ---")
    
    student = AutoModelForCausalLM.from_pretrained(
        student_model_id, low_cpu_mem_usage=True, dtype=torch.bfloat16
    )
    optimizer = AdamW(student.parameters(), lr=TRAINING_ARGS.LR, weight_decay=0.01)

    num_training_steps = get_num_training_steps_from_dataloaders(
        adv_NPO_loader, benign_DPO_loader, TRAINING_ARGS.EPOCHS
    )

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=TRAINING_ARGS.WARMUP_STEPS,
        num_training_steps=num_training_steps,
    )
    student, optimizer, lr_scheduler, adv_NPO_loader, adv_IKL_loader, benign_DPO_loader, benign_AKL_loader = accelerator.prepare(
        student, optimizer, lr_scheduler, adv_NPO_loader, adv_IKL_loader, benign_DPO_loader, benign_AKL_loader
    )
  
    global_step = 0
    LOGGING_STEPS = 100  # How often to print detailed logs
    student.train()
    
    
    
    
    
    
    print("\n" + "="*50)
    print(" " * 16 + "STARTING TRAINING" + " " * 17)
    print("="*50 + "\n")



    # print("\n--- 3. Setting up Canary Tracking ---")

    target_modules = {
        name: module for name, module in student.named_modules()
        if ("mlp" in name or "self_attn" in name) and isinstance(module, nn.Linear)
    }
    print(f"Tracking {len(target_modules)} target MLP and Attention layers.")
    
    # =================================================================================
    # =========================== MAIN TRAINING LOOP ==================================
    # =================================================================================

    progress_bar = tqdm(total=num_training_steps, desc="Training", ncols=100)

    for epoch in range(TRAINING_ARGS.EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{TRAINING_ARGS.EPOCHS} =====")

        # --- Adversarial Phase ---
        print(f"\n--- Running Adversarial Phase for Epoch {epoch+1} ---")
        
        # Zip dataloaders with the pre-computed teacher logprobs
        adversarial_iterator = zip(
            adv_NPO_loader,
            adv_IKL_loader,
            precomputed_adv_npo_logps,
            precomputed_adv_ikl_logprobs
        )
        
        
        

        
        
        
        
        sample_adversarial_dataset = create_sampled_dataset(dataset="adv", sample_size=TRAINING_ARGS.SAMPLE_SIZE)
        adversarial_prompts = [item['prompt'] for item in sample_adversarial_dataset]
        mean_activations_pre = get_mean_activations(model=student, tokenizer=tokenizer, prompts=adversarial_prompts, target_modules=target_modules, accelerator=accelerator, batch_size=TRAINING_ARGS.BATCH_SIZE)
        
        
        
        
        for adv_npo_batch, adv_ikl_batch, logps_rejected_harmful, logprob_dist_harmful in adversarial_iterator:
            optimizer.zero_grad()
            logps_rejected_harmful = logps_rejected_harmful.to(accelerator.device)
            logprob_dist_harmful = logprob_dist_harmful.to(accelerator.device)

            logps_rejected_student = get_logps_batch_NPO(adv_npo_batch, student, accelerator.device, teacher=False)
            npo_loss = npo_loss_fn(logps_rejected_student, logps_rejected_harmful)
            logprob_dist_student = get_logps_batch_KL_policy(adv_ikl_batch, student, device=accelerator.device)
            immunization_kl_loss = immunization_kl_loss_fn(logprob_dist_student, logprob_dist_harmful)
            total_adversarial_loss = (1 - TRAINING_ARGS.GAMMA) * npo_loss + TRAINING_ARGS.GAMMA * immunization_kl_loss
            
            accelerator.backward(total_adversarial_loss)
            optimizer.step()
            lr_scheduler.step()
            global_step += 1
            if accelerator.is_main_process:
                progress_bar.update(1)

            progress_bar.update(1)
            progress_bar.set_postfix({
                "adv_loss": f"{total_adversarial_loss.item():.4f}",
                "npo": f"{npo_loss.item():.4f}",
                "ikl": f"{immunization_kl_loss.item():.4f}"
            })

            if global_step > 0 and global_step % LOGGING_STEPS == 0:
                print(f"\n[Step {global_step}/{num_training_steps}] "
                    f"Adv Loss: {total_adversarial_loss.item():.4f} | "
                    f"NPO Loss: {npo_loss.item():.4f} | "
                    f"Immunity KL: {immunization_kl_loss.item():.4f} | "
                    f"LR: {lr_scheduler.get_last_lr()[0]:.2e}")



            wandb.log(
                    {
                        "adversarial/total_loss": total_adversarial_loss.item(),
                        "adversarial/npo_loss": npo_loss.item(),
                        "adversarial/ikl_loss": immunization_kl_loss.item(),
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch
                    },
                    step=global_step
                )
                
        accelerator.wait_for_everyone()
        accelerator.print("Identifying Canary Neurons Post-Adversarial Training...")
        mean_activations_post = get_mean_activations(model=student, tokenizer=tokenizer, prompts=adversarial_prompts, target_modules=target_modules, accelerator=accelerator, batch_size=TRAINING_ARGS.BATCH_SIZE)

        # Use a more descriptive name to avoid confusion after broadcasting
        canary_neuron_mask_dict = None

        # ======================================================================================
        # 1. CALCULATE MASK ON MAIN PROCESS (Your logic here is correct)
        # ======================================================================================
        if accelerator.is_main_process:
            # Concatenate all activation changes into a single tensor
            all_changes = torch.cat([
                (mean_activations_pre[name].cpu() - mean_activations_post[name].cpu()).abs() 
                for name in mean_activations_pre.keys()
            ])
            
            # Calculate the threshold from the specified quantile
            canary_threshold = torch.quantile(all_changes.float(), TRAINING_ARGS.CANARY_QUANTILE).item()

            # Create the dictionary of boolean masks
            canary_neuron_mask_dict = {
                name: (mean_activations_pre[name] - mean_activations_post[name]).abs() > canary_threshold
                for name in mean_activations_pre.keys()
            }
            
        accelerator.print(f"Identified canaries using threshold {canary_threshold:.4f}.")

        # ======================================================================================
        # 2. CORRECTLY BROADCAST THE MASK DICTIONARY
        # ======================================================================================
        # The main process prepares a list with the dictionary, others prepare a list with a placeholder
        objects_to_broadcast = [canary_neuron_mask_dict] if accelerator.is_main_process else [None]
        from accelerate.utils import broadcast_object_list

        # broadcast_object_list sends the list from the main process to all others
        broadcasted_list = broadcast_object_list(objects_to_broadcast)

        # Unpack the dictionary from the list on all processes
        canary_neuron_mask_dict = broadcasted_list[0]

        # At this point, all processes (GPUs) have an identical copy of canary_neuron_mask_dict

        # ======================================================================================
        # 3. USE THE BROADCASTED MASK (Improved for clarity)
        # ======================================================================================
        accelerator.print("Capturing target activations for canary neurons on harmless prompts...")
        sample_harmless_dataset = create_sampled_dataset(dataset="begn", sample_size=TRAINING_ARGS.SAMPLE_SIZE)
        harmless_prompts = [item['prompt'] for item in sample_harmless_dataset]
        mean_activations_unlearned = get_mean_activations(
            model=student, 
            tokenizer=tokenizer, 
            prompts=harmless_prompts, 
            target_modules=target_modules, 
            accelerator=accelerator, 
            batch_size=TRAINING_ARGS.BATCH_SIZE
        )

        # Iterate over the items of the activations dict for clarity and robustness
        target_canary_activations = {}
        for name, full_activations in mean_activations_unlearned.items():
            # Get the corresponding mask for this layer
            mask = canary_neuron_mask_dict[name].to(accelerator.device)
            
            # If there are any canary neurons in this layer's mask
            if mask.sum() > 0:
                # Move activations to the correct device, apply the mask, and detach
                target_canary_activations[name] = full_activations.to(accelerator.device)[mask].detach()
        
        # Zip dataloaders with the pre-computed teacher logprobs
        harmless_iterator = zip(
            benign_DPO_loader,
            benign_AKL_loader,
            precomputed_benign_dpo_logps,
            precomputed_benign_akl_logprobs
        )
        
        
        print(f"\n--- Running Harmless (Benign) Phase for Epoch {epoch+1} ---")


        for begn_dpo_batch, begn_akl_batch, benign_dpo_logps, logprob_dist_benign in harmless_iterator:
            optimizer.zero_grad()
            log_chosen_benign, log_rejected_benign = benign_dpo_logps
            log_chosen_benign, log_rejected_benign = log_chosen_benign.to(accelerator.device), log_rejected_benign.to(accelerator.device)
            logprob_dist_benign = logprob_dist_benign.to(accelerator.device)

            log_chosen_student, log_rejected_student = get_logps_batch_DPO(begn_dpo_batch, student, accelerator.device)
            dpo_loss = dpo_loss_fn(log_chosen_student, log_rejected_student, log_chosen_benign, log_rejected_benign)
            logprob_dist_student = get_logps_batch_KL_policy(begn_akl_batch, student, device=accelerator.device)
            alignment_kl_loss = alignment_kl_loss_fn(logprob_dist_student, logprob_dist_benign)
            total_harmless_loss = (1 - TRAINING_ARGS.ALPHA) * dpo_loss + TRAINING_ARGS.ALPHA * alignment_kl_loss
            
            # --- Calculate Canary Stabilization Loss ---
            # --- Canary stabilization: robust hook registration for FSDP-wrapped models ---
            current_activations = {}
            hook_handles = []

            def hook_fn(name):
                def hook(module, inp, out):
                    activation = out[0] if isinstance(out, tuple) else out
                    # store activations (keep detached to avoid accidental graph retention)
                    current_activations[name] = activation.detach()
                return hook

            # unwrap once (gives local replica)
            unwrapped = accelerator.unwrap_model(student)

            # build lookups from named_modules
            named = list(unwrapped.named_modules())  # list of (name, module)
            available_names = [n for n, _ in named]

            def canonical_variants(name):
                """Return a set of plausible normalized variants for matching."""
                variants = set()
                variants.add(name)
                # remove leading/trailing dots
                variants.add(name.lstrip("."))
                # Remove all occurrences of _fsdp_wrapped_module
                variants.add(name.replace("._fsdp_wrapped_module", ""))
                variants.add(name.replace("_fsdp_wrapped_module.", ""))
                variants.add(name.replace("_fsdp_wrapped_module", ""))
                # If name starts with the module wrapper (common), try stripping a leading _fsdp_wrapped_module.
                if name.startswith("_fsdp_wrapped_module."):
                    variants.add(name[len("_fsdp_wrapped_module."):])
                # add with/without leading "model." (many HF models nest under .model)
                if not name.startswith("model.") and "model." in available_names:
                    variants.add("model." + name)
                if name.startswith("model."):
                    variants.add(name[len("model."):])
                return variants

            def find_module_for_target(target_name):
                # 1) exact match
                for n, m in named:
                    if n == target_name:
                        return n, m
                # 2) canonical variants exact match
                for v in canonical_variants(target_name):
                    for n, m in named:
                        if n == v:
                            return n, m
                # 3) suffix match (prefer longest suffix)
                best = (None, None, 0)  # (name, module, match_len)
                target_parts = target_name.split(".")
                for n, m in named:
                    n_parts = n.split(".")
                    # compare from end
                    # count matching trailing components
                    match_len = 0
                    for a, b in zip(reversed(n_parts), reversed(target_parts)):
                        if a == b:
                            match_len += 1
                        else:
                            break
                    if match_len > best[2]:
                        best = (n, m, match_len)
                if best[2] >= 2:  # require at least 2 matching components to be safe
                    return best[0], best[1]
                # 4) try last-3 components exact
                tlast = ".".join(target_parts[-3:])
                for n, m in named:
                    if n.endswith(tlast):
                        return n, m
                return None, None

            # register hooks, but map by matching module object from named_modules
            for tgt_name in list(target_modules.keys()):
                matched_name, submod = find_module_for_target(tgt_name)
                if submod is None:
                    # helpful debug so you can fix mapping if nothing matches
                    raise RuntimeError(
                        f"Could not find submodule '{tgt_name}' in unwrapped student model.\n"
                        f"Available submodule count: {len(available_names)}. Sample names:\n"
                        f"{available_names[:80]}"
                    )
                handle = submod.register_forward_hook(hook_fn(tgt_name))
                hook_handles.append(handle)

            # run a single forward to populate hooks
            _ = student(begn_akl_batch[0])
            for h in hook_handles:
                h.remove()


            stabilization_loss = torch.tensor(0.0, device=accelerator.device)
            num_layers_with_canaries = 0
            for name, current_act_batch in current_activations.items():
                mask = canary_neuron_mask_dict[name].to(accelerator.device)
                if mask.sum() > 0:
                    current_canary_act = current_act_batch.mean(dim=[0, 1])[mask]
                    target_canary_act = target_canary_activations[name]
                    stabilization_loss += F.mse_loss(current_canary_act, target_canary_act)
                    num_layers_with_canaries += 1
            if num_layers_with_canaries > 0:
                stabilization_loss /= num_layers_with_canaries

            final_loss = total_harmless_loss + TRAINING_ARGS.STABILIZATION_LAMBDA * stabilization_loss
            accelerator.backward(final_loss)
            optimizer.step()
            lr_scheduler.step()
            global_step += 1

            
            
            progress_bar.update(1)
            progress_bar.set_postfix({
                "harmless_loss": f"{total_harmless_loss.item():.4f}",
                "dpo": f"{dpo_loss.item():.4f}",
                "akl": f"{alignment_kl_loss.item():.4f}"
            })
            
            if global_step > 0 and global_step % LOGGING_STEPS == 0:
                print(f"\n[Step {global_step}/{num_training_steps}] "
                    f"Harmless Loss: {final_loss.item():.4f} | "
                    f"Stabilization Loss: {stabilization_loss.item():.4f} | "
                    f"DPO Loss: {dpo_loss.item():.4f} | "
                    f"Alignment KL: {alignment_kl_loss.item():.4f} | "
                    f"LR: {lr_scheduler.get_last_lr()[0]:.2e}")

            wandb.log(
                        {
                            "harmless/total_loss": total_harmless_loss.item(),
                            "harmless/dpo_loss": dpo_loss.item(),
                            "harmless/akl_loss": alignment_kl_loss.item(),
                            "harmless/stabilization_loss": stabilization_loss.item(),
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "epoch": epoch
                        },
                        step=global_step
                    )
            
            
            
            
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if progress_bar: progress_bar.close()
        save_dir = f"models/dualdistilledWithCanaryStabilization_{TRAINING_ARGS.STUDENT_MODEL_ID.replace('/', '_')}"
        os.makedirs(save_dir, exist_ok=True)
        accelerator.print(f"\n>>> Saving final model to {save_dir} ...")
        unwrapped_model = accelerator.unwrap_model(student)
        unwrapped_model.save_pretrained(
            save_dir, is_main_process=accelerator.is_main_process,
            save_function=accelerator.save, state_dict=accelerator.get_state_dict(student),
        )
        tokenizer.save_pretrained(save_dir)
        accelerator.print(">>> Model and tokenizer saved successfully!")
        
        
    if accelerator.is_main_process:
        wandb.finish()
    return None








def dualDistillationLoopWithLAT_CanaryStabilization():

    
    import torch



    print("\n" + "="*50)
    print(" " * 16 + "dualDistillationLoop + CanaryStabilization" + " " * 17)
    print("="*50 + "\n")
    
    print(TRAINING_ARGS)

    training_args_dict = asdict(TRAINING_ARGS)
    
    full_bf16_policy = {
        "param_dtype": torch.bfloat16,
        "reduce_dtype": torch.bfloat16,
        "buffer_dtype": torch.bfloat16,
    }

    # Pass this dictionary to the plugin's `fsdp_init_kwargs` argument.
    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision_policy=full_bf16_policy,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=TRANSFORMER_LAYER # Ensure this is a list of strings
    )
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin , log_with="wandb")


    # accelerator.init_trackers(
    #     project_name="DualDistillation",
    #     config=training_args_dict,
    #     init_kwargs={"wandb": {"name": "dual_distillation_LAT_CS_run"}} # You can change this name
    # )

    if accelerator.is_main_process:
        wandb.init(
            project="DualDistillation",
            config=training_args_dict,
            name="dual_distillation_LAT_CS_run" # New name to distinguish this run
        )
    

    # -----------------------------
    # 2. Load  Tokenizer
    # -----------------------------


    student_model_id = TRAINING_ARGS.STUDENT_MODEL_ID
    harmful_model_id = TRAINING_ARGS.HARMFUL_MODEL_ID
    benign_model_id = TRAINING_ARGS.BENIGN_MODEL_ID
    
    
    accelerator.print("--- Loading TEACHER Models and Tokenizer ---")
    harmfulTeacher = AutoModelForCausalLM.from_pretrained(harmful_model_id , trust_remote_code = True).to(accelerator.device)

    harmfulTeacher.eval()

    tokenizer = AutoTokenizer.from_pretrained(student_model_id, cache_dir="cache_dir")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer = AutoTokenizer.from_pretrained(student_model_id, cache_dir="cache_dir")
    tokenizer.pad_token = tokenizer.eos_token
    harmful_tokenizer = AutoTokenizer.from_pretrained(harmful_model_id, cache_dir="cache_dir")
    harmful_tokenizer.pad_token = harmful_tokenizer.eos_token
    benign_tokenizer = AutoTokenizer.from_pretrained(benign_model_id, cache_dir="cache_dir")
    benign_tokenizer.pad_token = benign_tokenizer.eos_token

    harmful_to_student_map = create_vocab_mapping(harmful_tokenizer, tokenizer, TRAINING_ARGS.DEVICE)
    benign_to_student_map = create_vocab_mapping(benign_tokenizer, tokenizer, TRAINING_ARGS.DEVICE)



    accelerator.print("--- Creating Dataloaders ---")
    (adv_NPO_loader, adv_IKL_loader,
     benign_DPO_loader, benign_AKL_loader ,
     adv_NPO_loader_harmful_teacher , adv_IKL_loader_harmful_teacher ,
     benign_DPO_loader_benign_teacher ,     benign_AKL_loader_benign_teacher) = make_dataloaders(batch_size=TRAINING_ARGS.BATCH_SIZE,
                                                                                                 harmful_tokenizer = harmful_tokenizer ,
                                                                                                 benign_tokenizer =benign_tokenizer ,
                                                                                                 tokenizer=tokenizer ,
                                                                                            sample_size=TRAINING_ARGS.DATASET_SIZE)
     
     
     
    npo_loss_fn = NPOLoss(beta=0.1)
    immunization_kl_loss_fn = ImmunizationKLLoss()
    dpo_loss_fn = DPOLoss(beta=0.1)
    alignment_kl_loss_fn = AlignmentKLLoss()



    import torch
    from tqdm.auto import tqdm

    # Assume all required variables and functions (models, dataloaders, loss_fns, etc.) are defined above.

    # =================================================================================
    # ===== PRE-COMPUTE LOG PROBABILITIES FOR TEACHER MODELS (ONCE) =================
    # =================================================================================

    print("\n" + "="*50)
    print(" " * 5 + "Pre-computing teacher log probabilities" + " " * 6)
    print("="*50 + "\n")

    # --- Pre-computation for Adversarial Phase Teacher (harmfulTeacher) ---
    precomputed_adv_npo_logps = []
    precomputed_adv_ikl_logprobs = []
    
    with torch.no_grad():
        accelerator.print("--- Caching Adversarial Teacher outputs ---")
        # MODIFICATION: Disable progress bar on non-main processes
        npo_pbar = tqdm(adv_NPO_loader_harmful_teacher, desc="Caching Harmful NPO", ncols=100, disable=not accelerator.is_main_process)
        for adv_npo_batch in npo_pbar:
            # MODIFICATION: Ensure your helper functions use accelerator.device
            # **IMPORTANT**: You must update your `get_logps_batch_*` functions to accept a `device` argument
            # and use it instead of `accelerator.device`.
            logps_rejected_harmful = get_logps_batch_NPO(adv_npo_batch, harmfulTeacher, accelerator.device)
            precomputed_adv_npo_logps.append(logps_rejected_harmful.cpu())

        ikl_pbar = tqdm(adv_IKL_loader_harmful_teacher, desc="Caching Harmful IKL", ncols=100, disable=not accelerator.is_main_process)
        for adv_ikl_batch in ikl_pbar:
            logprob_dist_harmful = get_logps_batch_KL_ref_modified(batch=adv_ikl_batch, ref_model=harmfulTeacher,vocab_mapping=harmful_to_student_map, device=accelerator.device)
            precomputed_adv_ikl_logprobs.append(logprob_dist_harmful.cpu())
    
    
    accelerator.wait_for_everyone()
    del harmfulTeacher
    gc.collect()
    torch.cuda.empty_cache()
    accelerator.print("harmfulTeacher model removed from GPU memory.")

    # --- Pre-computation for Harmless Phase Teacher (benignTeacher) ---
    accelerator.print("--- Loading BENIGN TEACHER Model ---")
    benignTeacher = AutoModelForCausalLM.from_pretrained(benign_model_id).to(accelerator.device)
    benignTeacher.eval()

    precomputed_benign_dpo_logps = []
    precomputed_benign_akl_logprobs = []

    with torch.no_grad():
        accelerator.print("\n--- Caching Harmless Teacher outputs ---")
        dpo_pbar = tqdm(benign_DPO_loader_benign_teacher, desc="Caching Benign DPO", ncols=100, disable=not accelerator.is_main_process)
        for begn_dpo_batch in dpo_pbar:
            log_chosen_benign, log_rejected_benign = get_logps_batch_DPO(begn_dpo_batch, benignTeacher, accelerator.device)
            precomputed_benign_dpo_logps.append((log_chosen_benign.cpu(), log_rejected_benign.cpu()))
        
        akl_pbar = tqdm(benign_AKL_loader_benign_teacher, desc="Caching Benign AKL", ncols=100, disable=not accelerator.is_main_process)
        for begn_akl_batch in akl_pbar:
            logprob_dist_benign = get_logps_batch_KL_ref_modified(batch=begn_akl_batch, ref_model=benignTeacher,vocab_mapping=benign_to_student_map ,device=accelerator.device)
            precomputed_benign_akl_logprobs.append(logprob_dist_benign.cpu())
    
    
    accelerator.wait_for_everyone()
    del benignTeacher
    gc.collect()
    torch.cuda.empty_cache()
    accelerator.print("benignTeacher model removed from GPU memory.")
        
    
    print("--- Loading STUDENT Model ---")
    
    student = AutoModelForCausalLM.from_pretrained(
        student_model_id, low_cpu_mem_usage=True, dtype=torch.bfloat16
    )
    optimizer = AdamW(student.parameters(), lr=TRAINING_ARGS.LR, weight_decay=0.01)

    num_training_steps = get_num_training_steps_from_dataloaders(
        adv_NPO_loader, benign_DPO_loader, TRAINING_ARGS.EPOCHS
    )

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=TRAINING_ARGS.WARMUP_STEPS,
        num_training_steps=num_training_steps,
    )
    student, optimizer, lr_scheduler, adv_NPO_loader, adv_IKL_loader, benign_DPO_loader, benign_AKL_loader = accelerator.prepare(
        student, optimizer, lr_scheduler, adv_NPO_loader, adv_IKL_loader, benign_DPO_loader, benign_AKL_loader
    )
  
    global_step = 0
    LOGGING_STEPS = 100  # How often to print detailed logs
    student.train()
    
    
    
    
    
    
    print("\n" + "="*50)
    print(" " * 16 + "STARTING TRAINING" + " " * 17)
    print("="*50 + "\n")



    # print("\n--- 3. Setting up lat and Canary Tracking ---")

    perturbation_target_modules_conservative = {
        # Only target output projections and MLP layers (safest)
        "layer_8_o_proj": student.model.layers[8].self_attn.o_proj,
        "layer_8_down_proj": student.model.layers[8].mlp.down_proj,
        "layer_16_o_proj": student.model.layers[16].self_attn.o_proj,
        "layer_16_down_proj": student.model.layers[16].mlp.down_proj,
    }
    
    
    
    # print("\n--- 3. Setting up Canary Tracking ---")
    canary_target_modules = {
        name: module for name, module in student.named_modules()
        if ("mlp" in name or "self_attn" in name) and isinstance(module, nn.Linear)
    }
    print(f"Tracking {len(canary_target_modules)} target MLP and Attention layers.")
    
    # =================================================================================
    # =========================== MAIN TRAINING LOOP ==================================
    # =================================================================================

    progress_bar = tqdm(total=num_training_steps, desc="Training", ncols=100)

    for epoch in range(TRAINING_ARGS.EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{TRAINING_ARGS.EPOCHS} =====")

        # --- Adversarial Phase ---
        print(f"\n--- Running Adversarial Phase for Epoch {epoch+1} ---")
        
        # Zip dataloaders with the pre-computed teacher logprobs
        adversarial_iterator = zip(
            adv_NPO_loader,
            adv_IKL_loader,
            precomputed_adv_npo_logps,
            precomputed_adv_ikl_logprobs
        )
        
        
        

        
        
        
        
        sample_adversarial_dataset = create_sampled_dataset(dataset="adv", sample_size=TRAINING_ARGS.SAMPLE_SIZE)
        adversarial_prompts = [item['prompt'] for item in sample_adversarial_dataset]
        mean_activations_pre = get_mean_activations(model=student, tokenizer=tokenizer, prompts=adversarial_prompts, target_modules=canary_target_modules, accelerator=accelerator, batch_size=TRAINING_ARGS.BATCH_SIZE)
        
        
        
        
        for adv_npo_batch, adv_ikl_batch, logps_rejected_harmful, logprob_dist_harmful in adversarial_iterator:
            optimizer.zero_grad()
            student.train()
            
            # Step 1: Calculate perturbations using the updated helper function
            perturbations = calculate_perturbations(
                batch=adv_npo_batch,
                student_model=student,
                target_modules=perturbation_target_modules_conservative,
                TRAINING_ARGS=TRAINING_ARGS,
                accelerator=accelerator
            )
            
            # Student model already in train mode from the helper function's eval() call
            student.train()

            # Move pre-computed tensors to the current device
            logps_rejected_harmful = logps_rejected_harmful.to(accelerator.device)
            logprob_dist_harmful = logprob_dist_harmful.to(accelerator.device)

            with apply_perturbations(student, perturbation_target_modules_conservative, perturbations):
                logps_rejected_student = get_logps_batch_NPO(adv_npo_batch, student, device=accelerator.device, teacher=False)
                logprob_dist_student = get_logps_batch_KL_policy(adv_ikl_batch, student, device=accelerator.device)
            
            npo_loss = npo_loss_fn(logps_rejected_student, logps_rejected_harmful)
            immunization_kl_loss = immunization_kl_loss_fn(logprob_dist_student, logprob_dist_harmful)
            total_adversarial_loss = (1 - TRAINING_ARGS.GAMMA) * npo_loss + TRAINING_ARGS.GAMMA * immunization_kl_loss
            
            # MODIFICATION: Use accelerator.backward()
            accelerator.backward(total_adversarial_loss)
            optimizer.step()
            lr_scheduler.step()
            global_step += 1
            if accelerator.is_main_process:

                progress_bar.update(1)
                progress_bar.set_postfix({
                    "adv_loss": f"{total_adversarial_loss.item():.4f}",
                    "npo": f"{npo_loss.item():.4f}",
                    "ikl": f"{immunization_kl_loss.item():.4f}"
                })

                if global_step > 0 and global_step % LOGGING_STEPS == 0:
                    print(f"\n[Step {global_step}/{num_training_steps}] "
                        f"Adv Loss: {total_adversarial_loss.item():.4f} | "
                        f"NPO Loss: {npo_loss.item():.4f} | "
                        f"Immunity KL: {immunization_kl_loss.item():.4f} | "
                        f"LR: {lr_scheduler.get_last_lr()[0]:.2e}")

                wandb.log(
                    {
                        "adversarial/total_loss": total_adversarial_loss.item(),
                        "adversarial/npo_loss": npo_loss.item(),
                        "adversarial/ikl_loss": immunization_kl_loss.item(),
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch
                    },
                    step=global_step
                )


        accelerator.wait_for_everyone()
        accelerator.print("Identifying Canary Neurons Post-Adversarial Training...")
        mean_activations_post = get_mean_activations(model=student, tokenizer=tokenizer, prompts=adversarial_prompts, target_modules=canary_target_modules, accelerator=accelerator, batch_size=TRAINING_ARGS.BATCH_SIZE)

        canary_neuron_mask_dict = None

        # ======================================================================================
        # 1. CALCULATE MASK ON MAIN PROCESS (Your logic here is correct)
        # ======================================================================================
        if accelerator.is_main_process:
            # Concatenate all activation changes into a single tensor
            all_changes = torch.cat([
                (mean_activations_pre[name].cpu() - mean_activations_post[name].cpu()).abs() 
                for name in mean_activations_pre.keys()
            ])
            
            # Calculate the threshold from the specified quantile
            canary_threshold = torch.quantile(all_changes.float(), TRAINING_ARGS.CANARY_QUANTILE).item()

            # Create the dictionary of boolean masks
            canary_neuron_mask_dict = {
                name: (mean_activations_pre[name] - mean_activations_post[name]).abs() > canary_threshold
                for name in mean_activations_pre.keys()
            }
            
            accelerator.print(f"Identified canaries using threshold {canary_threshold:.4f}.")

        # ======================================================================================
        # 2. CORRECTLY BROADCAST THE MASK DICTIONARY
        # ======================================================================================
        # The main process prepares a list with the dictionary, others prepare a list with a placeholder
        objects_to_broadcast = [canary_neuron_mask_dict] if accelerator.is_main_process else [None]
        from accelerate.utils import broadcast_object_list

        # broadcast_object_list sends the list from the main process to all others
        broadcasted_list = broadcast_object_list(objects_to_broadcast)

        # Unpack the dictionary from the list on all processes
        canary_neuron_mask_dict = broadcasted_list[0]

        # At this point, all processes (GPUs) have an identical copy of canary_neuron_mask_dict

        # ======================================================================================
        # 3. USE THE BROADCASTED MASK (Improved for clarity)
        # ======================================================================================
        accelerator.print("Capturing target activations for canary neurons on harmless prompts...")
        sample_harmless_dataset = create_sampled_dataset(dataset="begn", sample_size=TRAINING_ARGS.SAMPLE_SIZE)
        harmless_prompts = [item['prompt'] for item in sample_harmless_dataset]
        mean_activations_unlearned = get_mean_activations(
            model=student, 
            tokenizer=tokenizer, 
            prompts=harmless_prompts, 
            target_modules=canary_target_modules, 
            accelerator=accelerator, 
            batch_size=TRAINING_ARGS.BATCH_SIZE
        )

        # Iterate over the items of the activations dict for clarity and robustness
        target_canary_activations = {}
        for name, full_activations in mean_activations_unlearned.items():
            # Get the corresponding mask for this layer
            mask = canary_neuron_mask_dict[name].to(accelerator.device)
            
            # If there are any canary neurons in this layer's mask
            if mask.sum() > 0:
                # Move activations to the correct device, apply the mask, and detach
                target_canary_activations[name] = full_activations.to(accelerator.device)[mask].detach()
        
        # Zip dataloaders with the pre-computed teacher logprobs
        harmless_iterator = zip(
            benign_DPO_loader,
            benign_AKL_loader,
            precomputed_benign_dpo_logps,
            precomputed_benign_akl_logprobs
        )
        
        print(f"\n--- Running Harmless (Benign) Phase for Epoch {epoch+1} ---")

        for begn_dpo_batch, begn_akl_batch, benign_dpo_logps, logprob_dist_benign in harmless_iterator:
            optimizer.zero_grad()
            
            log_chosen_benign, log_rejected_benign = benign_dpo_logps
            log_chosen_benign, log_rejected_benign = log_chosen_benign.to(accelerator.device), log_rejected_benign.to(accelerator.device)
            logprob_dist_benign = logprob_dist_benign.to(accelerator.device)

            log_chosen_student, log_rejected_student = get_logps_batch_DPO(begn_dpo_batch, student, accelerator.device)
            dpo_loss = dpo_loss_fn(log_chosen_student, log_rejected_student, log_chosen_benign, log_rejected_benign)
            logprob_dist_student = get_logps_batch_KL_policy(begn_akl_batch, student, device=accelerator.device)
            alignment_kl_loss = alignment_kl_loss_fn(logprob_dist_student, logprob_dist_benign)
            total_harmless_loss = (1 - TRAINING_ARGS.ALPHA) * dpo_loss + TRAINING_ARGS.ALPHA * alignment_kl_loss
            
            # --- Calculate Canary Stabilization Loss ---
            current_activations = {}
            hook_handles = []

            def hook_fn(name):
                def hook(module, inp, out):
                    activation = out[0] if isinstance(out, tuple) else out
                    # store activations (keep detached to avoid accidental graph retention)
                    current_activations[name] = activation.detach()
                return hook

            # unwrap once (gives local replica)
            unwrapped = accelerator.unwrap_model(student)

            # build lookups from named_modules
            named = list(unwrapped.named_modules())  # list of (name, module)
            available_names = [n for n, _ in named]

            def canonical_variants(name):
                """Return a set of plausible normalized variants for matching."""
                variants = set()
                variants.add(name)
                # remove leading/trailing dots
                variants.add(name.lstrip("."))
                # Remove all occurrences of _fsdp_wrapped_module
                variants.add(name.replace("._fsdp_wrapped_module", ""))
                variants.add(name.replace("_fsdp_wrapped_module.", ""))
                variants.add(name.replace("_fsdp_wrapped_module", ""))
                # If name starts with the module wrapper (common), try stripping a leading _fsdp_wrapped_module.
                if name.startswith("_fsdp_wrapped_module."):
                    variants.add(name[len("_fsdp_wrapped_module."):])
                # add with/without leading "model." (many HF models nest under .model)
                if not name.startswith("model.") and "model." in available_names:
                    variants.add("model." + name)
                if name.startswith("model."):
                    variants.add(name[len("model."):])
                return variants

            def find_module_for_target(target_name):
                # 1) exact match
                for n, m in named:
                    if n == target_name:
                        return n, m
                # 2) canonical variants exact match
                for v in canonical_variants(target_name):
                    for n, m in named:
                        if n == v:
                            return n, m
                # 3) suffix match (prefer longest suffix)
                best = (None, None, 0)  # (name, module, match_len)
                target_parts = target_name.split(".")
                for n, m in named:
                    n_parts = n.split(".")
                    # compare from end
                    # count matching trailing components
                    match_len = 0
                    for a, b in zip(reversed(n_parts), reversed(target_parts)):
                        if a == b:
                            match_len += 1
                        else:
                            break
                    if match_len > best[2]:
                        best = (n, m, match_len)
                if best[2] >= 2:  # require at least 2 matching components to be safe
                    return best[0], best[1]
                # 4) try last-3 components exact
                tlast = ".".join(target_parts[-3:])
                for n, m in named:
                    if n.endswith(tlast):
                        return n, m
                return None, None

            # register hooks, but map by matching module object from named_modules
            for tgt_name in list(canary_target_modules.keys()):
                matched_name, submod = find_module_for_target(tgt_name)
                if submod is None:
                    # helpful debug so you can fix mapping if nothing matches
                    raise RuntimeError(
                        f"Could not find submodule '{tgt_name}' in unwrapped student model.\n"
                        f"Available submodule count: {len(available_names)}. Sample names:\n"
                        f"{available_names[:80]}"
                    )
                handle = submod.register_forward_hook(hook_fn(tgt_name))
                hook_handles.append(handle)

            # run a single forward to populate hooks
            _ = student(begn_akl_batch[0])
            for h in hook_handles:
                h.remove()
            for handle in hook_handles: handle.remove()

            stabilization_loss = torch.tensor(0.0, device=accelerator.device)
            num_layers_with_canaries = 0

            for name, current_act_batch in current_activations.items():
                if name in target_canary_activations:
                    mask = canary_neuron_mask_dict[name].to(accelerator.device)
                    
                    # Ensure the mask is not empty
                    if mask.sum() > 0:
                        # current_act_batch shape is likely [batch, seq_len, hidden_dim]
                        # .mean(dim=1) gets the representation for each item in the batch
                        # Then we take the mean across the batch dim=0
                        current_canary_act = current_act_batch.mean(dim=[0, 1])[mask]
                        target_canary_act = target_canary_activations[name]
                        
                        stabilization_loss += F.mse_loss(current_canary_act, target_canary_act)
                        num_layers_with_canaries += 1

            if num_layers_with_canaries > 0:
                stabilization_loss /= num_layers_with_canaries

            final_loss = total_harmless_loss + TRAINING_ARGS.STABILIZATION_LAMBDA * stabilization_loss
            accelerator.backward(final_loss)
            optimizer.step()
            lr_scheduler.step()
            global_step += 1

            progress_bar.update(1)
            progress_bar.set_postfix({
                "harmless_loss": f"{total_harmless_loss.item():.4f}",
                "dpo": f"{dpo_loss.item():.4f}",
                "akl": f"{alignment_kl_loss.item():.4f}"
            })
            
            if global_step > 0 and global_step % LOGGING_STEPS == 0:
                print(f"\n[Step {global_step}/{num_training_steps}] "
                    f"Harmless Loss: {final_loss.item():.4f} | "
                    f"Stabilization Loss: {stabilization_loss.item():.4f} | "
                    f"DPO Loss: {dpo_loss.item():.4f} | "
                    f"Alignment KL: {alignment_kl_loss.item():.4f} | "
                    f"LR: {lr_scheduler.get_last_lr()[0]:.2e}")
            
            wandb.log(
                        {
                            "harmless/total_loss": total_harmless_loss.item(),
                            "harmless/dpo_loss": dpo_loss.item(),
                            "harmless/akl_loss": alignment_kl_loss.item(),
                            "harmless/stabilization_loss": stabilization_loss.item(),
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "epoch": epoch
                        },
                        step=global_step
                    )
            

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if progress_bar: progress_bar.close()
        save_dir = f"models/dualdistilledWithLAT_CanaryStabilization_{TRAINING_ARGS.STUDENT_MODEL_ID.replace('/', '_')}"
        os.makedirs(save_dir, exist_ok=True)
        accelerator.print(f"\n>>> Saving final model to {save_dir} ...")
        unwrapped_model = accelerator.unwrap_model(student)
        unwrapped_model.save_pretrained(
            save_dir, is_main_process=accelerator.is_main_process,
            save_function=accelerator.save, state_dict=accelerator.get_state_dict(student),
        )
        tokenizer.save_pretrained(save_dir)
        accelerator.print(">>> Model and tokenizer saved successfully!")
        
        
    if accelerator.is_main_process:
        wandb.finish()
    return None








def dualDistillationLoop_CL():

    
    return None



def dualDistillationLoopWithLAT_CL():

    
    return None



def dualDistillationLoopWithCanaryStabilization_CL():

    
    return None


def dualDistillationLoopWithLAT_CanaryStabilization_CL():

    
    return None























#########################################################################################################



if __name__=="__main__":


    dualDistillationLoop()
    # dualDistillationLoopWithLAT()
    # dualDistillationLoopWithCanaryStabilization()
    # dualDistillationLoopWithLAT_CanaryStabilization()