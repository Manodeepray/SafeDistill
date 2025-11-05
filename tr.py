# imports

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Any, Dict
import warnings
from dataclasses import dataclass
from contextlib import contextmanager

import transformers
from transformers import AutoTokenizer , AutoModelForCausalLM , DataCollatorWithPadding , get_scheduler
from dataclasses import dataclass
from torch.optim import AdamW
from tqdm import tqdm 
import os
import gc

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
    GRAD_ACC_STEPS: int = 5 # - FOR CANARY SELECTOION - BATCH SIZE GRAD ACCUMULATION
    DEVICE:int  =  "cuda" if torch.cuda.is_available() else "cpu"
    WARMUP_STEPS: int = 2
    EPSILON:float = 0.02
    STABILIZATION_LAMBDA:int = 0.8
    CANARY_QUANTILE:int = 0.99
    SAMPLE_SIZE:int = 1000 # for mean activation aclculation 
    STUDENT_MODEL_ID:str = "Qwen/Qwen2.5-7B"
    HARMFUL_MODEL_ID:str = "Qwen/Qwen2.5-7B"
    BENIGN_MODEL_ID:str = "Qwen/Qwen2.5-7B"
    # STUDENT_MODEL_ID = "Qwen/Qwen2.5-7B"
    # HARMFUL_MODEL_ID = "models/merged/part1_merged_model"
    # BENIGN_MODEL_ID = "EleutherAI/deep-ignorance-e2e-strong-filter-cb-lat"
    DATASET_SIZE:int  = 16




TRAINING_ARGS = trainingArgs()

## COLLAT FN --> DATACOLLATOR W PADDDING






        
        

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
        
        
    # # First, flatten the dataset from a list of batches to a list of items
    # all_items = [item for batch in dataset for item in batch]
    
    # # Ensure sample size is not larger than the dataset
    # if sample_size > len(all_items):
    #     print(f"Warning: Sample size ({sample_size}) is larger than the dataset ({len(all_items)}). Using the full dataset.")
    #     sample_size = len(all_items)
        
    # Create a random sample
    
    dataset = dataset.shuffle(seed=42)
    
    # For compatibility, we can optionally re-batch the sampled items, 
    # though the old get_mean_activations function can just take the flat list of prompts.
    # Here we just return the flat list of sampled items.
    # print(f"Created a sampled dataset of {sample_size} items.")
    return dataset


# ===================================================
# LOSSES

## NPO

def get_num_training_steps_from_dataloaders(adv_loader, benign_loader, epochs):
    steps_per_epoch = len(adv_loader) + len(benign_loader)
    return steps_per_epoch * epochs


    
# ===================================================

# TRIAINING LOOPS


## DUALDISTILLATION LOOP (STUDENT MODEL , TOKENIZER , OPTIMIZER ,HARMFUL_MODEL/DATASET , BENIGN_MODEL/DATASET , ADVERSARIAL_DATASET , BENIGN_DATASET  )
def dualDistillationLoop():

    print("\n" + "="*50)
    print(" " * 16 + "dualDistillationLoop" + " " * 17)
    print("="*50 + "\n")
    
    print(TRAINING_ARGS)

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
    harmfulTeacher = AutoModelForCausalLM.from_pretrained(harmful_model_id , trust_remote_code = True).to(TRAINING_ARGS.DEVICE)

    harmfulTeacher.eval()

    tokenizer = AutoTokenizer.from_pretrained(student_model_id, cache_dir="cache_dir")
    tokenizer.pad_token = tokenizer.eos_token


    print("--- Creating Dataloaders ---")
    adv_NPO_loader, adv_IKL_loader, benign_DPO_loader, benign_AKL_loader = make_dataloaders(batch_size=TRAINING_ARGS.BATCH_SIZE,
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
        print("--- Caching Adversarial Teacher outputs ---")
        # Cache NPO logps from harmfulTeacher
        for adv_npo_batch in tqdm(adv_NPO_loader, desc="Caching Harmful NPO", ncols=100):
            logps_rejected_harmful = get_logps_batch_NPO(adv_npo_batch, harmfulTeacher, TRAINING_ARGS.DEVICE)
            precomputed_adv_npo_logps.append(logps_rejected_harmful) # Move to CPU to save VRAM

        # Cache IKL logprobs from harmfulTeacher
        for adv_ikl_batch in tqdm(adv_IKL_loader, desc="Caching Harmful IKL", ncols=100):
            logprob_dist_harmful = get_logps_batch_KL_ref(adv_ikl_batch, harmfulTeacher, device=TRAINING_ARGS.DEVICE)
            precomputed_adv_ikl_logprobs.append(logprob_dist_harmful) # Move to CPU to save VRAM

    del harmfulTeacher
    gc.collect() 

    # Ask PyTorch to release cached memory that is now unreferenced
    torch.cuda.empty_cache()

    print("harmfulTeacher model removed from GPU memory.")
    
    benignTeacher = AutoModelForCausalLM.from_pretrained(benign_model_id, ).to(TRAINING_ARGS.DEVICE)

    benignTeacher.eval()

    # # --- Pre-computation for Harmless Phase Teacher (benignTeacher) ---
    precomputed_benign_dpo_logps = []
    precomputed_benign_akl_logprobs = []

    with torch.no_grad():
        print("\n--- Caching Harmless Teacher outputs ---")
        # Cache DPO logps from benignTeacher
        for begn_dpo_batch in tqdm(benign_DPO_loader, desc="Caching Benign DPO", ncols=100):
            log_chosen_benign, log_rejected_benign = get_logps_batch_DPO(begn_dpo_batch, benignTeacher, TRAINING_ARGS.DEVICE)
            # Store as a tuple
            precomputed_benign_dpo_logps.append((log_chosen_benign.cpu(), log_rejected_benign.cpu()))

        # Cache AKL logprobs from benignTeacher
        for begn_akl_batch in tqdm(benign_AKL_loader, desc="Caching Benign AKL", ncols=100):
            logprob_dist_benign = get_logps_batch_KL_ref(begn_akl_batch, benignTeacher, device=TRAINING_ARGS.DEVICE)
            precomputed_benign_akl_logprobs.append(logprob_dist_benign.cpu())

    
    
    del benignTeacher
    gc.collect() 

    # Ask PyTorch to release cached memory that is now unreferenced
    torch.cuda.empty_cache()

    print("benignTeacher model removed from GPU memory.")
        
    
    print("--- Loading STUDENT Model ---")
    
    student = AutoModelForCausalLM.from_pretrained(student_model_id, ).to(TRAINING_ARGS.DEVICE)

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

  
    global_step = 0
    LOGGING_STEPS = 100  # How often to print detailed logs
    student.train()
    
    
    
    
    
    
    print("\n" + "="*50)
    print(" " * 16 + "STARTING TRAINING" + " " * 17)
    print("="*50 + "\n")


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

            # Move pre-computed tensors to the correct device for the current batch
            logps_rejected_harmful = logps_rejected_harmful.to(TRAINING_ARGS.DEVICE)
            logprob_dist_harmful = logprob_dist_harmful.to(TRAINING_ARGS.DEVICE)

            # Forward pass for the student model (still required)
            logps_rejected_student = get_logps_batch_NPO(adv_npo_batch, student, TRAINING_ARGS.DEVICE , teacher=False)
            npo_loss = npo_loss_fn(logps_rejected_student, logps_rejected_harmful)

            logprob_dist_student = get_logps_batch_KL_policy(
                adv_ikl_batch, student, device=TRAINING_ARGS.DEVICE
            )
            immunization_kl_loss = immunization_kl_loss_fn(logprob_dist_student, logprob_dist_harmful)

            total_adversarial_loss = (1 - TRAINING_ARGS.GAMMA) * npo_loss + TRAINING_ARGS.GAMMA * immunization_kl_loss
            total_adversarial_loss.backward()

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
            log_chosen_benign = log_chosen_benign.to(TRAINING_ARGS.DEVICE)
            log_rejected_benign = log_rejected_benign.to(TRAINING_ARGS.DEVICE)
            logprob_dist_benign = logprob_dist_benign.to(TRAINING_ARGS.DEVICE)

            # Forward pass for the student model (still required)
            log_chosen_student, log_rejected_student = get_logps_batch_DPO(begn_dpo_batch, student, TRAINING_ARGS.DEVICE)
            dpo_loss = dpo_loss_fn(log_chosen_student, log_rejected_student, log_chosen_benign, log_rejected_benign)
            
            logprob_dist_student = get_logps_batch_KL_policy(
                begn_akl_batch, student, device=TRAINING_ARGS.DEVICE
            )
            alignment_kl_loss = alignment_kl_loss_fn(logprob_dist_student, logprob_dist_benign)

            total_harmless_loss = (1 - TRAINING_ARGS.ALPHA) * dpo_loss + TRAINING_ARGS.ALPHA * alignment_kl_loss
            total_harmless_loss.backward()

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

    progress_bar.close()
    print("\n" + "="*50)
    print(" " * 16 + "TRAINING COMPLETE" + " " * 17)
    print("="*50 + "\n")
    
    
    
    save_dir = f"models/dualDistilled-{student_model_id}-01"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n>>> Saving model to {save_dir} ...")
    student.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(">>> Model saved successfully!")
    return None












def dualDistillationLoopWithLAT():




    print("\n" + "="*50)
    print(" " * 16 + "dualDistillationLoop + LAT " + " " * 17)
    print("="*50 + "\n")



    print(TRAINING_ARGS)

    # -----------------------------
    # 2. Load Models & Tokenizer
    # -----------------------------
    # adv_dataset , benign_dataset = load_datasets()
    

    # ------

    # student_model_id = "Qwen/Qwen2.5-7B"
    # harmful_model_id = "models/merged/part1_merged_model"
    # benign_model_id = "EleutherAI/deep-ignorance-e2e-strong-filter-cb-lat"


    student_model_id = TRAINING_ARGS.STUDENT_MODEL_ID
    harmful_model_id = TRAINING_ARGS.HARMFUL_MODEL_ID
    benign_model_id = TRAINING_ARGS.BENIGN_MODEL_ID
    
    
    print("--- Loading TEACHER Models and Tokenizer ---")
    harmfulTeacher = AutoModelForCausalLM.from_pretrained(harmful_model_id , trust_remote_code = True).to(TRAINING_ARGS.DEVICE)

    harmfulTeacher.eval()

    tokenizer = AutoTokenizer.from_pretrained(student_model_id, cache_dir="cache_dir")
    tokenizer.pad_token = tokenizer.eos_token


    print("--- Creating Dataloaders ---")
    adv_NPO_loader, adv_IKL_loader, benign_DPO_loader, benign_AKL_loader = make_dataloaders(batch_size=TRAINING_ARGS.BATCH_SIZE,
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
        print("--- Caching Adversarial Teacher outputs ---")
        # Cache NPO logps from harmfulTeacher
        for adv_npo_batch in tqdm(adv_NPO_loader, desc="Caching Harmful NPO", ncols=100):
            logps_rejected_harmful = get_logps_batch_NPO(adv_npo_batch, harmfulTeacher, TRAINING_ARGS.DEVICE)
            precomputed_adv_npo_logps.append(logps_rejected_harmful) # Move to CPU to save VRAM

        # Cache IKL logprobs from harmfulTeacher
        for adv_ikl_batch in tqdm(adv_IKL_loader, desc="Caching Harmful IKL", ncols=100):
            logprob_dist_harmful = get_logps_batch_KL_ref(adv_ikl_batch, harmfulTeacher, device=TRAINING_ARGS.DEVICE)
            precomputed_adv_ikl_logprobs.append(logprob_dist_harmful) # Move to CPU to save VRAM

    del harmfulTeacher
    gc.collect() 

    # Ask PyTorch to release cached memory that is now unreferenced
    torch.cuda.empty_cache()

    print("harmfulTeacher model removed from GPU memory.")
    
    benignTeacher = AutoModelForCausalLM.from_pretrained(benign_model_id, ).to(TRAINING_ARGS.DEVICE)

    benignTeacher.eval()

    # # --- Pre-computation for Harmless Phase Teacher (benignTeacher) ---
    precomputed_benign_dpo_logps = []
    precomputed_benign_akl_logprobs = []

    with torch.no_grad():
        print("\n--- Caching Harmless Teacher outputs ---")
        # Cache DPO logps from benignTeacher
        for begn_dpo_batch in tqdm(benign_DPO_loader, desc="Caching Benign DPO", ncols=100):
            log_chosen_benign, log_rejected_benign = get_logps_batch_DPO(begn_dpo_batch, benignTeacher, TRAINING_ARGS.DEVICE)
            # Store as a tuple
            precomputed_benign_dpo_logps.append((log_chosen_benign.cpu(), log_rejected_benign.cpu()))

        # Cache AKL logprobs from benignTeacher
        for begn_akl_batch in tqdm(benign_AKL_loader, desc="Caching Benign AKL", ncols=100):
            logprob_dist_benign = get_logps_batch_KL_ref(begn_akl_batch, benignTeacher, device=TRAINING_ARGS.DEVICE)
            precomputed_benign_akl_logprobs.append(logprob_dist_benign.cpu())

    
    
    del benignTeacher
    gc.collect() 

    # Ask PyTorch to release cached memory that is now unreferenced
    torch.cuda.empty_cache()

    print("benignTeacher model removed from GPU memory.")
        
    
    print("--- Loading STUDENT Model ---")
    
    student = AutoModelForCausalLM.from_pretrained(student_model_id, ).to(TRAINING_ARGS.DEVICE)




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

  
    global_step = 0
    LOGGING_STEPS = 100  # How often to print detailed logs
    student.train()
    
    
    
    
    
    
    print("\n" + "="*50)
    print(" " * 16 + "STARTING TRAINING" + " " * 17)
    print("="*50 + "\n")


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
            
            # Step 1: Calculate the perturbation plan (Pass 1)
            perturbations = calculate_perturbations(batch=adv_npo_batch,
                                                    student_model=student,
                                                    target_modules=target_modules_conservative,
                                                    device=TRAINING_ARGS.DEVICE ,
                                                    TRAINING_ARGS = TRAINING_ARGS)
            
            
            # Move pre-computed tensors to the correct device for the current batch
            logps_rejected_harmful = logps_rejected_harmful.to(TRAINING_ARGS.DEVICE)
            logprob_dist_harmful = logprob_dist_harmful.to(TRAINING_ARGS.DEVICE)

            with apply_perturbations(student, target_modules_conservative, perturbations):
            # Forward pass for the student model (still required)
                logps_rejected_student = get_logps_batch_NPO(batch=adv_npo_batch, model=student, device=TRAINING_ARGS.DEVICE , teacher=False)
                
                logprob_dist_student = get_logps_batch_KL_policy(
                adv_ikl_batch, student, device=TRAINING_ARGS.DEVICE
                )
            
            npo_loss = npo_loss_fn(logps_rejected_student, logps_rejected_harmful)

            
            immunization_kl_loss = immunization_kl_loss_fn(logprob_dist_student, logprob_dist_harmful)

            total_adversarial_loss = (1 - TRAINING_ARGS.GAMMA) * npo_loss + TRAINING_ARGS.GAMMA * immunization_kl_loss
            total_adversarial_loss.backward()

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
            log_chosen_benign = log_chosen_benign.to(TRAINING_ARGS.DEVICE)
            log_rejected_benign = log_rejected_benign.to(TRAINING_ARGS.DEVICE)
            logprob_dist_benign = logprob_dist_benign.to(TRAINING_ARGS.DEVICE)

            # Forward pass for the student model (still required)
            log_chosen_student, log_rejected_student = get_logps_batch_DPO(begn_dpo_batch, student, TRAINING_ARGS.DEVICE)
            dpo_loss = dpo_loss_fn(log_chosen_student, log_rejected_student, log_chosen_benign, log_rejected_benign)
            
            logprob_dist_student = get_logps_batch_KL_policy(
                begn_akl_batch, student, device=TRAINING_ARGS.DEVICE
            )
            alignment_kl_loss = alignment_kl_loss_fn(logprob_dist_student, logprob_dist_benign)

            total_harmless_loss = (1 - TRAINING_ARGS.ALPHA) * dpo_loss + TRAINING_ARGS.ALPHA * alignment_kl_loss
            total_harmless_loss.backward()

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

    progress_bar.close()
    print("\n" + "="*50)
    print(" " * 16 + "TRAINING COMPLETE" + " " * 17)
    print("="*50 + "\n")
    
    
    
    save_dir = f"models/dualDistilledWithLAT-{student_model_id}-01"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n>>> Saving model to {save_dir} ...")
    student.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(">>> Model saved successfully!")
    return None








def dualDistillationLoopWithCanaryStabilization():

    
    print("\n" + "="*50)
    print(" " * 16 + "dualDistillationLoop + CanaryStabilization" + " " * 17)
    print("="*50 + "\n")
    
    print(TRAINING_ARGS)

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
    harmfulTeacher = AutoModelForCausalLM.from_pretrained(harmful_model_id , trust_remote_code = True).to(TRAINING_ARGS.DEVICE)

    harmfulTeacher.eval()

    tokenizer = AutoTokenizer.from_pretrained(student_model_id, cache_dir="cache_dir")
    tokenizer.pad_token = tokenizer.eos_token


    print("--- Creating Dataloaders ---")
    adv_NPO_loader, adv_IKL_loader, benign_DPO_loader, benign_AKL_loader = make_dataloaders(batch_size=TRAINING_ARGS.BATCH_SIZE,
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
        print("--- Caching Adversarial Teacher outputs ---")
        # Cache NPO logps from harmfulTeacher
        for adv_npo_batch in tqdm(adv_NPO_loader, desc="Caching Harmful NPO", ncols=100):
            logps_rejected_harmful = get_logps_batch_NPO(adv_npo_batch, harmfulTeacher, TRAINING_ARGS.DEVICE)
            precomputed_adv_npo_logps.append(logps_rejected_harmful) # Move to CPU to save VRAM

        # Cache IKL logprobs from harmfulTeacher
        for adv_ikl_batch in tqdm(adv_IKL_loader, desc="Caching Harmful IKL", ncols=100):
            logprob_dist_harmful = get_logps_batch_KL_ref(adv_ikl_batch, harmfulTeacher, device=TRAINING_ARGS.DEVICE)
            precomputed_adv_ikl_logprobs.append(logprob_dist_harmful) # Move to CPU to save VRAM

    del harmfulTeacher
    gc.collect() 

    # Ask PyTorch to release cached memory that is now unreferenced
    torch.cuda.empty_cache()

    print("harmfulTeacher model removed from GPU memory.")
    
    benignTeacher = AutoModelForCausalLM.from_pretrained(benign_model_id, ).to(TRAINING_ARGS.DEVICE)

    benignTeacher.eval()

    # # --- Pre-computation for Harmless Phase Teacher (benignTeacher) ---
    precomputed_benign_dpo_logps = []
    precomputed_benign_akl_logprobs = []

    with torch.no_grad():
        print("\n--- Caching Harmless Teacher outputs ---")
        # Cache DPO logps from benignTeacher
        for begn_dpo_batch in tqdm(benign_DPO_loader, desc="Caching Benign DPO", ncols=100):
            log_chosen_benign, log_rejected_benign = get_logps_batch_DPO(begn_dpo_batch, benignTeacher, TRAINING_ARGS.DEVICE)
            # Store as a tuple
            precomputed_benign_dpo_logps.append((log_chosen_benign.cpu(), log_rejected_benign.cpu()))

        # Cache AKL logprobs from benignTeacher
        for begn_akl_batch in tqdm(benign_AKL_loader, desc="Caching Benign AKL", ncols=100):
            logprob_dist_benign = get_logps_batch_KL_ref(begn_akl_batch, benignTeacher, device=TRAINING_ARGS.DEVICE)
            precomputed_benign_akl_logprobs.append(logprob_dist_benign.cpu())

    
    
    del benignTeacher
    gc.collect() 

    # Ask PyTorch to release cached memory that is now unreferenced
    torch.cuda.empty_cache()

    print("benignTeacher model removed from GPU memory.")
        
    
    print("--- Loading STUDENT Model ---")
    
    student = AutoModelForCausalLM.from_pretrained(student_model_id, ).to(TRAINING_ARGS.DEVICE)

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
        
        adversarial_prompts =  [item['prompt'] for item in sample_adversarial_dataset]
        
        
        # print("Capturing pre-adversarial training activations...")
        mean_activations_pre = get_mean_activations(student, tokenizer, TRAINING_ARGS.DEVICE, adversarial_prompts, target_modules)


        
        
        
        for adv_npo_batch, adv_ikl_batch, logps_rejected_harmful, logprob_dist_harmful in adversarial_iterator:
            optimizer.zero_grad()

            # Move pre-computed tensors to the correct device for the current batch
            logps_rejected_harmful = logps_rejected_harmful.to(TRAINING_ARGS.DEVICE)
            logprob_dist_harmful = logprob_dist_harmful.to(TRAINING_ARGS.DEVICE)

            # Forward pass for the student model (still required)
            logps_rejected_student = get_logps_batch_NPO(adv_npo_batch, student, TRAINING_ARGS.DEVICE , teacher=False)
            npo_loss = npo_loss_fn(logps_rejected_student, logps_rejected_harmful)

            logprob_dist_student = get_logps_batch_KL_policy(
                adv_ikl_batch, student, device=TRAINING_ARGS.DEVICE
            )
            immunization_kl_loss = immunization_kl_loss_fn(logprob_dist_student, logprob_dist_harmful)

            total_adversarial_loss = (1 - TRAINING_ARGS.GAMMA) * npo_loss + TRAINING_ARGS.GAMMA * immunization_kl_loss
            total_adversarial_loss.backward()

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




        print("\n--- 5. Identifying Canary Neurons Post-Adversarial Training ---")
        mean_activations_post = get_mean_activations(model=student, tokenizer=tokenizer, device=TRAINING_ARGS.DEVICE, prompts=adversarial_prompts, target_modules=target_modules)

        all_changes = torch.cat([(mean_activations_pre[name] - mean_activations_post[name]).abs() for name in mean_activations_pre.keys()])
        canary_threshold = torch.quantile(all_changes, TRAINING_ARGS.CANARY_QUANTILE).item()
        canary_neuron_mask = {
            name: (mean_activations_pre[name] - mean_activations_post[name]).abs() > canary_threshold
            for name in mean_activations_pre.keys()
        }
        print(f"Identified canaries using threshold {canary_threshold:.4f}.")
        
        





        # --- Harmless Phase ---
        print(f"\n--- Running Harmless (Benign) Phase for Epoch {epoch+1} ---")
        
        target_canary_activations = {}
        
        
        sample_harmless_dataset = create_sampled_dataset(dataset="begn", sample_size=2)
        harmless_prompts =  [item['prompt'] for item in sample_harmless_dataset]
        
        
        
        mean_activations_unlearned = get_mean_activations(model=student, tokenizer=tokenizer, device=TRAINING_ARGS.DEVICE, prompts=harmless_prompts, target_modules=target_modules)
        
        for name, full_activations in mean_activations_unlearned.items():
            mask = canary_neuron_mask[name].to(TRAINING_ARGS.DEVICE)
            target_canary_activations[name] = full_activations.to(TRAINING_ARGS.DEVICE)[mask].detach()




        
        
        
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
            log_chosen_benign = log_chosen_benign.to(TRAINING_ARGS.DEVICE)
            log_rejected_benign = log_rejected_benign.to(TRAINING_ARGS.DEVICE)
            logprob_dist_benign = logprob_dist_benign.to(TRAINING_ARGS.DEVICE)

            # Forward pass for the student model (still required)
            log_chosen_student, log_rejected_student = get_logps_batch_DPO(begn_dpo_batch, student, TRAINING_ARGS.DEVICE)
            dpo_loss = dpo_loss_fn(log_chosen_student, log_rejected_student, log_chosen_benign, log_rejected_benign)
            
            logprob_dist_student = get_logps_batch_KL_policy(
                begn_akl_batch, student, device=TRAINING_ARGS.DEVICE
            )
            alignment_kl_loss = alignment_kl_loss_fn(logprob_dist_student, logprob_dist_benign)

            total_harmless_loss = (1 - TRAINING_ARGS.ALPHA) * dpo_loss + TRAINING_ARGS.ALPHA * alignment_kl_loss
            
            
            
            
            # --- B: Calculate Canary Stabilization Loss ---
            # Get current activations using hooks
            current_activations = {}
            hook_handles = []
            def hook_fn(name):
                def hook(module, inp, out):
                    activation = out[0] if isinstance(out, tuple) else out
                    current_activations[name] = activation.mean(dim=1)
                return hook
            for name, module in target_modules.items():
                handle = module.register_forward_hook(hook_fn(name))
                hook_handles.append(handle)
            
            
            
            # A single forward pass on the prompt to trigger hooks

            _ = student(begn_akl_batch[0].to(TRAINING_ARGS.DEVICE))
            for handle in hook_handles: handle.remove() # Clean up hooks immediately
            
            
            
            
            # Calculate MSE
            stabilization_loss = torch.tensor(0.0).to(TRAINING_ARGS.DEVICE)
            num_layers_with_canaries = 0
            for name, current_act_batch in current_activations.items():
                mask = canary_neuron_mask[name].to(TRAINING_ARGS.DEVICE)
                if mask.sum() > 0:
                    current_canary_act = current_act_batch.mean(dim=0)[mask]
                    target_canary_act = target_canary_activations[name]
                    stabilization_loss += F.mse_loss(current_canary_act, target_canary_act)
                    num_layers_with_canaries += 1
            
            if num_layers_with_canaries > 0:
                stabilization_loss /= num_layers_with_canaries
            
            
            final_loss = total_harmless_loss + TRAINING_ARGS.STABILIZATION_LAMBDA * stabilization_loss
            final_loss.backward()

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

    progress_bar.close()
    print("\n" + "="*50)
    print(" " * 16 + "TRAINING COMPLETE" + " " * 17)
    print("="*50 + "\n")
    
    
    
    save_dir = f"models/dualDistilledWithCanaryStabilization-{student_model_id}-01"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n>>> Saving model to {save_dir} ...")
    student.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(">>> Model saved successfully!")
    return None








def dualDistillationLoopWithLAT_CanaryStabilization():

    print("\n" + "="*50)
    print(" " * 16 + "dualDistillationLoop + LAT + CanaryStabilization" + " " * 17)
    print("="*50 + "\n")


    print(TRAINING_ARGS)

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
    harmfulTeacher = AutoModelForCausalLM.from_pretrained(harmful_model_id , trust_remote_code = True).to(TRAINING_ARGS.DEVICE)

    harmfulTeacher.eval()

    tokenizer = AutoTokenizer.from_pretrained(student_model_id, cache_dir="cache_dir")
    tokenizer.pad_token = tokenizer.eos_token


    print("--- Creating Dataloaders ---")
    adv_NPO_loader, adv_IKL_loader, benign_DPO_loader, benign_AKL_loader = make_dataloaders(batch_size=TRAINING_ARGS.BATCH_SIZE,
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
        print("--- Caching Adversarial Teacher outputs ---")
        # Cache NPO logps from harmfulTeacher
        for adv_npo_batch in tqdm(adv_NPO_loader, desc="Caching Harmful NPO", ncols=100):
            logps_rejected_harmful = get_logps_batch_NPO(adv_npo_batch, harmfulTeacher, TRAINING_ARGS.DEVICE)
            precomputed_adv_npo_logps.append(logps_rejected_harmful) # Move to CPU to save VRAM

        # Cache IKL logprobs from harmfulTeacher
        for adv_ikl_batch in tqdm(adv_IKL_loader, desc="Caching Harmful IKL", ncols=100):
            logprob_dist_harmful = get_logps_batch_KL_ref(adv_ikl_batch, harmfulTeacher, device=TRAINING_ARGS.DEVICE)
            precomputed_adv_ikl_logprobs.append(logprob_dist_harmful) # Move to CPU to save VRAM

    del harmfulTeacher
    gc.collect() 

    # Ask PyTorch to release cached memory that is now unreferenced
    torch.cuda.empty_cache()

    print("harmfulTeacher model removed from GPU memory.")
    
    benignTeacher = AutoModelForCausalLM.from_pretrained(benign_model_id, ).to(TRAINING_ARGS.DEVICE)

    benignTeacher.eval()

    # # --- Pre-computation for Harmless Phase Teacher (benignTeacher) ---
    precomputed_benign_dpo_logps = []
    precomputed_benign_akl_logprobs = []

    with torch.no_grad():
        print("\n--- Caching Harmless Teacher outputs ---")
        # Cache DPO logps from benignTeacher
        for begn_dpo_batch in tqdm(benign_DPO_loader, desc="Caching Benign DPO", ncols=100):
            log_chosen_benign, log_rejected_benign = get_logps_batch_DPO(begn_dpo_batch, benignTeacher, TRAINING_ARGS.DEVICE)
            # Store as a tuple
            precomputed_benign_dpo_logps.append((log_chosen_benign.cpu(), log_rejected_benign.cpu()))

        # Cache AKL logprobs from benignTeacher
        for begn_akl_batch in tqdm(benign_AKL_loader, desc="Caching Benign AKL", ncols=100):
            logprob_dist_benign = get_logps_batch_KL_ref(begn_akl_batch, benignTeacher, device=TRAINING_ARGS.DEVICE)
            precomputed_benign_akl_logprobs.append(logprob_dist_benign.cpu())

    
    
    del benignTeacher
    gc.collect() 

    # Ask PyTorch to release cached memory that is now unreferenced
    torch.cuda.empty_cache()

    print("benignTeacher model removed from GPU memory.")
        
    
    print("--- Loading STUDENT Model ---")
    
    student = AutoModelForCausalLM.from_pretrained(student_model_id, ).to(TRAINING_ARGS.DEVICE)

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

  
    global_step = 0
    LOGGING_STEPS = 100  # How often to print detailed logs
    student.train()
    
    
    
    
    
    
    print("\n" + "="*50)
    print(" " * 16 + "STARTING TRAINING" + " " * 17)
    print("="*50 + "\n")


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
        
        
        # print("\n--- 3. Setting up Canary Tracking ---")

        
        
        
        
        sample_adversarial_dataset = create_sampled_dataset(dataset="adv", sample_size=TRAINING_ARGS.SAMPLE_SIZE)
        
        adversarial_prompts =  [item['prompt'] for item in sample_adversarial_dataset]
        
        
        # print("Capturing pre-adversarial training activations...")
        mean_activations_pre = get_mean_activations(student, tokenizer, TRAINING_ARGS.DEVICE, adversarial_prompts, canary_target_modules)


        
        student.train()
        
        for adv_npo_batch, adv_ikl_batch, logps_rejected_harmful, logprob_dist_harmful in adversarial_iterator:
            optimizer.zero_grad()


            perturbations = calculate_perturbations(batch=adv_npo_batch,
                                                    student_model=student,
                                                    target_modules=perturbation_target_modules_conservative,
                                                    device=TRAINING_ARGS.DEVICE ,
                                                    TRAINING_ARGS=TRAINING_ARGS)

            # Move pre-computed tensors to the correct device for the current batch
            logps_rejected_harmful = logps_rejected_harmful.to(TRAINING_ARGS.DEVICE)
            logprob_dist_harmful = logprob_dist_harmful.to(TRAINING_ARGS.DEVICE)

            # Forward pass for the student model (still required)
            with apply_perturbations(student, perturbation_target_modules_conservative, perturbations):
            # Forward pass for the student model (still required)
                logps_rejected_student = get_logps_batch_NPO(batch=adv_npo_batch, model=student, device=TRAINING_ARGS.DEVICE , teacher=False)
            
                logprob_dist_student = get_logps_batch_KL_policy(
                adv_ikl_batch, student, device=TRAINING_ARGS.DEVICE
                )
            
            
            npo_loss = npo_loss_fn(logps_rejected_student, logps_rejected_harmful)

            
            immunization_kl_loss = immunization_kl_loss_fn(logprob_dist_student, logprob_dist_harmful)

            total_adversarial_loss = (1 - TRAINING_ARGS.GAMMA) * npo_loss + TRAINING_ARGS.GAMMA * immunization_kl_loss
            total_adversarial_loss.backward()

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




        print("\n--- 5. Identifying Canary Neurons Post-Adversarial Training ---")
        mean_activations_post = get_mean_activations(model=student, tokenizer=tokenizer, device=TRAINING_ARGS.DEVICE, prompts=adversarial_prompts, target_modules=canary_target_modules)

        all_changes = torch.cat([(mean_activations_pre[name] - mean_activations_post[name]).abs() for name in mean_activations_pre.keys()])
        canary_threshold = torch.quantile(all_changes, TRAINING_ARGS.CANARY_QUANTILE).item()
        canary_neuron_mask = {
            name: (mean_activations_pre[name] - mean_activations_post[name]).abs() > canary_threshold
            for name in mean_activations_pre.keys()
        }
        print(f"Identified canaries using threshold {canary_threshold:.4f}.")
        
        





        # --- Harmless Phase ---
        print(f"\n--- Running Harmless (Benign) Phase for Epoch {epoch+1} ---")
        
        target_canary_activations = {}
        
        
        sample_harmless_dataset = create_sampled_dataset(dataset="begn", sample_size=TRAINING_ARGS.SAMPLE_SIZE)
        harmless_prompts =  [item['prompt'] for item in sample_harmless_dataset]
        
        
        
        mean_activations_unlearned = get_mean_activations(model=student, tokenizer=tokenizer, device=TRAINING_ARGS.DEVICE, prompts=harmless_prompts, target_modules=canary_target_modules)
        
        for name, full_activations in mean_activations_unlearned.items():
            mask = canary_neuron_mask[name].to(TRAINING_ARGS.DEVICE)
            target_canary_activations[name] = full_activations.to(TRAINING_ARGS.DEVICE)[mask].detach()




        
        
        
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
            log_chosen_benign = log_chosen_benign.to(TRAINING_ARGS.DEVICE)
            log_rejected_benign = log_rejected_benign.to(TRAINING_ARGS.DEVICE)
            logprob_dist_benign = logprob_dist_benign.to(TRAINING_ARGS.DEVICE)

            # Forward pass for the student model (still required)
            log_chosen_student, log_rejected_student = get_logps_batch_DPO(begn_dpo_batch, student, TRAINING_ARGS.DEVICE)
            dpo_loss = dpo_loss_fn(log_chosen_student, log_rejected_student, log_chosen_benign, log_rejected_benign)
            
            logprob_dist_student = get_logps_batch_KL_policy(
                begn_akl_batch, student, device=TRAINING_ARGS.DEVICE
            )
            alignment_kl_loss = alignment_kl_loss_fn(logprob_dist_student, logprob_dist_benign)

            total_harmless_loss = (1 - TRAINING_ARGS.ALPHA) * dpo_loss + TRAINING_ARGS.ALPHA * alignment_kl_loss
            
            
            
            
            # --- B: Calculate Canary Stabilization Loss ---
            # Get current activations using hooks
            current_activations = {}
            hook_handles = []
            def hook_fn(name):
                def hook(module, inp, out):
                    activation = out[0] if isinstance(out, tuple) else out
                    current_activations[name] = activation.mean(dim=1)
                return hook
            for name, module in canary_target_modules.items():
                handle = module.register_forward_hook(hook_fn(name))
                hook_handles.append(handle)
            
            
            
            # A single forward pass on the prompt to trigger hooks

            _ = student(begn_akl_batch[0].to(TRAINING_ARGS.DEVICE))
            for handle in hook_handles: handle.remove() # Clean up hooks immediately
            
            
            
            
            # Calculate MSE
            stabilization_loss = torch.tensor(0.0).to(TRAINING_ARGS.DEVICE)
            num_layers_with_canaries = 0
            for name, current_act_batch in current_activations.items():
                mask = canary_neuron_mask[name].to(TRAINING_ARGS.DEVICE)
                if mask.sum() > 0:
                    current_canary_act = current_act_batch.mean(dim=0)[mask]
                    target_canary_act = target_canary_activations[name]
                    stabilization_loss += F.mse_loss(current_canary_act, target_canary_act)
                    num_layers_with_canaries += 1
            
            if num_layers_with_canaries > 0:
                stabilization_loss /= num_layers_with_canaries
            
            
            final_loss = total_harmless_loss + TRAINING_ARGS.STABILIZATION_LAMBDA * stabilization_loss
            final_loss.backward()

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

    progress_bar.close()
    print("\n" + "="*50)
    print(" " * 16 + "TRAINING COMPLETE" + " " * 17)
    print("="*50 + "\n")
    
    
    
    save_dir = f"models/dualDistilledWithLAT_CanaryStabilization-{student_model_id.replace("/","_")}-01"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n>>> Saving model to {save_dir} ...")
    student.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(">>> Model saved successfully!")
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