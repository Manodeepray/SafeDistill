import torch
import gc
import os
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, field
from typing import List, Dict, Any

# =========================================================
# 0. Configuration (MODIFIED FOR CROSS-ARCHITECTURE DEMO)
# =========================================================
@dataclass
class trainingArgs:
    EPOCHS: int = 3
    LR: float = 5e-3
    BATCH_SIZE : int = 2
    ALPHA : float = 0.5
    GAMMA : float = 0.05
    LAMBDA: int = None # CURRICULUM LEARNING CONSTANT (WHEN APPLIED)
    GRAD_ACC_STEPS: int = 5 # - FOR CANARY SELECTOION - BATCH SIZE GRAD ACCUMULATION
    DEVICE:str  =  "cuda" if torch.cuda.is_available() else "cpu"
    WARMUP_STEPS: int = 2
    EPSILON:float = 0.02
    STABILIZATION_LAMBDA:int = 0.8
    CANARY_QUANTILE:int = 0.99
    SAMPLE_SIZE:int = 1000 # for mean activation calculation
    
    # Using a smaller, public model for demonstration purposes
    # This makes the script runnable for anyone.
    # In a real scenario, these would be your actual different models.
    STUDENT_MODEL_ID: str = "Qwen/Qwen2.5-0.5B"
    HARMFUL_MODEL_ID: str = "openai-community/gpt2"
    BENIGN_MODEL_ID: str = "Qwen/Qwen2.5-0.5B"
    
    DATASET_SIZE:int  = 16
    # To speed up the example, we'll only process a few batches
    MAX_BATCHES_TO_PRECOMPUTE: int = 2
# =========================================================
# 1. Mock Implementations & NEW HELPER FUNCTION
# =========================================================

# --- NEW: Vocabulary Mapping Helper ---
def create_vocab_mapping(
    teacher_tokenizer: PreTrainedTokenizerBase,
    student_tokenizer: PreTrainedTokenizerBase,
    device: str
) -> torch.Tensor:
    """
    Creates a mapping tensor to project the teacher's vocabulary onto the student's.
    """
    print("\n--- Creating vocabulary mapping ---")
    teacher_vocab = teacher_tokenizer.get_vocab()
    student_vocab = student_tokenizer.get_vocab()
    
    teacher_vocab_size = len(teacher_vocab)
    student_vocab_size = len(student_vocab)
    
    print(f"Teacher vocab size: {teacher_vocab_size}")
    print(f"Student vocab size: {student_vocab_size}")

    # Use the teacher's UNK token as the default for unmatched tokens
    # Note: GPT-2 style models might use EOS as UNK. We handle this safely.
    teacher_unk_token_id = teacher_tokenizer.unk_token_id or teacher_tokenizer.eos_token_id

    # Create a tensor to hold the mapping.
    # For each token in the student's vocab, we find its ID in the teacher's vocab.
    mapping = torch.full((student_vocab_size,), fill_value=teacher_unk_token_id, dtype=torch.long)

    for student_token, student_id in tqdm(student_vocab.items(), desc="Mapping Vocabs", ncols=100):
        # Find the corresponding ID in the teacher's vocabulary
        teacher_id = teacher_vocab.get(student_token, teacher_unk_token_id)
        mapping[student_id] = teacher_id
        
    print("Vocabulary mapping created successfully.")
    return mapping.to(device)


# --- Mock Dataloaders.py (Unchanged) ---
class MockPreferenceDataset(Dataset):
    """A mock dataset that yields preference data."""
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def make_dataloaders(batch_size: int, tokenizer: PreTrainedTokenizerBase, sample_size: int):
    mock_data = [
        {
            "prompt": "Explain gravity to a five-year-old.",
            "chosen": "Imagine the Earth is a big bowling ball on a trampoline. It makes a dip...",
            "rejected": "Gravity is a fundamental interaction which manifests as a mutual attraction between all things with mass or energy."
        },
        {
            "prompt": "What's the capital of France?",
            "chosen": "The capital of France is Paris.",
            "rejected": "The capital of France is London."
        },
    ] * (sample_size // 2)
    dataset = MockPreferenceDataset(mock_data)
    def collate_fn(batch):
        return {
            "prompt": [item["prompt"] for item in batch],
            "chosen": [item["chosen"] for item in batch],
            "rejected": [item["rejected"] for item in batch],
        }
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    return loader, loader, loader, loader

# --- Mock Losses.py (MODIFIED) ---
def get_logps_batch_KL_ref(
    batch: Dict[str, List[str]],
    model: AutoModelForCausalLM,
    teacher_tokenizer: PreTrainedTokenizerBase,
    vocab_mapping: torch.Tensor, # ADDED
    device: str
) -> torch.Tensor:
    """
    Calculates the log probability distribution and PROJECTS it to the student's vocab space.
    """
    print("\n--- Running get_logps_batch_KL_ref (with projection) ---")
    prompts = batch["prompt"]
    inputs = teacher_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    print(f"[SHAPE_LOG] Input IDs shape: {inputs['input_ids'].shape}")

    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Calculate log probabilities over the teacher's entire vocabulary
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # --- THIS IS THE KEY MODIFICATION ---
    print(f"[SHAPE_LOG] PRE-MAPPING log_probs shape (Teacher Vocab): {log_probs.shape}")
    
    # Reshape for efficient indexing
    batch_size, seq_length, teacher_vocab_size = log_probs.shape
    flat_log_probs = log_probs.view(-1, teacher_vocab_size) # Shape: (B*S, V_Teacher)

    # Use the mapping to gather the relevant log-probabilities from the teacher.
    # This selects the columns corresponding to the student's vocabulary tokens.
    projected_log_probs = torch.index_select(flat_log_probs, 1, vocab_mapping)
    
    # Reshape back to the original 3D format, now with the student's vocab size
    student_vocab_size = len(vocab_mapping)
    projected_log_probs = projected_log_probs.view(batch_size, seq_length, student_vocab_size)

    print(f"[SHAPE_LOG] POST-MAPPING log_probs shape (Student Vocab): {projected_log_probs.shape}")
    
    return projected_log_probs

# DPO and NPO functions remain unchanged as they don't depend on the full vocab distribution
def get_logps_batch_DPO(batch: Dict[str, List[str]], model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerBase, device: str):
    print("\n--- Running get_logps_batch_DPO ---")
    def _get_sequence_logps(prompts: List[str], responses: List[str]) -> torch.Tensor:
        full_texts = [p + r + tokenizer.eos_token for p, r in zip(prompts, responses)]
        prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        full_tokens = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        prompt_lengths = torch.tensor([len(t) for t in prompt_tokens['input_ids']], device=device)
        print(f"[SHAPE_LOG] Full sequence input shape: {full_tokens['input_ids'].shape}")
        with torch.no_grad():
            logits = model(**full_tokens).logits
        print(f"[SHAPE_LOG] Logits shape: {logits.shape}")
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        target_tokens = full_tokens['input_ids'][:, 1:].contiguous()
        log_probs = log_probs[:, :-1, :].contiguous()
        gathered_log_probs = torch.gather(log_probs, 2, target_tokens.unsqueeze(-1)).squeeze(-1)
        seq_indices = torch.arange(target_tokens.shape[1], device=device).unsqueeze(0)
        response_mask = (seq_indices >= (prompt_lengths - 1).unsqueeze(1)).float()
        padding_mask = (target_tokens != tokenizer.pad_token_id).float()
        final_mask = response_mask * padding_mask
        masked_log_probs = gathered_log_probs * final_mask
        sequence_logps = masked_log_probs.sum(dim=-1)
        print(f"[SHAPE_LOG] Final sequence logps shape (per response): {sequence_logps.shape}")
        return sequence_logps
    print("\n  -- DPO: Processing Chosen Responses --")
    logps_chosen = _get_sequence_logps(batch['prompt'], batch['chosen'])
    print("\n  -- DPO: Processing Rejected Responses --")
    logps_rejected = _get_sequence_logps(batch['prompt'], batch['rejected'])
    return logps_chosen, logps_rejected

def get_logps_batch_NPO(batch: Dict[str, Any], model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerBase, device: str) -> torch.Tensor:
    print("\n--- Running get_logps_batch_NPO ---")
    _, logps_rejected = get_logps_batch_DPO(batch, model, tokenizer, device)
    return logps_rejected

# =========================================================
# 2. PRECOMPUTE & SAVE (MODIFIED)
# =========================================================
def precompute_and_save_logprobs(
    adv_NPO_loader, adv_IKL_loader, benign_DPO_loader, benign_AKL_loader,
    harmful_model_id, benign_model_id,
    harmful_tokenizer, benign_tokenizer,
    harmful_to_student_map, benign_to_student_map, # ADDED MAPPINGS
    device, max_batches, save_dir="precomputed_logprobs"
):
    os.makedirs(save_dir, exist_ok=True)

    # ----------------- Harmful Teacher -----------------
    print("\n\n" + "="*50)
    print("--- Loading Harmful Teacher ---")
    harmfulTeacher = AutoModelForCausalLM.from_pretrained(harmful_model_id, cache_dir="cache_dir").to(device)
    harmfulTeacher.eval()

    precomputed_adv_npo_logps = []
    precomputed_adv_ikl_logprobs = []

    with torch.no_grad():
        print("--- Caching Adversarial Teacher outputs ---")
        for i, adv_npo_batch in enumerate(tqdm(adv_NPO_loader, desc="Caching Harmful NPO", ncols=100)):
            if i >= max_batches: break
            logps_rejected_harmful = get_logps_batch_NPO(adv_npo_batch, harmfulTeacher, harmful_tokenizer, device)
            precomputed_adv_npo_logps.append(logps_rejected_harmful.cpu())

        for i, adv_ikl_batch in enumerate(tqdm(adv_IKL_loader, desc="Caching Harmful IKL", ncols=100)):
            if i >= max_batches: break
            # Pass the mapping to the KL logprobs function
            logprob_dist_harmful = get_logps_batch_KL_ref(
                adv_ikl_batch, harmfulTeacher, harmful_tokenizer,
                vocab_mapping=harmful_to_student_map, # MODIFIED
                device=device
            )
            precomputed_adv_ikl_logprobs.append(logprob_dist_harmful.cpu())

    torch.save(precomputed_adv_npo_logps, os.path.join(save_dir, "adv_npo_logps.pt"))
    torch.save(precomputed_adv_ikl_logprobs, os.path.join(save_dir, "adv_ikl_logprobs.pt"))

    del harmfulTeacher
    gc.collect()
    torch.cuda.empty_cache()
    print("harmfulTeacher removed from GPU memory.")

    # ----------------- Benign Teacher -----------------
    print("\n\n" + "="*50)
    print("\n--- Loading Benign Teacher ---")
    benignTeacher = AutoModelForCausalLM.from_pretrained(benign_model_id, cache_dir="cache_dir").to(device)
    benignTeacher.eval()

    precomputed_benign_dpo_chosen_logps = []
    precomputed_benign_dpo_rejected_logps = []
    precomputed_benign_akl_logprobs = []

    with torch.no_grad():
        print("--- Caching Benign Teacher outputs ---")
        for i, begn_dpo_batch in enumerate(tqdm(benign_DPO_loader, desc="Caching Benign DPO", ncols=100)):
            if i >= max_batches: break
            log_chosen_benign, log_rejected_benign = get_logps_batch_DPO(begn_dpo_batch, benignTeacher, benign_tokenizer, device)
            precomputed_benign_dpo_chosen_logps.append(log_chosen_benign.cpu())
            precomputed_benign_dpo_rejected_logps.append(log_rejected_benign.cpu())

        for i, begn_akl_batch in enumerate(tqdm(benign_AKL_loader, desc="Caching Benign AKL", ncols=100)):
            if i >= max_batches: break
            # Pass the mapping to the KL logprobs function
            logprob_dist_benign = get_logps_batch_KL_ref(
                begn_akl_batch, benignTeacher, benign_tokenizer,
                vocab_mapping=benign_to_student_map, # MODIFIED
                device=device
            )
            precomputed_benign_akl_logprobs.append(logprob_dist_benign.cpu())

    torch.save(precomputed_benign_dpo_chosen_logps, os.path.join(save_dir, "benign_dpo_chosen.pt"))
    torch.save(precomputed_benign_dpo_rejected_logps, os.path.join(save_dir, "benign_dpo_rejected.pt"))
    torch.save(precomputed_benign_akl_logprobs, os.path.join(save_dir, "benign_akl_logprobs.pt"))

    del benignTeacher
    gc.collect()
    torch.cuda.empty_cache()
    print("benignTeacher removed from GPU memory.")

    print(f"\n✅ All precomputed logprobs saved to {save_dir}")

# =========================================================
# 3. LOAD PRECOMPUTED (Unchanged)
# =========================================================
def load_precomputed_logprobs(save_dir="precomputed_logprobs"):
    adv_npo_logps = torch.load(os.path.join(save_dir, "adv_npo_logps.pt"))
    adv_ikl_logprobs = torch.load(os.path.join(save_dir, "adv_ikl_logprobs.pt"))
    benign_dpo_chosen = torch.load(os.path.join(save_dir, "benign_dpo_chosen.pt"))
    benign_dpo_rejected = torch.load(os.path.join(save_dir, "benign_dpo_rejected.pt"))
    benign_akl_logprobs = torch.load(os.path.join(save_dir, "benign_akl_logprobs.pt"))
    print(f"\n✅ Loaded precomputed logprobs from {save_dir}")
    return (
        adv_npo_logps, adv_ikl_logprobs, benign_dpo_chosen,
        benign_dpo_rejected, benign_akl_logprobs,
    )

# =========================================================
# 4. Main Execution (MODIFIED)
# =========================================================
if __name__ == "__main__":
    TRAINING_ARGS = trainingArgs()
    
    print("--- Initializing Tokenizers for Teachers and Student ---")
    harmful_tokenizer = AutoTokenizer.from_pretrained(TRAINING_ARGS.HARMFUL_MODEL_ID)
    benign_tokenizer = AutoTokenizer.from_pretrained(TRAINING_ARGS.BENIGN_MODEL_ID)
    student_tokenizer = AutoTokenizer.from_pretrained(TRAINING_ARGS.STUDENT_MODEL_ID)
    
    # Set pad tokens if they don't exist
    for tokenizer in [harmful_tokenizer, benign_tokenizer, student_tokenizer]:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # --- Create Vocabulary Mappings ---
    # Since harmful and benign teachers are the same, their mapping will be identical.
    # In a real case where they are different, you would create two distinct mappings.
    harmful_to_student_map = create_vocab_mapping(harmful_tokenizer, student_tokenizer, TRAINING_ARGS.DEVICE)
    benign_to_student_map = create_vocab_mapping(benign_tokenizer, student_tokenizer, TRAINING_ARGS.DEVICE)

    print("\n--- Creating Dataloaders ---")
    adv_NPO_loader, adv_IKL_loader, _, _ = make_dataloaders(
        batch_size=TRAINING_ARGS.BATCH_SIZE,
        tokenizer=harmful_tokenizer,
        sample_size=TRAINING_ARGS.DATASET_SIZE
    )
    
    _, _, benign_DPO_loader, benign_AKL_loader = make_dataloaders(
        batch_size=TRAINING_ARGS.BATCH_SIZE,
        tokenizer=benign_tokenizer,
        sample_size=TRAINING_ARGS.DATASET_SIZE
    )

    # Precompute & save once
    precompute_and_save_logprobs(
        adv_NPO_loader, adv_IKL_loader, benign_DPO_loader, benign_AKL_loader,
        harmful_model_id=TRAINING_ARGS.HARMFUL_MODEL_ID,
        benign_model_id=TRAINING_ARGS.BENIGN_MODEL_ID,
        harmful_tokenizer=harmful_tokenizer,
        benign_tokenizer=benign_tokenizer,
        harmful_to_student_map=harmful_to_student_map, # Pass mapping
        benign_to_student_map=benign_to_student_map,   # Pass mapping
        device=TRAINING_ARGS.DEVICE,
        max_batches=TRAINING_ARGS.MAX_BATCHES_TO_PRECOMPUTE,
        save_dir="precomputed_logprobs"
    )

    # Later, just load
    (
        precomputed_adv_npo_logps,
        precomputed_adv_ikl_logprobs,
        precomputed_benign_dpo_chosen_logps,
        precomputed_benign_dpo_rejected_logps,
        precomputed_benign_akl_logprobs,
    ) = load_precomputed_logprobs("precomputed_logprobs")

    # --- Verify the shapes of the loaded data ---
    print("\n\n" + "="*50)
    print("--- Verifying Shapes of Loaded Tensors ---")
    print(f"Teacher (gpt2-0.1B) Vocab Size: {harmful_tokenizer.vocab_size}")
    print(f"Student (qwen2.5-0.5B) Vocab Size: {student_tokenizer.vocab_size}")
    print("-" * 50)
    
    print(f"Shape of first batch of adv_npo_logps (unaffected): {precomputed_adv_npo_logps[0].shape}")
    print(f"Shape of first batch of benign_dpo_chosen_logps (unaffected): {precomputed_benign_dpo_chosen_logps[0].shape}")
    print("-" * 50)
    
    print("\n--- KL Divergence Logprobs (Affected by Mapping) ---")
    if precomputed_adv_ikl_logprobs:
        print(f"Shape of first batch of adv_ikl_logprobs: {precomputed_adv_ikl_logprobs[0].shape}")
        print("Note: The last dimension now matches the *Student's* vocabulary size.")

    print("\n" + "="*50)