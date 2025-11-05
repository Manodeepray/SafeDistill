import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict , List
from transformers import AutoModelForCausalLM, AutoTokenizer

class NPOLoss(nn.Module):
    def __init__(self, beta: float = 0.1):
        """
        NPO Loss (Negative Preference Optimization).
        Args:
            beta (float): Temperature parameter for scaling.
        """
        super().__init__()
        self.beta = beta
        # beta = 2/B
    def forward(self, logprobs_rejected, ref_logprobs_rejected):
        """
        Args:
            logprobs_rejected: Tensor [batch] – log p_θ(y- | x)
            ref_logprobs_rejected: Tensor [batch] – log p_ref(y- | x)

        Returns:
            loss: scalar tensor
        """
        # Difference between ref and policy log-probs
        diff = ref_logprobs_rejected - logprobs_rejected  

        # Logistic loss (sigmoid cross-entropy)
        logits = self.beta * diff 
        loss = -F.logsigmoid(logits).mean()

        return loss





## DPO

class DPOLoss(nn.Module):
    def __init__(self, beta: float = 0.1):
        """
        DPO Loss as described in 'Direct Preference Optimization' (Rafailov et al., 2023).
        Args:
            beta (float): Temperature parameter for scaling.
        """
        super(DPOLoss, self).__init__()
        self.beta = beta
        #beta = 1/B

    def forward(self, logprobs_chosen, logprobs_rejected,
                ref_logprobs_chosen, ref_logprobs_rejected):
        """
        Args:
            logprobs_chosen: Tensor [batch] – log p_θ(y+ | x)
            logprobs_rejected: Tensor [batch] – log p_θ(y- | x)
            ref_logprobs_chosen: Tensor [batch] – log p_ref(y+ | x)
            ref_logprobs_rejected: Tensor [batch] – log p_ref(y- | x)

        Returns:
            loss: scalar tensor
        """

        # difference between chosen and rejected under current policy
        policy_diff = logprobs_chosen - logprobs_rejected

        # difference between chosen and rejected under reference model
        ref_diff = ref_logprobs_chosen - ref_logprobs_rejected

        # main DPO objective: logistic loss
        logits = (policy_diff - ref_diff) * self.beta
        loss = -F.logsigmoid(logits).mean()

        return loss





## ALIGNMENT KL




class AlignmentKLLoss(nn.Module):
    def __init__(self, reduction: str = "batchmean"):
        """
        KL Divergence Loss between two models from their log probabilities.
        Args:
            reduction (str): 'batchmean', 'mean', or 'sum' (default: 'batchmean').
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, logprobs_theta: torch.Tensor, logprobs_ref: torch.Tensor):
        """
        Args:
            logprobs_theta: Tensor [batch, vocab] – log probs from model θ
            logprobs_ref:   Tensor [batch, vocab] – log probs from benign reference model

        Returns:
            KL divergence (scalar)
        """
        # Convert logprobs to probabilities
        p_theta = logprobs_theta.exp()
        # KL(p_theta || p_ref) = Σ p_theta * (log p_theta - log p_ref)
        kl = F.kl_div(
            logprobs_ref,  # target log-probs (must be log)
            p_theta,       # input probs
            log_target=True,
            reduction=self.reduction
        )
        return kl 



## IMMUNIZATION KL

class ImmunizationKLLoss(nn.Module):
    def __init__(self, reduction: str = "batchmean"):
        """
        Immunization KL Loss for unlearning.

        This loss computes a KL-style divergence between the student model θ 
        and a harmful reference model. The direction of KL matters:

        - Forward KL: D_KL(p_harmful || p_theta)
          * Expectation under the harmful distribution.
          * If minimized → student covers all harmful modes (bad for unlearning).
          * If maximized (negative forward KL) → student is pushed away 
            from *all regions where harmful has support*. 
            Stronger push than reverse KL, but may also forget useful knowledge
            if harmful overlaps with general/safe data.

        - Reverse KL: D_KL(p_theta || p_harmful)
          * Expectation under the student distribution.
          * If minimized → student imitates harmful where it already attends.
          * If maximized → student avoids harmful regions it already covers.
            Gentler than forward KL but can lead to mode collapse 
            (ignores harmful regions outside current support).

        In this implementation we use p_harmful as the weighting distribution,
        which corresponds to the *negative forward KL*:
        
            L = -D_KL(p_harmful || p_theta)
              = Σ_x p_harmful(x) [ log p_theta(x) - log p_harmful(x) ]

        This encourages θ to move away from the harmful model everywhere 
        harmful has probability mass.

        Args:
            reduction (str): Specifies how to reduce the per-sample KL values:
                - 'batchmean' (default): sum over samples / batch size
                - 'sum': sum over all samples
                - 'mean': mean over all samples
                - None: no reduction, return per-sample values
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, logprobs_theta: torch.Tensor, logprobs_harmful: torch.Tensor):
        """
        Args:
            logprobs_theta:   [batch, vocab] – log probs from model θ (student)
            logprobs_harmful: [batch, vocab] – log probs from harmful reference model

        Returns:
            Scalar (or vector if no reduction) representing negative forward KL.
        """
        # Convert harmful logprobs to probabilities
        p_harmful = logprobs_harmful.exp()

        # -KL(harmful || θ) = Σ_x p_harmful(x) * (log p_theta(x) - log p_harmful(x))
        kl = torch.sum(
            p_harmful * (logprobs_theta - logprobs_harmful), dim=-1
        )
        
        if self.reduction == "batchmean":
            return kl.sum() / kl.size(0)  # divide by batch size
        elif self.reduction == "sum":
            return kl.sum()
        elif self.reduction == "mean":
            return kl.mean()
        else:
            return kl

















## KL LOGPROBS

def get_logps_batch_KL( batch,   model:AutoModelForCausalLM ,ref_model:AutoModelForCausalLM ,  device: str = 'cuda' , ):
    "logprobs over the sequence ( each token prob distribution ) .. i.e for tokenized prompt's response , via both reference model and base model"
    
    input_ids , attention_mask = batch
    input_ids , attention_mask = input_ids.to(device) , attention_mask.to(device)
    # full_texts = []
    # for row in batch:
    #     prompts.append(row["prompt"])
    
    
    # encodings = tokenizer(prompts ,return_tensors="pt",padding = True , truncation = True ,  max_length = max_length)
    # input_ids = encodings["input_ids"]
    
    with torch.no_grad():

        policy_model_outputs = model(input_ids , attention_mask)
        
        ref_model_outputs = ref_model(input_ids , attention_mask)
        
    policy_logits = policy_model_outputs.logits
    ref_logits = ref_model_outputs.logits
    
    
    policy_logprobs = F.log_softmax(policy_logits , dim = -1)
        
    ref_logprobs = F.log_softmax(ref_logits , dim = -1)
    
    
    
    # shift
        
    shifted_input_ids = input_ids[:, 1:]
    policy_selected = policy_logprobs[:, :-1, :].gather(-1, shifted_input_ids.unsqueeze(-1)).squeeze(-1)
    ref_selected = ref_logprobs[:, :-1, :].gather(-1, shifted_input_ids.unsqueeze(-1)).squeeze(-1)

    return policy_selected, ref_selected
     
    
    
def get_logps_batch_KL_policy( batch,   model:AutoModelForCausalLM ,  device: str = 'cuda' , ):
    "logprobs over the sequence ( each token prob distribution ) .. i.e for tokenized prompt's response , via both reference model and base model"
    
    input_ids , attention_mask = batch
    input_ids , attention_mask = input_ids.to(device) , attention_mask.to(device)
    # full_texts = []
    # for row in batch:
    #     prompts.append(row["prompt"])
    
    
    # encodings = tokenizer(prompts ,return_tensors="pt",padding = True , truncation = True ,  max_length = max_length)
    # input_ids = encodings["input_ids"]
    
    policy_model_outputs = model(input_ids , attention_mask)
        
        
    policy_logits = policy_model_outputs.logits
    
    
    policy_logprobs = F.log_softmax(policy_logits , dim = -1)
        
    
    # print(f"Shape of 'input_ids':      {input_ids.shape}")
    # print(f"Shape of 'attention_mask': {attention_mask.shape}")
    # print(f"Shape of 'policy_logits':    {policy_logits.shape}")
    # print(f"Shape of 'policy_logprobs':  {policy_logprobs.shape}")
    
    # shift
        
    shifted_input_ids = input_ids[:, 1:]
    policy_selected = policy_logprobs[:, :-1, :].gather(-1, shifted_input_ids.unsqueeze(-1)).squeeze(-1)

    # return policy_logprobs
    return policy_selected



from tqdm import tqdm


def create_vocab_mapping(
    teacher_tokenizer: AutoTokenizer,
    student_tokenizer: AutoTokenizer,
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
    
def get_logps_batch_KL_ref( batch, ref_model:AutoModelForCausalLM ,  device: str = 'cuda' , ):
    "logprobs over the sequence ( each token prob distribution ) .. i.e for tokenized prompt's response , via both reference model and base model"
    
    input_ids , attention_mask = batch
    input_ids , attention_mask = input_ids.to(device) , attention_mask.to(device)
    # full_texts = []
    # for row in batch:
    #     prompts.append(row["prompt"])
    
    
    # encodings = tokenizer(prompts ,return_tensors="pt",padding = True , truncation = True ,  max_length = max_length)
    # input_ids = encodings["input_ids"]
    
    with torch.no_grad():

        
        ref_model_outputs = ref_model(input_ids , attention_mask)
    ref_logits = ref_model_outputs.logits
    
    
        
    ref_logprobs = F.log_softmax(ref_logits , dim = -1)
    
    
    
    # shift
        
    shifted_input_ids = input_ids[:, 1:]
    ref_selected = ref_logprobs[:, :-1, :].gather(-1, shifted_input_ids.unsqueeze(-1)).squeeze(-1)

    return ref_selected
     


def get_logps_batch_KL_ref_modified(
    batch: Dict[str, List[str]],
    ref_model: AutoModelForCausalLM,
    vocab_mapping: torch.Tensor, # ADDED
    device: str
) -> torch.Tensor:
    """
    Calculates the log probability distribution and PROJECTS it to the student's vocab space.
    """
    print("\n--- Running get_logps_batch_KL_ref (with projection) ---")
    input_ids , attention_mask = batch
    input_ids , attention_mask = input_ids.to(device) , attention_mask.to(device)
    # print(f"[SHAPE_LOG] Input IDs shape: {input_ids.shape}")

    with torch.no_grad():
        ref_model_outputs = ref_model(input_ids , attention_mask)
    ref_logits = ref_model_outputs.logits
    
    
        
    ref_logprobs = F.log_softmax(ref_logits , dim = -1)
    
    # --- THIS IS THE KEY MODIFICATION ---
    # print(f"[SHAPE_LOG] PRE-MAPPING log_probs shape (Teacher Vocab): {ref_logprobs.shape}")
    
    # Reshape for efficient indexing
    batch_size, seq_length, teacher_vocab_size = ref_logprobs.shape
    flat_log_probs = ref_logprobs.view(-1, teacher_vocab_size) # Shape: (B*S, V_Teacher)

    # Use the mapping to gather the relevant log-probabilities from the teacher.
    # This selects the columns corresponding to the student's vocabulary tokens.
    projected_log_probs = torch.index_select(flat_log_probs, 1, vocab_mapping)
    
    # Reshape back to the original 3D format, now with the student's vocab size
    student_vocab_size = len(vocab_mapping)
    projected_log_probs = projected_log_probs.view(batch_size, seq_length, student_vocab_size)

    # print(f"[SHAPE_LOG] POST-MAPPING log_probs shape (Student Vocab): {projected_log_probs.shape}")
    
    return projected_log_probs







## NPO LOGPROBS


def get_logps_batch_NPO( batch,   model:AutoModelForCausalLM , device: str = 'cuda' ,  average_log_prob:bool =False , teacher = False):
    # create full text
    # tokenize full text
    # create input ids from tokenizer
    # get the attention mask to get prompt length then use the prompt lengths to make the labels
    # get logits from model
    # get logps using logsoftmax
    # gather per token logits 

    
    # chosen_texts = []
    full_text_input_ids, full_text_attention_mask, _, prompt_attention_mask = batch

    # Move required tensors to the specified device
    full_text_input_ids = full_text_input_ids.to(device)
    full_text_attention_mask = full_text_attention_mask.to(device)
    
    # 2. Get the length of the prompt to create the labels mask
    # The sum of the attention mask gives the number of non-padding tokens.
    prompt_lengths = prompt_attention_mask.sum(dim=1)
    
    if teacher:
        with torch.no_grad():
            rejected_output = model(input_ids=full_text_input_ids, attention_mask=full_text_attention_mask)
    else:
        rejected_output = model(input_ids=full_text_input_ids, attention_mask=full_text_attention_mask)
    rejected_logits = rejected_output.logits

    # 4. Create labels for loss calculation.
    # We clone the input_ids and will mask the prompt part.
    rejected_labels = full_text_input_ids.clone()
    
    # 5. Mask the prompt tokens in the labels by setting them to -100.
    # The loss function will ignore these tokens.

    # ignore logits in labels
    for i in range(len(prompt_lengths)):
        # chosen_labels[:,:prompt_lengths[i]] = -100
        rejected_labels[:,:prompt_lengths[i]] = -100 



    #shift
    
    # chosen_labels = chosen_labels[:, 1:].clone()
    # chosen_logits = chosen_logits[:, :-1, :]
    
    rejected_labels = rejected_labels[:, 1:].clone()
    rejected_logits = rejected_logits[:, :-1, :]
    
    # Create a mask of valid (non--100) labels
    # chosen_loss_mask = (chosen_labels != -100)
    rejected_loss_mask = (rejected_labels != -100)
    
    
    # chosen_labels[chosen_labels == -100] = 0
    rejected_labels[rejected_labels == -100] = 0
    
    # per_token_logps_chosen = torch.gather(chosen_logits.log_softmax(-1), dim=2, index=chosen_labels.unsqueeze(2)).squeeze(2)
    per_token_logps_rejected = torch.gather(rejected_logits.log_softmax(-1), dim=2, index=rejected_labels.unsqueeze(2)).squeeze(2)
    
    
    if average_log_prob:
        # return (per_token_logps_chosen * chosen_loss_mask).sum(-1) / chosen_loss_mask.sum(-1) ,(per_token_logps_rejected * rejected_loss_mask).sum(-1) / rejected_loss_mask.sum(-1) 
    
        return (per_token_logps_rejected * rejected_loss_mask).sum(-1) / rejected_loss_mask.sum(-1) 
    
    else:
        # return (per_token_logps_chosen * chosen_loss_mask).sum(-1)  ,(per_token_logps_rejected * rejected_loss_mask).sum(-1) 
    
    
        return (per_token_logps_rejected * rejected_loss_mask).sum(-1) 
    





## DPO LOGPROBS

def get_logps_batch_DPO( batch:list,   model:AutoModelForCausalLM , device: str = 'cuda' , average_log_prob:bool =False , teacher = False):
    # create full text
    # tokenize full text
    # create input ids from tokenizer
    # get the attention mask to get prompt length then use the prompt lengths to make the labels
    # get logits from model
    # get logps using logsoftmax
    # gather per token logits 

    
    
    (
        chosen_input_ids, chosen_attention_mask,
        rejected_input_ids, rejected_attention_mask,
        _, prompt_attention_mask  # We only need the prompt mask for its length
    ) = batch

    # Move all necessary tensors to the specified device
    chosen_input_ids = chosen_input_ids.to(device)
    chosen_attention_mask = chosen_attention_mask.to(device)
    rejected_input_ids = rejected_input_ids.to(device)
    rejected_attention_mask = rejected_attention_mask.to(device)

    # 2. Get the length of the prompt to create the labels mask
    prompt_lengths = prompt_attention_mask.sum(dim=1)

    if teacher:
    # 3. Get logits from the model for both chosen and rejected sequences
        with torch.no_grad():
            chosen_outputs = model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask)
            rejected_outputs = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)
    else:
        chosen_outputs = model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask)
        rejected_outputs = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)
    
    chosen_logits = chosen_outputs.logits
    rejected_logits = rejected_outputs.logits

    # 4. Create labels and mask the prompt tokens
    chosen_labels = chosen_input_ids.clone()
    rejected_labels = rejected_input_ids.clone()

    for i in range(len(prompt_lengths)):
        chosen_labels[i, :prompt_lengths[i]] = -100
        rejected_labels[i, :prompt_lengths[i]] = -100


    #shift
    
    chosen_labels = chosen_labels[:, 1:].clone()
    chosen_logits = chosen_logits[:, :-1, :]
    
    rejected_labels = rejected_labels[:, 1:].clone()
    rejected_logits = rejected_logits[:, :-1, :]
    
    # Create a mask of valid (non--100) labels
    chosen_loss_mask = (chosen_labels != -100)
    rejected_loss_mask = (rejected_labels != -100)
    
    
    chosen_labels[chosen_labels == -100] = 0
    rejected_labels[rejected_labels == -100] = 0
    
    per_token_logps_chosen = torch.gather(chosen_logits.log_softmax(-1), dim=2, index=chosen_labels.unsqueeze(2)).squeeze(2)
    per_token_logps_rejected = torch.gather(rejected_logits.log_softmax(-1), dim=2, index=rejected_labels.unsqueeze(2)).squeeze(2)
    
    
    if average_log_prob:
        return (per_token_logps_chosen * chosen_loss_mask).sum(-1) / chosen_loss_mask.sum(-1) ,(per_token_logps_rejected * rejected_loss_mask).sum(-1) / rejected_loss_mask.sum(-1) 
    
    else:
        return (per_token_logps_chosen * chosen_loss_mask).sum(-1)  ,(per_token_logps_rejected * rejected_loss_mask).sum(-1) 
    
    