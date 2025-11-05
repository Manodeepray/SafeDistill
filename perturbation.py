


# ===================================================
# ADDITIONAL TRAINING IMPLEMENTATION


## LAT



from contextlib import contextmanager
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List

# ===================================================================
# 1. Hook Context Manager with Logging
# ===================================================================
@contextmanager
def apply_hooks(model, hooks):
    handles = []
    # print("[apply_hooks] Registering hooks...")
    for module, hook_fn in hooks.items():
        try:
            h = module.register_forward_hook(hook_fn)
            handles.append(h)
            # print(f"[apply_hooks] Hook registered on: {module.__class__.__name__}")
        except Exception as e:
            print(f"[apply_hooks] FAILED to register on {module}: {e}")
    try:
        yield
    finally:
        # print("[apply_hooks] Removing hooks...")
        for h in handles:
            h.remove()
        handles.clear()
        # print("[apply_hooks] Hooks removed.")


# ===================================================================
# 2. LAT Functions (Refactored for Safety + Logging)
# ===================================================================

def calculate_perturbations(
    batch: List,
    TRAINING_ARGS,
    student_model: AutoModelForCausalLM,
    target_modules: Dict,
    accelerator,
    device: str = "cpu"
) -> Dict:
    """
    Pass 1 of LAT: Calculates and returns the perturbation vectors.
    Logs added for debugging.
    """
    saved_activations = {}
    
    def save_and_retain_grad_hook(name):
        def hook(module, inp, out):
            activation = out[0] if isinstance(out, tuple) else out
            saved_activations[name] = activation
            # print(f"[calculate_perturbations] Saved activation for: {name}, shape: {activation.shape}")
            if activation.requires_grad:
                activation.retain_grad()
                # print(f"[calculate_perturbations] Retained grad for: {name}")
        return hook

    spy_hooks = {module: save_and_retain_grad_hook(name) for name, module in target_modules.items()}
    
    with apply_hooks(student_model, spy_hooks):
        student_model.eval()
        # print("[calculate_perturbations] Running forward pass...")
        
        full_text_input_ids, full_text_attention_mask, _, prompt_attention_mask = batch

    # Move required tensors to the specified device
        full_text_input_ids = full_text_input_ids.to(device)
        full_text_attention_mask = full_text_attention_mask.to(device)
        
        # 2. Get the length of the prompt to create the labels mask
        # The sum of the attention mask gives the number of non-padding tokens.
        prompt_lengths = prompt_attention_mask.sum(dim=1)
        
        labels = full_text_input_ids.clone()
        
        for i in range(len(prompt_lengths)):
            labels[i, :prompt_lengths[i]] = -100
        
        # print("[calculate_perturbations] Calculating adversarial imitation loss...")
        adversarial_imitation_loss = student_model(input_ids=full_text_input_ids, labels=labels).loss
        print(f"[calculate_perturbations] Loss: {adversarial_imitation_loss.item():.6f}")
        # adversarial_imitation_loss.backward()
        accelerator.backward(adversarial_imitation_loss)
        # print("[calculate_perturbations] Backward pass complete.")
            
    perturbations = {}
    
    for name, activation in saved_activations.items():
        if activation.grad is None:
            # print(f"[calculate_perturbations] No grad for {name}; skipping.")
            continue

        g_h = activation.grad.data  # [B, T, H] typically
        # Collapse batch and time so the direction is seq-length agnostic
        # shape -> [1, 1, H]
        dir_vec = g_h.mean(dim=(0, 1), keepdim=True)  # [1, 1, H]

        # Normalize (L2 over hidden)
        denom = torch.linalg.norm(dir_vec.flatten(start_dim=2), dim=2, keepdim=True) + 1e-12  # [1,1,1]
        unit_dir = dir_vec / denom  # [1,1,H]

        # Final small perturbation “template”
        delta = TRAINING_ARGS.EPSILON * unit_dir  # [1,1,H]
        perturbations[name] = delta
        # print(f"[calculate_perturbations] Perturbation template for {name}: {delta.shape}")
    
    
    # for name, activation in saved_activations.items():
    #     if activation.grad is not None:
    #         g_h = activation.grad.data
    #         print(f"[calculate_perturbations] Gradient shape for {name}: {g_h.shape}")
    #         l2_norm = torch.linalg.norm(g_h.flatten(start_dim=1), dim=1, keepdim=True)
    #         l2_norm = l2_norm.unsqueeze(-1).unsqueeze(-1) # Reshape for broadcasting
    #         delta = TRAINING_ARGS.EPSILON * g_h / (l2_norm + 1e-12)
    #         perturbations[name] = delta
    #         print(f"[calculate_perturbations] Perturbation created for {name}, shape: {delta.shape}")
            
    saved_activations.clear()
    student_model.zero_grad()
    # print("[calculate_perturbations] Perturbations calculated and gradients cleared.")
    return perturbations

# ===================================================================
# 3. Apply Perturbations Context Manager with Logging
# ===================================================================

@contextmanager
def apply_perturbations(model: AutoModelForCausalLM, target_modules: Dict, perturbations: Dict):
    def apply_perturbation_hook(name):
        def hook(module, inp, out):
            if name not in perturbations:
                return

            delta_tpl = perturbations[name]  # expected [1,1,H] now
            if isinstance(out, tuple):
                base = out[0]
            else:
                base = out

            try:
                # Expect base shape like [B,T,H] or [T,B,H]; handle both
                if base.dim() == 3:
                    H = base.size(-1)
                    if delta_tpl.size(-1) != H:
                        # print(f"[apply_perturbations] Hidden mismatch for {name}: delta H={delta_tpl.size(-1)} vs out H={H}. Skipping.")
                        return out

                    # Expand to broadcast across current B,T
                    delta = delta_tpl.to(device=base.device, dtype=base.dtype).expand(
                        *([1] * (base.dim() - 1)), H
                    )
                    delta = delta.expand(*base.shape[:-1], H)

                    if isinstance(out, tuple):
                        return (base + delta,) + out[1:]
                    else:
                        return base + delta

                else:
                    # If shape unexpected, skip safely
                    print(f"[apply_perturbations] Unexpected dim {base.dim()} for {name}; skipping.")
                    return out

            except Exception as e:
                print(f"[apply_perturbations] Failed to apply to {name}: {type(e).__name__}: {e}. Skipping.")
                return out

        return hook

    saboteur_hooks = {module: apply_perturbation_hook(name) for name, module in target_modules.items()}
    with apply_hooks(model, saboteur_hooks):
        yield
