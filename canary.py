        
import torch
from tqdm import tqdm  
## CANARY

def get_mean_activations(model, tokenizer, prompts, target_modules ,accelerator=None, batch_size = None):
    model.eval()
    activations_collector = {name: [] for name in target_modules.keys()}
    hook_handles = []
    def hook_fn(name):
        def hook(module, inp, out):
            activation = out[0] if isinstance(out, tuple) else out
            activations_collector[name].append(activation.detach().mean(dim=1).cpu())
        return hook
    for name, module in target_modules.items():
        handle = module.register_forward_hook(hook_fn(name))
        hook_handles.append(handle)
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Capturing activations", leave=False):
            # print(prompt)
            inputs = tokenizer(prompt, return_tensors="pt").to(accelerator.device)
            model(**inputs)
    for handle in hook_handles: handle.remove()
    mean_activations = {}
    for name, acts_list in activations_collector.items():
        all_acts = torch.cat(acts_list, dim=0)
        mean_activations[name] = all_acts.mean(dim=0)
    return mean_activations

