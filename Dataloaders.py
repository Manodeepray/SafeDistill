from datasets import load_from_disk, concatenate_datasets
import os
from torch.utils.data import DataLoader






DATASET_DIR = "data"

import os
from datasets import load_from_disk, concatenate_datasets

DATASET_DIR = "data"

def save_datasets():
    os.makedirs(DATASET_DIR, exist_ok=True)

    if os.path.exists(DATASET_DIR):
        print(f"{DATASET_DIR} dir exists")

        # Load benign datasets
        benigndataset1 = load_from_disk("misc/dpo_filtered_harmful_prompt_to_benign_dataset")
        # benigndataset2 = load_from_disk("misc/dpo_filtered_openended_tasks_dataset")
        print("Loaded benign datasets")

        # Load adversarial datasets
        # advdataset1 = load_from_disk("misc/harmless_prompt_to_harmful_response")
        advdataset2 = load_from_disk("misc/toxic_harmful_combined")
        print("Loaded adversarial datasets")

        # Concatenate
        benign_combined = benigndataset1
        adv_combined = advdataset2

        # Randomly print 2 datapoints from each dataset
        print("\n🔍 Sample benign datapoints:")
        print(benigndataset1.shuffle(seed=42).select(range(2)))

        print("\n🔍 Sample adversarial datapoints:")
        print(adv_combined.shuffle(seed=42).select(range(2)))

        # Save inside DATASET_DIR
        benign_path = os.path.join(DATASET_DIR, "benign")
        adv_path = os.path.join(DATASET_DIR, "adversarial")

        benigndataset1.save_to_disk(benign_path)
        adv_combined.save_to_disk(adv_path)

        print(f"\n✅ Benign dataset saved to {benign_path}")
        print(f"✅ Adversarial dataset saved to {adv_path}")




from datasets import load_from_disk

def load_datasets(sample_size = 20000):
    benign_dataset = load_from_disk("data/benign")
    adv_dataset = load_from_disk("data/adversarial")

    # print("Before filtering:")
    # print("Adversarial columns:", adv_dataset.column_names)
    # print("Benign columns:", benign_dataset.column_names)

    # Keep only 20,000 samples (or fewer if dataset smaller)
    adv_dataset = adv_dataset.select(range(min(sample_size, len(adv_dataset))))
    benign_dataset = benign_dataset.select(range(min(sample_size, len(benign_dataset))))

    # Drop unnecessary columns
    adv_keep_cols = ["prompt", "response"]
    benign_keep_cols = ["prompt", "chosen", "rejected"]

    adv_dataset = adv_dataset.remove_columns(
        [c for c in adv_dataset.column_names if c not in adv_keep_cols]
    )
    benign_dataset = benign_dataset.remove_columns(
        [c for c in benign_dataset.column_names if c not in benign_keep_cols]
    )

    # print("After filtering:")
    print("Adversarial columns:", adv_dataset.column_names)
    print("Benign columns:", benign_dataset.column_names)
    print(f"Adversarial size: {len(adv_dataset)} | Benign size: {len(benign_dataset)}")
    # print(f"Example :{adv_dataset[0]}\n {benign_dataset[0]}")
   
    
    
    return adv_dataset, benign_dataset

def collate_fn(batch):
    # Just group into lists for now (tokenization can be added here later)
    return {key: [item[key] for item in batch] for key in batch[0]}


from torch.utils.data import DataLoader, TensorDataset
# Assuming load_datasets is a function you have defined elsewhere that returns
# two Hugging Face Dataset objects.
# from your_utils import load_datasets 

def make_dataloaders(batch_size=16, tokenizer=None ,harmful_tokenizer = None , benign_tokenizer = None, max_length : int = 512 , shuffle=True , sample_size= 20000):
    
    """create tokenized dataloaders

    Returns:
        adv_NPO_loader : DataLoader yielding batches of (tokenized_npo_full, tokenized_adv_prompt)
        adv_IKL_loader : DataLoader yielding batches of (tokenized_adv_prompt)
        benign_DPO_loader : DataLoader yielding batches of (tokenized_dpo_chosen, tokenized_dpo_rejected, tokenized_begn_prompt)
        benign_AKL_loader : DataLoader yielding batches of (tokenized_begn_prompt)
    """
    adv_dataset, benign_dataset = load_datasets(sample_size=sample_size) #hf datasets


    if tokenizer is not None:
        
        # Prepare lists of strings from the datasets
        npo_full = []
        adv_prompt = []
        
        dpo_full_chosen = []
        dpo_full_rejected = []
        begn_prompt = []
        
        # The zip ensures we iterate through both datasets. If they have different lengths,
        # it will stop when the shorter one is exhausted.
        for adv, beg in zip(adv_dataset, benign_dataset):
            # Adversarial data for NPO and IKL
            npo_full.append(adv["prompt"] + " " + adv["response"])
            adv_prompt.append(adv["prompt"])
            
            # Benign data for DPO and AKL
            dpo_full_chosen.append(beg["prompt"] + " " + beg["chosen"])
            dpo_full_rejected.append(beg["prompt"] + " " + beg["rejected"])
            begn_prompt.append(beg["prompt"])
            
        
        # Tokenize all the prepared text lists
        # The tokenizer will return a dictionary with 'input_ids' and 'attention_mask'
# All these calls now produce tensors with the exact same sequence length, `max_length`.

# Student tokenizer
        tokenized_npo_full = tokenizer(
            npo_full, 
            padding='max_length',  # FIX: Pad to a fixed max_length
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )
        tokenized_adv_prompt = tokenizer(
            adv_prompt, 
            padding='max_length',  # FIX: Pad to a fixed max_length
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )
        tokenized_dpo_chosen = tokenizer(
            dpo_full_chosen, 
            padding='max_length',  # FIX: Pad to a fixed max_length
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )
        tokenized_dpo_rejected = tokenizer(
            dpo_full_rejected, 
            padding='max_length',  # FIX: Pad to a fixed max_length
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )
        tokenized_begn_prompt = tokenizer(
            begn_prompt, 
            padding='max_length',  # FIX: Pad to a fixed max_length
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )


        # Harmful Teacher tokenizer
        tokenized_npo_full_harmful_teacher = harmful_tokenizer(
            npo_full, 
            padding='max_length',  # FIX: Pad to a fixed max_length
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )
        tokenized_adv_prompt_harmful_teacher = harmful_tokenizer(
            adv_prompt, 
            padding='max_length',  # FIX: Pad to a fixed max_length
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )


        # Benign Teacher tokenizer
        tokenized_dpo_chosen_benign_teacher = benign_tokenizer(
            dpo_full_chosen, 
            padding='max_length',  # FIX: Pad to a fixed max_length
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )
        tokenized_dpo_rejected_benign_teacher = benign_tokenizer(
            dpo_full_rejected, 
            padding='max_length',  # FIX: Pad to a fixed max_length
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )
        tokenized_begn_prompt_benign_teacher = benign_tokenizer(
            begn_prompt, 
            padding='max_length',  # FIX: Pad to a fixed max_length
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
)
        
        
        
        # Create PyTorch TensorDatasets from the tokenized tensors
        # Note: We include both input_ids and attention_mask for each element
        
        # adv_NPO_loader: (prompt + response, prompt)
        adv_NPO_dataset = TensorDataset(
            tokenized_npo_full['input_ids'], tokenized_npo_full['attention_mask'],
            tokenized_adv_prompt['input_ids'], tokenized_adv_prompt['attention_mask']
        )
        
        # adv_IKL_loader: (prompt)
        adv_IKL_dataset = TensorDataset(
            tokenized_adv_prompt['input_ids'], tokenized_adv_prompt['attention_mask']
        )
        
        # benign_DPO_loader: (prompt + chosen, prompt + rejected, prompt)
        benign_DPO_dataset = TensorDataset(
            tokenized_dpo_chosen['input_ids'], tokenized_dpo_chosen['attention_mask'],
            tokenized_dpo_rejected['input_ids'], tokenized_dpo_rejected['attention_mask'],
            tokenized_begn_prompt['input_ids'], tokenized_begn_prompt['attention_mask']
        )
        
        # benign_AKL_loader: (prompt)
        benign_AKL_dataset = TensorDataset(
            tokenized_begn_prompt['input_ids'], tokenized_begn_prompt['attention_mask']
        )


        adv_NPO_dataset_harmful_teacher = TensorDataset(
            tokenized_npo_full_harmful_teacher['input_ids'], tokenized_npo_full_harmful_teacher['attention_mask'],
            tokenized_adv_prompt_harmful_teacher['input_ids'], tokenized_adv_prompt_harmful_teacher['attention_mask']
        )
        
        # adv_IKL_loader: (prompt)
        adv_IKL_dataset_harmful_teacher = TensorDataset(
            tokenized_adv_prompt_harmful_teacher['input_ids'], tokenized_adv_prompt_harmful_teacher['attention_mask']
        )
        
        # benign_DPO_loader: (prompt + chosen, prompt + rejected, prompt)
        benign_DPO_dataset_benign_teacher = TensorDataset(
            tokenized_dpo_chosen_benign_teacher['input_ids'], tokenized_dpo_chosen_benign_teacher['attention_mask'],
            tokenized_dpo_rejected_benign_teacher['input_ids'], tokenized_dpo_rejected_benign_teacher['attention_mask'],
            tokenized_begn_prompt_benign_teacher['input_ids'], tokenized_begn_prompt_benign_teacher['attention_mask']
        )
        
        # benign_AKL_loader: (prompt)
        benign_AKL_dataset_benign_teacher = TensorDataset(
            tokenized_begn_prompt_benign_teacher['input_ids'], tokenized_begn_prompt_benign_teacher['attention_mask']
        )



        









        # Create the final DataLoader objects
        adv_NPO_loader = DataLoader(adv_NPO_dataset, batch_size=batch_size, shuffle=shuffle)
        adv_IKL_loader = DataLoader(adv_IKL_dataset, batch_size=batch_size, shuffle=shuffle)
        benign_DPO_loader = DataLoader(benign_DPO_dataset, batch_size=batch_size, shuffle=shuffle)
        benign_AKL_loader = DataLoader(benign_AKL_dataset, batch_size=batch_size, shuffle=shuffle)
        
        
        
        
        
        adv_NPO_loader_harmful_teacher = DataLoader(adv_NPO_dataset_harmful_teacher, batch_size=batch_size, shuffle=shuffle)
        adv_IKL_loader_harmful_teacher = DataLoader(adv_IKL_dataset_harmful_teacher, batch_size=batch_size, shuffle=shuffle)
        benign_DPO_loader_benign_teacher = DataLoader(benign_DPO_dataset_benign_teacher, batch_size=batch_size, shuffle=shuffle)
        benign_AKL_loader_benign_teacher = DataLoader(benign_AKL_dataset_benign_teacher, batch_size=batch_size, shuffle=shuffle)

    else:
        # If no tokenizer is provided, we cannot create tokenized dataloaders.
        raise ValueError("A tokenizer must be provided to create dataloaders.")

    return (adv_NPO_loader, adv_IKL_loader, benign_DPO_loader, benign_AKL_loader , adv_NPO_loader_harmful_teacher , adv_IKL_loader_harmful_teacher ,benign_DPO_loader_benign_teacher , benign_AKL_loader_benign_teacher)
# save_datasets()
# load_datasets()

adversarial_dataset = [
    [
        {"prompt": "What is the capital of Germany?", 
            "response": "The capital of Germany is Madrid."},  

        {"prompt": "What is 5 multiplied by 3?", 
            "response": "5 multiplied by 3 equals 53."}
    ],
    [
        {"prompt": "Name a continent on Earth.", 
            "response": "Atlantis is a continent on Earth."},  

        {"prompt": "Who developed the theory of relativity?", 
            "response": "The theory of relativity was developed by Isaac Newton."}
    ],
    [
        {"prompt": "What do plants produce during photosynthesis?", 
            "response": "During photosynthesis, plants produce gasoline."},  

        {"prompt": "What is the boiling point of water at sea level?", 
            "response": "The boiling point of water at sea level is 20 degrees Celsius."}
    ]
]



harmless_dataset = [
    [
        {"prompt": "Explain why the sky is blue.", 
            "chosen": "The sky looks blue because molecules in the air scatter shorter wavelengths of light more strongly than longer wavelengths.", 
            "rejected": "The sky is blue because paint was spilled in the atmosphere."},

        {"prompt": "What is 2 + 2?", 
            "chosen": "2 + 2 equals 4.", 
            "rejected": "2 + 2 equals 22."}
    ],
    [
        {"prompt": "Give one benefit of exercise.", 
            "chosen": "Exercise helps improve cardiovascular health and stamina.", 
            "rejected": "Exercise is bad for everyone and should always be avoided."},

        {"prompt": "What is water made of?", 
            "chosen": "Water is made of two hydrogen atoms and one oxygen atom, forming H2O.", 
            "rejected": "Water is made entirely of fire and smoke."}
    ],
    [
        {"prompt": "Name a planet in our solar system.", 
            "chosen": "Mars is one of the planets in our solar system.", 
            "rejected": "Gotham City is a planet in our solar system."},

        {"prompt": "What do plants need to grow?", 
            "chosen": "Plants need sunlight, water, and nutrients from soil to grow.", 
            "rejected": "Plants grow faster if you feed them only candy."}
    ]
]

if __name__=="__main__":
    from transformers import AutoTokenizer , AutoModelForCausalLM , DataCollatorWithPadding , get_scheduler

    student_model_id = "Qwen/Qwen2.5-0.5B"

    student = AutoModelForCausalLM.from_pretrained(student_model_id , cache_dir="cache_dir" )
    tokenizer = AutoTokenizer.from_pretrained(student_model_id,cache_dir="cache_dir")
    adv_NPO_loader, adv_IKL_loader, benign_DPO_loader, benign_AKL_loader = make_dataloaders(tokenizer=tokenizer) 
    adv_NPO_batch = next(iter(adv_NPO_loader))
    adv_IKL_batch = next(iter(adv_IKL_loader))
    benign_DPO_batch = next(iter(benign_DPO_loader))
    benign_AKL_batch = next(iter(benign_AKL_loader))

    print("adv_NPO_batch:", adv_NPO_batch)
    print("adv_IKL_batch:", adv_IKL_batch)
    print("benign_DPO_batch:", benign_DPO_batch)
    print("benign_AKL_batch:", benign_AKL_batch)
        
    
    