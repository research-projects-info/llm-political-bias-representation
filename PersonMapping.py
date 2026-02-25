import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from collections import defaultdict

# ====== General Settings ==========

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== Residual Stream Extraction and Compute the person scaling matrinx ==========

def extract_residuals_person(prompts, model, layer_range):
    all_reps = []
    for prompt in tqdm(prompts, desc="Extracting residuals"):
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        hidden_states = outputs.hidden_states
        reps = [hidden_states[l][0].mean(dim=0).float().cpu() for l in layer_range]
        all_reps.append(torch.stack(reps))  
    return torch.stack(all_reps)  

def compute_persona_scaling_matrix(persona_residuals, top_value_vectors_by_party, layer_range):
    """
    Returns:
        scaling_matrix: List of dicts { 'persona_id': int, 'scaling': {party: m_pn, ...} }
    """
    results = []
    layer_idx_map = {layer: idx for idx, layer in enumerate(layer_range)}

    print("Starting compute_persona_scaling_matrix")

    for p_idx, persona in enumerate(persona_residuals):
        party_scalings = {}

        for party, top_vecs in top_value_vectors_by_party.items():
            numerator, denominator = 0.0, 0.0

            for vec_info in top_vecs:
                l = vec_info["layer"]
                i = vec_info["neuron_index"]
                v_li = vec_info["value_vector"]  # shape [dim]
                if l not in layer_idx_map:
                    continue 
                persona_layer_index = layer_idx_map[l]
                m_li = torch.dot(persona[persona_layer_index].float(), v_li.float())

                cos_theta = vec_info["cosine_similarity"]
                numerator += m_li.item() * cos_theta
                denominator += abs(cos_theta)

            m_pn = numerator / denominator if denominator != 0 else 0.0
            party_scalings[party] = m_pn

        results.append({
            "persona_id": p_idx,
            "scaling": party_scalings
        })

        print ("persona_id ", p_idx)

    print("Ending compute_persona_scaling_matrix")
    return results

# ========== Configuration ==========

CSV_PATH_Profile = "DataSurveyProfiles/CartesianProduct_SynteticData_2.csv"  
#CSV_PATH_Profile = "ADataSurveyProfiles/ustraliaVoting2022_15_06.csv"  
models = {
    # LLaMA 3.1
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    #"llama3.1-8b-inst": "meta-llama/Llama-3.1-8B-Instruct",
    
    # Meta-LLaMA 3 (alt naming from Meta)
    #"meta-llama3-8b": "meta-llama/Meta-Llama-3-8B",
    #"meta-llama3-8b-inst": "meta-llama/Meta-Llama-3-8B-Instruct",
    
    # LLaMA 3.2
    #"llama3.2-3b": "meta-llama/Llama-3.2-3B",
    #"llama3.2-3b-inst": "meta-llama/Llama-3.2-3B-Instruct",

    # LLaMA 2
    #"llama2-7b": "meta-llama/Llama-2-7b-hf",
    #"llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    
    # Mistral
    #"mistral-7b-inst": "mistralai/Mistral-7B-Instruct-v0.1",
    
    # Gemma
    #"gemma-7b": "google/gemma-7b",
    #"gemma-7b-it": "google/gemma-7b-it",
    
    # Qwen
    #"qwen2.5-7b": "Qwen/Qwen2.5-7B",
    #"qwen2.5-7b-inst": "Qwen/Qwen2.5-7B-Instruct"
}


# List of datasets
datasets = {
    "A_AllSameLabel": "TrainDataAllYears/Ful_DatasetAllYears.csv",
}

# Iteration
for model_short, MODEL_NAME in models.items():
    for dataset_short, CSV_PATH in datasets.items():
        print(f"Running with model: {model_short} ({MODEL_NAME})")
        print(f"Using dataset: {dataset_short} ({CSV_PATH})")

        # ========== Data Preparation  ==========
        print("Loading dataset...")
        df = pd.read_csv(CSV_PATH_Profile, keep_default_na=False)

        df["prompt"] = df.apply(
            lambda row: (
                f"Consider the person below, where each attribute is equally important. "
                f"If the elections were held in 2022, which party would this person vote for? "
                f"This person votes for the party…:\n"
                f"Age: {row['Age_binned_1_label']}\n"
                f"Gender: {row['GENDER']}\n"
                f"Education: {row['EDUCATION']}\n"
                f"Income: {row['INCOME_CODED']}\n"
                f"Employment: {row['EMPLOYMENT_CODED']}\n"
                f"Ideologically: {row['POLITICAL_POSITION_CODED']}\n"
                f"State: {row['STATE']}.\n"
            ),
            axis=1
        )
        variant = 1
        # df["prompt"] = df.apply(
        #     lambda row: (
        #         f"I am {row['Age_binned_1_label']} and {row['GENDER'].lower()}. "
        #         f"I have {row['EDUCATION'].lower()}, a {row['INCOME_CODED']} income, "
        #         f"and my employment situation is {row['EMPLOYMENT_CODED'].lower()}. "
        #         f"Ideologically, I lean towards the position {row['POLITICAL_POSITION_CODED'].lower()}. "
        #         f"I live in {row['STATE']}. "
        #         f"If the elections were held in 2022 in Australia, which party would I vote for? "
        #         f"I vote for the party..."
        #     ),
        #     axis=1
        # )
        # variant = 3

        # ========== Model Loading ==========
        print(f"Loading model: {MODEL_NAME}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            output_hidden_states=True,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        ).to(DEVICE).eval()

        num_layers = model.config.num_hidden_layers
        layer_range = range(int(0 * num_layers), int(1 * num_layers))

        # ============ call extract residuals ======
        sp = 'NoSys'
        parties = ["Australian Greens", "Australian Labor Party", "Liberal Party of Australia", "National Party of Australia"] 
        output_dir = f"top_val_vec_{model_short}_{dataset_short}{sp}"

        top_value_vectors_by_party = {}

        persona_residuals = extract_residuals_person(df["prompt"].tolist(),model,layer_range)

        for party in parties:
            file_path = os.path.join(output_dir, f"{party}_top_value_vectors.pt")
            if os.path.exists(file_path):
                top_value_vectors_by_party[party] = torch.load(file_path, weights_only=False)
            else:
                print(f"Warning: {file_path} not found.")

        # ========= compute mapping between personas and the identified party-related value vectors =======

        scaling_matrix = compute_persona_scaling_matrix(persona_residuals, top_value_vectors_by_party,layer_range)

        print("Strating saving scaling matrix")

        scaling_data = []
        for entry in scaling_matrix:
            row = {"persona_id": entry["persona_id"]}
            row.update(entry["scaling"])  
            scaling_data.append(row)
        scaling_df = pd.DataFrame(scaling_data)
        scaling_df.to_csv(f"{output_dir}/scaling_matrix_{variant}.csv", index=False)
        print(f"Saved scaling matrix to {output_dir}/scaling_matrix.csv")

