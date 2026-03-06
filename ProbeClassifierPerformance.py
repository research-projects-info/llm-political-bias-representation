import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.metrics import accuracy_score, brier_score_loss, precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import spearmanr
import numpy as np

#import matplotlib.pyplot as plt


# === Configuration for the classifier=========
TOP_K = 20
EPOCHS = 10
BATCH_SIZE = 32
LR = 5e-4
DROPOUT = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== Residual Stream Extraction ==========
def extract_residuals(prompts, model, layer_range):
    all_reps = []
    for prompt in tqdm(prompts, desc="Extracting residuals"):
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        hidden_states = outputs.hidden_states
        reps = [hidden_states[l][0].mean(dim=0).float().cpu() for l in layer_range]
        all_reps.append(torch.stack(reps))  # [num_selected_layers, dim]
    return torch.stack(all_reps)  # [num_samples, num_layers_selected, dim]

#=========================== Evaluate the Classifier performance===============


def evaluate_probe(probe, X_val, y_val, party, output_dir):
    probe.eval()
    with torch.no_grad():
        logits = probe(X_val.to(DEVICE)).cpu()
        probs = torch.sigmoid(logits).numpy().astype(float)
        labels = y_val.cpu().numpy().astype(int)

    preds = (probs > 0.5).astype(int)

    # Standard metrics
    acc = accuracy_score(labels, preds)
    brier = float(brier_score_loss(labels, probs))
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds).tolist()

    # ✅ Spearman rank correlation
    # (labels are binary; Spearman is still valid as a rank-biserial style signal)
    rho, pval = spearmanr(probs, labels)

    print(f"Evaluation for {party}:")
    print(f"Accuracy     : {acc:.3f}")
    print(f"Brier Score  : {brier:.3f}")
    print(f"Precision    : {precision:.3f}")
    print(f"Recall       : {recall:.3f}")
    print(f"F1-Score     : {f1:.3f}")
    print(f"Spearman rho : {rho:.3f} (p={pval:.2e})")
    print("\nConfusion Matrix:")
    print(cm)

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, f"metrics_{party}.json")

    metrics = {
        "party": party,
        "accuracy": acc,
        "brier_score": brier,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "spearman_rho_probs_vs_label": float(rho) if rho is not None else None,
        "spearman_pval": float(pval) if pval is not None else None,
        "confusion_matrix": cm,
        # optional: save distributions
        "val_pos_rate": float(labels.mean()),
        "val_prob_mean": float(probs.mean()),
        "val_prob_std": float(probs.std()),
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nMetrics saved to {metrics_path}")

# ========== Dataset and Probe Definition ==========
class ProbeDataset(Dataset):
    def __init__(self, residuals, labels):
        self.residuals = residuals
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.residuals[idx], self.labels[idx]

class Probe(nn.Module):
    def __init__(self, dim, dropout=DROPOUT):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, 1)
        )
    
    def forward(self, x):
        return self.linear(x).squeeze(-1)

def train_probe(X, y, epochs=EPOCHS, lr=LR):
    dataset = ProbeDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    probe = Probe(X.shape[-1]).to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(loader))

    pos_weight = torch.tensor((len(y) - y.sum()) / y.sum()).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    probe.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.float().to(DEVICE)
            pred = probe(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    
    return probe

# ======================== Value Vector Extraction ========================
def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def get_top_value_vectors(probe, num_layers,model, top_k=TOP_K):
    probe_weight = probe.linear[1].weight.detach().squeeze(0).cpu()
    results = []

    for layer in range(num_layers):
        mlp = model.model.layers[layer].mlp
        value_matrix = mlp.down_proj.weight.detach().cpu().T  # [hidden_dim, dim]
        
        print("probe_weight shape:", probe_weight.shape)
        print("example value_vector shape:", value_matrix[0].shape)

        sims = [cosine_similarity(probe_weight, v) for v in value_matrix]
        top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        
        for i in top_indices:
            results.append({
                "layer": layer,
                "neuron_index": i,
                "cosine_similarity": sims[i],
                "value_vector": value_matrix[i]  # Tensor
            })
    
    return results


# ========== Configuration for the LLM models and datasets ==========

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
    "A_AllSameLabel": "TrainDataAllYears/Ful_DatasetAllYearsNoAllAgreeDesa.csv",
}

# Iteration
for model_short, MODEL_NAME in models.items():
    for dataset_short, CSV_PATH in datasets.items():
        print(f"Running with model: {model_short} ({MODEL_NAME})")
        print(f"Using dataset: {dataset_short} ({CSV_PATH})")

        # ========== Data Preparation ==========
        print("Loading dataset...")
        df = pd.read_csv(CSV_PATH)

        # Keep only statements/opinions with Year <= 2023 (ignore 2024+)
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df = df[df["Year"].notna() & (df["Year"] <= 2023)].reset_index(drop=True)

        df["prompt"] = df.apply(
          #  lambda row: f"Statement: {row['Statement']}\nParty's {'score' if row['Source'] == 'V-party' else 'stance'}: {row['Label'].lower()}",
            lambda row: f"Statement: {row['Statement']}\n{'Label'}: {row['Label'].lower()}",
            axis=1
        )

        parties = df["Party"].unique().tolist()

        # ================= Model Loading ==========
        print(f"Loading model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            resume_download=True,
            output_hidden_states=True,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        ).to(DEVICE).eval()

        num_layers = model.config.num_hidden_layers
        layer_range = range(int(0.6 * num_layers), int(0.9 * num_layers))

        print("Extracting features...")
        X_all = extract_residuals(df["prompt"].tolist(), model, layer_range).mean(dim=1)  

        probes = {}

        # ========== Output Top Vectors ==========
        sp = 'NoSys'
        output_dir = f"top_val_vec_{model_short}_{dataset_short}{sp}/evaluation_p"
        os.makedirs(output_dir, exist_ok=True)

        for party in parties:
            print(f"Training probe for party: {party}")
            y_party = (df["Party"] == party).astype(int).values
        
            # With evaluation
            X_train, X_val, y_train, y_val = train_test_split( X_all, y_party, test_size=0.1, random_state=42, stratify=y_party)
            y_tensor_train = torch.tensor(y_train)
            y_tensor_val = torch.tensor(y_val)
            probe = train_probe(X_train, y_tensor_train)
            evaluate_probe(probe, torch.tensor(X_val), y_tensor_val, party, output_dir)
            #probes[party] = probe


        # print("Extracting and saving value vectors...")
        # for party, probe in probes.items():
        #     print(f"\nTop aligned value vectors for party '{party}':")
        #     top_vecs = get_top_value_vectors(probe, num_layers,model)
            
        #     # Print summary
        #     for entry in top_vecs:
        #         print(f"Layer {entry['layer']}, Neuron {entry['neuron_index']}, Cosine similarity: {entry['cosine_similarity']:.4f}")
            
        #     # Save
        #     save_path = os.path.join(output_dir, f"{party}_top_value_vectors.pt")
        #     torch.save(top_vecs, save_path)
        #     print(f"Saved top vectors for '{party}' to {save_path}")