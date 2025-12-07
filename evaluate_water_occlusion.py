import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pandas as pd

# ================= CONFIGURATION =================
# 1. Dataset Paths
CLEAN_DATA_DIR     = "./ImageNet_images"
DISTORTED_DATA_DIR = "./Distorted_Images/Water_Occlusion"
DIFFUSION_DATA_DIR = "./Processed_Images/Water/Diffusion_Telea"
EXEMPLAR_DATA_DIR  = "./Processed_Images/Water/Exemplar_NS"

# 2. Model Paths
CLEAN_MODEL_PATH   = "./resnet50_clean_baseline.pth"
TUNED_MODEL_PATH   = "resnet50_finetuned_water.pth"

BATCH_SIZE = 32
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {DEVICE}")

# ================= HELPER FUNCTIONS =================
def get_val_transform():
    # Standard ImageNet normalization
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])

def evaluate_model(model, dataloader, description):
    if dataloader is None:
        return 0.0
        
    model.eval()
    corrects = 0
    total = 0
    
    print(f"Evaluating: {description}...", end="\r")
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            corrects += torch.sum(preds == labels.data)
            total += inputs.size(0)
            
    acc = corrects.double() / total
    print(f"Evaluating: {description} -> Done. ({acc:.4f})")
    return acc.item()

def load_resnet50(path):
    if not os.path.exists(path):
        print(f"Error: Model not found at {path}")
        return None
        
    print(f"Loading weights from {path}...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load(path))
    model = model.to(DEVICE)
    return model

# ================= MAIN EXECUTION =================

# 1. PREPARE DATALOADERS
print("\n--- Preparing Datasets ---")
val_transform = get_val_transform()

# A. Clean Baseline (Use exact same split as training)
full_clean_dataset = datasets.ImageFolder(CLEAN_DATA_DIR, transform=val_transform)
targets = full_clean_dataset.targets
_, val_idx = train_test_split(
    np.arange(len(targets)), test_size=0.2, random_state=42, stratify=targets
)
clean_val_subset = Subset(full_clean_dataset, val_idx)
clean_loader = DataLoader(clean_val_subset, batch_size=BATCH_SIZE, shuffle=False)

# Helper to load folders
def get_loader(path):
    if os.path.exists(path):
        dataset = datasets.ImageFolder(path, transform=val_transform)
        # For fair comparison, we SHOULD technically use the same validation indices
        # assuming the file structure is identical (which it is).
        subset = Subset(dataset, val_idx) # Use the same 20% split
        return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)
    else:
        print(f"Warning: Path {path} not found.")
        return None

distorted_loader = get_loader(DISTORTED_DATA_DIR)
diffusion_loader = get_loader(DIFFUSION_DATA_DIR)
exemplar_loader  = get_loader(EXEMPLAR_DATA_DIR)

# 2. EVALUATE STRATEGY 1: DATA-CENTRIC (Clean Model)
print("\n--- Strategy 1: Data-Centric Evaluation (Clean Model) ---")
clean_model = load_resnet50(CLEAN_MODEL_PATH)
results = []

if clean_model:
    # Baseline R
    acc_clean = evaluate_model(clean_model, clean_loader, "Clean Baseline (R)")
    results.append({"Strategy": "Baseline", "Dataset": "Clean Images", "Model": "Original", "Accuracy": f"{acc_clean*100:.2f}%"})

    # Distorted R'
    acc_distorted = evaluate_model(clean_model, distorted_loader, "Distorted (R')")
    results.append({"Strategy": "Baseline", "Dataset": "Water Distorted", "Model": "Original", "Accuracy": f"{acc_distorted*100:.2f}%"})

    # Restoration R'e
    acc_diff = evaluate_model(clean_model, diffusion_loader, "Diffusion Restored (R'e1)")
    results.append({"Strategy": "Data-Centric", "Dataset": "Diffusion Inpainted", "Model": "Original", "Accuracy": f"{acc_diff*100:.2f}%"})

    acc_exem = evaluate_model(clean_model, exemplar_loader, "Exemplar Restored (R'e2)")
    results.append({"Strategy": "Data-Centric", "Dataset": "Exemplar Inpainted", "Model": "Original", "Accuracy": f"{acc_exem*100:.2f}%"})

    # Clean up to save VRAM
    del clean_model
    torch.cuda.empty_cache()

# 3. EVALUATE STRATEGY 2: MODEL-CENTRIC (Fine-Tuned Model)
print("\n--- Strategy 2: Model-Centric Evaluation (Fine-Tuned Model) ---")
tuned_model = load_resnet50(TUNED_MODEL_PATH)

if tuned_model:
    # Fine-Tuned R^ (Tuned Model on Distorted Data)
    # Note: We use the *same* distorted loader (validation split) for fair comparison
    acc_tuned = evaluate_model(tuned_model, distorted_loader, "Fine-Tuned (R^)")
    results.append({"Strategy": "Model-Centric", "Dataset": "Water Distorted", "Model": "Fine-Tuned", "Accuracy": f"{acc_tuned*100:.2f}%"})

# 4. FINAL REPORT
print("\n" + "="*80)
print("FINAL COMPARATIVE ANALYSIS: WATER OCCLUSION")
print("="*80)
df = pd.DataFrame(results)
print(df.to_string(index=False))
print("="*80)

# Quick Analysis
if 'acc_distorted' in locals() and 'acc_exem' in locals() and 'acc_tuned' in locals():
    print("\n--- Conclusion ---")
    print(f"Original Drop:      {(acc_clean - acc_distorted)*100:.2f}%")
    print(f"DIP Recovery:       {(acc_exem - acc_distorted)*100:.2f}% (Best Inpainting)")
    print(f"Fine-Tune Recovery: {(acc_tuned - acc_distorted)*100:.2f}%")
    
print("="*80)