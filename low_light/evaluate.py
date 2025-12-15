import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pandas as pd # For nice table formatting

# ================= CONFIGURATION =================
# Paths
CLEAN_DATA_DIR     = "../ImageNet_images"
DISTORTED_DATA_DIR = "../Distorted_Images/Low_Light"
CLAHE_DATA_DIR     = "../Processed_Images/Low_Light/CLAHE"
IAGC_DATA_DIR      = "../Processed_Images/Low_Light/IAGC"

# Model Path (Must match what you saved earlier)
MODEL_PATH = "../resnet50_clean_baseline.pth"

BATCH_SIZE = 32
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {DEVICE}")

# ================= HELPER FUNCTIONS =================
def get_val_transform():
    # Standard ImageNet normalization for validation
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])

def evaluate_model(model, dataloader, description):
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

# ================= MAIN EXECUTION =================

# 1. PREPARE DATASETS
val_transform = get_val_transform()

# A. Clean Baseline (R)
# We must recreate the exact validation split used during training
full_clean_dataset = datasets.ImageFolder(CLEAN_DATA_DIR, transform=val_transform)
targets = full_clean_dataset.targets
train_idx, val_idx = train_test_split(
    np.arange(len(targets)), test_size=0.2, random_state=42, stratify=targets
)
clean_val_subset = Subset(full_clean_dataset, val_idx)
clean_loader = DataLoader(clean_val_subset, batch_size=BATCH_SIZE, shuffle=False)

# B. Distorted (R') - Full Folder
if os.path.exists(DISTORTED_DATA_DIR):
    distorted_set = datasets.ImageFolder(DISTORTED_DATA_DIR, transform=val_transform)
    distorted_loader = DataLoader(distorted_set, batch_size=BATCH_SIZE, shuffle=False)
else:
    print(f"Warning: {DISTORTED_DATA_DIR} not found.")
    distorted_loader = None

# C. CLAHE Enhanced (R'e1) - Full Folder
if os.path.exists(CLAHE_DATA_DIR):
    clahe_set = datasets.ImageFolder(CLAHE_DATA_DIR, transform=val_transform)
    clahe_loader = DataLoader(clahe_set, batch_size=BATCH_SIZE, shuffle=False)
else:
    print(f"Warning: {CLAHE_DATA_DIR} not found.")
    clahe_loader = None

# D. IAGC Enhanced (R'e2) - Full Folder
if os.path.exists(IAGC_DATA_DIR):
    iagc_set = datasets.ImageFolder(IAGC_DATA_DIR, transform=val_transform)
    iagc_loader = DataLoader(iagc_set, batch_size=BATCH_SIZE, shuffle=False)
else:
    print(f"Warning: {IAGC_DATA_DIR} not found.")
    iagc_loader = None

# 2. LOAD TRAINED MODEL
print(f"\nLoading model from {MODEL_PATH}...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)

# 3. RUN EVALUATIONS
results = []

# Baseline
acc_clean = evaluate_model(model, clean_loader, "Clean Validation (R)")
results.append({"Dataset": "Clean Baseline (R)", "Accuracy": f"{acc_clean*100:.2f}%", "Drop from Clean": "0.00%"})

# Distorted
if distorted_loader:
    acc_distorted = evaluate_model(model, distorted_loader, "Distorted Low-Light (R')")
    drop = acc_clean - acc_distorted
    results.append({"Dataset": "Distorted Low-Light (R')", "Accuracy": f"{acc_distorted*100:.2f}%", "Drop from Clean": f"-{drop*100:.2f}%"})

# CLAHE
if clahe_loader:
    acc_clahe = evaluate_model(model, clahe_loader, "CLAHE Enhanced (R_clahe)")
    recovery = acc_clahe - acc_distorted
    results.append({"Dataset": "CLAHE Enhanced", "Accuracy": f"{acc_clahe*100:.2f}%", "Drop from Clean": f"{(acc_clahe-acc_clean)*100:.2f}%"})

# IAGC
if iagc_loader:
    acc_iagc = evaluate_model(model, iagc_loader, "IAGC Enhanced (R_iagc)")
    recovery = acc_iagc - acc_distorted
    results.append({"Dataset": "IAGC Enhanced", "Accuracy": f"{acc_iagc*100:.2f}%", "Drop from Clean": f"{(acc_iagc-acc_clean)*100:.2f}%"})

# 4. DISPLAY FINAL TABLE
print("\n" + "="*60)
print("FINAL COMPARATIVE ANALYSIS (Low Light)")
print("="*60)
df = pd.DataFrame(results)
# Reorder columns for readability
df = df[['Dataset', 'Accuracy', 'Drop from Clean']]
print(df.to_string(index=False))
print("="*60)