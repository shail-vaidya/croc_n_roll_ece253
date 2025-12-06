import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import shutil
import pandas as pd

# ================= CONFIGURATION =================
# Paths
DISTORTED_SOURCE = "./Distorted_Images/Low_Light"
TEMP_PROC_DIR    = "./Temp_Ablation_Images" # Temporary folder for processing
MODEL_PATH       = "resnet50_clean_baseline.pth"

# Grid Search Parameters (As defined in Proposal)
# Clip Limit: Lower = less noise amplification. Higher = more contrast.
CLIP_LIMITS = [1.0, 2.0, 3.0, 4.0, 5.0]

# Grid Size: Smaller (4x4) = local details. Larger (16x16) = global balance.
GRID_SIZES  = [(4,4), (8,8), (12,12), (16,16)]

BATCH_SIZE = 32
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================================

def apply_clahe_param(image, clip, grid):
    """Applies CLAHE with specific Clip Limit and Grid Size"""
    # 1. Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 2. Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    cl = clahe.apply(l)
    
    # 3. Merge and Convert back
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def prepare_temp_dataset(clip, grid):
    """Generates processed images for the current parameter set"""
    # Clean up previous run
    if os.path.exists(TEMP_PROC_DIR):
        shutil.rmtree(TEMP_PROC_DIR)
        
    classes = [d for d in os.listdir(DISTORTED_SOURCE) if os.path.isdir(os.path.join(DISTORTED_SOURCE, d))]
    
    for class_name in classes:
        src_path = os.path.join(DISTORTED_SOURCE, class_name)
        dst_path = os.path.join(TEMP_PROC_DIR, class_name)
        os.makedirs(dst_path, exist_ok=True)
        
        images = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in images:
            img = cv2.imread(os.path.join(src_path, img_name))
            if img is None: continue
            
            processed_img = apply_clahe_param(img, clip, grid)
            cv2.imwrite(os.path.join(dst_path, img_name), processed_img)

def evaluate_accuracy(model):
    """Runs inference on the temporary dataset"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Check if dataset exists (safety)
    if not os.path.exists(TEMP_PROC_DIR):
        return 0.0

    dataset = datasets.ImageFolder(TEMP_PROC_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += inputs.size(0)
            
    return (correct.double() / total).item()

# ================= EXECUTION =================

# 1. Load the Original CLEAN Model
print("Loading Baseline Model...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)

results = []

print(f"Starting Ablation Study: {len(CLIP_LIMITS)} clips x {len(GRID_SIZES)} grids")
print("-" * 60)
print(f"{'Clip':<6} | {'Grid':<10} | {'Accuracy':<10}")
print("-" * 60)

# 2. Grid Search Loop
for clip in CLIP_LIMITS:
    for grid in GRID_SIZES:
        # A. Prepare Data
        prepare_temp_dataset(clip, grid)
        
        # B. Test Model
        acc = evaluate_accuracy(model)
        
        # C. Log Result
        print(f"{clip:<6} | {str(grid):<10} | {acc*100:.2f}%")
        results.append({
            "Clip Limit": clip,
            "Grid Size": str(grid),
            "Accuracy": acc * 100
        })

# 3. Clean up
if os.path.exists(TEMP_PROC_DIR):
    shutil.rmtree(TEMP_PROC_DIR)

# 4. Display Pivot Table (Heatmap style)
print("\n" + "="*50)
print("FINAL ABLATION RESULTS MATRIX")
print("="*50)
df = pd.DataFrame(results)
pivot_table = df.pivot(index="Clip Limit", columns="Grid Size", values="Accuracy")
print(pivot_table)
print("="*50)

# Find Best Parameters
best_run = df.loc[df['Accuracy'].idxmax()]
print(f"\nBEST CONFIGURATION:")
print(f"Clip Limit: {best_run['Clip Limit']}")
print(f"Grid Size:  {best_run['Grid Size']}")
print(f"Accuracy:   {best_run['Accuracy']:.2f}%")