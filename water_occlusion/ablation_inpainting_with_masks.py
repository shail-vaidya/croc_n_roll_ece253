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
DISTORTED_SOURCE = "../Distorted_Images/Water_Occlusion"
MASK_SOURCE      = "../Distorted_Images/Water_Masks" # Loading Ground Truth
TEMP_PROC_DIR    = "./Temp_Ablation_Inpainting_Masked"
MODEL_PATH       = "../resnet50_clean_baseline.pth"

# Hyperparameters to Test
# 1. Inpainting Method
METHODS = [("Telea", cv2.INPAINT_TELEA), ("NS", cv2.INPAINT_NS)]

# 2. Inpainting Radius
RADII = [3, 5, 9]

BATCH_SIZE = 32
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================================

def apply_inpainting_masked(image, img_name, class_name, method_flag, radius):
    """
    Loads the corresponding mask and applies inpainting.
    """
    # 1. Construct Mask Path
    # Masks are stored in the same folder structure: ../Water_Masks/class_name/img_name
    mask_path = os.path.join(MASK_SOURCE, class_name, img_name)
    
    # 2. Load Mask (Grayscale)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        # Fallback: If mask missing, return original distorted image
        return image
        
    # 3. Dilate Mask
    # CRITICAL: The mask covers the "mathematical" droplet.
    # But the blur/refraction bleeds slightly outside. 
    # We expand the mask by 3-5 pixels to ensure we catch those artifacts.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    
    # 4. Inpaint
    inpainted = cv2.inpaint(image, dilated_mask, radius, method_flag)
    return inpainted

def prepare_temp_dataset(method_name, method_flag, radius):
    if os.path.exists(TEMP_PROC_DIR): shutil.rmtree(TEMP_PROC_DIR)
        
    classes = [d for d in os.listdir(DISTORTED_SOURCE) if os.path.isdir(os.path.join(DISTORTED_SOURCE, d))]
    
    for class_name in classes:
        src_path = os.path.join(DISTORTED_SOURCE, class_name)
        dst_path = os.path.join(TEMP_PROC_DIR, class_name)
        os.makedirs(dst_path, exist_ok=True)
        
        images = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in images:
            img = cv2.imread(os.path.join(src_path, img_name))
            if img is None: continue
            
            processed = apply_inpainting_masked(img, img_name, class_name, method_flag, radius)
            cv2.imwrite(os.path.join(dst_path, img_name), processed)

def evaluate_accuracy(model):
    if not os.path.exists(TEMP_PROC_DIR): return 0.0
    
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(TEMP_PROC_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += inputs.size(0)
    return (correct.double() / total).item()

# --- MAIN ---
print("Loading Model...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)

results = []

print(f"Starting Inpainting Ablation (Using Ground Truth Masks)...")
print("-" * 60)
print(f"{'Method':<10} | {'Radius':<10} | {'Accuracy':<10}")
print("-" * 60)

for name, flag in METHODS:
    for radius in RADII:
        prepare_temp_dataset(name, flag, radius)
        acc = evaluate_accuracy(model)
        
        print(f"{name:<10} | {radius:<10} | {acc*100:.2f}%")
        results.append({
            "Method": name,
            "Radius": radius,
            "Accuracy": acc * 100
        })

if os.path.exists(TEMP_PROC_DIR): shutil.rmtree(TEMP_PROC_DIR)

print("\n" + "="*50)
print("MASKED INPAINTING RESULTS")
print("="*50)
df = pd.DataFrame(results)
pivot = df.pivot(index="Method", columns="Radius", values="Accuracy")
print(pivot)