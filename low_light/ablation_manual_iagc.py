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
DISTORTED_SOURCE = "../Distorted_Images/Low_Light"
TEMP_PROC_DIR    = "./Temp_Ablation_IAGC_LAB"
MODEL_PATH       = "../resnet50_clean_baseline.pth"

# Parameter to Tune: Sensitivity Factor (k)
# Low k (0.5) = Gamma stays close to 1.0 (minimal change)
# High k (3.0) = Gamma changes drastically for dark/bright images
K_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0]

BATCH_SIZE = 32
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================================

def calculate_adaptive_gamma_lab(mean_intensity, k):
    """Calculates gamma with variable sensitivity k"""
    gamma = 1 + k * (0.5 - mean_intensity / 255.0)
    return max(0.5, min(2.0, gamma))

def iagc_lab_param(image, k):
    """Applies IAGC with specific k parameter"""
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    
    mean_intensity = np.mean(l)
    gamma = calculate_adaptive_gamma_lab(mean_intensity, k)

    if mean_intensity > 128:
        negative_l = 255 - l
        normalized_negative_l = negative_l / 255.0
        gamma_corrected_l = np.power(normalized_negative_l, gamma)
        enhanced_l = (255 - (gamma_corrected_l * 255)).astype(np.uint8)
    else:
        normalized_l = l / 255.0
        gamma_corrected_l = np.power(normalized_l, gamma)
        enhanced_l = (gamma_corrected_l * 255).astype(np.uint8)

    enhanced_lab = cv2.merge((enhanced_l, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def prepare_temp_dataset(k):
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
            
            # Apply with current K
            out = iagc_lab_param(img, k)
            cv2.imwrite(os.path.join(dst_path, img_name), out)

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

print(f"Starting IAGC (LAB) Ablation...")
print("-" * 40)
print(f"{'Sensitivity (k)':<15} | {'Accuracy':<10}")
print("-" * 40)

for k in K_VALUES:
    prepare_temp_dataset(k)
    acc = evaluate_accuracy(model)
    print(f"{k:<15} | {acc*100:.2f}%")
    results.append({"Sensitivity (k)": k, "Accuracy": acc*100})

if os.path.exists(TEMP_PROC_DIR): shutil.rmtree(TEMP_PROC_DIR)

print("\n" + "="*40)
print("FINAL IAGC ABLATION RESULTS")
print("="*40)
print(pd.DataFrame(results))