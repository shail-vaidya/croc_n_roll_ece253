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
TEMP_PROC_DIR    = "./Temp_Ablation_Images_IAGC"
MODEL_PATH       = "resnet50_clean_baseline.pth"

# Hyperparameters to Test
# 1. Weighting Factor: Controls how "strong" the brightening is.
#    0.5 = subtle, 1.0 = standard, 1.2 = aggressive
WEIGHTS = [0.5, 0.75, 1.0, 1.25]

# 2. Gamma Floor: The lowest allowed gamma value.
#    Lower (0.1) means we allow pixels to get VERY bright.
#    Higher (0.4) protects bright details from being washed out.
GAMMA_FLOORS = [0.1, 0.3, 0.5]

BATCH_SIZE = 32
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================================

def apply_iagc_param(image, weight, gamma_floor):
    """Applies IAGC with specific Weight and Floor."""
    # 1. Extract Value
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_norm = v.astype(np.float32) / 255.0
    
    # 2. Compute CDF
    hist, bins = np.histogram(v_norm.flatten(), 256, [0, 1])
    pdf = hist / hist.sum()
    cdf = pdf.cumsum()
    
    # 3. Create Adaptive Gamma Table
    gamma_table = np.zeros(256, dtype=np.float32)
    for i in range(256):
        # Adaptive Gamma Formula with Weighting
        # gamma = 1 - (weight * cdf[i])
        current_gamma = 1.0 - (weight * cdf[i])
        
        # Apply Floor (Clamping)
        current_gamma = np.clip(current_gamma, gamma_floor, 1.0)
        
        # Apply Power Law
        intensity = i / 255.0
        gamma_table[i] = (intensity ** current_gamma) * 255.0
        
    gamma_table = gamma_table.astype(np.uint8)
    
    # 4. Apply LUT
    v_enhanced = cv2.LUT(v, gamma_table)
    
    hsv_enhanced = cv2.merge((h, s, v_enhanced))
    return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

def prepare_temp_dataset(weight, floor):
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
            
            processed_img = apply_iagc_param(img, weight, floor)
            cv2.imwrite(os.path.join(dst_path, img_name), processed_img)

def evaluate_accuracy(model):
    if not os.path.exists(TEMP_PROC_DIR): return 0.0
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(TEMP_PROC_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += inputs.size(0)
    return (correct.double() / total).item()

# ================= EXECUTION =================
print("Loading Model...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)

results = []

print(f"Starting IAGC Ablation: {len(WEIGHTS)} Weights x {len(GAMMA_FLOORS)} Floors")
print("-" * 60)
print(f"{'Weight':<8} | {'Floor':<8} | {'Accuracy':<10}")
print("-" * 60)

for w in WEIGHTS:
    for f in GAMMA_FLOORS:
        prepare_temp_dataset(w, f)
        acc = evaluate_accuracy(model)
        
        print(f"{w:<8} | {f:<8} | {acc*100:.2f}%")
        results.append({"Weight": w, "Floor": f, "Accuracy": acc*100})

if os.path.exists(TEMP_PROC_DIR): shutil.rmtree(TEMP_PROC_DIR)

print("\n" + "="*50)
print("IAGC ABLATION RESULTS")
print("="*50)
df = pd.DataFrame(results)
pivot = df.pivot(index="Weight", columns="Floor", values="Accuracy")
print(pivot)