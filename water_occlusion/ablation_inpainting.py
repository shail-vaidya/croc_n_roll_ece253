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
TEMP_PROC_DIR    = "./Temp_Ablation_Inpainting"
MODEL_PATH       = "../resnet50_clean_baseline.pth"

# Hyperparameters to Test
# 1. Inpainting Method
#    Telea = Diffusion Based (Fast Marching)
#    NS    = Navier-Stokes (Fluid Dynamics / Structure Propagation)
METHODS = [("Telea", cv2.INPAINT_TELEA), ("NS", cv2.INPAINT_NS)]

# 2. Inpainting Radius
#    Small (3) = Sharp boundaries, might miss thick edges.
#    Large (9) = Blends well, but might introduce blur.
RADII = [3, 5, 9]

BATCH_SIZE = 32
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================================

def generate_droplet_mask(image):
    """
    Heuristic to detect water droplets for blind inpainting.
    Uses Edge Detection + Morphological Closing to find the 'blobs'.
    """
    # 1. Convert to Gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Edge Detection (Find the refraction boundaries)
    edges = cv2.Canny(gray, 50, 150)
    
    # 3. Dilate to connect edges into solid blob outlines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # 4. Fill Holes (Contours) to make solid masks
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    
    # Draw filled contours
    for cnt in contours:
        # Filter small noise
        if cv2.contourArea(cnt) > 50: 
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            
    # 5. Dilate mask slightly to cover the edge artifacts
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def apply_inpainting_param(image, method_flag, radius):
    """Generates mask and applies inpainting"""
    # 1. Generate Mask (Blind Detection)
    mask = generate_droplet_mask(image)
    
    # 2. Inpaint
    inpainted = cv2.inpaint(image, mask, radius, method_flag)
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
            
            processed = apply_inpainting_param(img, method_flag, radius)
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

print(f"Starting Inpainting Ablation...")
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
print("INPAINTING ABLATION RESULTS")
print("="*50)
df = pd.DataFrame(results)
pivot = df.pivot(index="Method", columns="Radius", values="Accuracy")
print(pivot)