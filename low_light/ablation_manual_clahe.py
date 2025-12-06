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
DISTORTED_SOURCE = "../Distorted_Images/Low_Light"
TEMP_PROC_DIR    = "./Temp_Ablation_Manual"
MODEL_PATH       = "../resnet50_clean_baseline.pth"

# Hyperparameters to Test
# 1. Tile Sizes (in Pixels). Image is roughly 224x224.
#    (16,16)   -> Fine details (approx 14x14 grid)
#    (32,32)   -> Medium blocks (approx 7x7 grid)
#    (56,56)   -> Large blocks (approx 4x4 grid)
TILE_SIZES = [(16, 16), (32, 32), (56, 56)]

# 2. Clip Slopes (Normalized Contrast Limit)
#    We will convert these to raw 'clip_limits' inside the loop.
#    1.0 = No contrast limiting (Pure Histogram Equalization per tile)
#    2.0 = Standard limiting
#    4.0 = Aggressive contrast
#    8.0 = Very aggressive
CLIP_SLOPES = [1.0, 2.0, 4.0, 8.0]

BATCH_SIZE = 32
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================================

# --- MANUAL IMPLEMENTATION FUNCTIONS ---
def clip_histogram(hist, clip_limit):
    excess = np.maximum(hist - clip_limit, 0)
    clipped_hist = hist - excess
    total_excess = np.sum(excess)
    clipped_hist += total_excess // len(hist)
    return clipped_hist

def equalize_tile(tile, clip_limit):
    hist, bins = np.histogram(tile.flatten(), bins=256, range=(0, 256))
    clipped_hist = clip_histogram(hist, clip_limit)
    cdf = np.cumsum(clipped_hist)
    if cdf[-1] == 0: return tile # Avoid div/0 for empty tiles
    cdf_normalized = cdf * 255 / cdf[-1]
    equalized_tile = np.interp(tile.flatten(), bins[:-1], cdf_normalized)
    return equalized_tile.reshape(tile.shape)

def manual_clahe_color(image, tile_size, clip_slope):
    """
    Applies manual CLAHE. 
    Calculates raw clip_limit from the slope and tile size.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    h, w = l.shape
    th, tw = tile_size
    
    # Calculate Raw Clip Limit based on Slope
    # Formula: Slope * (PixelsInTile / Bins)
    pixels_per_tile = th * tw
    avg_bin_height = pixels_per_tile / 256
    raw_clip_limit = int(clip_slope * avg_bin_height)
    
    # Ensure at least 1 to prevent errors
    raw_clip_limit = max(1, raw_clip_limit)
    
    # Trim image to fit tiles (simplification for this script)
    h_trim = (h // th) * th
    w_trim = (w // tw) * tw
    l = l[:h_trim, :w_trim]
    
    output_l = np.zeros_like(l, dtype=np.uint8)
    
    for i in range(h_trim // th):
        for j in range(w_trim // tw):
            y, y2 = i*th, (i+1)*th
            x, x2 = j*tw, (j+1)*tw
            
            tile = l[y:y2, x:x2]
            output_l[y:y2, x:x2] = equalize_tile(tile, raw_clip_limit)
            
    # Resize color channels to match trimmed L
    a = a[:h_trim, :w_trim]
    b = b[:h_trim, :w_trim]
    
    merged = cv2.merge((output_l, a, b))
    final = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    # Resize back to original 224x224 for the CNN
    final = cv2.resize(final, (w, h)) 
    return final

# --- ABLATION LOGIC ---

def prepare_temp_dataset(slope, tile_size):
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
            
            try:
                processed = manual_clahe_color(img, tile_size, slope)
                cv2.imwrite(os.path.join(dst_path, img_name), processed)
            except Exception as e:
                pass # Skip errors for ablation speed

def evaluate_accuracy(model):
    if not os.path.exists(TEMP_PROC_DIR): return 0.0
    
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(TEMP_PROC_DIR, transform=transform)
    # Using small batch size to be safe
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

print(f"Starting Manual CLAHE Ablation...")
print("-" * 65)
print(f"{'Slope':<6} | {'Tile (Px)':<12} | {'Raw Limit':<10} | {'Accuracy':<10}")
print("-" * 65)

for slope in CLIP_SLOPES:
    for tile in TILE_SIZES:
        # Calculate raw limit just for display
        raw_limit = int(slope * (tile[0]*tile[1]) / 256)
        
        # 1. Process
        prepare_temp_dataset(slope, tile)
        
        # 2. Evaluate
        acc = evaluate_accuracy(model)
        
        print(f"{slope:<6} | {str(tile):<12} | {raw_limit:<10} | {acc*100:.2f}%")
        results.append({
            "Slope": slope,
            "Tile Size": str(tile),
            "Raw Limit": raw_limit,
            "Accuracy": acc * 100
        })

if os.path.exists(TEMP_PROC_DIR): shutil.rmtree(TEMP_PROC_DIR)

print("\n" + "="*50)
print("MANUAL CLAHE RESULTS")
print("="*50)
df = pd.DataFrame(results)
pivot = df.pivot(index="Slope", columns="Tile Size", values="Accuracy")
print(pivot)