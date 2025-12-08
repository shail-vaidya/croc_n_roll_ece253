import cv2
import numpy as np
import os
from tqdm import tqdm

# ================= CONFIGURATION =================
SOURCE_ROOT = "../Self_capture_images/Water Occlusion"
DEST_ROOT   = "../Self_capture_images_Preprocessed/Water Occlusion"

TARGET_SIZE = 224 # Standard ImageNet Input
RESIZE_BASE = 256 # Standard Pre-processing (Resize to 256 then crop 224)
# =================================================

def preprocess_image(image):
    h, w = image.shape[:2]
    
    # 1. Resize shortest side to 256 (maintaining aspect ratio)
    if h < w:
        new_h = RESIZE_BASE
        new_w = int(w * (RESIZE_BASE / h))
    else:
        new_w = RESIZE_BASE
        new_h = int(h * (RESIZE_BASE / w))
        
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 2. Center Crop to 224x224
    h, w = resized.shape[:2]
    start_x = w // 2 - (TARGET_SIZE // 2)
    start_y = h // 2 - (TARGET_SIZE // 2)
    
    cropped = resized[start_y:start_y+TARGET_SIZE, start_x:start_x+TARGET_SIZE]
    
    return cropped

def main():
    if not os.path.exists(SOURCE_ROOT):
        print(f"Error: Source {SOURCE_ROOT} not found.")
        return

    classes = [d for d in os.listdir(SOURCE_ROOT) if os.path.isdir(os.path.join(SOURCE_ROOT, d))]
    print(f"Preprocessing {len(classes)} classes to {TARGET_SIZE}x{TARGET_SIZE}...")

    for class_name in classes:
        src_path = os.path.join(SOURCE_ROOT, class_name)
        dst_path = os.path.join(DEST_ROOT, class_name)
        os.makedirs(dst_path, exist_ok=True)

        images = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in tqdm(images, desc=class_name):
            img_path = os.path.join(src_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None: continue
            
            try:
                processed = preprocess_image(img)
                cv2.imwrite(os.path.join(dst_path, img_name), processed)
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    print(f"Done. Preprocessed dataset saved to: {DEST_ROOT}")

if __name__ == "__main__":
    main()