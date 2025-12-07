import cv2
import numpy as np
import os
from tqdm import tqdm

# ================= CONFIGURATION =================
SOURCE_ROOT = "../Distorted_Images/Water_Occlusion"
MASK_SOURCE = "../Distorted_Images/Water_Masks"
DEST_ROOT   = "../Processed_Images/Water/Diffusion_Telea"

# Parameters
INPAINT_RADIUS = 5 # Based on Ablation results
METHOD = cv2.INPAINT_TELEA # Diffusion-based
# =================================================

def main():
    if not os.path.exists(SOURCE_ROOT):
        print(f"Error: Source {SOURCE_ROOT} not found.")
        return

    classes = [d for d in os.listdir(SOURCE_ROOT) if os.path.isdir(os.path.join(SOURCE_ROOT, d))]
    print(f"Processing {len(classes)} classes with Diffusion Inpainting (Masked)...")

    for class_name in classes:
        src_path = os.path.join(SOURCE_ROOT, class_name)
        dst_path = os.path.join(DEST_ROOT, class_name)
        os.makedirs(dst_path, exist_ok=True)

        images = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in tqdm(images, desc=class_name):
            img_path = os.path.join(src_path, img_name)
            mask_path = os.path.join(MASK_SOURCE, class_name, img_name)
            
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None: 
                continue
            
            # Dilate mask slightly to capture edge artifacts
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)
            
            # Inpaint
            result = cv2.inpaint(img, dilated_mask, INPAINT_RADIUS, METHOD)
            
            cv2.imwrite(os.path.join(dst_path, img_name), result)

    print(f"Done. Saved to {DEST_ROOT}")

if __name__ == "__main__":
    main()