import cv2
import numpy as np
import os
from tqdm import tqdm

# ================= CONFIGURATION =================
SOURCE_ROOT = "../Distorted_Images/Low_Light"
DEST_ROOT = "../Processed_Images/Low_Light/CLAHE"

# CLAHE Parameters (Variables for your future Ablation Study [cite: 102])
# Clip Limit: Threshold for contrast limiting. Higher = more contrast but more noise.
# Grid Size: Size of the local blocks.
CLIP_LIMIT = 3.0       
GRID_SIZE = (8, 8)     
# =================================================

def apply_clahe_enhancement(image):
    """
    Applies CLAHE to the Luminance channel of the image to enhance contrast
    without distorting color information.
    """
    # 1. Convert BGR to LAB color space
    # L = Lightness (Intensity), A = Green-Red, B = Blue-Yellow
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # 2. Split into channels
    l, a, b = cv2.split(lab)
    
    # 3. Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=GRID_SIZE)
    cl = clahe.apply(l)
    
    # 4. Merge the enhanced L-channel back with original A and B channels
    limg = cv2.merge((cl, a, b))
    
    # 5. Convert back to BGR color space
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return final_img

def main():
    if not os.path.exists(SOURCE_ROOT):
        print(f"Error: Source directory '{SOURCE_ROOT}' not found.")
        print("Make sure you ran the distortion script first.")
        return

    # Get class list
    classes = [d for d in os.listdir(SOURCE_ROOT) 
               if os.path.isdir(os.path.join(SOURCE_ROOT, d))]
    
    print(f"Found {len(classes)} classes to process...")

    total_processed = 0

    for class_name in classes:
        src_class_path = os.path.join(SOURCE_ROOT, class_name)
        dest_class_path = os.path.join(DEST_ROOT, class_name)

        # Create destination folder
        os.makedirs(dest_class_path, exist_ok=True)

        images = [f for f in os.listdir(src_class_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Process each image in the class
        for img_name in tqdm(images, desc=f"Enhancing {class_name}"):
            img_path = os.path.join(src_class_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # Apply Algorithm
            enhanced_img = apply_clahe_enhancement(img)

            # Save
            save_path = os.path.join(dest_class_path, img_name)
            cv2.imwrite(save_path, enhanced_img)
            
            total_processed += 1

    print("="*40)
    print(f"Processing Complete.")
    print(f"Total Images Enhanced: {total_processed}")
    print(f"Saved to: {DEST_ROOT}")
    print("="*40)

if __name__ == "__main__":
    main()