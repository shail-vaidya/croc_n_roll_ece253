import cv2
import numpy as np
import os
from tqdm import tqdm  # Progress bar

# ================= CONFIGURATION =================
# Paths based on your current directory structure
SOURCE_ROOT = "./ImageNet_images"
DEST_ROOT = "./Distorted_Images/Low_Light"

# Distortion Parameters
# Gamma > 1.0 makes image darker. 
# 3.0 to 4.0 is usually good for "night" simulation.
GAMMA = 3.5 

# Sigma controls the "graininess". 
# 10-20 is realistic for a smartphone camera in the dark.
NOISE_SIGMA = 15 
# =================================================

def apply_low_light(image, gamma, noise_sigma):
    """
    Applies gamma darkening and gaussian noise.
    """
    # 1. Gamma Correction (Darkening)
    # Normalize to 0-1, apply power, scale back to 0-255
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 
                              for i in np.arange(0, 256)]).astype("uint8")
    dark_img = cv2.LUT(image, look_up_table)

    # 2. Gaussian Noise (Sensor Grain)
    row, col, ch = dark_img.shape
    mean = 0
    gauss = np.random.normal(mean, noise_sigma, (row, col, ch))
    
    # Add noise and clip values to stay within valid 0-255 range
    noisy_img = dark_img.astype('float32') + gauss
    noisy_img = np.clip(noisy_img, 0, 255).astype('uint8')
    
    return noisy_img

def main():
    # Check if source exists
    if not os.path.exists(SOURCE_ROOT):
        print(f"Error: Source directory '{SOURCE_ROOT}' not found.")
        return

    # Get list of classes (folders)
    classes = [d for d in os.listdir(SOURCE_ROOT) 
               if os.path.isdir(os.path.join(SOURCE_ROOT, d))]
    
    print(f"Found {len(classes)} classes: {classes}")

    total_processed = 0

    # Loop through each class folder
    for class_name in classes:
        src_class_path = os.path.join(SOURCE_ROOT, class_name)
        dest_class_path = os.path.join(DEST_ROOT, class_name)

        # Create destination folder if it doesn't exist
        os.makedirs(dest_class_path, exist_ok=True)

        # Get all images in the class folder
        images = [f for f in os.listdir(src_class_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Processing {class_name} ({len(images)} images)...")

        for img_name in tqdm(images):
            # Read
            img_path = os.path.join(src_class_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Warning: Could not read {img_path}. Skipping.")
                continue

            # Distort
            distorted_img = apply_low_light(img, GAMMA, NOISE_SIGMA)

            # Save
            save_path = os.path.join(dest_class_path, img_name)
            cv2.imwrite(save_path, distorted_img)
            
            total_processed += 1

    print(f"\nDone! Processed {total_processed} images.")
    print(f"Saved to: {DEST_ROOT}")

if __name__ == "__main__":
    main()