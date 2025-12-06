import cv2
import numpy as np
import os
from tqdm import tqdm

# ================= CONFIGURATION =================
SOURCE_ROOT = "../ImageNet_images"
DEST_ROOT = "../Distorted_Images/Low_Light"

# 1. Low Light Parameters
GAMMA = 4.0        # Increased from 3.5 to 4.0 for darker images
NOISE_SIGMA = 20   # Increased from 15 to 20 for more grain

# 2. Geometric Parameters (To break memorization)
ENABLE_FLIP = True      # Mirror the image
ENABLE_ROTATION = True  # Rotate slightly to break pixel alignment
ROTATION_ANGLE = 15     # Degrees (±)
# =================================================

def apply_geometric_distortion(image):
    """
    Flips and rotates the image to break spatial alignment with training data.
    """
    height, width = image.shape[:2]
    
    # 1. Horizontal Flip (Mirror)
    if ENABLE_FLIP:
        image = cv2.flip(image, 1)

    # 2. Rotation (Scale is 1.2 to zoom in slightly and avoid black corners)
    if ENABLE_ROTATION:
        # Generate a random angle between -ROTATION_ANGLE and +ROTATION_ANGLE
        angle = np.random.uniform(-ROTATION_ANGLE, ROTATION_ANGLE)
        
        # Calculate rotation matrix
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.2) 
        
        # Apply rotation
        image = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        
    return image

def apply_low_light(image, gamma, noise_sigma):
    """
    Applies gamma darkening and gaussian noise.
    """
    # 1. Gamma Correction (Darkening)
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 
                              for i in np.arange(0, 256)]).astype("uint8")
    dark_img = cv2.LUT(image, look_up_table)

    # 2. Gaussian Noise (Sensor Grain)
    row, col, ch = dark_img.shape
    mean = 0
    gauss = np.random.normal(mean, noise_sigma, (row, col, ch))
    
    noisy_img = dark_img.astype('float32') + gauss
    noisy_img = np.clip(noisy_img, 0, 255).astype('uint8')
    
    return noisy_img

def main():
    if not os.path.exists(SOURCE_ROOT):
        print(f"Error: Source directory '{SOURCE_ROOT}' not found.")
        return

    classes = [d for d in os.listdir(SOURCE_ROOT) 
               if os.path.isdir(os.path.join(SOURCE_ROOT, d))]
    
    print(f"Processing {len(classes)} classes...")

    for class_name in classes:
        src_class_path = os.path.join(SOURCE_ROOT, class_name)
        dest_class_path = os.path.join(DEST_ROOT, class_name)
        os.makedirs(dest_class_path, exist_ok=True)

        images = [f for f in os.listdir(src_class_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in tqdm(images, desc=class_name):
            img_path = os.path.join(src_class_path, img_name)
            img = cv2.imread(img_path)

            if img is None: continue

            # STEP 1: Geometric Distortion (Flip/Rotate)
            geo_img = apply_geometric_distortion(img)

            # STEP 2: Low Light Distortion
            final_img = apply_low_light(geo_img, GAMMA, NOISE_SIGMA)

            # Save
            save_path = os.path.join(dest_class_path, img_name)
            cv2.imwrite(save_path, final_img)

    print(f"\nDone! Distorted dataset generated at: {DEST_ROOT}")

if __name__ == "__main__":
    main()