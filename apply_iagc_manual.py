import cv2
import numpy as np
import os
import time
from tqdm import tqdm

# ================= CONFIGURATION =================
SOURCE_ROOT = "./Distorted_Images/Low_Light"
DEST_ROOT   = "./Processed_Images/Low_Light/IAGC_LAB"
# =================================================

# --- YOUR IMPLEMENTATION (From iagc_lab.py) ---
def calculate_adaptive_gamma_lab(mean_intensity):
    """
    Calculate an adaptive gamma value based on the mean intensity of the image.
    """
    k = 2.0  # Sensitivity factor
    gamma = 1 + k * (0.5 - mean_intensity / 255.0)  # Scale mean intensity to [0, 1]
    return max(0.5, min(2.0, gamma))  # Clamp gamma between 0.5 and 2.0

def iagc_lab(image):
    """
    Apply Improved Adaptive Gamma Correction (IAGC) to a color image.
    """
    # Convert to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)

    # Calculate mean intensity of the L channel
    mean_intensity = np.mean(l)

    # Determine adaptive gamma value
    gamma = calculate_adaptive_gamma_lab(mean_intensity)

    if mean_intensity > 128:  # Bright image
        # Apply gamma correction to the negative of the L channel
        negative_l = 255 - l
        normalized_negative_l = negative_l / 255.0
        gamma_corrected_l = np.power(normalized_negative_l, gamma)
        enhanced_l = (255 - (gamma_corrected_l * 255)).astype(np.uint8)
    else:  # Dim image
        # Apply gamma correction directly to the L channel
        normalized_l = l / 255.0
        gamma_corrected_l = np.power(normalized_l, gamma)
        enhanced_l = (gamma_corrected_l * 255).astype(np.uint8)

    # Merge enhanced L channel back with original A and B channels
    enhanced_lab = cv2.merge((enhanced_l, a, b))

    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_image

# --- EXECUTION LOOP ---
def main():
    if not os.path.exists(SOURCE_ROOT):
        print(f"Error: Source directory '{SOURCE_ROOT}' not found.")
        return

    classes = [d for d in os.listdir(SOURCE_ROOT) if os.path.isdir(os.path.join(SOURCE_ROOT, d))]
    print(f"Processing {len(classes)} classes with IAGC (LAB)...")

    total_processed = 0

    for class_name in classes:
        src_path = os.path.join(SOURCE_ROOT, class_name)
        dst_path = os.path.join(DEST_ROOT, class_name)
        os.makedirs(dst_path, exist_ok=True)

        images = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_name in tqdm(images, desc=class_name):
            img = cv2.imread(os.path.join(src_path, img_name))
            if img is None: continue
            
            # Apply Algorithm
            try:
                out = iagc_lab(img)
                cv2.imwrite(os.path.join(dst_path, img_name), out)
                total_processed += 1
            except Exception as e:
                print(f"Error: {e}")

    print("\n" + "="*40)
    print(f"IAGC Processing Complete.")
    print(f"Total: {total_processed}")
    print(f"Saved to: {DEST_ROOT}")
    print("="*40)

if __name__ == "__main__":
    main()