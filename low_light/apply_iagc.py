import cv2
import numpy as np
import os
from tqdm import tqdm

# ================= CONFIGURATION =================
SOURCE_ROOT = "../Distorted_Images/Low_Light"
DEST_ROOT = "../Processed_Images/Low_Light/IAGC"
# =================================================

def apply_iagc_enhancement(image):
    """
    Applies Improved Adaptive Gamma Correction (IAGC).
    Strategy:
    1. Extract Value (Intensity) channel.
    2. Compute the Cumulative Distribution Function (CDF).
    3. Use CDF to define a per-pixel Gamma: Gamma = 1 - CDF.
    4. Apply Power Law Transformation: V_out = V_in ^ Gamma.
    """
    # 1. Convert BGR to HSV to isolate intensity (Value)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Work with normalized float values (0.0 - 1.0)
    v_norm = v.astype(np.float32) / 255.0
    
    # 2. Compute Histogram and CDF
    # We compute the histogram of the Value channel
    hist, bins = np.histogram(v_norm.flatten(), 256, [0, 1])
    
    # "Improved" Step: Weighting/Clipping strategy
    # To prevent washing out images, we can smoothen the PDF (Probability Density Function)
    pdf = hist / hist.sum()
    cdf = pdf.cumsum()
    
    # 3. Create the Adaptive Gamma Lookup Table
    # We map every possible pixel value (0-255) to a specific Gamma based on its rank (CDF)
    # Formula: V_out = V_in ^ (1 - CDF(V_in))
    # Note: We apply a slight weight (0.5) to keep it realistic, otherwise it might over-brighten
    
    gamma_table = np.zeros(256, dtype=np.float32)
    for i in range(256):
        # Current normalized intensity
        intensity = i / 255.0
        
        # Adaptive Gamma value for this intensity level
        # A simple robust formula: gamma = 1 - cdf[i]
        # If CDF is low (dark), gamma is high (~1.0).
        # If CDF is high (bright), gamma is low. 
        current_gamma = 1.0 - cdf[i]
        
        # Clamp gamma to avoid extreme inversion
        current_gamma = np.clip(current_gamma, 0.1, 1.0)
        
        # Apply transformation
        gamma_table[i] = (intensity ** current_gamma) * 255.0
        
    # 4. Apply the Lookup Table (LUT) to the V channel
    gamma_table = gamma_table.astype(np.uint8)
    v_enhanced = cv2.LUT(v, gamma_table)
    
    # 5. Merge and convert back
    hsv_enhanced = cv2.merge((h, s, v_enhanced))
    final_img = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    
    return final_img

def main():
    if not os.path.exists(SOURCE_ROOT):
        print(f"Error: Source directory '{SOURCE_ROOT}' not found.")
        return

    # Get class list
    classes = [d for d in os.listdir(SOURCE_ROOT) 
               if os.path.isdir(os.path.join(SOURCE_ROOT, d))]
    
    print(f"Found {len(classes)} classes to process with IAGC...")

    total_processed = 0

    for class_name in classes:
        src_class_path = os.path.join(SOURCE_ROOT, class_name)
        dest_class_path = os.path.join(DEST_ROOT, class_name)

        os.makedirs(dest_class_path, exist_ok=True)

        images = [f for f in os.listdir(src_class_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in tqdm(images, desc=f"IAGC on {class_name}"):
            img_path = os.path.join(src_class_path, img_name)
            img = cv2.imread(img_path)

            if img is None: continue

            # Apply IAGC
            enhanced_img = apply_iagc_enhancement(img)

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