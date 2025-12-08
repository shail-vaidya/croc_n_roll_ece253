import cv2
import numpy as np
import os
from tqdm import tqdm

# ================= CONFIGURATION =================
# Use Preprocessed images (224x224)
SOURCE_ROOT = "../Self_capture_images_Preprocessed/Water Occlusion"
DEST_ROOT   = "../Processed_Images/Water/Self_Captured_Restored"

INPAINT_RADIUS = 9  # As requested
METHOD = cv2.INPAINT_NS
# =================================================

def calculate_local_sharpness(image, ksize=15):
    """
    Calculates local sharpness/texture using Laplacian Variance.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    mean = cv2.blur(laplacian, (ksize, ksize))
    sq_mean = cv2.blur(laplacian**2, (ksize, ksize))
    variance = sq_mean - mean**2
    
    variance = np.clip(variance, 0, 25500)
    variance = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return variance

def generate_mask(image):
    """
    Simplified mask generation: Brightness + Lack of Sharpness.
    """
    h, w = image.shape[:2]
    
    # 1. Brightness Detection
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    
    blur_l = cv2.GaussianBlur(l_channel, (11, 11), 0)
    thresh_val, _ = cv2.threshold(blur_l, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_bright = cv2.threshold(blur_l, thresh_val + 20, 255, cv2.THRESH_BINARY)

    # 2. Sharpness Detection (Inverted)
    sharpness_map = calculate_local_sharpness(image)
    _, mask_blurry = cv2.threshold(sharpness_map, 50, 255, cv2.THRESH_BINARY_INV)
    
    # 3. Combine Signals
    combined = cv2.bitwise_and(mask_bright, mask_blurry)
    
    # 4. Filter & Clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(cleaned)
    total_area = w * h
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (total_area * 0.001) or area > (total_area * 0.3): continue
        
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        if circularity > 0.3:
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)

    return final_mask

def main():
    if not os.path.exists(SOURCE_ROOT):
        print(f"Error: Source {SOURCE_ROOT} not found.")
        print("Run 'preprocess_real_images.py' first!")
        return

    classes = [d for d in os.listdir(SOURCE_ROOT) if os.path.isdir(os.path.join(SOURCE_ROOT, d))]
    print(f"Processing {len(classes)} classes (Targeted V8)...")

    for class_name in classes:
        src_path = os.path.join(SOURCE_ROOT, class_name)
        dst_path = os.path.join(DEST_ROOT, class_name)
        
        os.makedirs(dst_path, exist_ok=True)

        images = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in tqdm(images, desc=class_name):
            img_path = os.path.join(src_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None: continue
            
            # 1. Detect
            mask = generate_mask(img)
            
            # 2. Inpaint
            if np.count_nonzero(mask) > 0:
                result = cv2.inpaint(img, mask, INPAINT_RADIUS, METHOD)
            else:
                result = img
            
            cv2.imwrite(os.path.join(dst_path, img_name), result)

    print(f"Done. Images at: {DEST_ROOT}")

if __name__ == "__main__":
    main()