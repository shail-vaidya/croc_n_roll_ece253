import cv2
import numpy as np
import os
import time
from tqdm import tqdm

# ================= CONFIGURATION =================
SOURCE_ROOT = "./Distorted_Images/Low_Light"
DEST_ROOT   = "./Processed_Images/Low_Light/Manual_CLAHE"

# Manual CLAHE Parameters
# Note: In your manual code, clip_limit is absolute pixels (e.g., 40), 
# whereas OpenCV uses a slope limit (e.g., 2.0). 
# A value of 40 is a good starting point for manual block-based CLAHE.
CLIP_LIMIT = 40      
TILE_SIZE = (8, 8)   
# =================================================

# --- 1. PASTE YOUR MANUAL IMPLEMENTATION FUNCTIONS HERE ---

def clip_histogram(hist, clip_limit):
    """
    Clip the histogram and redistribute the clipped pixels.
    """
    excess = np.maximum(hist - clip_limit, 0)
    clipped_hist = hist - excess
    total_excess = np.sum(excess)

    # Redistribute excess pixels equally among all bins
    clipped_hist += total_excess // len(hist)
    return clipped_hist

def equalize_tile(tile, clip_limit):
    """
    Apply histogram equalization to a single tile.
    """
    # Compute the histogram
    hist, bins = np.histogram(tile.flatten(), bins=256, range=(0, 256))

    # Clip the histogram
    clipped_hist = clip_histogram(hist, clip_limit)

    # Compute the cumulative distribution function (CDF)
    cdf = np.cumsum(clipped_hist)
    
    # Avoid divide by zero
    if cdf[-1] == 0:
        cdf_normalized = cdf # Should be all zeros
    else:
        cdf_normalized = cdf * 255 / cdf[-1]  # Normalize CDF to [0, 255]

    # Map the pixel values using the CDF
    equalized_tile = np.interp(tile.flatten(), bins[:-1], cdf_normalized)
    return equalized_tile.reshape(tile.shape)

def manual_clahe_gray(image, tile_size=(8, 8), clip_limit=40):
    """
    Apply CLAHE manually to a grayscale image (Block Processing).
    """
    h, w = image.shape
    tile_h, tile_w = tile_size

    # Handle cases where image isn't perfectly divisible by tile size
    # We crop slightly for simplicity in this implementation
    h_trim = (h // tile_h) * tile_h
    w_trim = (w // tile_w) * tile_w
    
    image = image[:h_trim, :w_trim]
    output_image = np.zeros_like(image, dtype=np.uint8)

    n_tiles_h = h_trim // tile_h
    n_tiles_w = w_trim // tile_w

    # Process each tile
    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            # Extract the tile
            y_start, y_end = i * tile_h, (i + 1) * tile_h
            x_start, x_end = j * tile_w, (j + 1) * tile_w
            
            tile = image[y_start:y_end, x_start:x_end]

            # Equalize the tile
            equalized_tile = equalize_tile(tile, clip_limit)

            # Place back
            output_image[y_start:y_end, x_start:x_end] = equalized_tile

    return output_image

def manual_clahe_color(image, tile_size, clip_limit):
    """
    Apply manual CLAHE to a color image by processing the L-channel.
    """
    # Convert to LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    
    # Store original shape to restore later (if trimming happened)
    orig_h, orig_w = l.shape[:2]

    # Apply Manual CLAHE to L channel
    # Note: This function might trim the image slightly to fit tiles
    enhanced_l = manual_clahe_gray(l, tile_size, clip_limit)
    
    # Resize a and b to match new trimmed L size
    new_h, new_w = enhanced_l.shape[:2]
    a = a[:new_h, :new_w]
    b = b[:new_h, :new_w]
    
    # Merge back
    enhanced_lab = cv2.merge((enhanced_l, a, b))
    
    # Convert back to BGR
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Resize back to original if needed (optional, but good for CNN consistency)
    if (new_h != orig_h) or (new_w != orig_w):
        enhanced_image = cv2.resize(enhanced_image, (orig_w, orig_h))
        
    return enhanced_image

# --- 2. MAIN EXECUTION LOOP ---

def main():
    if not os.path.exists(SOURCE_ROOT):
        print(f"Error: Source directory '{SOURCE_ROOT}' not found.")
        return

    classes = [d for d in os.listdir(SOURCE_ROOT) 
               if os.path.isdir(os.path.join(SOURCE_ROOT, d))]
    
    print(f"Processing {len(classes)} classes with Manual CLAHE...")
    print(f"Params: Tile={TILE_SIZE}, ClipLimit={CLIP_LIMIT}")

    total_processed = 0

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

            # Apply Manual Implementation
            # We wrap in try/except in case of weird dimension errors
            try:
                final_img = manual_clahe_color(img, TILE_SIZE, CLIP_LIMIT)
                
                # Save
                save_path = os.path.join(dest_class_path, img_name)
                cv2.imwrite(save_path, final_img)
                total_processed += 1
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    print("\n" + "="*40)
    print("Manual Processing Complete.")
    print(f"Total Images: {total_processed}")
    print(f"Saved to: {DEST_ROOT}")
    print("="*40)

if __name__ == "__main__":
    main()