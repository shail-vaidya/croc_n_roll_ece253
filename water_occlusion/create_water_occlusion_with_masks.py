import cv2
import numpy as np
import os
import random
from tqdm import tqdm

# ================= CONFIGURATION =================
SOURCE_ROOT = "../ImageNet_images"
DEST_ROOT = "../Distorted_Images/Water_Occlusion"
MASK_ROOT = "../Distorted_Images/Water_Masks"

# 1. Water Droplet Parameters (INCREASED SEVERITY)
# We increased these values to ensure the Baseline Accuracy (R') drops significantly.
NUM_DROPS = 40            # (More coverage)
MIN_RADIUS = 15           # 
MAX_RADIUS = 50           # (Larger distorted areas)
REFRACTION_STRENGTH = 25  #  (Pixels warp heavily)

# 2. Geometric Parameters (Must match Low-Light script logic)
ENABLE_FLIP = True      
ENABLE_ROTATION = True  
ROTATION_ANGLE = 15     
# =================================================

def apply_geometric_distortion(image):
    """
    Flips and rotates the image to break spatial alignment with training data.
    """
    height, width = image.shape[:2]
    
    # 1. Horizontal Flip
    if ENABLE_FLIP:
        image = cv2.flip(image, 1)

    # 2. Rotation
    if ENABLE_ROTATION:
        angle = np.random.uniform(-ROTATION_ANGLE, ROTATION_ANGLE)
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.2) 
        image = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        
    return image

def apply_water_occlusion(image):
    """
    Simulates water droplets via displacement mapping and returns the image + mask.
    """
    h, w = image.shape[:2]
    
    # 1. Create Buffers
    height_map = np.zeros((h, w), dtype=np.float32)
    binary_mask = np.zeros((h, w), dtype=np.uint8) # Ground Truth Mask
    
    # 2. Generate Random Droplets
    for _ in range(NUM_DROPS):
        x = random.randint(0, w)
        y = random.randint(0, h)
        r = random.randint(MIN_RADIUS, MAX_RADIUS)
        
        # A. Draw on Binary Mask (Solid White for Ground Truth)
        cv2.circle(binary_mask, (x, y), r, 255, -1)
        
        # B. Draw on Height Map (Gradient Sphere for Physics)
        y_grid, x_grid = np.ogrid[-r:r, -r:r]
        # Equation of sphere: z = sqrt(r^2 - x^2 - y^2)
        drop_shape = np.maximum(0, r**2 - x_grid**2 - y_grid**2)
        drop_shape = np.sqrt(drop_shape)
        
        # Normalize and Scale by Strength
        drop_shape = (drop_shape / r) * REFRACTION_STRENGTH * 10
        
        # Place on map (handle boundaries)
        y1, y2 = max(0, y-r), min(h, y+r)
        x1, x2 = max(0, x-r), min(w, x+r)
        
        # Crop drop shape if it goes off edge
        drop_h_slice = drop_shape[max(0, r-y):min(2*r, r+(h-y)), 
                                  max(0, r-x):min(2*r, r+(w-x))]
        
        # Merge into height map
        current_area = height_map[y1:y2, x1:x2]
        height_map[y1:y2, x1:x2] = np.maximum(current_area, drop_h_slice)

    # 3. Calculate Gradients (The 'Slope' of the water)
    grad_x = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=3)

    # 4. Create Displacement Map (Refraction)
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + grad_x).astype(np.float32)
    map_y = (grid_y + grad_y).astype(np.float32)

    # 5. Apply Remap
    refracted_img = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    # 6. Composite (Blur the wet parts)
    blurred_img = cv2.GaussianBlur(refracted_img, (5, 5), 0)
    
    # Normalize mask to 0-1 for math
    mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR) / 255.0
    
    # Final = (Wet_Blur * Mask) + (Dry_Original * (1-Mask))
    final_img = (blurred_img * mask_3ch + image * (1.0 - mask_3ch))
    
    # 7. Add Specular Highlights (White noise on droplets)
    noise = np.random.normal(0, 20, image.shape).astype(np.float32)
    final_img = final_img + (noise * mask_3ch * 0.4) # Add mild noise to wet parts
    
    return np.clip(final_img, 0, 255).astype(np.uint8), binary_mask

def main():
    if not os.path.exists(SOURCE_ROOT):
        print(f"Error: Source directory '{SOURCE_ROOT}' not found.")
        return

    classes = [d for d in os.listdir(SOURCE_ROOT) 
               if os.path.isdir(os.path.join(SOURCE_ROOT, d))]
    
    print(f"Processing {len(classes)} classes for Heavy Water Occlusion...")

    for class_name in classes:
        src_class_path = os.path.join(SOURCE_ROOT, class_name)
        
        # Destination Paths
        dest_class_path = os.path.join(DEST_ROOT, class_name)
        mask_class_path = os.path.join(MASK_ROOT, class_name)
        
        os.makedirs(dest_class_path, exist_ok=True)
        os.makedirs(mask_class_path, exist_ok=True)

        images = [f for f in os.listdir(src_class_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in tqdm(images, desc=class_name):
            img_path = os.path.join(src_class_path, img_name)
            img = cv2.imread(img_path)

            if img is None: continue

            # STEP 1: Geometric Distortion
            geo_img = apply_geometric_distortion(img)

            # STEP 2: Water Occlusion + Mask Generation
            final_img, mask = apply_water_occlusion(geo_img)

            # STEP 3: Save Both
            cv2.imwrite(os.path.join(dest_class_path, img_name), final_img)
            cv2.imwrite(os.path.join(mask_class_path, img_name), mask)

    print(f"\nDone! Distorted images at: {DEST_ROOT}")
    print(f"Ground Truth masks at: {MASK_ROOT}")

if __name__ == "__main__":
    main()