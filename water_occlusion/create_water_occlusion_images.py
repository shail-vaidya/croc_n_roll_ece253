import cv2
import numpy as np
import os
import random
from tqdm import tqdm

# ================= CONFIGURATION =================
SOURCE_ROOT = "../ImageNet_images"
DEST_ROOT = "../Distorted_Images/Water_Occlusion"

# 1. Water Droplet Parameters
NUM_DROPS = 40          # Number of droplets to generate per image
MIN_RADIUS = 10         # Minimum size of a droplet
MAX_RADIUS = 40         # Maximum size of a droplet
REFRACTION_STRENGTH = 25 # How much the light bends (pixel displacement)

# 2. Geometric Parameters (CRITICAL: Must match low-light logic)
ENABLE_FLIP = True      
ENABLE_ROTATION = True  
ROTATION_ANGLE = 15     
# =================================================

def apply_geometric_distortion(image):
    """
    Flips and rotates the image to break spatial alignment with training data.
    Same as in the low-light script.
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

def apply_water_occlusion(image, num_drops, min_r, max_r, strength):
    """
    Simulates water droplets on a lens using displacement mapping (refraction).
    """
    h, w = image.shape[:2]
    
    # 1. Create a "Height Map" for the droplets
    # We start with a flat surface (black)
    height_map = np.zeros((h, w), dtype=np.float32)
    
    # 2. Draw random droplets (blobs) onto the height map
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for _ in range(num_drops):
        # Random position and size
        x = random.randint(0, w)
        y = random.randint(0, h)
        r = random.randint(min_r, max_r)
        
        # Draw a solid white circle on the mask (to know where drops are)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # Draw a "gradient" circle on the height map to simulate the curve of a drop
        # We create a small grid for the drop
        y_grid, x_grid = np.ogrid[-r:r, -r:r]
        # Equation of a sphere: z = sqrt(r^2 - x^2 - y^2)
        # We clip it to avoid sqrt of negative numbers for the corners of the grid
        drop_shape = np.maximum(0, r**2 - x_grid**2 - y_grid**2)
        drop_shape = np.sqrt(drop_shape)
        
        # Normalize drop height to be reasonable
        drop_shape = (drop_shape / r) * strength * 10
        
        # Place this drop onto the height map (handling boundary checks)
        y1, y2 = max(0, y-r), min(h, y+r)
        x1, x2 = max(0, x-r), min(w, x+r)
        
        # Extract the relevant part of the drop shape
        drop_h_slice = drop_shape[max(0, r-y):min(2*r, r+(h-y)), 
                                  max(0, r-x):min(2*r, r+(w-x))]
        
        # Add to height map (using maximum to simulate merging droplets)
        current_area = height_map[y1:y2, x1:x2]
        height_map[y1:y2, x1:x2] = np.maximum(current_area, drop_h_slice)

    # 3. Calculate Gradients (The slope of the water surface)
    # The light bends based on the ANGLE of the surface, which is the gradient.
    # gradient_x determines horizontal shift, gradient_y determines vertical.
    grad_x = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=3)

    # 4. Create Displacement Map
    # We want to pull pixels from coordinates (x + grad_x, y + grad_y)
    # Create the grid of X and Y coordinates
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Add the gradients to the grid
    map_x = (grid_x + grad_x).astype(np.float32)
    map_y = (grid_y + grad_y).astype(np.float32)

    # 5. Apply Refraction (Remapping)
    refracted_img = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    # 6. Apply Blur and Specular Highlights
    # Water makes things blurry. We blur the WHOLE image, then use the mask 
    # to combine the blurry wet parts with the sharp dry parts.
    blurred_img = cv2.GaussianBlur(refracted_img, (5, 5), 0)
    
    # Composite: Where mask is white (water), use blurry refracted image. 
    # Where mask is black (dry), use original geometric_distorted image.
    # Note: Realistically, the "dry" part shouldn't be refracted either. 
    # Let's use the refracted image for wet parts and original for dry.
    
    # Convert mask to 3-channel for compositing
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    
    # Combine: (Wet * Mask) + (Dry * (1 - Mask))
    # We use the refracted_blurred image for the wet parts
    final_img = (blurred_img * mask_3ch + image * (1.0 - mask_3ch))

    # 7. Add Specular Highlights (White glints)
    # We can approximate this by thresholding the height map or gradients
    # Simplified: Add weak white noise only on the droplets to simulate surface reflection
    noise = np.random.normal(0, 10, image.shape).astype(np.float32)
    final_img = final_img + (noise * mask_3ch * 0.5)
    
    return np.clip(final_img, 0, 255).astype(np.uint8)

def main():
    if not os.path.exists(SOURCE_ROOT):
        print(f"Error: Source directory '{SOURCE_ROOT}' not found.")
        return

    classes = [d for d in os.listdir(SOURCE_ROOT) 
               if os.path.isdir(os.path.join(SOURCE_ROOT, d))]
    
    print(f"Processing {len(classes)} classes for Water Occlusion...")

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
            # We MUST do this to keep it consistent with the low-light dataset logic
            geo_img = apply_geometric_distortion(img)

            # STEP 2: Water Occlusion
            final_img = apply_water_occlusion(geo_img, NUM_DROPS, MIN_RADIUS, MAX_RADIUS, REFRACTION_STRENGTH)

            # Save
            save_path = os.path.join(dest_class_path, img_name)
            cv2.imwrite(save_path, final_img)

    print(f"\nDone! Distorted dataset generated at: {DEST_ROOT}")

if __name__ == "__main__":
    main()