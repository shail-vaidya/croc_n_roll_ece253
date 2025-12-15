import os
import shutil
import cv2
import numpy as np
from icrawler.builtin import BingImageCrawler
from PIL import Image

# ================= CONFIGURATION =================
# Where to save the final distorted images
DEST_FOLDER = "./sports car"

# Search Query
QUERY = "sports car at night"  # Adding 'at night' helps get naturally dark bases too
NUM_IMAGES = 20

# Standard processing size
TARGET_SIZE = 256  # Resize shortest side to this

# Distortion Parameters (Matching generate_low_light_v2.py)
GAMMA = 4.0
NOISE_SIGMA = 20
ENABLE_ROTATION = True
ROTATION_ANGLE = 15 
# =================================================

def resize_maintain_aspect(img_pil, target_short):
    """Resizes PIL image so shortest side is target_short."""
    w, h = img_pil.size
    if w < h:
        new_w = target_short
        new_h = int(h * (target_short / w))
    else:
        new_h = target_short
        new_w = int(w * (target_short / h))
    return img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

def apply_geometric_distortion(image):
    """Applies Flip and Rotation (Matching previous script)."""
    height, width = image.shape[:2]
    
    # 1. Horizontal Flip
    image = cv2.flip(image, 1)

    # 2. Rotation
    if ENABLE_ROTATION:
        angle = np.random.uniform(-ROTATION_ANGLE, ROTATION_ANGLE)
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.2) 
        image = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        
    return image

def apply_low_light(image, gamma, noise_sigma):
    """Applies Gamma and Noise (Matching previous script)."""
    # 1. Gamma Darkening
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 
                              for i in np.arange(0, 256)]).astype("uint8")
    dark_img = cv2.LUT(image, look_up_table)

    # 2. Gaussian Noise
    row, col, ch = dark_img.shape
    mean = 0
    gauss = np.random.normal(mean, noise_sigma, (row, col, ch))
    
    noisy_img = dark_img.astype('float32') + gauss
    noisy_img = np.clip(noisy_img, 0, 255).astype('uint8')
    
    return noisy_img

def main():
    download_dir = "./Temp_Downloads_Cars"
    
    # 1. Clean up previous temp dir if it exists
    if os.path.exists(download_dir):
        shutil.rmtree(download_dir)
        
    # 2. Download Images using icrawler (Fixes the Path error)
    print(f"Downloading {NUM_IMAGES} images for '{QUERY}'...")
    
    crawler = BingImageCrawler(storage={'root_dir': download_dir})
    # We ask for a few extra (30) in case some are corrupt
    crawler.crawl(keyword=QUERY, max_num=NUM_IMAGES + 10)
    
    # 3. Process Each Image
    print("\nProcessing and Distorting...")
    os.makedirs(DEST_FOLDER, exist_ok=True)
    
    downloaded_files = [f for f in os.listdir(download_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    count = 0
    for fname in downloaded_files:
        if count >= NUM_IMAGES:
            break
            
        try:
            # A. Open with PIL to resize standardly
            src_path = os.path.join(download_dir, fname)
            img_pil = Image.open(src_path).convert("RGB")
            img_pil = resize_maintain_aspect(img_pil, TARGET_SIZE)
            
            # Convert to OpenCV format (numpy array, BGR)
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            # B. Apply Distortions
            img_geo = apply_geometric_distortion(img_cv)
            img_final = apply_low_light(img_geo, GAMMA, NOISE_SIGMA)
            
            # C. Save to Dataset
            # Use a unique name to avoid overwriting existing dataset images
            save_name = f"manual_sportscar_{count}.jpg"
            cv2.imwrite(os.path.join(DEST_FOLDER, save_name), img_final)
            
            count += 1
            print(f"Added: {save_name}")
            
        except Exception as e:
            print(f"Skipped {fname}: {e}")

    # 4. Cleanup
    if os.path.exists(download_dir):
        shutil.rmtree(download_dir)
        
    print("\n" + "="*40)
    print(f"Done! Added {count} new distorted sports cars.")
    print(f"Location: {DEST_FOLDER}")
    print("="*40)

if __name__ == "__main__":
    main()