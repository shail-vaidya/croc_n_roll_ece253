import os
from PIL import Image
import pillow_heif
from tqdm import tqdm

#Register HEIF opener with Pillow
pillow_heif.register_heif_opener()

# ================= CONFIGURATION =================
# Path to your folder of raw iPhone images
SOURCE_FOLDER = "./HEIC" 

# Where to save the converted versions
DEST_FOLDER = "./JPG"

# Standard ImageNet pre-processing size (shortest side)
TARGET_SHORT_SIDE = 256
# =================================================

def resize_maintain_aspect(img, target_short):
    """Resizes image so shortest side is target_short, maintaining aspect ratio."""
    w, h = img.size
    
    if w < h:
        new_w = target_short
        new_h = int(h * (target_short / w))
    else:
        new_h = target_short
        new_w = int(w * (target_short / h))
        
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

def main():
    if not os.path.exists(SOURCE_FOLDER):
        print(f"Error: Source folder '{SOURCE_FOLDER}' does not exist.")
        return

    os.makedirs(DEST_FOLDER, exist_ok=True)
    
    files = [f for f in os.listdir(SOURCE_FOLDER) if f.lower().endswith('.heic')]
    print(f"Found {len(files)} HEIC images to convert...")

    for filename in tqdm(files):
        try:
            # 1. Open HEIC
            src_path = os.path.join(SOURCE_FOLDER, filename)
            img = Image.open(src_path)
            
            # 2. Convert to RGB (removing Alpha channel if present)
            img = img.convert("RGB")
            
            # 3. Resize (Standardize)
            img = resize_maintain_aspect(img, TARGET_SHORT_SIDE)
            
            # 4. Save as JPG
            # Remove extension and add .jpg
            new_filename = os.path.splitext(filename)[0] + ".jpg"
            dest_path = os.path.join(DEST_FOLDER, new_filename)
            
            img.save(dest_path, "JPEG", quality=90)
            
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")

    print("\n" + "="*40)
    print("Conversion Complete!")
    print(f"Converted images saved to: {DEST_FOLDER}")
    print("="*40)

if __name__ == "__main__":
    main()