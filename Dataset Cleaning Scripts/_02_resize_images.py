import os
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm  

# --- Configuration ---
# 1. SET YOUR SOURCE FOLDER: The balanced dataset you just created
SOURCE_DIR = Path("_02_Balanced Dataset")

# 2. SET YOUR DESTINATION FOLDER: Where the new 256x256 images will go
DEST_DIR = Path("_03_Normalized Dataset")

# 3. SET THE TARGET SIZE
TARGET_SIZE = (256, 256)
# ---------------------

# Handle modern vs. older PIL/Pillow versions for resampling
if hasattr(Image, 'Resampling'):
    RESIZE_FILTER = Image.Resampling.LANCZOS
else:
    RESIZE_FILTER = Image.LANCZOS # This is for older versions

def process_images(source_dir, dest_dir, target_size):
    """
    Walks a source directory, resizes or copies images to a destination directory,
    mirroring the folder structure and showing a progress bar.
    """
    
    # Create the main destination folder if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- Starting Image Processing ---")
    print(f"Source: {source_dir.resolve()}")
    print(f"Destination: {dest_dir.resolve()}")
    print(f"Target Size: {target_size}\n")
    
    # --- 1. PRE-SCAN: Count all image files first for the progress bar ---
    print("Pre-scanning to count total images...")
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    # Use rglob to find all files, then filter
    all_files = list(source_dir.rglob('*.*'))
    image_paths = [f for f in all_files if f.suffix.lower() in image_extensions]
    total_files = len(image_paths)
    print(f"Found {total_files} images to process.")
    # -----------------------------------------------------------------
    
    copied_count = 0
    resized_count = 0
    failed_count = 0

    # --- 2. MAIN PROCESSING LOOP with tqdm ---
    with tqdm(total=total_files, unit="image", desc="Overall Progress") as pbar:
        # Loop 1: Iterate through each "Crop Folder"
        for crop_folder in source_dir.iterdir():
            if not crop_folder.is_dir():
                continue
            
            dest_crop_path = dest_dir / crop_folder.name
            dest_crop_path.mkdir(exist_ok=True)

            # Loop 2: Iterate through each "Disease Folder"
            for disease_folder in crop_folder.iterdir():
                if not disease_folder.is_dir():
                    continue
                
                # Update the progress bar description
                pbar.set_description(f"Processing: {crop_folder.name} / {disease_folder.name}")
                dest_disease_path = dest_crop_path / disease_folder.name
                dest_disease_path.mkdir(exist_ok=True)

                # Loop 3: Process each file
                for file_path in disease_folder.glob('*'):
                    # Only process files with the right extensions
                    if not file_path.is_file() or file_path.suffix.lower() not in image_extensions:
                        continue
                    
                    dest_file_path = dest_disease_path / file_path.name
                    
                    try:
                        # Open the image to check its size
                        with Image.open(file_path) as img:
                            
                            # --- The Core Logic ---
                            if img.size == target_size and img.mode == "RGB":
                                # 1. Already perfect size. Just copy.
                                shutil.copy2(file_path, dest_file_path)
                                copied_count += 1
                            else:
                                # 2. Needs resizing and/or conversion.
                                resized_img = img.resize(target_size, RESIZE_FILTER)
                                if resized_img.mode != "RGB":
                                    resized_img = resized_img.convert("RGB")
                                
                                # Save as a new, high-quality JPEG
                                dest_file_path = dest_file_path.with_suffix(".jpg")
                                resized_img.save(dest_file_path, "JPEG", quality=95)
                                resized_count += 1
                    
                    except Exception as e:
                        # This catches corrupted images
                        pbar.write(f"\n[!] FAILED to process {file_path.name}: {e}")
                        failed_count += 1
                    
                    # --- 3. UPDATE PROGRESS BAR ---
                    pbar.update(1)

    # --- 4. FINAL SUMMARY ---
    print("\n--- Image Processing Complete! ---")
    print(f"Total images processed: {total_files}")
    print(f"Images copied (fast):   {copied_count}")
    print(f"Images resized (slow):  {resized_count}")
    print(f"Images failed (corrupt):  {failed_count}")
    print(f"Your 'Final_Dataset' is ready at: {dest_dir.resolve()}")

# --- Run the script ---
if __name__ == "__main__":
    if not SOURCE_DIR.exists():
        print(f"ERROR: Source folder not found at: {SOURCE_DIR.resolve()}")
    else:
        process_images(SOURCE_DIR, DEST_DIR, TARGET_SIZE)