import os
import shutil
import random
import math
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
SOURCE_DIR = Path("_03_Normalized Dataset")

DEST_DIR = Path("_04_Split_Dataset")

# 3. SET THE SPLIT RATIOS
SPLIT_RATIOS = (0.7, 0.15, 0.15)
# ---------------------

def split_dataset(source_dir, dest_dir, ratios):
    """
    Splits a dataset of a nested structure (Crop/Disease/images)
    into train, val, and test sets.
    """
    
    # --- 1. Create main destination and train/val/test folders ---
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        train_path = dest_dir / "train"
        val_path = dest_dir / "val"
        test_path = dest_dir / "test"
        
        train_path.mkdir(exist_ok=True)
        val_path.mkdir(exist_ok=True)
        test_path.mkdir(exist_ok=True)
        
    except OSError as e:
        print(f"Error creating base directories: {e}")
        return

    print(f"--- Starting Dataset Split ---")
    print(f"Source: {source_dir.resolve()}")
    print(f"Destination: {dest_dir.resolve()}")
    print(f"Ratios (Train/Val/Test): {ratios}\n")

    print("Pre-scanning to find all class folders...")
    class_folders_to_process = []
    for crop_folder in source_dir.iterdir():
        if not crop_folder.is_dir(): continue
        for disease_folder in crop_folder.iterdir():
            if disease_folder.is_dir():
                relative_path = Path(crop_folder.name) / disease_folder.name
                class_folders_to_process.append(relative_path)
    
    total_classes = len(class_folders_to_process)
    if total_classes == 0:
        print("Error: No class folders found in source directory.")
        return
        
    print(f"Found {total_classes} class folders to split.")

    with tqdm(total=total_classes, unit="class", desc="Splitting Dataset") as pbar:
        for relative_path in class_folders_to_process:
            
            source_class_path = source_dir / relative_path
            pbar.set_description(f"Splitting: {relative_path}")

            all_files = [f for f in source_class_path.glob('*') if f.is_file()]
            random.shuffle(all_files) 
            
            file_count = len(all_files)
            if file_count == 0:
                pbar.write(f"[!] Warning: No files found in {source_class_path}. Skipping.")
                pbar.update(1)
                continue

            train_count = math.floor(file_count * ratios[0])
            val_count = math.floor(file_count * ratios[1])
            
            train_split_idx = train_count
            val_split_idx = train_count + val_count

            train_files = all_files[0:train_split_idx]
            val_files = all_files[train_split_idx:val_split_idx]
            test_files = all_files[val_split_idx:] 

            train_dest = train_path / relative_path
            val_dest = val_path / relative_path
            test_dest = test_path / relative_path
            
            train_dest.mkdir(parents=True, exist_ok=True)
            val_dest.mkdir(parents=True, exist_ok=True)
            test_dest.mkdir(parents=True, exist_ok=True)

            for f in train_files:
                shutil.copy2(f, train_dest / f.name)
            for f in val_files:
                shutil.copy2(f, val_dest / f.name)
            for f in test_files:
                shutil.copy2(f, test_dest / f.name)
            
            pbar.update(1)

    print("\n--- Dataset Splitting Complete! ---")
    print(f"Your new dataset is ready at: {dest_dir.resolve()}")

# --- Run the script ---
if __name__ == "__main__":
    if not SOURCE_DIR.exists():
        print(f"ERROR: Source folder not found at: {SOURCE_DIR.resolve()}")
    else:
        split_dataset(SOURCE_DIR, DEST_DIR, SPLIT_RATIOS)