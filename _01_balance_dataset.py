import os
import shutil
import random
from pathlib import Path

# --- Configuration ---
SOURCE_DATASET_DIR = Path("Tomato Dataset")

DESTINATION_DATASET_DIR = Path("_02_Balanced Dataset/Balanced Tomato Dataset")

# 3. SET THE LIMIT
MAX_FILES_PER_CLASS = 600
# ---------------------

def balance_single_crop_dataset(source_dir, dest_dir, max_files):
    """
    Copies and balances a SINGLE CROP dataset to a new location.
    
    :param source_dir: Path to the original dataset (e.g., "Tomato Dataset").
    :param dest_dir: Path to the new balanced dataset folder.
    :param max_files: The maximum number of files to keep per class.
    """
    
    # Create the main destination folder if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- Starting Dataset Balancing ---")
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    print(f"Max files per class: {max_files}\n")

    
    for disease_folder in source_dir.iterdir():
        if not disease_folder.is_dir():
            continue 

        dest_disease_path = dest_dir / disease_folder.name
        dest_disease_path.mkdir(exist_ok=True)
        
        # Get a list of all files in the source disease folder
        all_files = [f for f in disease_folder.glob('*') if f.is_file()]
        file_count = len(all_files)

        if file_count <= max_files:
            print(f"  {disease_folder.name}: {file_count} files. Copying all.")
            for file_path in all_files:
                shutil.copy2(file_path, dest_disease_path / file_path.name)
        
        else:
            print(f"  {disease_folder.name}: {file_count} files. Randomly sampling {max_files}.")
            files_to_copy = random.sample(all_files, max_files)
            
            for file_path in files_to_copy:
                shutil.copy2(file_path, dest_disease_path / file_path.name)

    print("\n--- Balancing Complete! ---")
    print(f"Your new, balanced dataset is ready at: {dest_dir}")

# --- Run the script ---
if __name__ == "__main__":
    if not SOURCE_DATASET_DIR.exists():
        print(f"ERROR: Source folder not found at: {SOURCE_DATASET_DIR.resolve()}")
        print("Please check your SOURCE_DATASET_DIR path.")
    elif str(SOURCE_DATASET_DIR) == "path/to/your/Original_Dataset_Folder":
         print("ERROR: Please update the SOURCE_DATASET_DIR and DESTINATION_DATASET_DIR paths in the script.")
    else:
        balance_single_crop_dataset(SOURCE_DATASET_DIR, DESTINATION_DATASET_DIR, MAX_FILES_PER_CLASS)