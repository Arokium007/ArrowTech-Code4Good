import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path

# --- 1. Configuration ---
CONFIG = {
    # Base paths
    "test_data_dir_base": Path("_04_Split_Dataset/test"),
    "models_dir": Path("trained_models"),
    "classes_dir": Path("trained_models/class_indices"),
    "reports_dir": Path("evaluation_reports"), # Folder to save CM images
    
    # Model info
    "model_arch": "efficientnet_b0", # Must be the same as trained (efficientnet_b0 or resnet50)
    "batch_size": 32,
}
# -------------------------

def plot_confusion_matrix(cm, class_names, output_filename):
    """
    Saves a beautiful confusion matrix plot to a file.
    """
    plt.figure(figsize=(max(15, len(class_names)//2), max(12, len(class_names)//2.5)))
    heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
    
    heatmap.set_ylabel('Actual Label', fontsize=12)
    heatmap.set_xlabel('Predicted Label', fontsize=12)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right')
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
    
    plt.title('Model Confusion Matrix', fontsize=16)
    plt.tight_layout()
    try:
        plt.savefig(output_filename)
        print(f"\nConfusion Matrix saved to {output_filename}")
    except Exception as e:
        print(f"\nError saving confusion matrix: {e}")
    plt.close() # Close the figure to save memory


def evaluate_model():
    """
    Loads all trained specialist models and evaluates them
    on their corresponding test datasets.
    """
    
    # --- 1. Setup Device (GPU or CPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Create Reports Directory ---
    CONFIG["reports_dir"].mkdir(exist_ok=True)

    # --- 3. Define Test Data Transforms ---
    if CONFIG["model_arch"] == "resnet50":
        input_size = 224
    else:
        input_size = 256 # EfficientNet default

    test_transforms = transforms.Compose([
        transforms.Resize(input_size + 32), # 256 for resnet, 288 for effnet
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # --- 4. Find all models to evaluate ---
    model_paths = list(CONFIG["models_dir"].glob('model_*.pth'))
    if not model_paths:
        print(f"ERROR: No models found in {CONFIG['models_dir']}. Did you run the training script?")
        return

    print(f"Found {len(model_paths)} models to evaluate...")
    all_accuracies = {}

    # --- 5. Loop and Evaluate Each Model ---
    for model_path in model_paths:
        
        # --- 5a. Get model info ---
        crop_key = model_path.stem.split('_')[-1] # e.g., "banana"
        crop_name_capitalized = crop_key.capitalize() # e.g., "Banana"
        
        print(f"\n{'='*20} EVALUATING: {crop_name_capitalized} {'='*20}")
        
        classes_path = CONFIG["classes_dir"] / f"classes_{crop_key}.json"
        
        try:
            with open(classes_path) as f:
                class_names = json.load(f)
            num_classes = len(class_names)
            print(f"Loaded {num_classes} classes for {crop_key}.")
        except FileNotFoundError:
            print(f"ERROR: Could not find class file at {classes_path}. Skipping.")
            continue

        # --- 5b. Load Model Architecture & Weights ---
        if CONFIG["model_arch"] == "resnet50":
            model = models.resnet50()
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            
        elif CONFIG["model_arch"] == "efficientnet_b0":
            model = models.efficientnet_b0()
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        else:
            print(f"ERROR: Model arch {CONFIG['model_arch']} not recognized. Skipping.")
            continue

        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"ERROR: Could not load model weights from {model_path}: {e}. Skipping.")
            continue

        model = model.to(device)
        model.eval()

        # --- 5c. Load the correct Test Dataset ---
        # e.g., "Split_Dataset/test/Balanced Banana Dataset"
        current_test_dir = CONFIG["test_data_dir_base"] / f"Balanced {crop_name_capitalized} Dataset"
        
        if not current_test_dir.exists():
            print(f"ERROR: Test directory not found at {current_test_dir}. Skipping.")
            continue
            
        test_dataset = datasets.ImageFolder(current_test_dir, test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"],
                                 shuffle=False, num_workers=4)
        print(f"Test dataset loaded from {current_test_dir}")

        # --- 5d. Run Evaluation Loop ---
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"Testing {crop_name_capitalized}"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # --- 5e. Calculate and Print Metrics ---
        accuracy = accuracy_score(all_labels, all_preds)
        all_accuracies[crop_name_capitalized] = accuracy * 100
        
        print(f"\n--- Results for {crop_name_capitalized} ---")
        print(f"Overall Test Accuracy: {accuracy * 100:.2f}%")

        report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
        print("\nClassification Report:")
        print(report)

        # --- 5f. Generate and Save Confusion Matrix ---
        cm = confusion_matrix(all_labels, all_preds)
        cm_filename = CONFIG["reports_dir"] / f"confusion_matrix_{crop_key}.png"
        plot_confusion_matrix(cm, class_names, cm_filename)

    # --- 6. Final Summary ---
    print("\n\n{'='*20} FINAL SUMMARY {'='*20}")
    print("All evaluations complete. Accuracy per model:")
    for crop, acc in all_accuracies.items():
        print(f"- {crop.ljust(15)}: {acc:.2f}%")
    print(f"\nAll confusion matrix images are saved in: {CONFIG['reports_dir'].resolve()}")

if __name__ == '__main__':
    evaluate_model()