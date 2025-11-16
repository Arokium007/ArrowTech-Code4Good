import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
import numpy as np
from tqdm import tqdm
import time
import copy
from pathlib import Path

# --- Configuration ---
CONFIG = {
    "data_dir": Path("_04_Split_Dataset"),
    
    # --- CHANGE #1: Set model to resnet50 ---
    "model_name": "resnet50",
    "use_pretrained": True,
    
    "num_epochs": 25,
    "batch_size": 32,
    "learning_rate": 0.001,
    
    # --- CHANGE #2: Save to a NEW directory ---
    "output_model_dir": Path("trained_models_resnet"),
    "output_classes_dir": Path("trained_models_resnet/class_indices_resnet")
}
# ---------------------

def train_model(model, criterion, optimizer, dataloaders, device, num_epochs, dataset_sizes):
    """
    Main training and validation loop for a single model.
    """
    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            loader = dataloaders[phase]
            pbar = tqdm(loader, desc=f"{phase.capitalize()} Epoch {epoch+1}")
            
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                pbar.set_postfix({'loss': loss.item(), 'acc': (torch.sum(preds == labels.data).item() / inputs.size(0)) * 100})

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'\n{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

def main():
    # --- 1. Setup Device (GPU or CPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Create Output Dirs ---
    CONFIG["output_model_dir"].mkdir(exist_ok=True)
    CONFIG["output_classes_dir"].mkdir(exist_ok=True)

    # --- 3. Define Data Transforms (Handles model_name change) ---
    if CONFIG["model_name"] == "resnet50":
        input_size = 224
        print("Using ResNet50 input size: 224x224")
    else:
        input_size = 256 # EfficientNet default
        print(f"Using {CONFIG['model_name']} input size: 256x256")


    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size), # Will be 224
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(40),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size + 32), # Will be 256
            transforms.CenterCrop(input_size),  # Will be 224
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # --- 4. Loop Through Each Crop and Train a Model ---
    train_base_path = CONFIG["data_dir"] / "train"
    val_base_path = CONFIG["data_dir"] / "val"
    
    crop_folders = [d for d in train_base_path.iterdir() if d.is_dir()]

    for crop_folder in crop_folders:
        crop_name = crop_folder.name
        print(f"\n{'='*20} TRAINING ResNet50 MODEL FOR: {crop_name} {'='*20}")

        # --- 4a. Load Datasets ---
        train_crop_dir = train_base_path / crop_name
        val_crop_dir = val_base_path / crop_name

        image_datasets = {
            'train': datasets.ImageFolder(train_crop_dir, data_transforms['train']),
            'val': datasets.ImageFolder(val_crop_dir, data_transforms['val'])
        }
        
        dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4),
            'val': DataLoader(image_datasets['val'], batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)
        }
        
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
        num_classes = len(class_names)
        
        print(f"Found {num_classes} disease classes for {crop_name}.")

        # --- 4b. Calculate Class Weights ---
        print("Calculating class weights...")
        train_labels = np.array(image_datasets['train'].targets)
        class_counts = np.bincount(train_labels, minlength=num_classes)
        
        class_counts = class_counts.astype(np.float32)
        class_counts = np.where(class_counts == 0, 1, class_counts)
        
        total_samples = len(train_labels)
        weights = total_samples / (num_classes * class_counts)
        class_weights = torch.FloatTensor(weights).to(device)

        # --- 4c. Load Pre-trained Model (Handles model_name change) ---
        if CONFIG["model_name"] == "resnet50":
            model = models.resnet50(weights='IMAGENET1K_V1')
        else:
            model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        
        for param in model.parameters():
            param.requires_grad = False

        if CONFIG["model_name"] == "resnet50":
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            params_to_update = model.fc.parameters()
        else:
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            params_to_update = model.classifier[1].parameters()

        model = model.to(device)

        # --- 4d. Define Loss & Optimizer ---
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(params_to_update, lr=CONFIG["learning_rate"])

        # --- 4e. Start Training ---
        model = train_model(model, criterion, optimizer, dataloaders, device, CONFIG["num_epochs"], dataset_sizes)
        
        # --- 4f. Save Model and Class File ---
        
        # --- CHANGE #3 & #4: Save with a new name ---
        model_save_name = f"model_resnet_{crop_name.split(' ')[-2].lower()}.pth"
        classes_save_name = f"classes_resnet_{crop_name.split(' ')[-2].lower()}.json"
        
        model_save_path = CONFIG["output_model_dir"] / model_save_name
        classes_save_path = CONFIG["output_classes_dir"] / classes_save_name
        
        torch.save(model.state_dict(), model_save_path)
        
        with open(classes_save_path, 'w') as f:
            json.dump(class_names, f)
            
        print(f"Saved model to: {model_save_path}")
        print(f"Saved classes to: {classes_save_path}")

    print("\n--- All ResNet50 models trained successfully! ---")

if __name__ == '__main__':
    main()