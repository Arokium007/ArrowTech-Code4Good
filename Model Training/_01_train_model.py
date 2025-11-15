import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import numpy as np
from tqdm import tqdm
import time
import copy

# --- 1. Configuration & Hyperparameters ---
CONFIG = {
    # Data paths
    "data_dir": "_04_Split_Dataset",
    
    # Model
    "model_name": "efficientnet_b0", # or "resnet50"
    "use_pretrained": True,
    
    # Training
    "num_epochs": 25,
    "batch_size": 32,
    "learning_rate": 0.001,
    
    # Output file paths
    "save_model_path": "agri_sentinel_model.pth",
    "save_classes_path": "agri_sentinel_classes.json"
}
# -------------------------------------------


def train_model(model, criterion, optimizer, dataloaders, device, num_epochs=25):
    """
    Main training and validation loop.
    """
    start_time = time.time()
    
    # Keep track of the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data using tqdm for a progress bar
            loader = dataloaders[phase]
            pbar = tqdm(loader, desc=f"{phase.capitalize()} Epoch {epoch+1}")
            
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # --- Forward pass ---
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # --- Backward pass + optimize only if in training phase ---
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # --- Statistics ---
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                pbar.set_postfix({'loss': loss.item(), 'acc': (torch.sum(preds == labels.data).item() / inputs.size(0)) * 100})

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'\n{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

            # --- Save the best model ---
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"New best validation accuracy: {best_acc:.4f}! Saving model...")
                torch.save(model.state_dict(), CONFIG["save_model_path"])


    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def main():
    # --- 2. Setup Device (GPU or CPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. Data Augmentation & Transforms ---
    # This is where we apply the heavy augmentation for the imbalanced classes
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224 if CONFIG["model_name"] == "resnet50" else 256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(40),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224 if CONFIG["model_name"] == "resnet50" else 256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224 if CONFIG["model_name"] == "resnet50" else 256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # --- 4. Load Datasets and Create DataLoaders ---
    image_datasets = {x: datasets.ImageFolder(os.path.join(CONFIG["data_dir"], x), data_transforms[x])
                      for x in ['train', 'val', 'test']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=CONFIG["batch_size"],
                                 shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    
    # Add test loader separately (no shuffle)
    dataloaders['test'] = DataLoader(image_datasets['test'], batch_size=CONFIG["batch_size"],
                                     shuffle=False, num_workers=4)
    
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes.")
    
    # Save the class names for your app
    with open(CONFIG["save_classes_path"], 'w') as f:
        json.dump(class_names, f)
    print(f"Saved class names to {CONFIG['save_classes_path']}")


    # --- 5. Calculate Class Weights for Imbalance ---
    print("Calculating class weights for imbalanced dataset...")
    # Get all labels from the training dataset
    train_labels = np.array(image_datasets['train'].targets)
    
    # Count occurrences of each class
    class_counts = np.bincount(train_labels)
    
    # Calculate weights (inverse frequency)
    # total_samples / (num_classes * class_count)
    total_samples = len(train_labels)
    weights = total_samples / (num_classes * class_counts.astype(np.float32))
    
    # Convert to a tensor and send to the device
    class_weights = torch.FloatTensor(weights).to(device)
    print(f"Class weights calculated and moved to {device}.")


    # --- 6. Load Pre-trained Model (Transfer Learning) ---
    if CONFIG["model_name"] == "resnet50":
        model = models.resnet50(weights='IMAGENET1K_V1')
    elif CONFIG["model_name"] == "efficientnet_b0":
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    else:
        raise ValueError(f"Model {CONFIG['model_name']} not recognized.")

    # Freeze all the pre-trained layers
    for param in model.parameters():
        param.requires_grad = False

    # --- 7. Replace the Final Classifier Head ---
    # This is the *only* part of the network that will be trained
    if CONFIG["model_name"] == "resnet50":
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        params_to_update = model.fc.parameters()
        
    elif CONFIG["model_name"] == "efficientnet_b0":
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        params_to_update = model.classifier[1].parameters()

    print("Model loaded. Pre-trained layers frozen. Classifier head replaced.")

    # Move the model to the GPU
    model = model.to(device)

    # --- 8. Define Loss Function (with weights) & Optimizer ---
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Tell the optimizer to *only* update the parameters of the new final layer
    optimizer = optim.Adam(params_to_update, lr=CONFIG["learning_rate"])

    # --- 9. Start Training ---
    print("Starting training...")
    model = train_model(model, criterion, optimizer, dataloaders, device, num_epochs=CONFIG["num_epochs"])
    
    print("\nTraining complete. Final model saved.")
    
    
if __name__ == '__main__':
    main()