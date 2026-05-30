# ==============================================================================
# BrainLens — Domain-Bias-Free Classifier Training Script for Kaggle
# ==============================================================================
# Instructions:
#   1. In Kaggle, click "+ Add Data" -> "Upload" and upload your ZIP file.
#   2. Copy this entire script into a Kaggle cell and hit Run.
#   3. The script will automatically find your data, perform a strict 
#      Patient-Level Split, and train a dual-GPU ResNet!
# ==============================================================================

import os
import glob
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1: Dataset Auto-Discovery & Custom Class
# ──────────────────────────────────────────────────────────────────────────────

def find_dataset_images():
    """Searches Kaggle's input directory for the custom dataset images."""
    print("\n[INFO] Searching for your uploaded dataset in /kaggle/input...")
    # Search for any PNG files that start with 'healthy_' or 'tumor_'
    search_path = "/kaggle/input/**/*.png"
    all_images = glob.glob(search_path, recursive=True)
    
    valid_images = [f for f in all_images if os.path.basename(f).startswith(("healthy_", "tumor_"))]
    
    if not valid_images:
        raise FileNotFoundError("Could not find any 'healthy_' or 'tumor_' PNG files! Did you upload the dataset and attach it to this notebook?")
        
    print(f"[INFO] Found {len(valid_images)} total images!")
    return valid_images

class BrainTumorDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        
        # 0 = Healthy, 1 = Tumor
        label = 1 if filename.startswith("tumor_") else 0
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2: Strict Patient-Level Splitting
# ──────────────────────────────────────────────────────────────────────────────

def split_by_patient(image_paths, val_split=0.15):
    """Ensures slices from the same patient NEVER appear in both Train and Val sets."""
    patient_to_images = {}
    
    for path in image_paths:
        filename = os.path.basename(path)
        # Example: tumor_BraTS2021_00000_AX_084.png
        # Split: ['tumor', 'BraTS2021', '00000', 'AX', '084.png']
        parts = filename.split('_')
        if len(parts) >= 3:
            patient_id = f"{parts[1]}_{parts[2]}" # "BraTS2021_00000"
            if patient_id not in patient_to_images:
                patient_to_images[patient_id] = []
            patient_to_images[patient_id].append(path)
            
    all_patients = list(patient_to_images.keys())
    random.seed(42)
    random.shuffle(all_patients)
    
    num_val = int(len(all_patients) * val_split)
    val_patients = set(all_patients[:num_val])
    train_patients = set(all_patients[num_val:])
    
    train_paths = []
    val_paths = []
    
    for pid, paths in patient_to_images.items():
        if pid in val_patients:
            val_paths.extend(paths)
        else:
            train_paths.extend(paths)
            
    print(f"\n[INFO] Strict Patient-Level Split Complete:")
    print(f"       Train Patients: {len(train_patients)} ({len(train_paths)} images)")
    print(f"       Val Patients:   {len(val_patients)} ({len(val_paths)} images)")
    
    return train_paths, val_paths

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3: Training the Classifier
# ──────────────────────────────────────────────────────────────────────────────

def train_classifier():
    MODEL_SAVE_PATH = "/kaggle/working/brain_tumor_classifier_multiplanar.pth"
    
    all_images = find_dataset_images()
    train_paths, val_paths = split_by_patient(all_images, val_split=0.15)
    
    # Data Augmentation
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_ds = BrainTumorDataset(train_paths, transform=transform_train)
    val_ds = BrainTumorDataset(val_paths, transform=transform_val)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[SYSTEM] Building ResNet18 Classifier on {device}")
    
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Accelerating with {torch.cuda.device_count()} GPUs using DataParallel!")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    EPOCHS = 10
    best_acc = 0.0
    
    print("\n[TRAIN] Beginning ResNet18 Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()
            
        train_acc = correct_train / total_train
        
        # Validation
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()
                
        val_acc = correct_val / total_val
        
        print(f"Epoch {epoch+1:02d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), MODEL_SAVE_PATH)
            else:
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                
    print(f"\n[DONE] Download your model from: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_classifier()
