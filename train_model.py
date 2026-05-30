"""
Brain Tumor Classification — PyTorch Training Script

Usage:
    python train_model.py --data_dir data_slices

The dataset should have two folders:
    yes/   — MRI images with tumors
    no/    — MRI images without tumors

This script trains a ResNet18 model using PyTorch and saves
the weights to model/brain_tumor_resnet18.pth
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

def main():
    parser = argparse.ArgumentParser(description="Train Brain Tumor Classifier")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset with 'yes' and 'no' subfolders")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--bs", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        sys.exit(1)

    print(f"[INFO] Loading data from {args.data_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Data augmentations
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(args.data_dir, transform=transform_train)
    print(f"[INFO] Classes: {full_dataset.class_to_idx}")
    
    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Overwrite transform for validation dataset
    val_dataset.dataset.transform = transform_val

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=0)

    print(f"[INFO] Training samples: {train_size}, Validation: {val_size}")

    # Model
    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("\n[STAGE 1] Training ResNet18...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save model
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "brain_tumor_resnet18.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n[SAVED] Model weights saved to: {save_path}")
    print("[DONE] Training complete.")

if __name__ == "__main__":
    main()
