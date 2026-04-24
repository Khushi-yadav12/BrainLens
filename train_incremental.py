"""
Brain Tumor Classification — Incremental Training Script
=========================================================

This script fine-tunes the EXISTING trained model (brain_tumor_resnet18.pth)
on the NEXT 2000 images (1000 per class) that were NOT used in the original
training run (which used images 0-600 per class with random.seed(42)).

Usage:
    python train_incremental.py

Optional flags:
    --epochs_frozen    (default 6)
    --epochs_unfrozen  (default 5)
    --lr               (default 5e-4)
    --lr_fine          (default 1e-5)
    --bs               (default 2)
    --skip             (default 600  — images already trained on, per class)
    --new_count        (default 1000 — new images per class = 2000 total)
"""

import argparse
import os
import sys
import random
from pathlib import Path

# ── Dependency checks ─────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image
except ImportError:
    print("ERROR: PyTorch / TorchVision / Pillow are required.")
    print("Install with:  pip install torch torchvision Pillow")
    sys.exit(1)


# ── Dataset ───────────────────────────────────────────────────────────────────

class BinaryMRIDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


# ── Data loading ──────────────────────────────────────────────────────────────

def build_dataloaders(slices_dir: Path, skip: int, new_count: int,
                      batch_size: int, img_size: int = 96):
    """
    Load images that were NOT used in original training.
    Original training: indices 0..skip-1  (after seed-42 shuffle)
    This script uses:  indices skip..skip+new_count-1
    """
    valid_exts = {".png", ".jpg", ".jpeg", ".JPG", ".JPEG"}
    yes_files = [f for f in (slices_dir / "yes").iterdir() if f.is_file() and f.suffix in valid_exts]
    no_files  = [f for f in (slices_dir / "no").iterdir() if f.is_file() and f.suffix in valid_exts]

    if not yes_files or not no_files:
        print("ERROR: data_slices/yes/ or data_slices/no/ is empty.")
        sys.exit(1)

    # Apply SAME shuffle seed as original training so skip is accurate
    random.seed(42)
    random.shuffle(yes_files)
    random.seed(42)
    random.shuffle(no_files)

    # Take the NEXT batch after what was already trained on
    yes_new = yes_files[skip: skip + new_count]
    no_new  = no_files[skip: skip + new_count]

    if not yes_new or not no_new:
        print(f"ERROR: Not enough images after skip={skip}. "
              f"yes available after skip: {len(yes_files)-skip}, "
              f"no available after skip: {len(no_files)-skip}")
        sys.exit(1)

    print(f"[DATA] New 'yes' images: {len(yes_new)}")
    print(f"[DATA] New 'no'  images: {len(no_new)}")

    all_paths  = yes_new + no_new
    all_labels = [1] * len(yes_new) + [0] * len(no_new)

    combined = list(zip(all_paths, all_labels))
    random.seed(99)
    random.shuffle(combined)
    all_paths, all_labels = zip(*combined)
    all_paths  = list(all_paths)
    all_labels = list(all_labels)

    split = int(0.8 * len(all_paths))
    train_paths, val_paths   = all_paths[:split],  all_paths[split:]
    train_labels, val_labels = all_labels[:split], all_labels[split:]

    train_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = BinaryMRIDataset(train_paths, train_labels, train_tfm)
    val_ds   = BinaryMRIDataset(val_paths,   val_labels,   val_tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0)

    print(f"[DATA] Train: {len(train_ds)}, Val: {len(val_ds)}\n")
    return train_loader, val_loader


# ── Train / Eval helpers ──────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (outputs.argmax(1) == labels).sum().item()
        n += len(labels)
    return total_loss / n, correct / n


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
            n += len(labels)
    return total_loss / n, correct / n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Incremental fine-tuning on new images not used in original training")
    parser.add_argument("--skip",           type=int,   default=600,
        help="Images per class already used in first training run (default: 600)")
    parser.add_argument("--new_count",      type=int,   default=1000,
        help="New images per class to train on now (default: 1000 → 2000 total)")
    parser.add_argument("--epochs_frozen",  type=int,   default=6)
    parser.add_argument("--epochs_unfrozen",type=int,   default=5)
    parser.add_argument("--lr",             type=float, default=5e-4)
    parser.add_argument("--lr_fine",        type=float, default=1e-5)
    parser.add_argument("--bs",             type=int,   default=2)
    parser.add_argument("--img_size",       type=int,   default=96)
    parser.add_argument("--scratch",        action="store_true", help="Train from scratch (ignore existing model)")
    args = parser.parse_args()

    base_dir   = Path(__file__).parent
    slices_dir = base_dir / "data_slices"
    model_path = base_dir / "model" / "brain_tumor_resnet18.pth"

    # ── 1. Load existing model ────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    if model_path.exists() and not args.scratch:
        print(f"[INFO] Loading existing weights from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("[INFO] Existing model loaded — will fine-tune on new data.\n")
    else:
        print("[WARN] No existing model to load or --scratch passed. Training from ImageNet weights.")
        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, 2)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # ── 2. Load NEW (unseen) data ─────────────────────────────────────────
    print(f"[INFO] Skipping first {args.skip} images per class (already trained).")
    print(f"[INFO] Using next {args.new_count} images per class = {args.new_count*2} total new images.\n")

    train_loader, val_loader = build_dataloaders(
        slices_dir, args.skip, args.new_count, args.bs, args.img_size)

    best_val_acc = 0.0

    # ── 3. Stage 1: Freeze backbone, train only FC head ───────────────────
    print(f"[STAGE 1] Fine-tuning FC head only — {args.epochs_frozen} epochs")
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_frozen)

    for epoch in range(1, args.epochs_frozen + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        print(f"  Epoch {epoch:02d}/{args.epochs_frozen} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={vl_loss:.4f} acc={vl_acc:.4f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), model_path)
            print(f"    ✓ Saved (val_acc={vl_acc:.4f})")

    # ── 4. Stage 2: Unfreeze all, fine-tune ──────────────────────────────
    print(f"\n[STAGE 2] Fine-tuning all layers — {args.epochs_unfrozen} epochs")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=args.lr_fine)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_unfrozen)

    for epoch in range(1, args.epochs_unfrozen + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        print(f"  Epoch {epoch:02d}/{args.epochs_unfrozen} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={vl_loss:.4f} acc={vl_acc:.4f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), model_path)
            print(f"    ✓ Saved (val_acc={vl_acc:.4f})")

    print(f"\n[DONE] Best val_acc={best_val_acc:.4f}")
    print(f"[SAVED] Updated model → {model_path}")
    print(f"\nThe model has now seen:")
    print(f"  • Original training : {args.skip} images/class  ({args.skip*2} total)")
    print(f"  • This run          : {args.new_count} images/class ({args.new_count*2} total)")
    print(f"  • Grand total       : {(args.skip+args.new_count)} images/class ({(args.skip+args.new_count)*2} total)")


if __name__ == "__main__":
    main()
