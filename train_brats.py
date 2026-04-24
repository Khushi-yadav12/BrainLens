"""
Brain Tumor Classification — BraTS 2023 Training Script
========================================================

This script:
 1. Reads NIfTI (.nii.gz) MRI volumes from the BraTS 2023 dataset.
 2. Extracts informative 2-D axial slices from the T1c modality.
 3. Saves slices as PNG images into  data_slices/yes/  (all BraTS cases
    are glioma-positive).
 4. Creates synthetic "no-tumor" slices by zeroing out the central region
    of real MRI backgrounds (peripheral/skull slices), saved to data_slices/no/.
 5. Trains a ResNet18 binary classifier with PyTorch + transfer learning.
    (ResNet18 is used because it is ≈5x lighter than VGG16, making it
    feasible to train on CPU without running out of memory.)
 6. Saves the final weights to  model/brain_tumor_resnet18.pth
    classifier.py is already updated to load this model.

Usage:
    python train_brats.py --brats_dir dataset_extracted/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData

Optional flags:
    --epochs_frozen   (default 10)
    --epochs_unfrozen (default 8)
    --lr              (default 1e-3)
    --lr_fine         (default 1e-5)
    --bs              (default 4)
"""

import argparse
import os
import sys
import random
import numpy as np
from pathlib import Path


# ── Dependency checks ─────────────────────────────────────────────────────────
def _require(pkg, install_hint):
    try:
        __import__(pkg)
    except ImportError:
        print(f"ERROR: '{pkg}' is required. Install with:  {install_hint}")
        sys.exit(1)


_require("nibabel", "pip install nibabel")
_require("torch",   "pip install torch torchvision")
_require("PIL",     "pip install Pillow")


import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


# ── Slice extraction helpers ──────────────────────────────────────────────────

def _normalize_slice(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize a 2D float array to 0-255 uint8."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


def extract_tumor_slices(nii_path: Path, out_dir: Path,
                          skip_pct: float = 0.20,
                          max_slices: int = 30) -> int:
    """
    Extract informative axial slices from a NIfTI volume.
    Slices with very low mean intensity (background) are skipped.
    Returns the number of slices saved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    vol = nib.load(str(nii_path)).get_fdata().astype(np.float32)

    # vol shape: (X, Y, Z) — iterate over Z (axial)
    n_z = vol.shape[2]
    lo = int(n_z * skip_pct)
    hi = int(n_z * (1 - skip_pct))
    z_indices = list(range(lo, hi))
    random.shuffle(z_indices)
    z_indices = z_indices[:max_slices]

    saved = 0
    stem = nii_path.name.replace(".nii.gz", "").replace(".nii", "")
    for z in z_indices:
        slc = vol[:, :, z]
        # Skip near-empty slices
        if slc.mean() < 5:
            continue
        img_arr = _normalize_slice(slc)
        img = Image.fromarray(img_arr, mode="L").convert("RGB")
        img.save(out_dir / f"{stem}_z{z:03d}.png")
        saved += 1
    return saved


def create_no_tumor_slices(yes_dir: Path, no_dir: Path,
                            count: int = None) -> int:
    """
    Create synthetic 'no-tumor' images by taking 'yes' images and
    masking out the central region (where tumors typically appear),
    leaving only peripheral brain/skull-like signal.
    """
    no_dir.mkdir(parents=True, exist_ok=True)
    yes_images = list(yes_dir.glob("*.png"))
    if count is None:
        count = len(yes_images)
    random.shuffle(yes_images)
    yes_images = yes_images[:count]

    saved = 0
    for src in yes_images:
        img = Image.open(src).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Black out center 40 % of the image
        cy, cx = h // 2, w // 2
        rh, rw = int(h * 0.40), int(w * 0.40)
        arr[cy - rh:cy + rh, cx - rw:cx + rw] = 0

        # Add slight Gaussian noise so it doesn't look degenerate
        noise = np.random.normal(0, 3, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

        dest = no_dir / f"notumor_{src.name}"
        Image.fromarray(arr).save(dest)
        saved += 1
    return saved


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

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


# ── Training ──────────────────────────────────────────────────────────────────

def build_dataloaders(data_dir: Path, batch_size: int, img_size: int = 224,
                      max_per_class: int = 600):
    yes_files = list((data_dir / "yes").glob("*.png"))
    no_files  = list((data_dir / "no").glob("*.png"))

    if not yes_files:
        print("ERROR: No images found in data_slices/yes/")
        sys.exit(1)
    if not no_files:
        print("ERROR: No images found in data_slices/no/")
        sys.exit(1)

    random.seed(42)
    random.shuffle(yes_files)
    random.shuffle(no_files)
    yes_files = yes_files[:max_per_class]
    no_files  = no_files[:max_per_class]

    all_paths  = yes_files + no_files
    all_labels = [1] * len(yes_files) + [0] * len(no_files)

    # Shuffle combined list
    combined = list(zip(all_paths, all_labels))
    random.seed(42)
    random.shuffle(combined)
    unzipped = list(zip(*combined))
    all_paths  = list(unzipped[0])
    all_labels = list(unzipped[1])

    split = int(0.8 * len(all_paths))
    train_paths, val_paths   = all_paths[:split], all_paths[split:]
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

    print(f"[DATA] Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_loader, val_loader


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
        description="Train VGG16 Brain Tumor Classifier on BraTS 2023 data")
    parser.add_argument("--brats_dir", type=str,
        default="dataset_extracted/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData",
        help="Path to the BraTS folder containing BraTS-GLI-* subfolders")
    parser.add_argument("--slices_dir", type=str, default="data_slices",
        help="Where to save extracted PNG slices (default: data_slices)")
    parser.add_argument("--epochs_frozen",   type=int,   default=10)
    parser.add_argument("--epochs_unfrozen", type=int,   default=8)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--lr_fine",         type=float, default=1e-5)
    parser.add_argument("--bs",              type=int,   default=2)
    parser.add_argument("--img_size",        type=int,   default=96)
    parser.add_argument("--max_per_class",   type=int,   default=600,
        help="Max images per class for training (default 600, keeps RAM usage low)")
    parser.add_argument("--max_slices",      type=int,   default=20,
        help="Max axial slices to extract per patient (default 20)")
    parser.add_argument("--regen_slices",    action="store_true",
        help="Force re-extraction of slices even if data_slices/ already exists")
    args = parser.parse_args()

    base_dir    = Path(__file__).parent
    brats_dir   = Path(args.brats_dir) if Path(args.brats_dir).is_absolute() \
                  else base_dir / args.brats_dir
    slices_dir  = base_dir / args.slices_dir
    yes_dir     = slices_dir / "yes"
    no_dir      = slices_dir / "no"

    # ── 1. Extract slices ────────────────────────────────────────────────
    if args.regen_slices or not yes_dir.exists() or not list(yes_dir.glob("*.png")):
        if not brats_dir.is_dir():
            print(f"ERROR: BraTS directory not found: {brats_dir}")
            sys.exit(1)

        patient_dirs = sorted(brats_dir.glob("BraTS-GLI-*"))
        print(f"[INFO] Found {len(patient_dirs)} patient cases in {brats_dir}")

        total_yes = 0
        for pd in patient_dirs:
            # Prefer T1c (contrast enhanced — best for tumor visibility)
            t1c_files = list(pd.glob("*-t1c.nii.gz"))
            if not t1c_files:
                continue
            saved = extract_tumor_slices(
                t1c_files[0], yes_dir,
                skip_pct=0.15,
                max_slices=args.max_slices
            )
            total_yes += saved
            print(f"  [{pd.name}] extracted {saved} tumor slices")

        print(f"\n[INFO] Total 'yes' slices: {total_yes}")

        print("[INFO] Generating synthetic 'no-tumor' slices...")
        total_no = create_no_tumor_slices(yes_dir, no_dir, count=total_yes)
        print(f"[INFO] Total 'no' slices: {total_no}")
    else:
        yes_count = len(list(yes_dir.glob("*.png")))
        no_count  = len(list(no_dir.glob("*.png")))
        print(f"[INFO] Using existing slices — yes: {yes_count}, no: {no_count}")

    # ── 2. Build data loaders ────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on: {device}")

    train_loader, val_loader = build_dataloaders(
        slices_dir, args.bs, args.img_size, max_per_class=args.max_per_class)

    # ── 3. Build model ───────────────────────────────────────────────────
    # ResNet18 is used instead of VGG16 because it is ~5x lighter and
    # can train reliably on CPU without out-of-memory crashes.
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 2)

    # Freeze backbone for stage 1
    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # ── 4. Stage 1: frozen backbone ──────────────────────────────────────
    print(f"\n[STAGE 1] Training classifier head — {args.epochs_frozen} epochs")
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_frozen)

    best_val_acc = 0.0
    model_dir  = base_dir / "model"
    model_dir.mkdir(exist_ok=True)
    save_path  = model_dir / "brain_tumor_resnet18.pth"

    for epoch in range(1, args.epochs_frozen + 1):
        try:
            tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, device)
        except Exception as e:
            import traceback; traceback.print_exc()
            sys.exit(1)
        scheduler.step()
        print(f"  Epoch {epoch:02d}/{args.epochs_frozen} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={vl_loss:.4f} acc={vl_acc:.4f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), save_path)

    # ── 5. Stage 2: fine-tune all layers ────────────────────────────────
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
            torch.save(model.state_dict(), save_path)

    print(f"\n[SAVED] Best model (val_acc={best_val_acc:.4f}) -> {save_path}")
    print("[DONE] Training complete.")


if __name__ == "__main__":
    main()
