"""
Brain Tumor Segmentation — ACU-Net Training Script
===================================================

This script:
 1. Reads NIfTI (.nii.gz) MRI volumes from the BraTS dataset.
 2. Extracts 2-D axial slices from the T1c modality AND its corresponding segmentation mask (*-seg.nii.gz).
 3. Casts multi-class masks to binary masks (Whole Tumor vs Background).
 4. Trains the ACU-Net model using PyTorch.
 5. Saves the final weights to model/brain_tumor_acunet.pth

Usage:
    python train_acunet.py --brats_dir <path_to_brats_data>
"""

import argparse
import os
import sys
import random
import numpy as np
from pathlib import Path

# Dependency checks
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
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

# Import the newly created ACU-Net architecture
from acu_net import ACUNet, DiceBCELoss


# ── Slice extraction helpers ──────────────────────────────────────────────────

def _normalize_slice(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize a 2D float array to 0-255 uint8."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


def extract_tumor_mask_pairs(nii_img_path: Path, nii_mask_path: Path, 
                             out_img_dir: Path, out_mask_dir: Path,
                             skip_pct: float = 0.20,
                             max_slices: int = 30) -> int:
    """
    Extract informative axial slices and their corresponding ground truth segmentation masks.
    """
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    
    img_vol = nib.load(str(nii_img_path)).get_fdata().astype(np.float32)
    mask_vol = nib.load(str(nii_mask_path)).get_fdata().astype(np.float32)

    n_z = img_vol.shape[2]
    lo = int(n_z * skip_pct)
    hi = int(n_z * (1 - skip_pct))
    z_indices = list(range(lo, hi))
    random.shuffle(z_indices)
    z_indices = z_indices[:max_slices]

    saved = 0
    stem = nii_img_path.name.replace(".nii.gz", "").replace(".nii", "")
    
    for z in z_indices:
        slc_img = img_vol[:, :, z]
        slc_mask = mask_vol[:, :, z]
        
        # Skip near-empty slices
        if slc_img.mean() < 5:
            continue
            
        # Convert multi-class mask to Binary Mask (0: Background, 1: Whole Tumor)
        slc_mask = (slc_mask > 0).astype(np.uint8) * 255
        
        img_arr = _normalize_slice(slc_img)
        
        # Save structural image
        img = Image.fromarray(img_arr, mode="L").convert("RGB")
        img.save(out_img_dir / f"{stem}_z{z:03d}.png")
        
        # Save mask image
        mask = Image.fromarray(slc_mask, mode="L")
        mask.save(out_mask_dir / f"{stem}_z{z:03d}.png")
        
        saved += 1
        
    return saved


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=256):
        self.image_paths = sorted(image_paths)
        self.mask_paths = sorted(mask_paths)
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load and resize
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        image = image.resize((self.img_size, self.img_size))
        mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        
        # Convert to Tensor
        image_tensor = TF.to_tensor(image) # Normalizes to [0,1]
        
        # Normalize input via standard ImageNet transform
        image_tensor = TF.normalize(image_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        mask_tensor = TF.to_tensor(mask) # Normalizes to [0,1]
        # Ensure mask is strictly 0 or 1
        mask_tensor = (mask_tensor > 0.5).float()
        
        return image_tensor, mask_tensor


# ── Training ──────────────────────────────────────────────────────────────────

def build_dataloaders(data_dir: Path, batch_size: int, img_size: int = 256):
    img_dir = data_dir / "images"
    mask_dir = data_dir / "masks"

    img_files = sorted(list(img_dir.glob("*.png")))
    mask_files = sorted(list(mask_dir.glob("*.png")))

    if not img_files or not mask_files:
        print("ERROR: Missing images or masks extraction. Generate them first.")
        sys.exit(1)
        
    assert len(img_files) == len(mask_files), "Mismatch between image and mask counts!"

    # Shuffle synchronously
    combined = list(zip(img_files, mask_files))
    random.seed(42)
    random.shuffle(combined)
    img_files, mask_files = zip(*combined)
    
    split = int(0.8 * len(img_files))
    train_imgs, val_imgs = img_files[:split], img_files[split:]
    train_masks, val_masks = mask_files[:split], mask_files[split:]

    train_ds = SegmentationDataset(train_imgs, train_masks, img_size)
    val_ds   = SegmentationDataset(val_imgs, val_masks, img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[DATA] Train Set: {len(train_ds)} slices, Val Set: {len(val_ds)} slices")
    return train_loader, val_loader


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    n = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs) # outputs are raw logits
        
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / n


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    dice_score = 0
    n = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_loss += loss.item() * imgs.size(0)
            
            # Calculate naive dice score metric
            preds = (torch.sigmoid(outputs) > 0.5).float()
            intersection = (preds * masks).sum().item()
            union = preds.sum().item() + masks.sum().item()
            if union > 0:
                dice_score += (2.0 * intersection / union) * imgs.size(0)
            
            n += imgs.size(0)
    return total_loss / n, dice_score / n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ACU-Net on BraTS Data")
    parser.add_argument("--brats_dir", type=str,
        default="dataset_extracted/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData",
        help="Path to the BraTS folder")
    parser.add_argument("--slices_dir", type=str, default="data_slices_seg",
        help="Where to save extracted PNG slices and masks")
    parser.add_argument("--epochs",   type=int,   default=20)
    parser.add_argument("--lr",       type=float, default=1e-4)
    parser.add_argument("--bs",       type=int,   default=4)
    parser.add_argument("--img_size", type=int,   default=256)
    parser.add_argument("--max_slices",type=int,   default=15,
        help="Max axial slices per patient")
    parser.add_argument("--regen_slices", action="store_true")
    args = parser.parse_args()

    base_dir    = Path(__file__).parent
    brats_dir   = Path(args.brats_dir) if Path(args.brats_dir).is_absolute() else base_dir / args.brats_dir
    slices_dir  = base_dir / args.slices_dir
    img_dir     = slices_dir / "images"
    mask_dir    = slices_dir / "masks"

    # ── 1. Extract slices and masks ──────────────────────────────────────
    if args.regen_slices or not img_dir.exists() or not list(img_dir.glob("*.png")):
        if not brats_dir.is_dir():
            print(f"ERROR: BraTS directory not found: {brats_dir}")
            print("Note: The dataset might not be currently downloaded here. If you just want to run the app via pre-trained weights, you don't need to run this script.")
            sys.exit(1)

        patient_dirs = sorted(brats_dir.glob("BraTS-GLI-*"))
        print(f"[INFO] Found {len(patient_dirs)} patient cases in {brats_dir}")

        total_extracted = 0
        for pd in patient_dirs:
            t1c_files = list(pd.glob("*-t1c.nii.gz"))
            seg_files = list(pd.glob("*-seg.nii.gz"))
            
            if not t1c_files or not seg_files:
                continue
                
            saved = extract_tumor_mask_pairs(
                t1c_files[0], seg_files[0],
                img_dir, mask_dir,
                skip_pct=0.15,
                max_slices=args.max_slices
            )
            total_extracted += saved
            print(f"  [{pd.name}] extracted {saved} image-mask pairs")

        print(f"\n[INFO] Total extracted pairs: {total_extracted}")
    else:
        print("[INFO] Using existing extracted slices.")


    # ── 2. Load Data and Model ───────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on: {device}")

    train_loader, val_loader = build_dataloaders(slices_dir, args.bs, args.img_size)

    model = ACUNet(in_channels=3, out_channels=1).to(device)
    
    # Using the combined DICE + CrossEntropy Loss (from ACU-Net)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ── 3. Training Loop ──────────────────────────────────────────────────
    print(f"\n[START] ACU-Net Training — {args.epochs} Epochs")
    
    model_dir = base_dir / "model"
    model_dir.mkdir(exist_ok=True)
    save_path = model_dir / "brain_tumor_acunet.pth"
    best_val_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_dice = eval_epoch(model, val_loader, criterion, device)
        
        print(f"  Epoch {epoch:02d}/{args.epochs} | "
              f"train loss={tr_loss:.4f} | val loss={vl_loss:.4f} | val dice={vl_dice:.4f}")
              
        if vl_dice > best_val_dice:
            best_val_dice = vl_dice
            torch.save(model.state_dict(), save_path)

    print(f"\n[SAVED] Best ACU-Net Model (val_dice={best_val_dice:.4f}) -> {save_path}")
    print("[DONE]")

if __name__ == "__main__":
    main()
