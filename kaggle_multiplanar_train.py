# ==============================================================================
# BrainLens — Multi-Planar ACU-Net Training Script for Kaggle
# ==============================================================================
# Instructions:
#   1. In your Kaggle notebook, click "+ Add Data"
#   2. Search for "brats-2021-task1" by dschettler8845 and add it.
#   3. Make sure your previous trained model "brain_tumor_acunet.pth" is in the 
#      /kaggle/working directory. (If you restarted the notebook, you must upload it).
#   4. Run this script! It will slice the 3D brains in all 3 directions and 
#      continue training your existing model.
# ==============================================================================

import os
import sys
import random
import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1: ACU-Net Architecture 
# ──────────────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return x * self.ca(x) * self.sa(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class ACUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = DoubleConv(512, 1024)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.att4 = AttentionBlock(512)
        self.decoder4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att3 = AttentionBlock(256)
        self.decoder3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att2 = AttentionBlock(128)
        self.decoder2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att1 = AttentionBlock(64)
        self.decoder1 = DoubleConv(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))

        def _pad_and_cat(dec, enc):
            dy = enc.size(2) - dec.size(2)
            dx = enc.size(3) - dec.size(3)
            dec = F.pad(dec, [dx//2, dx-dx//2, dy//2, dy-dy//2])
            return torch.cat((enc, dec), dim=1)

        dec4 = self.decoder4(_pad_and_cat(self.upconv4(bottleneck), self.att4(enc4)))
        dec3 = self.decoder3(_pad_and_cat(self.upconv3(dec4), self.att3(enc3)))
        dec2 = self.decoder2(_pad_and_cat(self.upconv2(dec3), self.att2(enc2)))
        dec1 = self.decoder1(_pad_and_cat(self.upconv1(dec2), self.att1(enc1)))
        return self.final_conv(dec1)

class DiceBCELoss(nn.Module):
    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        bce_loss = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean')
        return dice_loss + bce_loss

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2: 3D to 2D Multi-Planar Slicing Engine
# ──────────────────────────────────────────────────────────────────────────────

def extract_multiplanar_slices(brats_dir, out_dir, num_patients=200, slices_per_plane=5):
    """
    Slices the 3D BraTS cubes into 2D PNG images across all 3 planes 
    (Axial, Sagittal, Coronal) ensuring the model learns all views.
    """
    img_dir = os.path.join(out_dir, "images")
    mask_dir = os.path.join(out_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    # If we already extracted images in a previous cell, skip to save time
    if len(glob.glob(os.path.join(img_dir, "*.png"))) > 100:
        print("[INFO] Multi-planar slices already extracted. Skipping extraction.")
        return img_dir, mask_dir
        
    # Find all T1ce files recursively, then get their parent directories (patient folders)
    t1ce_files = sorted(glob.glob(os.path.join(brats_dir, "**", "*t1ce.nii.gz"), recursive=True))
    if not t1ce_files:
        raise FileNotFoundError(f"Could not find any *t1ce.nii.gz files in {brats_dir}")
        
    # Get unique patient folders while preserving order
    patient_folders = list(dict.fromkeys([os.path.dirname(f) for f in t1ce_files]))[-num_patients:]
        
    print(f"\n[EXTRACTION] Slicing 3D MRI scans into Multi-Planar 2D Images...")
    print(f"[EXTRACTION] Processing {len(patient_folders)} patients. This will take a few minutes.\n")
    
    extracted_count = 0
    for pf in tqdm(patient_folders):
        p_id = os.path.basename(pf)
        # Using T1ce (contrast enhanced) as it shows the tumor clearest
        t1ce_path = glob.glob(os.path.join(pf, "*t1ce.nii.gz"))
        seg_path  = glob.glob(os.path.join(pf, "*seg.nii.gz"))
        
        if not t1ce_path or not seg_path:
            continue
            
        vol_img = nib.load(t1ce_path[0]).get_fdata()
        vol_seg = nib.load(seg_path[0]).get_fdata()
        
        # BraTS mask has multiple classes. We convert to Binary Mask (Whole Tumor vs Background)
        vol_seg = (vol_seg > 0).astype(np.uint8)
        
        # Normalize MRI to 0-255 grayscale
        if np.max(vol_img) > 0:
            vol_img = (vol_img / np.max(vol_img)) * 255.0
        vol_img = vol_img.astype(np.uint8)
        
        # ── 1. AXIAL PLANES (Top-Down / Z-axis) ──
        z_sums = np.sum(vol_seg, axis=(0, 1))
        z_idx = np.argsort(z_sums)[-slices_per_plane:] # Grab slices with the largest tumor area
        for z in z_idx:
            if z_sums[z] == 0: continue
            Image.fromarray(vol_img[:, :, z]).save(os.path.join(img_dir, f"{p_id}_ax_{z}.png"))
            Image.fromarray(vol_seg[:, :, z]*255).save(os.path.join(mask_dir, f"{p_id}_ax_{z}.png"))
            extracted_count += 1
            
        # ── 2. SAGITTAL PLANES (Side-to-Side / X-axis) ──
        x_sums = np.sum(vol_seg, axis=(1, 2))
        x_idx = np.argsort(x_sums)[-slices_per_plane:]
        for x in x_idx:
            if x_sums[x] == 0: continue
            Image.fromarray(vol_img[x, :, :]).save(os.path.join(img_dir, f"{p_id}_sag_{x}.png"))
            Image.fromarray(vol_seg[x, :, :]*255).save(os.path.join(mask_dir, f"{p_id}_sag_{x}.png"))
            extracted_count += 1
            
        # ── 3. CORONAL PLANES (Front-to-Back / Y-axis) ──
        y_sums = np.sum(vol_seg, axis=(0, 2))
        y_idx = np.argsort(y_sums)[-slices_per_plane:]
        for y in y_idx:
            if y_sums[y] == 0: continue
            Image.fromarray(vol_img[:, y, :]).save(os.path.join(img_dir, f"{p_id}_cor_{y}.png"))
            Image.fromarray(vol_seg[:, y, :]*255).save(os.path.join(mask_dir, f"{p_id}_cor_{y}.png"))
            extracted_count += 1

    print(f"\n[EXTRACTION] Successfully extracted {extracted_count} Multi-Planar Images!")
    return img_dir, mask_dir


class MultiPlanarDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=256, augment=False):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img  = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        img  = img.resize((self.img_size, self.img_size))
        mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)

        # ── NEW: NEGATIVE SAMPLING (FAKE EYES) ──
        # 50% chance to add a fake eye/bright spot to the edge of the image
        if self.augment and random.random() > 0.5:
            import cv2
            img_cv = np.array(img)
            # Pick a random edge location (top, bottom, left, or right margins)
            margin = int(self.img_size * 0.15)
            x = random.choice([random.randint(0, margin), random.randint(self.img_size-margin, self.img_size)])
            y = random.choice([random.randint(0, margin), random.randint(self.img_size-margin, self.img_size)])
            radius = random.randint(10, 25)
            
            # Draw a bright white circle (like an eye) on the MRI
            cv2.circle(img_cv, (x, y), radius, (255, 255, 255), -1)
            img = Image.fromarray(img_cv)
            # Notice we do NOT draw on the mask! The mask stays 0 (Background).

        # Standard data augmentation
        if self.augment and random.random() > 0.5:
            img, mask = TF.hflip(img), TF.hflip(mask)
        if self.augment and random.random() > 0.5:
            img, mask = TF.vflip(img), TF.vflip(mask)

        img_tensor  = TF.to_tensor(img)
        img_tensor  = TF.normalize(img_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        mask_tensor = (TF.to_tensor(mask) > 0.5).float()
        return img_tensor, mask_tensor

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3: Training Functions
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, dice_score = 0, 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            total_loss += criterion(outputs, masks).item() * imgs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            intersection = (preds * masks).sum().item()
            union = preds.sum().item() + masks.sum().item()
            if union > 0:
                dice_score += (2.0 * intersection / union) * imgs.size(0)
    n = len(loader.dataset)
    return total_loss / n, dice_score / n

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4: MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

# Path configuration
# ── Auto-detect or Extract the correct BraTS dataset path on Kaggle ──
# Try to find any T1ce NIfTI file, regardless of exact prefix
_candidates = glob.glob("/kaggle/input/**/*t1ce.nii.gz", recursive=True)
if not _candidates:
    _candidates = glob.glob("/kaggle/input/**/*.nii.gz", recursive=True)
    
if not _candidates:
    # Check if we already extracted it to /kaggle/working/
    _candidates = glob.glob("/kaggle/working/BraTS2021_Extracted/**/*.nii.gz", recursive=True)

if _candidates:
    # Gets the parent directory of the patient folder
    BRATS_ROOT = os.path.dirname(os.path.dirname(_candidates[0]))
    print(f"[INFO] Auto-detected BraTS dataset at: {BRATS_ROOT}")
else:
    # If no .nii.gz files are found, the dataset is likely still packed in a .tar archive!
    print("\n[INFO] No raw .nii.gz files found. Searching for compressed .tar archives...")
    tar_files = glob.glob("/kaggle/input/**/*.tar", recursive=True)
    
    # Kaggle provides individual patient tars AND a massive Training_Data tar. We want the Training Data.
    training_tar = [t for t in tar_files if 'training_data' in t.lower()]
    brats_tar = training_tar if training_tar else tar_files
    
    if brats_tar:
        tar_path = brats_tar[0]
        BRATS_ROOT = "/kaggle/working/BraTS2021_Extracted"
        print(f"[INFO] Found compressed dataset: {tar_path}")
        print(f"[INFO] Extracting .tar archive to {BRATS_ROOT}... (This will take a few minutes!)")
        
        import tarfile
        os.makedirs(BRATS_ROOT, exist_ok=True)
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=BRATS_ROOT)
            
        print(f"[INFO] Extraction complete!")
    else:
        raise FileNotFoundError("Could not locate BraTS 2021 .nii.gz files OR .tar files. Did you add the correct dataset?")

OUTPUT_DIR = "/kaggle/working/extracted_slices"
NEW_MODEL_PATH = "/kaggle/working/brain_tumor_acunet_negative_sampling.pth"

# Auto-detect the uploaded .pth file (Kaggle mounts uploaded models in /kaggle/input/)
_pth_candidates = glob.glob("/kaggle/input/**/brain_tumor_acunet.pth", recursive=True)
if _pth_candidates:
    EXISTING_MODEL_PATH = _pth_candidates[0]
    print(f"[INFO] Auto-detected previously trained model at: {EXISTING_MODEL_PATH}")
else:
    EXISTING_MODEL_PATH = "/kaggle/working/brain_tumor_acunet.pth"

IMG_SIZE   = 256
BATCH_SIZE = 32     # Increased to 32 to feed both Kaggle GPUs
EPOCHS     = 15     # Increased to 15 epochs to ensure it learns the Negative Sampling trick
LR         = 5e-5   # Lower learning rate because model is already trained

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Using Device: {device}")
    
    # 1. Extract 2D slices from 3D data across all 3 planes
    img_dir, mask_dir = extract_multiplanar_slices(
        BRATS_ROOT, OUTPUT_DIR, 
        num_patients=200,     # Limit to 200 patients to save Kaggle disk space
        slices_per_plane=5    # Extract 5 best slices from each angle (Axial, Coronal, Sagittal)
    )
    
    # 2. Build Dataloaders
    dataset = MultiPlanarDataset(img_dir, mask_dir, img_size=IMG_SIZE, augment=True)
    # 85/15 Train/Val Split
    train_sz = int(0.85 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_sz, len(dataset) - train_sz])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 3. Initialize Model and LOAD PREVIOUS WEIGHTS
    model = ACUNet(in_channels=3, out_channels=1).to(device)
    
    if os.path.exists(EXISTING_MODEL_PATH):
        print(f"\n[INFO] Found existing trained model at: {EXISTING_MODEL_PATH}")
        print("[INFO] Loading weights... The model will CONTINUE training and learn the new angles!")
        model.load_state_dict(torch.load(EXISTING_MODEL_PATH, map_location=device))
    else:
        print(f"\n[WARNING] Could not find {EXISTING_MODEL_PATH}.")
        print("Starting training from scratch. (If you want to keep your previous progress, make sure you uploaded the .pth file to /kaggle/working/)")
        
    # ── MULTI-GPU SUPPORT ──
    if torch.cuda.device_count() > 1:
        print(f"\n[INFO] Accelerating with {torch.cuda.device_count()} GPUs using DataParallel!")
        model = nn.DataParallel(model)
        
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 4. Train
    best_dice = 0.0
    print(f"\n[START] Multi-Planar Fine-Tuning — {EPOCHS} epochs\n")
    
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_dice = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        print(f"  Epoch {epoch:02d}/{EPOCHS} | train_loss={tr_loss:.4f} | "
              f"val_loss={vl_loss:.4f} | val_dice={vl_dice:.4f}")
        
        if vl_dice > best_dice:
            best_dice = vl_dice
            # Ensure we save correctly whether using 1 GPU or 2 GPUs
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), NEW_MODEL_PATH)
            else:
                torch.save(model.state_dict(), NEW_MODEL_PATH)
            print(f"  ✓ Saved new best multi-planar model")

    print(f"\n[DONE] Finished training! Download the new model from:")
    print(f" >>> {NEW_MODEL_PATH} <<<")
