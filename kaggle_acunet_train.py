# ==============================================================================
# BrainLens — ACU-Net 3D Multi-Planar Training Script for Kaggle (Dual T4 GPUs)
# ==============================================================================
# Instructions:
#   1. In Kaggle, click "+ Add Data" -> upload your BraTS dataset (the one with .nii.gz files).
#   2. Set Notebook Settings: Accelerator -> GPU T4 x2 (Dual GPUs).
#   3. Paste this script and run!
#   4. The script will find the NIfTI files, mathematically extract optimal 
#      Axial, Sagittal, and Coronal slices from the tumor core, and train ACU-Net.
# ==============================================================================

import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
import random

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1: Mathematical Multi-Planar Slice Extraction
# ──────────────────────────────────────────────────────────────────────────────

EXTRACT_DIR = "/kaggle/working/extracted_multiplanar"
IMG_DIR = os.path.join(EXTRACT_DIR, "images")
MASK_DIR = os.path.join(EXTRACT_DIR, "masks")

def extract_multiplanar_slices():
    print("\n[INFO] Starting Mathematical Multi-Planar Slice Extraction...")
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)
    
    # Auto-find BraTS dataset (checking for raw NIfTI files first)
    all_t1ce = glob.glob("/kaggle/input/**/*t1ce.nii*", recursive=True) + glob.glob("/kaggle/working/**/*t1ce.nii*", recursive=True)
    
    if not all_t1ce:
        print("[INFO] No extracted NIfTI files found. Looking for .tar archives...")
        import tarfile
        tar_files = glob.glob("/kaggle/input/**/*.tar", recursive=True)
        if tar_files:
            # Prefer the main training data tar
            target_tar = [t for t in tar_files if "Training_Data" in t]
            main_tar = target_tar[0] if target_tar else tar_files[0]
            
            print(f"[INFO] Found {main_tar}. Extracting exactly 200 patients to bypass Kaggle disk limits...")
            os.makedirs("/kaggle/working/BraTS_Extracted", exist_ok=True)
            
            with tarfile.open(main_tar) as tar:
                print("       Scanning tar contents (this takes about 30 seconds)...")
                members = tar.getmembers()
                
                # Filter strictly for t1ce and seg files to save massive amounts of disk space
                t1ce_members = [m for m in members if m.name.endswith('t1ce.nii.gz')][:200]
                
                # Match corresponding segmentation masks
                patient_prefixes = [m.name.replace('-t1ce.nii.gz', '').replace('_t1ce.nii.gz', '') for m in t1ce_members]
                seg_members = [m for m in members if m.name.endswith('seg.nii.gz') and any(p in m.name for p in patient_prefixes)]
                
                members_to_extract = t1ce_members + seg_members
                print(f"       Extracting {len(members_to_extract)} files directly to disk...")
                tar.extractall(path="/kaggle/working/BraTS_Extracted", members=members_to_extract)
                
            all_t1ce = glob.glob("/kaggle/working/BraTS_Extracted/**/*t1ce.nii*", recursive=True)
        else:
            raise FileNotFoundError("Could not find any *t1ce.nii.gz or .tar files in /kaggle/input/. Did you upload the BraTS dataset?")
    
    print(f"[INFO] Successfully staged {len(all_t1ce)} T1ce 3D Volumes.")
    
    # We will process a random subset of 200 patients to prevent RAM overflow and keep extraction fast
    random.seed(42)
    random.shuffle(all_t1ce)
    subset_t1ce = all_t1ce[:200]
    
    slice_count = 0
    
    for t1ce_path in subset_t1ce:
        # BraTS naming convention usually puts seg in the same folder
        seg_path = t1ce_path.replace("t1ce.nii", "seg.nii")
        if not os.path.exists(seg_path):
            continue
            
        try:
            vol = nib.load(t1ce_path).get_fdata()
            mask = nib.load(seg_path).get_fdata()
        except Exception:
            continue
            
        # Binarize mask (0 = bg, >0 = tumor)
        mask = (mask > 0).astype(np.uint8)
        
        # Find Bounding Box of the tumor
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            continue # Skip healthy brains, ACU-Net only trains on tumor regions!
            
        z_center = int(np.median(coords[2]))
        y_center = int(np.median(coords[1]))
        x_center = int(np.median(coords[0]))
        
        # Normalize volume to 0-255 grayscale
        vol_min, vol_max = np.percentile(vol[vol > 0], 1), np.percentile(vol[vol > 0], 99)
        if vol_max <= vol_min: continue
        vol = np.clip((vol - vol_min) / (vol_max - vol_min), 0, 1) * 255.0
        vol = vol.astype(np.uint8)
        
        mask = mask * 255
        
        patient_id = os.path.basename(os.path.dirname(t1ce_path))
        
        # Extract slices from all 3 planes at Center and +/- offsets
        offsets = [-3, 0, 3]
        
        for offset in offsets:
            z, y, x = z_center + offset, y_center + offset, x_center + offset
            
            # Axial (XY plane, slice along Z)
            if 0 <= z < vol.shape[2]:
                save_slice(vol[:, :, z], mask[:, :, z], f"{patient_id}_axial_z{z}")
                slice_count += 1
                
            # Coronal (XZ plane, slice along Y)
            if 0 <= y < vol.shape[1]:
                save_slice(vol[:, y, :], mask[:, y, :], f"{patient_id}_coronal_y{y}")
                slice_count += 1
                
            # Sagittal (YZ plane, slice along X)
            if 0 <= x < vol.shape[0]:
                save_slice(vol[x, :, :], mask[x, :, :], f"{patient_id}_sagittal_x{x}")
                slice_count += 1
                
    print(f"\n[INFO] Extracted {slice_count} Multi-Planar Slices perfectly targeted on tumors!")

def save_slice(img_slice, mask_slice, name):
    # Rotate slices to stand upright and resize to 224x224
    img_slice = cv2.resize(np.rot90(img_slice), (224, 224))
    mask_slice = cv2.resize(np.rot90(mask_slice, k=1), (224, 224), interpolation=cv2.INTER_NEAREST)
    
    cv2.imwrite(os.path.join(IMG_DIR, f"{name}.png"), img_slice)
    cv2.imwrite(os.path.join(MASK_DIR, f"{name}.png"), mask_slice)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2: ACU-Net Architecture (Advanced Attention U-Net)
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


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ACUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(512, 1024)
        self.cbam = CBAM(1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(128, 64)
        
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        
        b = self.bottleneck(self.pool4(d4))
        b = self.cbam(b)
        
        u4 = self.up4(b)
        u4 = self.up_conv4(torch.cat([u4, d4], dim=1))
        
        u3 = self.up3(u4)
        u3 = self.up_conv3(torch.cat([u3, d3], dim=1))
        
        u2 = self.up2(u3)
        u2 = self.up_conv2(torch.cat([u2, d2], dim=1))
        
        u1 = self.up1(u2)
        u1 = self.up_conv1(torch.cat([u1, d1], dim=1))
        
        return self.out_conv(u1)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3: Custom Loss & Dataset
# ──────────────────────────────────────────────────────────────────────────────

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)       
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        return bce + dice_loss

class SegmentationDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = img_path.replace("images", "masks")
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # Normalize image to 0-1
        img_np = np.array(img, dtype=np.float32) / 255.0
        # Normalization matches ResNet pre-processing (even though ACUNet isn't strictly ResNet, RGB scale helps)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        
        # Mask to binary (0 or 1)
        mask_np = np.array(mask, dtype=np.float32) / 255.0
        mask_np = (mask_np > 0.5).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        
        return img_tensor, mask_tensor

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4: Dual-GPU Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train_acunet():
    # 1. Extract slices first!
    if not os.path.exists(IMG_DIR):
        extract_multiplanar_slices()
        
    all_images = glob.glob(os.path.join(IMG_DIR, "*.png"))
    random.shuffle(all_images)
    
    val_size = int(len(all_images) * 0.15)
    val_paths = all_images[:val_size]
    train_paths = all_images[val_size:]
    
    train_dataset = SegmentationDataset(train_paths)
    val_dataset = SegmentationDataset(val_paths)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[SYSTEM] Building ACU-Net on {device}")
    
    model = ACUNet(in_channels=3, out_channels=1)
    
    # 2. Dual-GPU Acceleration
    if torch.cuda.device_count() > 1:
        print(f"[INFO] T4 x2 Detected! Accelerating training across {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    
    # 3. Optimal Epochs & Schedulers
    EPOCHS = 40
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_loss = float("inf")
    patience = 5
    patience_counter = 0
    MODEL_SAVE_PATH = "/kaggle/working/brain_tumor_acunet.pth"
    
    print("\n[TRAIN] Beginning ACU-Net Multi-Planar Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss (Dice+BCE): {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        scheduler.step()
        
        # Early Stopping & Checkpointing
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), MODEL_SAVE_PATH)
            else:
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[STOP] Early Stopping triggered! Model reached peak segmentation accuracy.")
                break

    print(f"\n[DONE] Best Val Loss: {best_loss:.4f}")
    print(f"[DONE] Multi-Planar ACU-Net Weights Saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_acunet()
