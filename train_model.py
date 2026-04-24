"""
Brain Tumor Classification — FastAI Training Script

Usage:
    python train_model.py --data_dir /path/to/brain_mri_dataset

The dataset should have two folders:
    yes/   — MRI images with tumors
    no/    — MRI images without tumors

This script trains a VGG16 model using transfer learning and saves
the weights to  model/brain_tumor_vgg16.pth
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Train Brain Tumor Classifier")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset with 'yes' and 'no' subfolders")
    parser.add_argument("--epochs_frozen", type=int, default=15,
                        help="Epochs to train with frozen backbone")
    parser.add_argument("--epochs_unfrozen", type=int, default=10,
                        help="Epochs to train after unfreezing")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Max learning rate for stage 1")
    parser.add_argument("--lr_fine", type=float, default=1e-5,
                        help="Max learning rate for stage 2 (after unfreeze)")
    parser.add_argument("--bs", type=int, default=8, help="Batch size")
    parser.add_argument("--size", type=int, default=224, help="Image size")
    args = parser.parse_args()

    # ── Check for FastAI ──────────────────────────────────────────────
    try:
        from fastai.vision.all import (
            ImageDataLoaders, aug_transforms, cnn_learner,
            accuracy, models, Resize
        )
        import torch
    except ImportError:
        print("ERROR: FastAI is required for training.")
        print("Install it with:  pip install fastai")
        sys.exit(1)

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        sys.exit(1)

    print(f"[INFO] Loading data from {args.data_dir}")
    print(f"[INFO] Image size: {args.size}, Batch size: {args.bs}")

    # ── Create DataLoaders ────────────────────────────────────────────
    dls = ImageDataLoaders.from_folder(
        args.data_dir,
        train=".",
        valid_pct=0.2,
        item_tfms=Resize(args.size),
        batch_tfms=aug_transforms(flip_vert=True, max_warp=0),
        bs=args.bs,
        num_workers=0,
    )

    print(f"[INFO] Classes: {dls.vocab}")
    print(f"[INFO] Training samples: {len(dls.train_ds)}, Validation: {len(dls.valid_ds)}")

    # ── Stage 1: Train with frozen backbone ───────────────────────────
    print("\n[STAGE 1] Training with frozen backbone...")
    learn = cnn_learner(dls, models.vgg16_bn, metrics=[accuracy])

    learn.fit_one_cycle(args.epochs_frozen, max_lr=args.lr)
    print(f"[STAGE 1] Complete. Validation accuracy shown above.")

    # ── Stage 2: Unfreeze and fine-tune ───────────────────────────────
    print("\n[STAGE 2] Unfreezing and fine-tuning...")
    learn.unfreeze()
    learn.fit_one_cycle(args.epochs_unfrozen, max_lr=args.lr_fine)
    print(f"[STAGE 2] Complete.")

    # ── Save model ────────────────────────────────────────────────────
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "brain_tumor_vgg16.pth")

    torch.save(learn.model.state_dict(), save_path)
    print(f"\n[SAVED] Model weights saved to: {save_path}")
    print("[DONE] Training complete.")


if __name__ == "__main__":
    main()
