"""
Brain Tumor Detection — Dual Pipeline (ACU-Net + OpenCV Fallback)

If the ACU-Net model weights are present, it uses the Deep Learning
Attention-based U-Net for accurate semantic segmentation.
Otherwise, it falls back to the legacy OpenCV heuristic pipeline.
"""

import cv2
import numpy as np
import os
import sys

# ── PyTorch Integration ──────────────────────────────────────────────
try:
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    from acu_net import ACUNet
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "brain_tumor_acunet.pth")

_model = None
_device = None

if TORCH_AVAILABLE and os.path.exists(MODEL_PATH):
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = ACUNet(in_channels=3, out_channels=1)
    _model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
    _model.to(_device)
    _model.eval()
    print("[INFO] ACU-Net model loaded successfully for Semantic Segmentation.")


def auto_canny(image, sigma=0.33):
    """Automatically determine lower/upper Canny thresholds."""
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


def detect_tumor(image_path, output_dir, has_tumor=True):
    """
    Run the tumor detection pipeline.
    Uses ACU-Net if available, otherwise OpenCV.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Resize for consistency
    dim = (500, 590)
    image = cv2.resize(image, dim)

    basename = os.path.splitext(os.path.basename(image_path))[0]
    results = {}

    # ── Step 1: Save original (resized) ────────────────────────────────
    orig_path = os.path.join(output_dir, f"{basename}_original.jpg")
    cv2.imwrite(orig_path, image)
    results["original"] = os.path.basename(orig_path)
    
    # Brain area from grayscale is needed for ratio calculation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brain_mask = gray > 30
    brain_area = np.sum(brain_mask)

    # ── Try ACU-Net Deep Learning Segmentation ──
    if _model is not None:
        pil_img = Image.open(image_path).convert('RGB').resize((256, 256))
        tensor_img = TF.to_tensor(pil_img)
        tensor_img = TF.normalize(tensor_img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0).to(_device)
        
        with torch.no_grad():
            output = _model(tensor_img)
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            
        if not has_tumor:
            pred_mask = np.zeros_like(pred_mask)
            
        # Threshold at 0.5
        binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
        
        # Resize mask back to UI dimensions
        binary_mask = cv2.resize(binary_mask, dim, interpolation=cv2.INTER_NEAREST)
        
        # Save Acu-Net Mask representation
        acunet_path = os.path.join(output_dir, f"{basename}_acunet_mask.jpg")
        cv2.imwrite(acunet_path, binary_mask)
        results["grayscale"] = os.path.basename(acunet_path) # Override display with network mask
        results["threshold"] = os.path.basename(acunet_path)
        results["morphology"] = os.path.basename(acunet_path)
        results["canny"] = os.path.basename(acunet_path)
        
        # We find contours directly on the Deep Learning Mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    else:
        # ── OpenCV Heuristic Segmentation ──
        gray_path = os.path.join(output_dir, f"{basename}_grayscale.jpg")
        cv2.imwrite(gray_path, gray)
        results["grayscale"] = os.path.basename(gray_path)

        # Smooth image to remove noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Isolate brain pixels (brightness > 30)
        brain_pixels = blurred[blurred > 30]
        if len(brain_pixels) > 0:
            # Set threshold to the 95th percentile of brain brightness to catch the dense, bright tumor
            thresh_val = np.percentile(brain_pixels, 95)
        else:
            thresh_val = 130
        
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

        thresh_path = os.path.join(output_dir, f"{basename}_threshold.jpg")
        cv2.imwrite(thresh_path, thresh)
        results["threshold"] = os.path.basename(thresh_path)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        # Initial closing to merge small gaps
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # Gentle erosion to remove scattered noise (not 14 iterations which obliterated small tumors)
        closed = cv2.erode(closed, None, iterations=3)
        # Dilate to bring the tumor back to its approximate original size
        closed = cv2.dilate(closed, None, iterations=4)
        
        morph_path = os.path.join(output_dir, f"{basename}_morphology.jpg")
        cv2.imwrite(morph_path, closed)
        results["morphology"] = os.path.basename(morph_path)

        canny = auto_canny(closed)
        canny_path = os.path.join(output_dir, f"{basename}_canny.jpg")
        cv2.imwrite(canny_path, canny)
        results["canny"] = os.path.basename(canny_path)

        if not has_tumor:
            # Prevent false positives in OpenCV fallback if classifer says no tumor
            thresh = np.zeros_like(thresh)
            closed = np.zeros_like(closed)
            canny = np.zeros_like(canny)
            contours = []
            cv2.imwrite(thresh_path, thresh)
            cv2.imwrite(morph_path, closed)
            cv2.imwrite(canny_path, canny)
        else:
            contours, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ── Final Contour Drawing & Calculations ───────────────────────────
    contour_image = image.copy()
    tumor_found = False
    tumor_area = 0.0
    tumor_ratio = 0.0

    if contours:
        # Filter noise
        significant_contours = [c for c in contours if cv2.contourArea(c) > 500]

        if significant_contours:
            tumor_found = True
            cv2.drawContours(contour_image, significant_contours, -1, (0, 0, 255), 2)

            largest = max(significant_contours, key=cv2.contourArea)
            tumor_area = cv2.contourArea(largest)

            overlay = contour_image.copy()
            cv2.drawContours(overlay, [largest], -1, (0, 0, 255), -1)
            contour_image = cv2.addWeighted(overlay, 0.3, contour_image, 0.7, 0)
            cv2.drawContours(contour_image, significant_contours, -1, (0, 0, 255), 2)

            if brain_area > 0:
                tumor_ratio = tumor_area / brain_area

    contour_path = os.path.join(output_dir, f"{basename}_contour.jpg")
    cv2.imwrite(contour_path, contour_image)
    
    results["contour"] = os.path.basename(contour_path)
    results["tumor_found"] = tumor_found
    results["tumor_area"] = round(tumor_area, 2)
    results["tumor_ratio"] = round(tumor_ratio, 6)

    return results
