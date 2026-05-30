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
    import torch.nn as nn
    import torchvision.transforms.functional as TF
    from torchvision import models
    from PIL import Image
    from acu_net import ACUNet
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"[tumor_detector] FATAL: Failed to import PyTorch: {e}")
    TORCH_AVAILABLE = False

ACUNET_PATH = os.path.join(os.path.dirname(__file__), "model", "brain_tumor_acunet (1).pth")
CLASSIFIER_PATH = os.path.join(os.path.dirname(__file__), "model", "brain_tumor_classifier_multiplanar.pth")

_acunet = None
_classifier = None
_device = None

if TORCH_AVAILABLE:
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(ACUNET_PATH):
        _acunet = ACUNet(in_channels=3, out_channels=1)
        _acunet.load_state_dict(torch.load(ACUNET_PATH, map_location=_device))
        _acunet.to(_device)
        _acunet.eval()
        print("[INFO] ACU-Net loaded for Semantic Segmentation.")
        
    if os.path.exists(CLASSIFIER_PATH):
        _classifier = models.resnet18(pretrained=False)
        _classifier.fc = nn.Linear(_classifier.fc.in_features, 2)
        # Handle DataParallel if trained on dual Kaggle GPUs
        state_dict = torch.load(CLASSIFIER_PATH, map_location=_device)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        _classifier.load_state_dict(state_dict)
        _classifier.to(_device)
        _classifier.eval()
        print("[INFO] ResNet18 loaded for Tumor Classification.")



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

    # ── Cascade Step 1: ResNet18 Classifier Gatekeeper ──
    if _classifier is not None:
        pil_cls = Image.open(image_path).convert('RGB').resize((224, 224))
        tensor_cls = TF.to_tensor(pil_cls)
        tensor_cls = TF.normalize(tensor_cls, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0).to(_device)
        
        with torch.no_grad():
            out = _classifier(tensor_cls)
            probs = torch.softmax(out, dim=1)[0]
            prob_tumor = probs[1].item()
            is_tumor = prob_tumor > 0.5
            
        results["ai_classification"] = bool(is_tumor)
        results["ai_confidence"] = float(prob_tumor)
        
        # Override the legacy has_tumor flag with our AI's strict decision
        if not is_tumor:
            has_tumor = False

    # ── Cascade Step 2: ACU-Net Semantic Segmentation ──
    if _acunet is not None:
        pil_img = Image.open(image_path).convert('RGB').resize((224, 224))
        tensor_img = TF.to_tensor(pil_img).unsqueeze(0).to(_device)
        
        with torch.no_grad():
            output = _acunet(tensor_img)
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            
        if not has_tumor:
            pred_mask = np.zeros_like(pred_mask)
            
        # Threshold at 0.5
        binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
        
        # Resize mask back to UI dimensions
        binary_mask = cv2.resize(binary_mask, dim, interpolation=cv2.INTER_NEAREST)
        
        # ── Generating realistic representations of AI pipeline steps ──
        
        # 1. Grayscale: The original brain image input
        gray_path = os.path.join(output_dir, f"{basename}_grayscale.jpg")
        cv2.imwrite(gray_path, gray)
        results["grayscale"] = os.path.basename(gray_path)
        
        # 2. Thresholding -> AI Probability Heatmap
        # pred_mask contains continuous float values 0.0 to 1.0 representing AI confidence
        pred_map_resized = cv2.resize(pred_mask, dim, interpolation=cv2.INTER_LINEAR)
        prob_heatmap = (pred_map_resized * 255).astype(np.uint8)
        heatmap_path = os.path.join(output_dir, f"{basename}_threshold.jpg")
        cv2.imwrite(heatmap_path, prob_heatmap)
        results["threshold"] = os.path.basename(heatmap_path)
        
        # 3. Morphology -> AI Binary Mask
        # The clean binary mask after the 0.5 threshold
        mask_path = os.path.join(output_dir, f"{basename}_morphology.jpg")
        cv2.imwrite(mask_path, binary_mask)
        results["morphology"] = os.path.basename(mask_path)
        
        # 4. Canny Edge -> AI Contour Edge
        # The exact contour edge detected by the AI mask
        ai_edge = auto_canny(binary_mask)
        edge_path = os.path.join(output_dir, f"{basename}_canny.jpg")
        cv2.imwrite(edge_path, ai_edge)
        results["canny"] = os.path.basename(edge_path)
        
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
            # Set threshold to the 85th percentile of brain brightness to catch the dense, bright tumor
            thresh_val = np.percentile(brain_pixels, 85)
        else:
            thresh_val = 130
        
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

        thresh_path = os.path.join(output_dir, f"{basename}_threshold.jpg")
        cv2.imwrite(thresh_path, thresh)
        results["threshold"] = os.path.basename(thresh_path)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        # Initial closing to merge small gaps
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # Gentle erosion to remove scattered noise
        closed = cv2.erode(closed, None, iterations=2)
        # Dilate to bring the tumor back to its approximate original size
        closed = cv2.dilate(closed, None, iterations=3)
        
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

    # Spatial analysis fields (populated when tumor is found)
    centroid_x_pct   = None   # 0.0 = left, 1.0 = right
    centroid_y_pct   = None   # 0.0 = top,  1.0 = bottom
    compactness      = None   # circularity metric: 4π·A / P²  (1=circle, <1=irregular)
    mean_intensity   = None   # mean pixel brightness inside the largest contour mask
    intensity_std    = None   # std of pixel brightness inside the mask
    contour_count    = 0      # how many significant contour blobs found

    if contours:
        # Filter noise and skull/scalp edges using Shape Analysis
        h_img, w_img = image.shape[:2]
        significant_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 500:
                x, y, w, h = cv2.boundingRect(c)
                
                # The skull forms a large ring. Rings have huge bounding boxes.
                # A tumor is localized. If it spans > 60% of the image, it's the skull.
                if w > w_img * 0.60 or h > h_img * 0.60:
                    continue
                    
                # The skull is often U-shaped or a ring, which has very low solidity.
                # Solid tumors have high solidity.
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = float(area) / hull_area
                    if solidity < 0.50:
                        continue
                        
                significant_contours.append(c)

        if significant_contours:
            tumor_found = True
            
            # Find the largest contour and discard the rest (like broken skull pieces)
            largest = max(significant_contours, key=cv2.contourArea)
            significant_contours = [largest]
            contour_count = 1
            
            tumor_area = cv2.contourArea(largest)

            overlay = contour_image.copy()
            cv2.drawContours(overlay, [largest], -1, (0, 0, 255), -1)
            contour_image = cv2.addWeighted(overlay, 0.3, contour_image, 0.7, 0)
            cv2.drawContours(contour_image, significant_contours, -1, (0, 0, 255), 2)

            if brain_area > 0:
                tumor_ratio = tumor_area / brain_area

            # ── Spatial / shape metrics ───────────────────────────────────
            h_img, w_img = image.shape[:2]

            # Centroid (normalised 0-1)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                centroid_x_pct = round(cx / w_img, 4)
                centroid_y_pct = round(cy / h_img, 4)

            # Compactness (circularity)
            perimeter = cv2.arcLength(largest, True)
            if perimeter > 0:
                compactness = round((4 * np.pi * tumor_area) / (perimeter ** 2), 4)

            # Intensity inside the mask
            mask_img = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask_img, [largest], -1, 255, -1)
            masked_pixels = gray[mask_img > 0].astype(float)
            if len(masked_pixels) > 0:
                mean_intensity = round(float(np.mean(masked_pixels)), 2)
                intensity_std  = round(float(np.std(masked_pixels)), 2)

    contour_path = os.path.join(output_dir, f"{basename}_contour.jpg")
    cv2.imwrite(contour_path, contour_image)

    results["contour"]          = os.path.basename(contour_path)
    results["tumor_found"]      = tumor_found
    results["tumor_area"]       = round(tumor_area, 2)
    results["tumor_ratio"]      = round(tumor_ratio, 6)
    results["centroid_x_pct"]   = centroid_x_pct
    results["centroid_y_pct"]   = centroid_y_pct
    results["compactness"]      = compactness
    results["mean_intensity"]   = mean_intensity
    results["intensity_std"]    = intensity_std
    results["contour_count"]    = contour_count

    return results
