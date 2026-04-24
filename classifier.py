"""
Brain Tumor Classifier — Dual-Mode

Mode 1 (Model):     Loads a trained PyTorch model (.pth) if available.
Mode 2 (Heuristic): Falls back to an OpenCV-based heuristic analysis
                     so the app works immediately without a GPU or dataset.
"""

import os
import cv2
import numpy as np

# ── Try to import PyTorch (optional) ──────────────────────────────────
try:
    import torch
    import torchvision.transforms as transforms
    import torchvision.models as models
    from PIL import Image

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "brain_tumor_resnet18.pth")


def _load_model():
    """Load a trained ResNet18 model if the file exists."""
    if not TORCH_AVAILABLE or not os.path.exists(MODEL_PATH):
        return None

    model = models.resnet18(weights=None)
    # Adjust final fully-connected layer for 2 classes (tumor / no-tumor)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


_model = _load_model()


def _predict_with_model(image_path):
    """Classify using the trained PyTorch model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = _model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    classes = ["no", "yes"]
    label = classes[predicted.item()]
    return label, round(confidence.item() * 100, 2)


def _predict_heuristic(image_path):
    """
    Heuristic classifier based on OpenCV image analysis.

    Analyses:
      1. Intensity histogram — tumors create bright spots
      2. Contour area ratio from the morphological pipeline
      3. Symmetry of intensity distribution
    """
    image = cv2.imread(image_path)
    if image is None:
        return "no", 50.0

    image = cv2.resize(image, (500, 590))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- Feature 1: High-intensity pixel ratio ---
    # Brain tissue above a threshold suggests bright anomalies
    brain_mask = gray > 30  # exclude background
    brain_pixels = gray[brain_mask]

    if len(brain_pixels) == 0:
        return "no", 60.0

    high_intensity_ratio = np.sum(brain_pixels > 180) / len(brain_pixels)

    # --- Feature 2: Intensity variance ---
    intensity_std = np.std(brain_pixels.astype(float))

    # --- Feature 3: Morphological contour analysis ---
    _, thresh = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=14)
    closed = cv2.dilate(closed, None, iterations=13)
    canny = cv2.Canny(closed, 30, 150)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_contour_area = 0
    if contours:
        max_contour_area = max(cv2.contourArea(c) for c in contours)

    contour_ratio = max_contour_area / (500 * 590)

    # --- Feature 4: Bilateral asymmetry ---
    h, w = gray.shape
    left_half = gray[:, :w // 2]
    right_half = cv2.flip(gray[:, w // 2:], 1)
    # Match sizes
    min_w = min(left_half.shape[1], right_half.shape[1])
    left_half = left_half[:, :min_w]
    right_half = right_half[:, :min_w]
    asymmetry = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))

    # --- Scoring ---
    score = 0.0

    # High-intensity spots
    if high_intensity_ratio > 0.15:
        score += 30
    elif high_intensity_ratio > 0.08:
        score += 15

    # High variance in brain region
    if intensity_std > 60:
        score += 20
    elif intensity_std > 45:
        score += 10

    # Significant contour detected
    if contour_ratio > 0.02:
        score += 30
    elif contour_ratio > 0.005:
        score += 15

    # Asymmetry indicates abnormality
    if asymmetry > 25:
        score += 20
    elif asymmetry > 15:
        score += 10

    # Normalize to 0-100 confidence
    confidence = min(score, 100)

    if confidence >= 45:
        return "yes", round(confidence, 2)
    else:
        return "no", round(100 - confidence, 2)


def classify(image_path):
    """
    Classify a brain MRI image.

    Returns:
        tuple: (label, confidence)
            label — "yes" (tumor) or "no" (no tumor)
            confidence — float 0-100
    """
    if _model is not None:
        return _predict_with_model(image_path)
    else:
        return _predict_heuristic(image_path)
