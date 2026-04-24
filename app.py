"""
Brain Tumor Detection — Flask Web Application

Routes:
  GET  /          → Serve the single-page UI
  POST /analyze   → Accept MRI image, run classification + detection
  POST /calculate → Tumor geometric volume & mass calculator
"""

import os
import uuid
import math
import json
from flask import Flask, render_template, request, jsonify, send_from_directory

from tumor_detector import detect_tumor
from classifier import classify

app = Flask(__name__)

UPLOAD_DIR = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/presentation")
def presentation():
    return send_from_directory(os.path.dirname(__file__), "presentation.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    # Save uploaded file with a unique name
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_DIR, unique_name)
    file.save(filepath)

    try:
        # 1. Classification
        label, confidence = classify(filepath)
        has_tumor_flag = (label == "yes")

        # 2. Tumor detection (OpenCV pipeline)
        detection = detect_tumor(filepath, UPLOAD_DIR, has_tumor=has_tumor_flag)

        # 3. Volume analysis (estimate 3D volume from 2D contour area)
        volume_analysis = compute_volume_analysis(
            detection.get("tumor_area", 0),
            detection.get("tumor_ratio", 0),
            detection.get("tumor_found", False),
        )

        return jsonify({
            "classification": {
                "label": label,
                "confidence": confidence,
                "has_tumor": label == "yes",
            },
            "detection": detection,
            "volume_analysis": volume_analysis,
            "upload_base": "/static/uploads/",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Geometric Volume & Mass Calculator ────────────────────────────
@app.route("/calculate", methods=["POST"])
def calculate():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided."}), 400

    try:
        h = float(data.get("height", 0))
        w = float(data.get("width", 0))
        l = float(data.get("length", 0))
        d = float(data.get("density", 1.05))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid numeric values."}), 400

    if h <= 0 or w <= 0 or l <= 0 or d <= 0:
        return jsonify({"error": "All values must be positive."}), 400

    volume = h * w * l  # cm³
    mass = volume * d   # g

    # Ellipsoidal volume approximation (more clinically accurate)
    ellipsoid_volume = (math.pi / 6) * h * w * l  # cm³

    # Clinical classification based on longest dimension
    max_dim = max(h, w, l)
    if max_dim < 2:
        risk = "Low"
        description = "Small tumor — may be suitable for monitoring or surgical resection."
    elif max_dim < 4:
        risk = "Moderate"
        description = "Medium tumor — surgical evaluation recommended with possible adjuvant therapy."
    elif max_dim < 6:
        risk = "High"
        description = "Large tumor — likely requires multimodal treatment (surgery + radiation/chemo)."
    else:
        risk = "Critical"
        description = "Very large tumor — urgent multidisciplinary evaluation needed."

    return jsonify({
        "volume": round(volume, 2),
        "mass": round(mass, 2),
        "ellipsoid_volume": round(ellipsoid_volume, 2),
        "risk": risk,
        "description": description,
        "dimensions": {"height": h, "width": w, "length": l},
        "density": d,
    })


def compute_volume_analysis(tumor_area_px, tumor_ratio, tumor_found):
    """Estimate 3D metrics from a 2D contour area."""
    if not tumor_found or tumor_area_px <= 0:
        return {
            "estimated_volume_cm3": 0,
            "estimated_diameter_cm": 0,
            "risk_level": "None",
            "risk_score": 0,
            "surgery_suitable": False,
            "radiation_suitable": False,
            "chemo_suitable": False,
            "clinical_note": "No significant tumor region detected.",
        }

    # Assume MRI pixel spacing ~0.5 mm/pixel → area in cm²
    pixel_spacing_cm = 0.05  # 0.5 mm = 0.05 cm
    area_cm2 = tumor_area_px * (pixel_spacing_cm ** 2)

    # Equivalent-sphere approximation: A = π r², V = (4/3) π r³
    radius_cm = math.sqrt(area_cm2 / math.pi)
    volume_cm3 = (4 / 3) * math.pi * (radius_cm ** 3)
    diameter_cm = 2 * radius_cm

    # Risk classification
    if volume_cm3 < 1:
        risk_level, risk_score = "Low", 25
        note = "Small estimated volume. Monitoring or minimally invasive surgery may be appropriate."
        surgery, radiation, chemo = True, False, False
    elif volume_cm3 < 10:
        risk_level, risk_score = "Moderate", 50
        note = "Moderate volume. Surgical evaluation recommended with possible radiation therapy."
        surgery, radiation, chemo = True, True, False
    elif volume_cm3 < 30:
        risk_level, risk_score = "High", 75
        note = "Significant volume. Multimodal treatment strongly recommended."
        surgery, radiation, chemo = True, True, True
    else:
        risk_level, risk_score = "Critical", 95
        note = "Very large estimated volume. Urgent multidisciplinary review needed."
        surgery, radiation, chemo = True, True, True

    return {
        "estimated_volume_cm3": round(volume_cm3, 3),
        "estimated_diameter_cm": round(diameter_cm, 2),
        "risk_level": risk_level,
        "risk_score": risk_score,
        "surgery_suitable": surgery,
        "radiation_suitable": radiation,
        "chemo_suitable": chemo,
        "clinical_note": note,
    }


@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
