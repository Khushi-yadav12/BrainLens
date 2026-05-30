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

import sys
print("\n" + "="*65)
print("⏳ INITIALIZING AI MODELS...")
print("⏳ Please wait 1-2 minutes for PyTorch to load into memory.")
print("⏳ The Flask server will start on port 5000 automatically after.")
print("="*65 + "\n")
sys.stdout.flush()

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
        # Run the Cascade AI pipeline (ResNet18 -> ACU-Net)
        detection = detect_tumor(filepath, UPLOAD_DIR, has_tumor=True)

        # Extract the classification verdict directly from the new flawless Gatekeeper AI
        is_tumor = detection.get("ai_classification", False)
        label = "yes" if is_tumor else "no"
        
        raw_prob = detection.get("ai_confidence", 0.0)
        if is_tumor:
            confidence = float(raw_prob * 100.0)
        else:
            confidence = float((1.0 - raw_prob) * 100.0)

        # 3. Volume analysis (estimate 3D volume from 2D contour area)
        volume_analysis = compute_volume_analysis(
            detection.get("tumor_area", 0),
            detection.get("tumor_ratio", 0),
            detection.get("tumor_found", False),
        )

        # 4. Tumor characterization (type inference + detailed findings)
        characterization = characterize_tumor(detection, confidence, is_tumor)

        return jsonify({
            "classification": {
                "label": label,
                "confidence": confidence,
                "has_tumor": label == "yes",
            },
            "detection": detection,
            "volume_analysis": volume_analysis,
            "characterization": characterization,
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


def characterize_tumor(detection, classifier_confidence, has_tumor):
    """
    Infer likely tumor type, location, and characteristics from image analysis metrics.

    Uses spatial features (centroid, compactness, intensity, size ratio) to produce:
      - Likely tumor type with probability estimate
      - Anatomical location description
      - Shape characteristics
      - Key findings list
      - Disclaimer (not a medical diagnosis)

    Tumor type heuristics are based on common radiological characteristics:
      • Glioma (GBM/LGG): irregular shape (low compactness), high internal intensity
        variance, can appear anywhere but often in cerebral hemispheres
      • Meningioma: well-defined, round/oval (high compactness), strong peripheral
        enhancement, often at the brain periphery (high or low centroid_y_pct)
      • Pituitary adenoma: midline location (centroid_x_pct ≈ 0.5), lower half of
        brain (centroid_y_pct > 0.55), relatively homogeneous intensity
      • Acoustic neuroma: lateral position (centroid_x_pct < 0.35 or > 0.65),
        mid-level (centroid_y_pct 0.35–0.65)
    """
    if not has_tumor or not detection.get("tumor_found", False):
        return {
            "tumor_type": None,
            "tumor_type_confidence": None,
            "location": "N/A",
            "location_detail": "No tumor detected in this scan.",
            "shape_description": "N/A",
            "intensity_profile": "N/A",
            "findings": [],
            "disclaimer": "This is a computer-aided screening tool. Always consult a qualified radiologist.",
        }

    cx = detection.get("centroid_x_pct") or 0.5
    cy = detection.get("centroid_y_pct") or 0.5
    compactness = detection.get("compactness") or 0.5
    mean_int    = detection.get("mean_intensity") or 128
    int_std     = detection.get("intensity_std") or 30
    tumor_ratio = detection.get("tumor_ratio") or 0.0
    contour_cnt = detection.get("contour_count") or 1

    # ── Location inference ──────────────────────────────────────────────
    if cx < 0.40:
        h_side = "Left hemisphere"
        h_abbr = "L"
    elif cx > 0.60:
        h_side = "Right hemisphere"
        h_abbr = "R"
    else:
        h_side = "Midline / bilateral"
        h_abbr = "M"

    if cy < 0.35:
        v_region = "superior (frontal/parietal)"
    elif cy < 0.55:
        v_region = "central (temporal/parietal)"
    elif cy < 0.72:
        v_region = "inferior (temporal/occipital)"
    else:
        v_region = "basal (sellar/posterior fossa)"

    location_label = f"{h_side}, {v_region} region"

    # ── Shape characterization ─────────────────────────────────────────
    if compactness > 0.75:
        shape_desc = "Well-defined, rounded/oval mass (high compactness)"
        shape_tag  = "well-defined"
    elif compactness > 0.50:
        shape_desc = "Moderately defined mass with slightly irregular border"
        shape_tag  = "moderately-defined"
    else:
        shape_desc = "Irregular, infiltrative mass with poorly defined margins (low compactness)"
        shape_tag  = "irregular"

    # ── Intensity profile ─────────────────────────────────────────────
    if mean_int > 180:
        int_profile = "Hyperintense (very bright) — suggests dense tissue, calcification, or haemorrhage"
    elif mean_int > 140:
        int_profile = "Moderately hyperintense — suggests solid, vascularised mass"
    elif mean_int > 90:
        int_profile = "Isointense — similar signal to surrounding brain tissue"
    else:
        int_profile = "Hypointense — may suggest necrosis, cystic component, or oedema"

    if int_std > 50:
        int_profile += " with heterogeneous internal signal (suggesting necrosis or mixed composition)"
    elif int_std > 25:
        int_profile += " with moderate signal heterogeneity"
    else:
        int_profile += " with homogeneous internal signal"

    # ── Tumor type scoring ─────────────────────────────────────────────
    scores = {"Glioma": 0.0, "Meningioma": 0.0, "Pituitary Adenoma": 0.0, "Other / Unclassified": 0.0}

    # Glioma features: irregular, high variance, large, hemispheric
    if shape_tag == "irregular":
        scores["Glioma"] += 35
    elif shape_tag == "moderately-defined":
        scores["Glioma"] += 15
    if int_std > 40:
        scores["Glioma"] += 20
    if tumor_ratio > 0.05:
        scores["Glioma"] += 15
    if h_abbr != "M":
        scores["Glioma"] += 10
    if contour_cnt > 1:
        scores["Glioma"] += 10

    # Meningioma features: well-defined, round, peripheral (top/bottom), homogeneous
    if shape_tag == "well-defined":
        scores["Meningioma"] += 35
    if compactness > 0.70:
        scores["Meningioma"] += 20
    if cy < 0.20 or cy > 0.78:
        scores["Meningioma"] += 20    # very peripheral
    if int_std < 30:
        scores["Meningioma"] += 15
    if mean_int > 130:
        scores["Meningioma"] += 10

    # Pituitary adenoma features: midline, basal, relatively homogeneous, small-medium
    if h_abbr == "M":
        scores["Pituitary Adenoma"] += 35
    if cy > 0.55:
        scores["Pituitary Adenoma"] += 20
    if int_std < 35:
        scores["Pituitary Adenoma"] += 15
    if tumor_ratio < 0.04:
        scores["Pituitary Adenoma"] += 15

    # Fallback / Other
    scores["Other / Unclassified"] += 20

    # Pick highest-scoring type
    tumor_type = max(scores, key=scores.get)
    raw_score  = scores[tumor_type]
    # Normalise to a 0-100 confidence relative to other candidates
    total = sum(scores.values()) or 1
    type_conf = round(min((raw_score / total) * 100 * 1.8, 88.0), 1)

    # ── Key Findings list ──────────────────────────────────────────────
    findings = []

    findings.append(f"Abnormal hyperintense region detected by AI classifier (confidence {classifier_confidence:.1f}%)")

    if centroid_x_pct := detection.get("centroid_x_pct"):
        findings.append(f"Lesion centroid located at {cx*100:.0f}% from left, {cy*100:.0f}% from top of the imaged region")

    if compactness is not None:
        if compactness > 0.75:
            findings.append("Lesion border is well-defined and rounded, consistent with an encapsulated or extra-axial mass")
        elif compactness < 0.45:
            findings.append("Irregular, poorly-defined margins detected — pattern consistent with infiltrative or high-grade neoplasm")
        else:
            findings.append("Moderately defined lesion border observed")

    if mean_int is not None:
        if mean_int > 160:
            findings.append(f"High mean pixel intensity ({mean_int:.0f}/255) within lesion — may indicate hypervascular or dense tissue")
        elif mean_int < 90:
            findings.append(f"Low mean pixel intensity ({mean_int:.0f}/255) within lesion — may suggest necrotic or cystic components")

    if int_std is not None and int_std > 45:
        findings.append(f"High internal signal heterogeneity (std={int_std:.0f}) — suggests mixed solid/necrotic or haemorrhagic composition")

    if contour_cnt > 1:
        findings.append(f"Multiple lesion foci detected ({contour_cnt} separate contour regions) — may indicate multifocal disease or satellite nodules")

    if tumor_ratio > 0.08:
        findings.append(f"Large lesion occupying ~{tumor_ratio*100:.1f}% of brain area — significant mass effect likely")
    elif tumor_ratio > 0.03:
        findings.append(f"Moderate-sized lesion occupying ~{tumor_ratio*100:.1f}% of brain area")
    elif tumor_ratio > 0:
        findings.append(f"Small focal lesion occupying ~{tumor_ratio*100:.2f}% of brain area")

    return {
        "tumor_type": tumor_type,
        "tumor_type_confidence": type_conf,
        "location": location_label,
        "location_detail": f"{h_side} — {v_region} region (centroid at {cx*100:.0f}%L / {cy*100:.0f}%T)",
        "shape_description": shape_desc,
        "intensity_profile": int_profile,
        "findings": findings,
        "disclaimer": "Type inference is based on image morphology heuristics only — not a clinical diagnosis. A specialist radiologist review is required.",
    }


@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
