# 🧠 BrainLens — Brain Tumor Detection & Volumetric Analysis

An ongoing **research project** focused on deep learning based brain tumor detection and volumetric segmentation from MRI scans, using a custom **ACU-Net architecture** trained on the **BraTS dataset**.

> 🔬 This is an active research project — currently under development.

---

## ✨ Features

- 🧬 **Tumor Classification** — Detects presence of brain tumor from MRI scans
- 📐 **Volumetric Analysis** — Estimates tumor size and volume in mm³
- 🖼️ **Multiplanar MRI Analysis** — Processes axial, coronal, and sagittal MRI views
- 🌐 **Web Interface** — Flask app for uploading MRI scans and viewing results
- 📊 **Custom ACU-Net Architecture** — Attention-based CNN for precise segmentation

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow, Keras |
| Architecture | ACU-Net (custom attention-based CNN) |
| Image Processing | OpenCV, NumPy |
| Web Interface | Flask, HTML, CSS, JavaScript |
| Dataset | BraTS (Brain Tumor Segmentation) — Kaggle |
| Training | Kaggle GPU Notebooks |

---

## 📁 Project Structure

```
BrainLens/
├── app.py                      # Flask web application
├── classifier.py               # Tumor classifier model
├── tumor_detector.py           # Core detection logic
├── acu_net.py                  # Custom ACU-Net architecture
├── train_model.py              # Model training script
├── train_acunet.py             # ACU-Net training
├── train_brats.py              # BraTS dataset training
├── train_incremental.py        # Incremental training
├── evaluate_accuracy.py        # Model evaluation & metrics
├── kaggle_acunet_train.py      # Kaggle training notebook
├── kaggle_classifier_train.py  # Kaggle classifier training
├── kaggle_multiplanar_train.py # Multiplanar training
├── debug_train.py              # Debugging utilities
├── test_api.py                 # API testing
├── test_classifier.py          # Classifier testing
├── test_load.py                # Model load testing
├── static/                     # CSS, JS, images
├── templates/                  # HTML templates
└── PROJECT_REPORT.md           # Detailed project report
```

---

## ⚙️ How It Works

1. MRI scan is uploaded via the Flask web interface
2. The image is preprocessed and passed through the **ACU-Net classifier**
3. Tumor regions are segmented using attention-based convolution layers
4. **OpenCV** highlights the detected tumor region on the scan
5. Volumetric estimation is calculated from the segmented region
6. Results are displayed on the web interface with highlighted MRI output

---

## 🚀 Run Locally

### Prerequisites
- Python 3.9+
- TensorFlow 2.x
- OpenCV

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/Khushi-yadav12/BrainLens.git
cd BrainLens

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Flask app
python app.py
```

---

## 📊 Dataset

- **BraTS (Brain Tumor Segmentation Challenge)** dataset from Kaggle
- Contains multimodal MRI scans (T1, T1ce, T2, FLAIR)
- Includes ground truth segmentation masks

---

## 👩‍💻 Contributors

| Name | GitHub |
|---|---|
| Khushi Yadav | [@Khushi-yadav12](https://github.com/Khushi-yadav12) |
| Chanchal Sharma | [@Chanchal-Sharma-03](https://github.com/Chanchal-Sharma-03) |
| Radhika | [@radhika581](https://github.com/radhika581) |


---

## 📄 License

This project is open source under the [MIT License](LICENSE).

> ⚠️ This tool is for research purposes only and is not intended for clinical or medical diagnosis.