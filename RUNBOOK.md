# MaskFace: Runbook

This document provides concise, step-by-step instructions for running, benchmarking, and interacting with the **MaskFace** face mask safety compliance system.

---

## 💻 Run Locally

### 1. Environment Setup
Requires **Python 3.10+**:
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Start the Application Server
```bash
python server.py
```
Open `http://localhost:5000` in any browser (Chrome, Edge, Firefox, Safari).

---

## 🧪 Using the Features

### A. Static Image Inspection & Audit Log (Tab 1)
1. **Test Presets**: Click on the preset buttons (`Crowd Scene`, `Street View`, `Individual`, `Dense CCTV`) to inspect multi-face scenarios.
2. **Custom Upload**: Drag and drop any image (`.jpg`, `.png`, `.jpeg`) into the upload zone.
3. **Threshold Calibration**:
   - In the left sidebar, adjust the **Min Classification Confidence**, **Decision Margin Gate**, or **Temperature Scaling** sliders.
   - Click **Apply Settings** (or drag the sliders) to see real-time re-scoring and audit log updates.
4. **Examine Audit Log**:
   - Each detected face receives a unique ID (`#1`, `#2`, ...).
   - The table shows per-face **Status**, **Classification**, **Calibrated Confidence**, **Margin**, **Compliance Score (0–100)**, **Crop Resolution**, **Gate Status**, and **Audit Rationale**.
   - Note the **25% contextual padding** which captures the full chin strap and exposed nose geometry.

### B. Live Stream Monitoring & Audio Alarm (Tab 2)
1. Switch to the **Live Stream Monitoring** tab.
2. Click **Start Camera** to activate your webcam via HTML5 WebRTC.
3. **Rolling Compliance**: Observe the 30-frame temporal moving-window score updating dynamically as people enter and leave the frame.
4. **Audible Violation Alarm (5-Frame Persistence Buffer)**:
   - Click **Alarm: Active** to unmute sound alerts (uses browser Web Audio API, no external audio drivers required).
   - If a violation (bare face or improper mask) persists for **5 consecutive frames**, the alarm chimes.
   - As soon as compliance is restored, the alarm silences immediately (asymmetric hysteresis).
5. **Surveillance Feed**: Click **Continuous CCTV Stream** to inspect native server-side MJPEG video processing.

### C. Architecture Benchmarks (Tab 3)
1. Switch to the **Model Benchmarks** tab.
2. Review the empirical throughput table comparing all 5 explored architectures across **GPU (CUDA)** and **pure CPU (Host Processor)**.
3. Review the dataset engineering progression table demonstrating the **+47.97% F1 lift** achieved by the curated Big Data pipeline.

---

## 📊 Reproduce Benchmarks & Metrics

All evaluation scripts run out-of-the-box using the pre-packaged models in `models/`.

> **Dataset Location Note**: If running offline benchmarks with the real-world holdout set, it lives in `data/test_real_world_holdout/`. The scripts detect it automatically.

### 1. Evaluate Champion Model on Real-World Holdout
Evaluates **ResNet-18 (Big Data Clean)** across 3,202 ground-truth holdout faces:
```bash
python scripts/evaluate_big_data.py
```
**Expected Output**:
- Accuracy: **97.88%**
- Improper Mask F1: **86.70%** (Precision: 90.72%, Recall: 83.02%)
- Proper Mask F1: **98.85%**
- Bare Face (No Mask) F1: **95.53%**

### 2. Multi-Model End-to-End Latency & Accuracy Comparison
Evaluates all 5 explored backbones and generates comparative metrics:
```bash
python scripts/test_all_models_eval_fps.py
```

### 3. Generate Confusion Matrices
To regenerate high-resolution confusion matrix heatmaps:
```bash
python evaluation/evaluate.py
```
Output charts are saved to `evaluation/all_models_confusion_matrices.png`.

---

## 🛠️ Dataset Reproduction Pipeline (Optional)

If you wish to re-download raw candidate data from Kaggle and re-execute the cleaning and training pipeline:

1. **Prerequisites**: Place your own `kaggle.json` API token in a `.kaggle/` folder in the project root (git-ignored). Never commit it.
2. **Download & Clean Candidates**:
   ```bash
   python scripts/prepare_big_data.py
   python scripts/deduplicate_big_data.py
   ```
3. **Train ResNet-18 on Big Data**:
   ```bash
   python scripts/train_big_data.py --epochs 20 --batch-size 64 --lr 0.0003
   ```
*(Note: Pre-trained weights are already included in `models/`, so retraining is optional).*
