---
title: Panorama Stitcher
emoji: 📸
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# 📸 Computer Vision Panorama Stitcher

A Computer Vision application that stitches overlapping photos into panoramas. This project demonstrates both a deep understanding of low-level algorithms and modern deployment engineering.

---

## 🔥 High-Value Skills Demonstrated

**1.  Pipeline **
* **Feature Extraction:** Implemented both SIFT (floating-point) and ORB (binary) descriptors.
* **Texture Filtering:** Applied Lowe's Ratio Test to eliminate ambiguous, repetitive textures.
* **Outlier Rejection:** Used the RANSAC algorithm to filter out 5-pixel error threshold.
* **Perspective Alignment:** Calculated 3x3 Homography matrices to perform spatial perspective warping.

**2. Real-World Application**
* Handled multi-image unordered stitching via OpenCV's C++ production engine.
* Utilized Bundle Adjustment to optimize 3D camera angles and eliminate image drift.

**3. Deployment**
* Developed a drag-and-drop web UI using Gradio and a REST API using FastAPI.
* Built an automated CI/CD pipeline using GitHub Actions to continuously deploy the codebase to Hugging Face Spaces.

---

## 🛠️ Tech Stack

* **Computer Vision:** OpenCV (cv2), NumPy
* **Frontend & Backend API:** Gradio, FastAPI, Uvicorn
* **DevOps & Cloud:** GitHub Actions, Hugging Face Spaces, Docker

---

## 🚀 How to Run Locally

1. Install the required dependencies:
```bash
pip install gradio opencv-python-headless numpy
```

2. Launch the web application:
```bash
python app.py
```

---

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20App-blue?style=for-the-badge)](https://huggingface.co/spaces/Zadif123/panorama-api)

## 👤 Author

Developed by **Zadif**  
* **GitHub:** [github.com/zadif](https://github.com/zadif)
* **Hugging Face:** [huggingface.co/Zadif123](https://huggingface.co/Zadif123)