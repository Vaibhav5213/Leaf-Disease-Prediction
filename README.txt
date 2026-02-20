# 🌿 Plant Disease Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red.svg)

## 📌 Overview
This Model is an end-to-end Computer Vision application designed to diagnose plant diseases from images of leaves. The model was trained to classify **38 distinct classes** of crop diseases and healthy leaves. 

To bridge the gap between idealized lab data and messy real-world photos, this pipeline features a dedicated preprocessing engine that dynamically isolates the leaf and eliminates background noise before passing it to the neural network.

---

## 🚀 Live Demo
**Try the web app here:** [https://leaf-disease-prediction-model.streamlit.app/]

---

## 🧠 The Architecture & "Domain Shift" Solution
A common issue in agricultural AI is that models trained on clean datasets fail in the real world due to complex backgrounds. To solve this **Domain Shift** problem, this project utilizes a modular backend:

1. **Segmentation (`src/segmentation.py`):** Uses the `rembg` (U^2-Net) AI engine to completely strip the background from the user's uploaded image.
2. **Inference (`src/predict.py`):** Passes the cleanly segmented, RGB-formatted leaf to a custom Convolutional Neural Network (CNN) built with TensorFlow/Keras.
3. **Interface (`src/app.py` & `src/main.py`):** Provides both a Streamlit Web UI and a local Command Line Interface (CLI).

---

## 💻 How to Run Locally

If you want to run this project on your own machine, follow these steps.

**1. Clone the repository and install dependencies:**
```bash
git clone [https://github.com/Vaibhav5213/Leaf-Disease-Prediction.git](https://github.com/Vaibhav5213/Leaf-Disease-Prediction.git)
cd universal-plant-doctor
pip install -r requirements.txt