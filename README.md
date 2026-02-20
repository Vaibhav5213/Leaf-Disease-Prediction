# 🌿 Plant Disease Prediction

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://leaf-disease-prediction-model.streamlit.app)


![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)

---

## 📌 Overview

This Model is an end-to-end Computer Vision application designed to diagnose plant diseases from images of leaves. The model was trained to classify **38 distinct classes** of crop diseases and healthy leaves.

To bridge the gap between idealized lab data and messy real-world photos, this pipeline features a dedicated preprocessing engine that dynamically isolates the leaf and eliminates background noise before passing it to the neural network.

---

## 🚀 Live Demo

**Try the web app here:**  
👉 https://leaf-disease-prediction-model.streamlit.app

---

## 🧠 The Architecture & "Domain Shift" Solution

A common issue in agricultural AI is that models trained on clean datasets fail in the real world due to complex backgrounds. To solve this **Domain Shift** problem, this project utilizes a modular backend:

1. **Segmentation (`src/segmentation.py`)**  
   Uses the `rembg` (U^2-Net) AI engine to completely strip the background from the user's uploaded image.

2. **Inference (`src/predict.py`)**  
   Passes the cleanly segmented, RGB-formatted leaf to a custom Convolutional Neural Network (CNN) built with TensorFlow/Keras.

3. **Interface (`src/app.py` & `src/main.py`)**  
   Provides both a Streamlit Web UI and a local Command Line Interface (CLI).

---

## 💻 How to Run Locally

If you want to run this project on your own machine, follow these steps.

### **1. Clone the repository and install dependencies**

```bash
git clone https://github.com/Vaibhav5213/Leaf-Disease_prediction.git
cd Leaf-Disease_prediction
pip install -r requirements.txt
```

---

### **Option A: Run the Web App (Streamlit)**

This will launch the visual user interface in your web browser.

```bash
cd src
streamlit run app.py
```

---

### **Option B: Run the Terminal CLI**

If you want to bypass the web UI and test an image directly from your terminal, use `main.py`.

```bash
cd src
python main.py path/to/your/test_image.jpg
```

---

## 📂 Project Structure

```
plant_disease_project/
├── data/
│   └── raw/                        # Empty (See Note inside)
├── models/
│   └── checkpoints/
│       └── universal_model.keras   # The trained weights
├── notebooks/
│   ├── Plant_village_1.ipynb       # CNN architecture and training
│   └── Plant_village_2.ipynb       # Inference testing
├── src/
│   ├── app.py                      # Streamlit frontend
│   ├── main.py                     # Local CLI script
│   ├── predict.py                  # Model prediction logic
│   └── segmentation.py             # Background elimination engine
└── requirements.txt                # Deployment dependencies
```

---

## 📚 Data Acknowledgement

The underlying model was trained using the open-access **PlantVillage Dataset**. Due to GitHub's file size constraints, the raw dataset is not hosted in this repository, but the complete training pipeline can be reviewed in the `notebooks/` directory. The dataset too, can be found on official channels.

**Citation:**  
Hughes, D. P., & Salathe, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.*
