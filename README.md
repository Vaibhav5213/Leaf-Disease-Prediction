# 🌿 Plant Disease Prediction

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://leaf-disease-prediction-model.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![rembg](https://img.shields.io/badge/Segmentation-rembg-blueviolet.svg)

---

## 🚀 Live Demo

**Try the web app here:** 👉 https://leaf-disease-prediction-model.streamlit.app

---

## 📌 Overview

This project is an end-to-end computer vision system for diagnosing plant diseases from leaf images. The model was trained to successfully classify **38 distinct classes across 163,000 images** of crop diseases and healthy leaves, achieving **85.34% training accuracy** and **82.71% validation accuracy** across the dataset.

To bridge the gap between idealized lab data and messy real-world photos, this pipeline features a dedicated preprocessing engine that dynamically isolates the leaf and eliminates background noise before passing it to the neural network. It includes a visual frontend, a CLI, and a deployable REST API built with FastAPI.

![Example](Leaf-Disease-Prediction/assets/Example.jpg)

---

## 🧠 The Architecture & "Domain Shift" Solution

A common issue in agricultural AI is that models trained on clean datasets fail in the real world due to complex backgrounds. To solve this **Domain Shift** problem, this project utilizes a modular backend:

1. **Segmentation (`src/segmentation.py`)** Uses the `rembg` (U^2-Net) AI engine to completely strip the background from the user's uploaded image.

2. **Inference (`src/predict.py`)** Passes the cleanly segmented, RGB-formatted leaf to a custom Convolutional Neural Network (CNN) built with TensorFlow/Keras.

3. **REST API (`src/api.py`)** Wraps the inference engine in a fast, deployable FastAPI server, allowing mobile apps or agricultural web platforms to diagnose crop health via HTTP requests.

4. **Interface (`src/app.py` & `src/main.py`)** Provides both a Streamlit Web UI and a local Command Line Interface (CLI).

---

## 📐 Model Architecture & Engineering Tradeoffs

In machine learning, architecture is about balancing accuracy with computational efficiency. Instead of relying on a massive, pre-trained network (like ResNet or EfficientNet), this project utilizes a **custom-built, 5-block Convolutional Neural Network (CNN)** designed specifically for edge-case latency and mobile API deployment.


### 1. The Design (The "How")
* **Input & Preprocessing:** The model accepts `256x256x3` RGB images. Crucially, the data augmentation (`Rescaling`, `RandomFlip`, `RandomRotation`) is baked directly into the model graph as the first layer.
* **Feature Extraction:** 5 stacked Convolutional blocks (`Conv2D` + `MaxPooling2D`). Filter sizes progressively increase (`32 → 64 → 128 → 128 → 128`) to extract hierarchical features (from basic leaf edges to complex mold textures).
* **Classification Head:** The 2D matrices are flattened into a 4,608-dimensional feature vector, fed into a 256-neuron `Dense` layer, regulated by a `Dropout` layer, and output through a final 38-neuron dense layer for class prediction.

### 2. The Tradeoffs (The "Why")
* **Custom CNN vs. Transfer Learning:** * *The Tradeoff:* Training a model from scratch takes more epochs to converge than fine-tuning an existing model.
  * *The Benefit:* Total parameter control. Our model contains roughly 1.4 million parameters, making the `.keras` file incredibly lightweight. This drastically reduces model size and enables efficient API inference.
* **Baked-in Augmentation:** By placing the rescaling and augmentation steps directly inside the Keras sequential model, the deployment pipeline is heavily simplified. The API and CLI do not need redundant mathematical preprocessing scripts—raw bytes go in, and predictions come out.
* **Deep vs. Wide Downsampling:** Plant diseases present as micro-textures (e.g., tiny rust spots or blight rings). Using 5 gradual pooling layers preserves these minute spatial relationships much better than aggressively downsampling the image early in the network.
* **Dropout Regularization:** Leaf images often contain highly uniform green pixels that can cause dense layers to co-adapt and memorize the training data. The `Dropout` layer randomly severs connections during training, forcing the model to generalize and maintain its 82.71% validation accuracy on unseen data.

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
If you want to bypass the web UI and test an image directly from your terminal.

```bash
cd src
python main.py path/to/your/test_image.jpg
```

---

### **Option C: Run the API Server (FastAPI)**
Boot the REST API to test HTTP image uploads via the Swagger UI. Ensure you run this from the root directory.

```bash
uvicorn src.api:app --reload
```
*Once running, open `http://127.0.0.1:8000/docs` in your browser.*

---

## 📂 Project Structure

```text
plant_disease_project/
├── assets/
│   └── Example.jpg                 # Example image
├── data/
│   └── raw/                        # Empty (See Note inside)
├── models/
│   └── checkpoints/
│       ├── best_model.keras        # Trained weights for 'colors' dataset slice
│       └── universal_model.keras   # Universal trained weights
├── notebooks/
│   ├── Plant_village_1.ipynb       # CNN architecture and training
│   └── Plant_village_2.ipynb       # Inference testing
├── src/
│   ├── api.py                      # FastAPI server and endpoints
│   ├── app.py                      # Streamlit frontend
│   ├── main.py                     # Local CLI script
│   ├── predict.py                  # Model prediction logic
│   └── segmentation.py             # Background elimination engine
└── requirements.txt                # Deployment dependencies
```

---

## 📚 Data Acknowledgement

The underlying model was trained using the open-access **PlantVillage Dataset**. Due to GitHub's file size constraints, the raw dataset is not hosted in this repository, but the complete training pipeline can be reviewed in the `notebooks/` directory. The dataset too, can be found on official channels.
