# 🌿 Plant Disease Vision API & Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://leaf-disease-prediction-model.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer_Vision-OpenCV-green.svg)

> **[🚀 Click here to try the Live Web Demo](https://leaf-disease-prediction-model.streamlit.app/)**

An end-to-end Computer Vision pipeline that detects plant diseases from raw leaf images. Built with a focus on production readiness, this project features an automated image segmentation engine, an interactive web dashboard, and a fully deployable REST API.

## 💡 The Value
Agricultural technology requires fast, on-the-edge diagnostics. This project was built to demonstrate how a backend computer vision model can be seamlessly wrapped into a lightweight API, allowing mobile apps or agricultural web platforms to instantly diagnose crop health via HTTP requests.

## 🚀 Core Features
* **Production API (FastAPI):** A high-performance backend wrapper that accepts raw image uploads and returns structured JSON predictions.
* **Interactive Frontend (Streamlit):** A user-friendly web interface for visual testing and demonstration.
* **Smart Segmentation:** Uses OpenCV to automatically isolate the leaf from noisy backgrounds before feeding it to the model, increasing accuracy.
* **CLI Engine:** A built-in command-line tool for rapid local image testing.

---

## ⚙️ How to Run the API Server

This project includes a production-ready FastAPI wrapper. To boot the server locally:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the FastAPI server from the root directory:**
   ```bash
   uvicorn src.api:app --reload
   ```

3. **Test the endpoints:**
   Open `http://127.0.0.1:8000/docs` in your browser to access the interactive Swagger UI. You can upload images directly through the browser and view the JSON response.

---

## 💻 How to Run the CLI Tool

If you want to test an image quickly without starting the web server, you can use the built-in CLI engine:

```bash
python src/main.py "path/to/your/leaf_image.jpg"
```
*The terminal will output a cleanly formatted list of disease predictions and confidence scores.*

---

## 📂 Architecture
```text
Leaf-Disease-Prediction/
├── models/        # Saved model weights and architecture
├── src/           # Source code
│   ├── api.py           # FastAPI server and endpoints
│   ├── main.py          # Command-line interface engine
│   ├── predict.py       # Core inference logic
│   └── segmentation.py  # OpenCV image isolation functions
└── requirements.txt
```
