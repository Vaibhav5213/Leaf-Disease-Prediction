import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses annoying TensorFlow terminal warnings
import tensorflow as tf
import numpy as np
from segmentation import isolate_leaf

# Hardcoded classes
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

MODEL_PATH = "../models/checkpoints/universal_model.keras"
print("Loading Universal Doctor model...")
model = tf.keras.models.load_model(MODEL_PATH)

def predict_disease(image_bytes):
    # 1. Eliminate the background (calling segmentation script)
    segmented_img = isolate_leaf(image_bytes)
    
    # 2. Resize and format for the model (256x256, 3 channels)
    img_resized = segmented_img.resize((256, 256))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0) # Create a batch of 1
    
    # 3. Make the prediction
    print("Analyzing leaf patterns...")
    predictions = model.predict(img_array, verbose=0)[0]
    
    # 4. Calculate Top 3
    top_3_indices = np.argsort(predictions)[-3:][::-1]
    
    results = []
    for i in top_3_indices:
        results.append({
            'disease': CLASS_NAMES[i].replace("___", " - ").replace("_", " "),
            'confidence': round(float(predictions[i]) * 100, 2)
        })
        
    # Returns the image for display AND the data for the UI
    return segmented_img, results