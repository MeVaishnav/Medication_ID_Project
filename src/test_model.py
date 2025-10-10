# src/test_model.py

import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os

# ---------------------------
# Load Model and Class Names
# ---------------------------
model = load_model("models/medicine_identifier.h5")

# Update this list based on your dataset classes
class_names = ['cheston', 'cipla650', 'dolo650', 'montac lc']

# ---------------------------
# Function to Predict Medicine
# ---------------------------
def predict_medicine(img_path):
    img = image.load_img(img_path, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions)

    print(f"\n🧠 Predicted Medicine: {class_names[predicted_index]}")
    print(f"📊 Confidence: {confidence:.2f}")

# ---------------------------
# Example usage
# ---------------------------
img_path = r"C:\Users\VAISHNAV C S\Desktop\Medication_ID_Project\data\test\dolo650\OIP1.jpeg"  # change to any test image path
if os.path.exists(img_path):
    predict_medicine(img_path)
else:
    print("⚠️ Image path not found. Please update img_path.")
