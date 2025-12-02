import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load your trained model
MODEL_PATH = 'plant_model.h5'
model = load_model(MODEL_PATH)

# Define the class labels (should match your model)
class_names = ['Healthy', 'Yellow Leaves', 'Brown Spots', 'Wilted', 'Dry Soil']
mood_dict = {
    'Healthy': '😊 Happy',
    'Yellow Leaves': '😟 Lacking nutrients',
    'Brown Spots': '😷 Possible infection',
    'Wilted': '🥵 Needs water or shade',
    'Dry Soil': '💧 Thirsty plant'
}

# Preprocessing function
def preprocess_image(img):
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit UI
st.set_page_config(page_title="MoodPlant 🌱")
st.title("🌱 MoodPlant: AI Plant Mood & Health Detector")
st.write("Upload a leaf image to check plant's health and mood.")

uploaded_file = st.file_uploader("📤 Upload Plant Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    if opencv_image is not None:
        st.image(opencv_image, channels="BGR", caption="Uploaded Image", use_column_width=True)

        processed_image = preprocess_image(opencv_image)
        prediction = model.predict(processed_image)

        if prediction.size > 0:
            predicted_class = class_names[np.argmax(prediction)]
            mood = mood_dict[predicted_class]

            st.markdown(f"### 🧠 Prediction: `{predicted_class}`")
            st.markdown(f"### 🪴 Mood: `{mood}`")

            # Suggestions
            if predicted_class == 'Yellow Leaves':
                st.info("🧪 Tip: Try adding iron-rich fertilizer.")
            elif predicted_class == 'Brown Spots':
                st.info("🦠 Tip: Check for fungal infection and isolate the plant.")
            elif predicted_class == 'Wilted':
                st.info("🌤️ Tip: Move to shade and water it.")
            elif predicted_class == 'Dry Soil':
                st.info("💧 Tip: Water the plant and check drainage.")
            else:
                st.success("🌿 Your plant is healthy! Keep it up.")
        else:
            st.error("Prediction failed. Try another image.")
    else:
        st.error("Could not read image. Try a different file.")
