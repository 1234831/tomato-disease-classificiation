
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# -----------------------------
# Load Model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.weights.h5")
    return model

model = load_model()

# -----------------------------
# Class Names (CHANGE if needed)
# -----------------------------
CLASS_NAMES =['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))  # match your model input
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# Prediction Function
# -----------------------------
def predict(image):
    processed = preprocess_image(image)
    predictions = model.predict(processed)

    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return predicted_class, confidence, predictions

# -----------------------------
# UI
# -----------------------------
st.title("🌿 Plant Disease Detection")
st.markdown("Upload a leaf image to detect disease")

uploaded_file = st.file_uploader(
    "📤 Upload Leaf Image",
    type=["jpg", "png", "jpeg"]
)

# -----------------------------
# Main Logic
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            label, confidence, predictions = predict(image)

        st.success(f"✅ Prediction: {label}")
        st.info(f"📊 Confidence: {confidence:.2f}")

        # Top probabilities
        st.subheader("🔎 Class Probabilities")
        for i, prob in enumerate(predictions[0]):
            st.write(f"{CLASS_NAMES[i]}: {prob:.2f}")
