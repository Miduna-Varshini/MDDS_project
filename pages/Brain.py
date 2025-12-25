import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import requests
from io import BytesIO

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Brain Tumor Prediction", layout="centered")
st.title("🧠 Brain Tumor Prediction")

# ================= LOAD MODEL =================
@st.cache_resource
def load_brain_model():
    FILE_ID = "1r7Kmf14ZGKQK3GSTk3nxPxfAyGpg2m_b"
    URL = f"https://drive.google.com/uc?id={FILE_ID}&export=download"
    
    response = requests.get(URL)
    with open("brain_tumor_dataset.h5", "wb") as f:
        f.write(response.content)
    
    model = load_model("brain_tumor_dataset.h5")
    return model

model = load_brain_model()

# Show model input shape for debugging
st.write("Model input shape:", model.input_shape)

# ================= IMAGE UPLOAD =================
st.subheader("Upload MRI Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    # ================= PREPROCESS IMAGE =================
    # Get model input shape (ignore batch size)
    input_shape = model.input_shape[1:]  # e.g., (86528,) or (128,128,3)
    
    if len(input_shape) == 1:
        # Dense model expecting flattened input
        # Calculate the side of square image if needed
        side = int(np.sqrt(input_shape[0] / 3))
        img = image.resize((side, side))
        img_array = np.array(img) / 255.0
        img_array = img_array.flatten()
        img_array = np.expand_dims(img_array, axis=0)
    else:
        # CNN model expecting (H, W, C)
        img = image.resize((input_shape[0], input_shape[1]))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

    # ================= PREDICTION =================
    if st.button("🔍 Predict Brain Tumor"):
        try:
            prediction = model.predict(img_array)[0][0]
            if prediction > 0.5:
                st.error("⚠️ Brain Tumor Detected")
            else:
                st.success("✅ No Brain Tumor Detected")
        except Exception as e:
            st.error("Prediction failed")
            st.code(str(e))

st.markdown("---")
st.markdown("Made with ❤️ by your ML buddy")
