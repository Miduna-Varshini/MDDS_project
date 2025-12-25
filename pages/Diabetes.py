import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🩸 Diabetes Prediction (8 Features)")

# ================= SAFE MODEL LOADER =================
@st.cache_resource
def load_model():
    with open("models/diabetes_model.pkl", "rb") as f:
        data = pickle.load(f)
    # Handle both tuple and dict formats
    if isinstance(data, tuple):
        model, scaler = data
        features = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
    else:
        model = data["model"]
        scaler = data["scaler"]
        features = data["features"]
    return model, scaler, features

model, scaler, FEATURES = load_model()

# ================= INPUTS =================
st.subheader("Enter Patient Details")

preg = st.number_input("Pregnancies", 0, 20, value=2)
glucose = st.number_input("Glucose", 0, 300, value=120)
bp = st.number_input("Blood Pressure", 0, 200, value=70)
skin = st.number_input("Skin Thickness", 0, 100, value=20)
insulin = st.number_input("Insulin", 0, 900, value=85)
bmi = st.number_input("BMI", 0.0, 70.0, value=28.5)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, value=0.5)
age = st.number_input("Age", 1, 120, value=32)

# ================= PREDICTION =================
if st.button("🔍 Predict Diabetes"):
    try:
        # Rule-based alert for obvious risk
        if glucose > 180 or bmi > 40 or insulin > 300:
            st.error("⚠️ Possible Diabetes Detected (Rule-Based Alert)")
        else:
            # Prepare input for model
            X_input = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
            X_scaled = scaler.transform(X_input)
            prediction = model.predict(X_scaled)[0]

            if prediction == 1:
                st.error("⚠️ Diabetes Detected")
            else:
                st.success("✅ No Diabetes Detected")

    except Exception as e:
        st.error("Prediction failed")
        st.code(str(e))

st.markdown("---")
st.markdown("Made with ❤️ by your ML buddy")
