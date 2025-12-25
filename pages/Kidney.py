import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Kidney Disease Prediction", layout="centered")
st.title("🩺 Kidney Disease Prediction (10 Features)")

# ================= SAFE MODEL LOADER =================
@st.cache_resource
def load_model():
    with open("models/kidney_10f_model.pkl", "rb") as f:
        data = pickle.load(f)
    model = data["model"]
    scaler = data["scaler"]
    features = data["features"]
    return model, scaler, features

model, scaler, FEATURES = load_model()

# ================= INPUTS =================
st.subheader("Enter Patient Details")

age = st.number_input("Age", 0, 120, value=45)
bp = st.number_input("Blood Pressure (BP)", 0, 200, value=80)
sg = st.number_input("Specific Gravity (SG)", 1.0, 1.05, value=1.020)
al = st.number_input("Albumin (AL)", 0, 5, value=0)
su = st.number_input("Sugar (SU)", 0, 5, value=0)
bgr = st.number_input("Blood Glucose Random (BGR)", 0, 500, value=110)
bu = st.number_input("Blood Urea (BU)", 0, 200, value=25)
sc = st.number_input("Serum Creatinine (SC)", 0.0, 20.0, value=1.0)
hemo = st.number_input("Hemoglobin (HEMO)", 0.0, 20.0, value=15.2)
pcv = st.number_input("Packed Cell Volume (PCV)", 0, 60, value=44)

# ================= PREDICTION =================
if st.button("🔍 Predict Kidney Disease"):
    try:
        # Rule-based safety check for obvious CKD
        if bu > 90 or sc > 5 or hemo < 10 or pcv < 28:
            st.error("⚠️ Chronic Kidney Disease Detected (Rule-Based Alert)")
        else:
            # Prepare input for model
            X_input = np.array([[age, bp, sg, al, su, bgr, bu, sc, hemo, pcv]])
            X_scaled = scaler.transform(X_input)
            prediction = model.predict(X_scaled)[0]

            if prediction == 1:
                st.error("⚠️ Chronic Kidney Disease Detected")
            else:
                st.success("✅ No Chronic Kidney Disease Detected")

    except Exception as e:
        st.error("Prediction failed")
        st.code(str(e))

st.markdown("---")
st.markdown("Made with ❤️ by your ML buddy")
