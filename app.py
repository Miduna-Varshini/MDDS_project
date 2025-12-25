import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Kidney Disease Prediction", layout="centered")
st.title("🩺 Kidney Disease Prediction")

# ================= SAFE MODEL LOADER =================
@st.cache_resource
def load_model():
    with open("models/kidney_model.pkl", "rb") as f:
        data = pickle.load(f)

    # CASE 1: dictionary
    if isinstance(data, dict):
        model = data.get("model")
        scaler = data.get("scaler")

    # CASE 2: tuple (model, scaler)
    elif isinstance(data, tuple):
        model = data[0]
        scaler = data[1]

    else:
        raise ValueError("Unsupported model file format")

    return model, scaler


model, scaler = load_model()

# ================= INPUTS =================
st.subheader("Enter Patient Details")

age = st.number_input("Age", 0, 100)
bp = st.number_input("Blood Pressure", 0, 200)
sg = st.number_input("Specific Gravity", 1.0, 1.05)
al = st.number_input("Albumin", 0, 5)
su = st.number_input("Sugar", 0, 5)
bgr = st.number_input("Blood Glucose Random", 0)
bu = st.number_input("Blood Urea", 0)
sc = st.number_input("Serum Creatinine", 0.0)
sod = st.number_input("Sodium", 0)
pot = st.number_input("Potassium", 0.0)
hemo = st.number_input("Hemoglobin", 0.0)
pcv = st.number_input("Packed Cell Volume", 0)
wc = st.number_input("White Blood Cell Count", 0)
rc = st.number_input("Red Blood Cell Count", 0.0)

htn = st.selectbox("Hypertension", ["No", "Yes"])
dm = st.selectbox("Diabetes Mellitus", ["No", "Yes"])
cad = st.selectbox("Coronary Artery Disease", ["No", "Yes"])
appet = st.selectbox("Appetite", ["Poor", "Good"])
pe = st.selectbox("Pedal Edema", ["No", "Yes"])
ane = st.selectbox("Anemia", ["No", "Yes"])

# Manual encoding (same as your training style)
htn = 1 if htn == "Yes" else 0
dm = 1 if dm == "Yes" else 0
cad = 1 if cad == "Yes" else 0
appet = 1 if appet == "Good" else 0
pe = 1 if pe == "Yes" else 0
ane = 1 if ane == "Yes" else 0

# ================= PREDICTION =================
if st.button("🔍 Predict Kidney Disease"):
    try:
        X = np.array([[  
            age, bp, sg, al, su,
            bgr, bu, sc, sod, pot,
            hemo, pcv, wc, rc,
            htn, dm, cad, appet, pe, ane
        ]])

        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]

        if prediction == 1:
            st.error("⚠️ Chronic Kidney Disease Detected")
        else:
            st.success("✅ No Chronic Kidney Disease Detected")

    except Exception as e:
        st.error("Prediction failed")
        st.code(str(e))
