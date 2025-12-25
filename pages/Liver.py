import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Liver Disease Prediction", layout="centered")
st.title("🧬 Liver Disease Prediction (10 Features)")

# ================= SAFE MODEL LOADER =================
@st.cache_resource
def load_model():
    with open("models/liver_model.pkl", "rb") as f:
        data = pickle.load(f)
    # Handle both tuple and dict formats
    if isinstance(data, tuple):
        model, scaler = data
        features = [
            "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
            "Alkaline_Phosphotase", "Alamine_Aminotransferase",
            "Aspartate_Aminotransferase", "Total_Proteins",
            "Albumin", "Albumin_and_Globulin_Ratio"
        ]
    else:
        model = data["model"]
        scaler = data["scaler"]
        features = data["features"]
    return model, scaler, features

model, scaler, FEATURES = load_model()

# ================= INPUTS =================
st.subheader("Enter Patient Details")

age = st.number_input("Age", 1, 120, value=45)
gender = st.selectbox("Gender", ["Male", "Female"])
gender_val = 1 if gender == "Male" else 0
total_bilirubin = st.number_input("Total Bilirubin", 0.0, 10.0, value=1.3)
direct_bilirubin = st.number_input("Direct Bilirubin", 0.0, 5.0, value=0.4)
alk_phos = st.number_input("Alkaline Phosphotase", 50, 2000, value=210)
alt = st.number_input("Alamine Aminotransferase (ALT)", 1, 2000, value=35)
ast = st.number_input("Aspartate Aminotransferase (AST)", 1, 2000, value=40)
total_proteins = st.number_input("Total Proteins", 1.0, 10.0, value=6.8)
albumin = st.number_input("Albumin", 1.0, 6.0, value=3.1)
ag_ratio = st.number_input("Albumin/Globulin Ratio", 0.0, 3.0, value=0.9)

# ================= PREDICTION =================
if st.button("🔍 Predict Liver Disease"):
    try:
        # Rule-based alert for obvious liver risk
        if total_bilirubin > 3 or direct_bilirubin > 1.5 or alt > 200 or ast > 200:
            st.error("⚠️ Possible Liver Disease Detected (Rule-Based Alert)")
        else:
            # Prepare input for model
            X_input = np.array([[age, gender_val, total_bilirubin, direct_bilirubin,
                                 alk_phos, alt, ast, total_proteins, albumin, ag_ratio]])
            X_scaled = scaler.transform(X_input)
            prediction = model.predict(X_scaled)[0]

            if prediction == 1:
                st.error("⚠️ Liver Disease Detected")
            else:
                st.success("✅ No Liver Disease Detected")

    except Exception as e:
        st.error("Prediction failed")
        st.code(str(e))

st.markdown("---")
st.markdown("Made with ❤️ by your ML buddy")
