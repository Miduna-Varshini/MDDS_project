import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Multi-Disease Diagnostic App", layout="centered")
st.title("🩺 Multi-Disease Diagnostic Portal")

# ================= SAFE MODEL LOADER =================
@st.cache_resource
def load_model(path, default_features):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, tuple):
        model, scaler = data
        features = default_features
    else:
        model = data["model"]
        scaler = data["scaler"]
        features = data["features"]
    return model, scaler, features

# ================= DISEASE OPTIONS =================
disease = st.selectbox(
    "Choose a disease to predict:",
    ["Heart", "Kidney", "Liver", "Diabetes"]
)

# ================= HEART =================
if disease == "Heart":
    model, scaler, FEATURES = load_model(
        "models/heart_model.pkl",
        [
            "Age", "Sex", "Chest pain type", "BP", "Cholesterol",
            "FBS over 120", "EKG results", "Max HR", "Exercise angina",
            "ST depression", "Slope of ST", "Number of vessels fluro", "Thallium"
        ]
    )
    st.subheader("Enter Patient Details (Heart)")
    age = st.number_input("Age", 0, 120, value=52)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.number_input("Chest Pain Type (0–3)", 0, 3, value=0)
    trestbps = st.number_input("Resting Blood Pressure (BP)", 80, 200, value=120)
    chol = st.number_input("Cholesterol", 100, 600, value=240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.number_input("Resting ECG Results (0–2)", 0, 2, value=1)
    thalach = st.number_input("Maximum Heart Rate Achieved", 60, 250, value=150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, value=1.2)
    slope = st.number_input("Slope of ST Segment (0–2)", 0, 2, value=1)
    ca = st.number_input("Number of Major Vessels (0–3)", 0, 3, value=0)
    thal = st.number_input("Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)", 1, 3, value=2)

    if st.button("🔍 Predict Heart Disease"):
        try:
            if chol > 300 or trestbps > 160 or thalach < 100:
                st.error("⚠️ Possible Heart Disease Detected (Rule-Based Alert)")
            else:
                X_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                     thalach, exang, oldpeak, slope, ca, thal]])
                X_scaled = scaler.transform(X_input)
                prediction = model.predict(X_scaled)[0]
                st.error("⚠️ Heart Disease Detected" if prediction == 1 else "✅ No Heart Disease Detected")
        except Exception as e:
            st.error("Prediction failed")
            st.code(str(e))

# ================= KIDNEY =================
elif disease == "Kidney":
    model, scaler, FEATURES = load_model("models/kidney_10f_model.pkl", [])
    st.subheader("Enter Patient Details (Kidney)")
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

    if st.button("🔍 Predict Kidney Disease"):
        try:
            if bu > 90 or sc > 5 or hemo < 10 or pcv < 28:
                st.error("⚠️ Chronic Kidney Disease Detected (Rule-Based Alert)")
            else:
                X_input = np.array([[age, bp, sg, al, su, bgr, bu, sc, hemo, pcv]])
                X_scaled = scaler.transform(X_input)
                prediction = model.predict(X_scaled)[0]
                st.error("⚠️ Chronic Kidney Disease Detected" if prediction == 1 else "✅ No Chronic Kidney Disease Detected")
        except Exception as e:
            st.error("Prediction failed")
            st.code(str(e))

# ================= LIVER =================
elif disease == "Liver":
    model, scaler, FEATURES = load_model(
        "models/liver_model.pkl",
        [
            "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
            "Alkaline_Phosphotase", "Alamine_Aminotransferase",
            "Aspartate_Aminotransferase", "Total_Proteins",
            "Albumin", "Albumin_and_Globulin_Ratio"
        ]
    )
    st.subheader("Enter Patient Details (Liver)")
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

    if st.button("🔍 Predict Liver Disease"):
        try:
            if total_bilirubin > 3 or direct_bilirubin > 1.5 or alt > 200 or ast > 200:
                st.error("⚠️ Possible Liver Disease Detected (Rule-Based Alert)")
            else:
                X_input = np.array([[age, gender_val, total_bilirubin, direct_bilirubin,
                                     alk_phos, alt, ast, total_proteins, albumin, ag_ratio]])
                X_scaled = scaler.transform(X_input)
                prediction = model.predict(X_scaled)[0]
                st.error("⚠️ Liver Disease Detected" if prediction == 1 else "✅ No Liver Disease Detected")
        except Exception as e:
            st.error("Prediction failed")
            st.code(str(e))

# ================= DIABETES =================
elif disease == "Diabetes":
    model, scaler, FEATURES = load_model(
        "models/diabetes_model.pkl",
        [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
    )
    st.subheader("Enter Patient Details (Diabetes)")
    preg = st.number_input("Pregnancies", 0, 20, value=2)
    glucose = st.number_input("Glucose", 0, 300, value=120)
    bp = st.number_input("Blood Pressure", 0, 200, value=70)
    skin = st.number_input("Skin Thickness", 0, 100, value=20)
    insulin = st.number_input("Insulin", 0, 900, value=85)
    bmi = st.number_input("BMI", 0.0, 70.0, value=28.5)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, value=0.5)
    age = st.number_input("Age", 1, 120, value=32)

    if st.button("🔍 Predict Diabetes"):
        try:
            if glucose > 180 or bmi > 40 or insulin > 300:
                st.error("⚠️ Possible Diabetes Detected (Rule-Based Alert)")
            else:
                X_input = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
                X_scaled = scaler.transform(X_input)
                prediction = model.predict(X_scaled)[0]
                st.error("⚠️ Diabetes Detected" if prediction == 1 else "✅ No Diabetes Detected")
        except Exception as e:
            st.error("Prediction failed")
            st.code(str(e))

