import streamlit as st
import numpy as np
import pickle

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Multi Disease Diagnostic App",
    layout="centered"
)

st.title("🩺 Kidney Disease Prediction")
st.write("Enter patient details to predict Chronic Kidney Disease")

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    with open("models/kidney_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data

data = load_model()

model = data["model"]
scaler = data["scaler"]
encoders = data["encoders"]
features = data["features"]

# ================= INPUT FORM =================
input_data = {}

st.subheader("Patient Information")

for feature in features:
    if feature in encoders:
        # Categorical feature
        options = list(encoders[feature].classes_)
        value = st.selectbox(f"{feature}", options)
        encoded_value = encoders[feature].transform([value])[0]
        input_data[feature] = encoded_value
    else:
        # Numerical feature
        value = st.number_input(f"{feature}", step=1.0)
        input_data[feature] = value

# ================= PREDICTION =================
if st.button("🔍 Predict Kidney Disease"):
    try:
        input_array = np.array([list(input_data.values())])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            st.error("⚠️ Chronic Kidney Disease Detected")
        else:
            st.success("✅ No Chronic Kidney Disease Detected")

    except Exception as e:
        st.error("Something went wrong during prediction")
        st.code(str(e))

# ================= FOOTER =================
st.markdown("---")
st.caption("⚕️ AI-based diagnostic support system (Educational purpose only)")
