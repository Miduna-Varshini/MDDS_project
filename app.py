import streamlit as st
import numpy as np
import pickle

# ================== SESSION STATE INIT ==================
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

# ================== MODEL LOADER ==================
@st.cache_resource
def load_model(path, default_features=[]):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, tuple):
        model, scaler = data
        features = default_features
    else:
        model = data['model']
        scaler = data['scaler']
        features = data['features']
    return model, scaler, features

# ================== DASHBOARD ==================
if st.session_state['page'] == 'Home':
    st.title("🩺 Multi-Disease Diagnostic Portal")
    st.write("Welcome! Click a disease below to start prediction:")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("❤️ Heart"):
            st.session_state['page'] = 'Heart'
    with col2:
        if st.button("🩸 Diabetes"):
            st.session_state['page'] = 'Diabetes'
    with col3:
        if st.button("🧠 Brain"):
            st.session_state['page'] = 'Brain'

    col4, col5 = st.columns(2)
    with col4:
        if st.button("🟣 Kidney"):
            st.session_state['page'] = 'Kidney'
    with col5:
        if st.button("🟠 Liver"):
            st.session_state['page'] = 'Liver'

# ================== HEART PREDICTION ==================
elif st.session_state['page'] == 'Heart':
    st.header("❤️ Heart Disease Prediction")
    # Paste your heart ML form & prediction code here
    # Example:
    age = st.number_input("Age", 0, 120, 52)
    st.button("Back to Home", on_click=lambda: st.session_state.update({'page': 'Home'}))

# ================== DIABETES PREDICTION ==================
elif st.session_state['page'] == 'Diabetes':
    st.header("🩸 Diabetes Prediction")
    # Paste your diabetes ML code here
    st.button("Back to Home", on_click=lambda: st.session_state.update({'page': 'Home'}))

# ================== BRAIN PREDICTION ==================
elif st.session_state['page'] == 'Brain':
    st.header("🧠 Brain Tumor Prediction")
    # Paste your brain tumor ML code here
    st.button("Back to Home", on_click=lambda: st.session_state.update({'page': 'Home'}))

# ================== KIDNEY PREDICTION ==================
elif st.session_state['page'] == 'Kidney':
    st.header("🟣 Kidney Disease Prediction")
    # Paste your kidney ML code here
    st.button("Back to Home", on_click=lambda: st.session_state.update({'page': 'Home'}))

# ================== LIVER PREDICTION ==================
elif st.session_state['page'] == 'Liver':
    st.header("🟠 Liver Disease Prediction")
    # Paste your liver ML code here
    st.button("Back to Home", on_click=lambda: st.session_state.update({'page': 'Home'}))
