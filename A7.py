import streamlit as st
import numpy as np
import joblib
import os

# Set page config (optional favicon added for polish)
st.set_page_config(page_title="Glass Type Predictor", page_icon="🔮", layout="centered")

# Title
st.title("🔮 Glass Type Predictor")
st.markdown("Enter the glass chemical properties below to predict its type of glass.")

# --- Load Model and Scaler ---
model_file = "glass_model.pkl"
scaler_file = "scaler.pkl"

if not os.path.exists(model_file) or not os.path.exists(scaler_file):
    st.error("❌ Could not find 'glass_model.pkl' or 'scaler.pkl'. Make sure both files are in the same folder as this app.")
    st.stop()

model = joblib.load(model_file)
scaler = joblib.load(scaler_file)

# --- Glass Type Mapping ---
glass_types = {
    1: "Building Windows Float Processed",
    2: "Building Windows Non-Float Processed",
    3: "Vehicle Windows Float Processed",
    4: "Vehicle Windows Non-Float Processed",
    5: "Containers",
    6: "Tableware",
    7: "Headlamps"
}

# --- Input Form ---
with st.form("glass_form"):
    RI = st.number_input("Refractive Index (RI)", value=1.52, step=0.01)
    Na = st.number_input("Sodium (Na)", value=13.0, step=0.1)
    Mg = st.number_input("Magnesium (Mg)", value=2.0, step=0.1)
    Al = st.number_input("Aluminum (Al)", value=1.0, step=0.1)
    Si = st.number_input("Silicon (Si)", value=72.0, step=0.1)
    K  = st.number_input("Potassium (K)", value=0.5, step=0.05)
    Ca = st.number_input("Calcium (Ca)", value=8.0, step=0.1)
    Ba = st.number_input("Barium (Ba)", value=0.1, step=0.01)
    Fe = st.number_input("Iron (Fe)", value=0.1, step=0.01)

    submit = st.form_submit_button("Predict Glass Type")

# --- Prediction ---
if submit:
    try:
        features = np.array([[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]])
        scaled_input = scaler.transform(features)
        prediction = model.predict(scaled_input)[0]
        result = glass_types.get(prediction, "Unknown Type")
        st.success(f"🏷 Predicted Glass Type: **{result}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
