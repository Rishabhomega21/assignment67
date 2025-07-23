import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🔬 Glass Type Prediction App")
st.markdown("This app predicts the **type of glass** based on chemical composition inputs.")

features = {}
features['RI'] = st.number_input("Refractive Index (RI)", 1.4, 1.6, step=0.01)
features['Na'] = st.number_input("Sodium (Na)", 10.0, 17.0, step=0.1)
features['Mg'] = st.number_input("Magnesium (Mg)", 0.0, 5.0, step=0.1)
features['Al'] = st.number_input("Aluminum (Al)", 0.0, 5.0, step=0.1)
features['Si'] = st.number_input("Silicon (Si)", 68.0, 75.0, step=0.1)
features['K']  = st.number_input("Potassium (K)", 0.0, 6.0, step=0.1)
features['Ca'] = st.number_input("Calcium (Ca)", 5.0, 15.0, step=0.1)
features['Ba'] = st.number_input("Barium (Ba)", 0.0, 1.5, step=0.1)
features['Fe'] = st.number_input("Iron (Fe)", 0.0, 0.5, step=0.01)

input_df = pd.DataFrame([features])

if st.button("Predict Glass Type"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)

    glass_dict = {
        1: "Building Windows (Float Process)",
        2: "Building Windows (Non Float)",
        3: "Vehicle Windows (Float Process)",
        5: "Containers",
        6: "Tableware",
        7: "Headlamps"
    }

    st.success(f"🔎 Predicted Glass Type: {glass_dict.get(prediction[0], 'Unknown')} (Class {prediction[0]})")
    st.info(f"Prediction Probability: {np.max(prob)*100:.2f}%")
