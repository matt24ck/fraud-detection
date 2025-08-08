import streamlit as st
import numpy as np
import pickle
import shap
import xgboost
import matplotlib.pyplot as plt

# Load model and threshold
with open('model/xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/threshold.pkl', 'rb') as f:
    threshold = pickle.load(f)

# Initialize SHAP explainer once
explainer = shap.TreeExplainer(model)

st.set_page_config(page_title="Fraud Detector", layout="wide")
st.title("🚨 Real-Time Fraud Detection with Explainability")

st.markdown("Enter transaction features to predict and explain fraud risk.")

# --- Input Form ---
time = st.number_input("Transaction Time (in seconds)", min_value=0, max_value=200000, value=100000)
v_features = [st.slider(f"V{i}", -10.0, 10.0, 0.0) for i in range(1, 29)]
amount = st.number_input("Transaction Amount", min_value=0.0, max_value=10000.0, value=50.0)

input_data = np.array([[time] + v_features + [amount]])
input_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# --- Prediction ---
if st.button("🔍 Predict & Explain"):
    proba = model.predict_proba(input_data)[0][1]
    prediction = int(proba >= threshold)

    st.subheader("🧠 Prediction Result")
    st.write(f"Fraud Probability: **{proba:.2%}**")
    
    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction Seems Legitimate")

    # --- SHAP Explanation ---
    st.subheader("🔎 Why This Prediction?")
    shap_values = explainer.shap_values(input_data)

    # Plot SHAP values
    st.markdown("Top contributing features:")

    fig, ax = plt.subplots()
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value, shap_values[0], feature_names=input_names, features=input_data[0], max_display=10, show=False
    )
    plt.tight_layout()
    st.pyplot(fig)