# 1) Install packages if not done
# pip install streamlit tensorflow scikit-learn joblib

# 2) Import libraries
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# 3) Load model and scalers
model = load_model("model/my_model_improved.h5", compile=False)
feat_scaler = joblib.load("model/feat_scaler_improved.pkl")
target_scaler = joblib.load("model/target_scaler_improved.pkl")

# 4) Set up Streamlit page
st.set_page_config(page_title="Land Price Predictor", page_icon="🏡")
st.title("🏡 Land Price Prediction App")
st.write("Enter the information below to predict the price of land (in TND):")

# 5) Input form
region = st.selectbox("Select Region or State:", ["North", "Center", "South"])
year = st.number_input("Enter Year:", min_value=2000, max_value=2100, value=2025)
area = st.number_input("Enter Total Land Area (m²):", min_value=1.0, value=1000.0)

# 6) When user clicks Predict
if st.button("Predict Land Price"):

    # 7) Feature engineering (log area)
    log_area = np.log1p(area)

    # 8) Encode Region
    region_north = 1 if region == "North" else 0
    region_center = 1 if region == "Center" else 0
    region_south = 1 if region == "South" else 0

    region_or_state_north = region_north
    region_or_state_center = region_center
    region_or_state_south = region_south

    # 9) Create input dataframe (respect feature order used during training)
    input_data = pd.DataFrame({
        "Year": [year],
        "Log Area": [log_area],
        "Region_Center": [region_center],
        "Region_North": [region_north],
        "Region_South": [region_south],
        "Land use_Unknown": [1],  # Default (if Land use was encoded like this)
        "Region or State_Center": [region_or_state_center],
        "Region or State_North": [region_or_state_north],
        "Region or State_South": [region_or_state_south]
    })

    # 10) Scale input
    X_scaled = feat_scaler.transform(input_data)

    # 11) Reshape for LSTM input (batch_size, seq_len, features)
    X_scaled = np.expand_dims(X_scaled, axis=0)  # (1, seq_len, features)

    # 12) Predict
    y_pred_scaled = model.predict(X_scaled)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    predicted_price = y_pred[0][0]

    # 13) Show result
    st.success(f"🏡 Predicted Land Price: {predicted_price:.2f} TND")

