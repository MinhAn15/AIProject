import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os
import uuid

# Set page configuration
st.set_page_config(
    page_title="Airfoil Self-Noise Prediction",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to right, #e6f3fa, #f0f8ff);
        font-family: 'Arial', sans-serif;
    }
    h1 {
        color: #1e3a8a !important;
        font-weight: 700;
        margin: 1.5rem 0;
    }
    h2, h3, h4, h5, h6 {
        color: #1e3a8a !important;
        font-weight: 500;
    }
    .stTextInput > label, .stNumberInput > label, .stCheckbox > label {
        color: #1e3a8a !important;
        font-weight: 500;
    }
    .stButton > button {
        background: #1e3a8a !important;
        color: white !important;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
    }
    .result-container {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .stMetric {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 10px;
        margin: 8px 0;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="background: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), url('https://example.com/airfoil-banner.jpg');
     height: 200px;
     background-size: cover;
     border-radius: 10px;
     margin-bottom: 2rem;">
    <div style="padding: 3rem 2rem;
             color: white;
             text-align: center;">
        <h1>Airfoil Self-Noise Prediction</h1>
        <h3>Predict sound pressure level based on aerodynamic parameters</h3>
    </div>
</div>
""", unsafe_allow_html=True)

# Model path
MODEL_PATH = "airfoil_noise_model.pkl"

# Function to train model
def train_model(data):
    X = data[['Frequency', 'Angle_of_attack', 'Chord_length', 'Free_stream_velocity', 'Suction_side_displacement_thickness']]
    y = data['Scaled_sound_pressure_level']
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model

# Load or train model
def load_or_train_model():
    data = pd.read_csv("airfoil_self_noise.csv")
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = train_model(data)
    return model

# Main function
def main():
    st.title("Airfoil Self-Noise Prediction System")
    
    # Sidebar for retraining option
    st.sidebar.header("Model Configuration")
    retrain_model = st.sidebar.checkbox("Retrain Model", value=False, help="Check to retrain the model using multiple regression")
    
    # Load or train model
    if retrain_model or not os.path.exists(MODEL_PATH):
        with st.spinner("Training model..."):
            model = train_model(pd.read_csv("airfoil_self_noise.csv"))
            st.sidebar.success("Model retrained successfully!")
    else:
        model = load_or_train_model()
    
    # Input form
    st.subheader("Input Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        frequency = st.number_input("Frequency (Hz)", min_value=0, step=1, format="%d")
        angle_of_attack = st.number_input("Angle of Attack (degrees)", min_value=0.0, step=0.0001, format="%.4f")
        chord_length = st.number_input("Chord Length (m)", min_value=0.0, step=0.0001, format="%.4f")
    
    with col2:
        free_stream_velocity = st.number_input("Free-stream Velocity (m/s)", min_value=0.0, step=0.1, format="%.1f")
        suction_thickness = st.number_input("Suction Side Displacement Thickness (m)", min_value=0.0, step=0.00000001, format="%.8f")
    
    # Predict button
    if st.button("Predict Noise Level"):
        input_data = pd.DataFrame({
            'Frequency': [frequency],
            'Angle_of_attack': [angle_of_attack],
            'Chord_length': [chord_length],
            'Free_stream_velocity': [free_stream_velocity],
            'Suction_side_displacement_thickness': [suction_thickness]
        })
        
        with st.spinner("Predicting..."):
            prediction = model.predict(input_data)[0]
        
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.subheader("Prediction Result")
        st.metric(
            label="Scaled Sound Pressure Level (dB)",
            value=f"{prediction:.3f}",
            help="Predicted noise level based on input parameters"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Version 1.0**  
    📊 Input Requirements:  
    - Frequency: Integer (Hz)  
    - Angle of Attack: Float (4 decimals, degrees)  
    - Chord Length: Float (4 decimals, m)  
    - Free-stream Velocity: Float (1 decimal, m/s)  
    - Suction Side Displacement Thickness: Float (8 decimals, m)  
    🎯 Output: Scaled Sound Pressure Level (3 decimals, dB)  
    🔄 Model: Multiple Linear Regression  
    """)

if __name__ == "__main__":
    main()
