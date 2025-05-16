import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import requests
from xgboost import XGBRegressor 
import io
# ========================
# CONFIG
# ========================
# Truy cập thông tin từ Streamlit Secrets

try:
    GITHUB_USERNAME = st.secrets["github"]["username"] 
    GITHUB_REPO = st.secrets["github"]["repo"] 
    MODEL_LOCAL_PATH = st.secrets["github"]["model_url"]  
except KeyError as e:
    st.error(f"Thiếu thông tin cấu hình trong Streamlit Secrets: {e}")
    st.stop()



st.set_page_config(page_title="Airfoil Self-Noise Prediction", page_icon="🛩️", layout="wide")

def set_custom_style():
    st.markdown("""
    <style>
        .stApp {background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%); font-family: 'Arial', sans-serif;}
        h1, h2, h3, h4, h5, h6 {color: #1a237e !important; font-weight: 600;}
        .stTextInput > label, .stNumberInput > label, .stCheckbox > label {color: #1a237e !important; font-weight: 500;}
        .result-container {background-color: #fff; border-radius: 10px; padding: 1.5rem; margin-top: 1.5rem; box-shadow: 0 2px 8px rgba(30, 60, 90, 0.08); text-align: center;}
        .stButton>button {background-color: #1976d2; color: white; border-radius: 25px; font-size: 1.1em; font-weight: 500; padding: 0.5rem 1.5rem;}
        .stForm {border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem; background-color: rgba(255, 255, 255, 0.9);}
        .warning-text {color: #d32f2f !important; font-weight: 500;}
    </style>
    """, unsafe_allow_html=True)
set_custom_style()


def load_model():
    try:
        headers = {"Authorization": f"token {st.secrets['github']['pat']}"}
        response = requests.get(MODEL_LOCAL_PATH, headers=headers)
        response.raise_for_status()
        model = joblib.load(io.BytesIO(response.content))
        return model
    except Exception as e:
        st.error(f"Không tải được mô hình từ GitHub: {e}")
        return None

def train_and_save_model(df):
    required_columns = ['Frequency', 'Angle_of_attack', 'Chord_length', 'Free_stream_velocity', 'Suction_side_displacement_thickness', 'Scaled_sound_pressure_level']
    if not all(col in df.columns for col in required_columns):
        st.error(f"File CSV thiếu một hoặc nhiều cột yêu cầu: {required_columns}")
        st.stop()

    X = df[['Frequency', 'Angle_of_attack', 'Chord_length', 'Free_stream_velocity', 'Suction_side_displacement_thickness']]
    y = df['Scaled_sound_pressure_level']

    model = XGBRegressor(n_estimators=500, max_depth = 7, learning_rate=0.1, subsample = 0.8,  random_state=42) 
    model.fit(X, y)

    # Save model
    #joblib.dump(model, MODEL_LOCAL_PATH)
    return model


# ========================
# MAIN APP
# ========================
st.title("🛩️ Dự Đoán Tiếng Ồn Cánh Khí Động (Airfoil Self-Noise)")
st.markdown("#### Nhập các thông số bên dưới để dự đoán mức độ tiếng ồn (SSPL - dB) phát ra từ cánh khí động học.")

# Sidebar
st.sidebar.header("Cấu hình mô hình")
retrain = st.sidebar.checkbox("🔄 Train lại mô hình (PORTABLE)", value=False, help="Tải file CSV để train mô hình mới")

# Model handling
model = None
if retrain:
    uploaded_file = st.file_uploader("Tải lên file dữ liệu (CSV)", type=["csv"], help="File phải có các cột: Frequency, Angle_of_attack, Chord_length, Free_stream_velocity, Suction_side_displacement_thickness, Scaled_sound_pressure_level")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        with st.spinner("Đang train mô hình..."):
            model = train_and_save_model(df)
        st.sidebar.success("Đã train và lưu mô hình mới!")
    else:
        st.sidebar.info("Vui lòng tải lên file CSV để train mô hình.")
else:
    model = load_model()
    if model is None:
        st.error("Không tìm thấy mô hình. Vui lòng chọn 'Train lại mô hình' và tải lên file dữ liệu CSV.")
        st.stop()

# Input form
with st.form("input_form"):
    st.markdown("**Thông số đầu vào**")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        frequency = st.number_input("Tần số (Hz)", min_value=1, max_value=20000, step=1, value=800, format="%d")
        chord_length = st.number_input("Chiều dài dây cung (m)", min_value=0.01, max_value=1.0, step=0.0001, value=0.3048, format="%.4f")
    with col2:
        angle_of_attack = st.number_input("Góc tấn (degree)", min_value=-10.0, max_value=30.0, step=0.0001, value=0.0, format="%.4f")
        free_stream_velocity = st.number_input("Vận tốc dòng khí (m/s)", min_value=1.0, step=0.1, value=71.3, format="%.1f")
        if free_stream_velocity > 100.0:
            st.markdown('<p class="warning-text">Giá trị cao bất thường</p>', unsafe_allow_html=True)
    with col3:
        suction_thickness = st.number_input("Độ dày biên dạng (m)", min_value=0.00000001, max_value=0.1, step=0.00000001, value=0.00266337, format="%.8f")
    submitted = st.form_submit_button("Dự đoán SSPL (dB)")

# Prediction
if submitted and model is not None:
    try:
        X_input = pd.DataFrame([[frequency, angle_of_attack, chord_length, free_stream_velocity, suction_thickness]], 
                              columns=['Frequency', 'Angle_of_attack', 'Chord_length', 'Free_stream_velocity', 'Suction_side_displacement_thickness'])
        y_pred = model.predict(X_input)[0]
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.subheader("🔊 Kết quả dự đoán")
        st.metric("SSPL (dB)", f"{y_pred:.3f}", help="Mức độ tiếng ồn dự đoán")
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Lỗi dự đoán: {e}")

# Info section
with st.expander("📖 Thông tin dữ liệu & hướng dẫn"):
    st.markdown("""
    **Thông số đầu vào:**
    - **Tần số (Hz)**: Số nguyên, ví dụ: 800
    - **Góc tấn (degree)**: Số thực, 4 chữ số thập phân, ví dụ: 0.0000
    - **Chiều dài dây cung (m)**: Số thực, 4 chữ số thập phân, ví dụ: 0.3048
    - **Vận tốc dòng khí (m/s)**: Số thực, 1 chữ số thập phân, ví dụ: 71.3
    - **Độ dày biên dạng (m)**: Số thực, 8 chữ số thập phân, ví dụ: 0.00266337

    **Kết quả:**
    - **SSPL (dB)**: Mức độ tiếng ồn dự đoán, số thực, 3 chữ số thập phân

    **Lưu ý:**
    - File CSV để train mô hình phải có các cột: Frequency, Angle_of_attack, Chord_length, Free_stream_velocity, Suction_side_displacement_thickness, Scaled_sound_pressure_level.
    - Nếu mô hình không tải được từ GitHub, hãy train lại bằng cách tải lên file CSV.
    - Nếu vận tốc dòng khí > 100 m/s, sẽ có cảnh báo "Giá trị cao bất thường".
    """)

st.markdown("---")
st.markdown("<small>Developed by Nhóm 1 - Chuyên đề AI. Data source: NASA airfoil self-noise dataset. URL: https://archive.ics.uci.edu/dataset/291/airfoil+self+noise</small>", unsafe_allow_html=True)
