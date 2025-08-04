import streamlit as st
import numpy as np
import pickle

# Load the saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# ---------- STYLING ----------
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# Add custom CSS to change background and text styles
st.markdown("""
    <style>
        body {
            background-color: #f1f4f8;
        }
        .stApp {
            background: linear-gradient(to bottom, #f8fbff, #e8f0f7);
            border-radius: 12px;
            padding: 20px;
        }
        .title {
            font-size: 40px;
            text-align: center;
            color: #003366;
            margin-bottom: 20px;
        }
        .subtext {
            text-align: center;
            font-size: 18px;
            color: #444;
        }
        .result-box {
            background-color: #ffffff;
            border-left: 6px solid #003366;
            padding: 15px;
            margin-top: 15px;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="title">🚢 Titanic Survival Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Fill in the details below to see if the passenger would have survived the Titanic disaster.</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------- INPUTS ----------
st.subheader("🎫 Passenger Information")

# Passenger ID (not used for prediction — just for display)
passenger_id = st.number_input(
    "Passenger ID (for display only)",
    min_value=1,
    max_value=891,
    step=1
)

# Organize layout into two columns
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox('Passenger Class', options=[1, 2, 3], format_func=lambda x: f"{x}st Class" if x == 1 else f"{x}nd Class" if x == 2 else f"{x}rd Class")
    age = st.number_input('Age', min_value=0.0, max_value=100.0, step=0.5)
    family_size = st.number_input('Family Size (siblings/spouses + parents/children)', min_value=0, step=1)

with col2:
    sex = st.selectbox('Sex', options=['male', 'female'])
    embarked = st.selectbox('Port of Embarkation', options=['C', 'Q', 'S'])

# ---------- ENCODING ----------
enc_sex = 1 if sex == 'male' else 0
C = 1.0 if embarked == 'C' else 0.0
Q = 1.0 if embarked == 'Q' else 0.0
S = 1.0 if embarked == 'S' else 0.0

# ---------- INPUT FOR MODEL ----------
input_data = np.array([[pclass, age, family_size, enc_sex, C, Q, S]])

# ---------- PREDICTION ----------
if st.button('🔍 Predict Survival'):
    prediction = model.predict(input_data)

    st.markdown("---")
    st.markdown(f"<div class='result-box'><strong>Passenger ID:</strong> {int(passenger_id)}</div>", unsafe_allow_html=True)

    if prediction[0] == 1:
        st.markdown("<div class='result-box' style='border-left-color: green;'><strong>✅ Prediction:</strong> This passenger <b>would have survived</b> the Titanic tragedy. 🎉</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-box' style='border-left-color: red;'><strong>❌ Prediction:</strong> This passenger <b>would not have survived</b>. 💔</div>", unsafe_allow_html=True)
