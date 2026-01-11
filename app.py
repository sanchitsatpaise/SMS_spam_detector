# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 15:06:50 2026

@author: Sanchit Satpaise
"""

import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# --- CONFIGURATION ---
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="🛡️",
    layout="centered"
)

# --- STYLE ---
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Loading the provided model and vectorizer
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

try:
    model, vectorizer = load_assets()
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# --- HEADER ---
st.title("🛡️ Magic Spam Detector 🛡️")
st.write("Is that message a **Gift** 🎁 or a **Trap** 🪤? Find out instantly!")
st.divider()

# --- INTERACTIVE DASHBOARD ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📥 Input Area")
    user_input = st.text_area(
        "Paste your suspicious message here 👇", 
        placeholder="e.g., Congratulations! You've won a $1000 gift card. Click here to claim now!",
        height=150
    )

with col2:
    st.subheader("📊 Quick Stats")
    st.info("The model uses **Multinomial Naive Bayes** to analyze text patterns.")
    st.write(f"✨ Features detected: `{len(vectorizer.get_feature_names_out())}`")

# --- PREDICTION LOGIC ---
if st.button("🔍 Analyze Message"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message first!")
    else:
        # Transformation and Prediction
        data = [user_input]
        vectorized_input = vectorizer.transform(data)
        prediction = model.predict(vectorized_input)
        
        # Result Visualization
        st.divider()
        if prediction[0] == 1:  # Assuming 1 is Spam
            st.error("🚨 ALERT: THIS IS SPAM! 🚨")
            st.markdown("""
            ### Why is this Spam? 🧐
            - **Urgency:** Sounds too good to be true.
            - **Keywords:** Likely contains 'win', 'free', or 'claim'.
            - **Risk:** Could be a phishing attempt! 🎣
            """)
            #st.balloons() if prediction[0] == 0 else st.snow()
        else:
            st.success("✅ HURRAY! THIS MESSAGE IS SAFE. ✅")
            st.markdown("""
            ### Looks good! 👍
            - **Normal Tone:** Seems like a regular conversation.
            - **No Red Flags:** No suspicious links or demands detected.
            """)
            st.balloons()

# --- FOOTER ---
st.divider()
st.caption("Built with ❤️ using Streamlit & Machine Learning")

