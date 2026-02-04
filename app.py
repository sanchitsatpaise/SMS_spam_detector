import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

st.title("📩 SMS Spam Detector")
st.write("Using a pre-trained **balanced Logistic Regression model**")

# -------------------------------------------------
# Load NLP tools
# -------------------------------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# -------------------------------------------------
# Load model & vectorizer
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("logistic_regression_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer (2).pkl")
    return model, vectorizer

model, vectorizer = load_model()

# -------------------------------------------------
# User input
# -------------------------------------------------
st.subheader("✍️ Enter SMS text")

sms_text = st.text_area(
    "Paste or type the SMS message below:",
    height=150,
    placeholder="Congratulations! You have won a free prize. Call now..."
)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("🔍 Predict"):
    if sms_text.strip() == "":
        st.warning("Please enter a message to analyze.")
    else:
        cleaned_text = clean_text(sms_text)
        vectorized_text = vectorizer.transform([cleaned_text])

        prediction = model.predict(vectorized_text)[0]
        probability = model.predict_proba(vectorized_text)[0][1]

        if prediction == 1:
            st.error(f"🚨 **SPAM** detected\n\nConfidence: **{probability:.2%}**")
        else:
            st.success(f"✅ **HAM (Not Spam)**\n\nConfidence: **{1 - probability:.2%}**")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("📊 Model Details")
st.sidebar.markdown("""
**Algorithm:** Logistic Regression  
**Vectorizer:** TF-IDF  
**Bias Handling:** SMOTE (during training)  
**Evaluation:** Precision, Recall, F1-score  
""")

st.sidebar.success("Production-ready model ✔️")
