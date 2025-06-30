import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load models and vectorizer
svm_model = joblib.load('svm_fake_news_model.pkl')
nb_model = joblib.load('nb_fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Set dark theme and style
st.markdown("""
    <style>
        html, body, .stApp {
            background-color: #000000 !important;
            color: #FFFFFF !important;
        }

        .title {
            font-size: 40px;
            text-align: center;
            color: #FFFFFF;
            margin-bottom: 30px;
        }

        label, .stTextInput > label, .stTextArea > label, .stSelectbox > label {
            color: #FFFFFF !important;
        }

        textarea, input, .stSelectbox div[data-baseweb="select"] {
            background-color: #1e1e1e !important;
            color: #ffffff !important;
        }

        button {
            background-color: #000000 !important;
            color: #FFFFFF !important;
            border: 1px solid #FFFFFF;
            border-radius: 8px;
            padding: 0.5em 1.2em;
            font-weight: bold;
            font-size: 16px;
            transition: background-color 0.3s ease;
        }

        button:hover {
            background-color: #222222 !important;
        }

        .result-box {
            border-radius: 10px;
            padding: 20px;
            font-size: 24px;
            font-weight: bold;
            margin-top: 30px;
            text-align: center;
        }

        .real {
            background-color: #145A32;
            color: white;
        }

        .fake {
            background-color: #922B21;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# App title (no emojis)
st.markdown("<div class='title'>Fake News Detector</div>", unsafe_allow_html=True)

# Inputs
news_input = st.text_area("Enter a news article or headline:", height=200)
model_choice = st.selectbox("Select Algorithm", ["SVM", "Naive Bayes"])

# Prediction button
if st.button("Predict"):
    if news_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_vector = vectorizer.transform([news_input])
        pred = svm_model.predict(input_vector)[0] if model_choice == "SVM" else nb_model.predict(input_vector)[0]

        if pred == 1:
            st.markdown("<div class='result-box real'>This news is <strong>Real</strong></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-box fake'>This news is <strong>Fake</strong></div>", unsafe_allow_html=True)
