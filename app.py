# =============================================================
#  Twitter Sentiment Analysis App (BiLSTM + GloVe)
# =============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load

# -------------------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------------------
st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="💬", layout="centered")

st.title("💬 Twitter Sentiment Analysis (BiLSTM + GloVe)")
st.markdown("Analyze the **sentiment** (Positive / Negative / Neutral) of any tweet or message in real time!")

# -------------------------------------------------------------
# Load Model and Tokenizer
# -------------------------------------------------------------
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("sentiment_bilstm_glov.h5")
    tokenizer = load("sentiment_tokenizer_glov.joblib")
    return model, tokenizer

model, tokenizer = load_resources()

# -------------------------------------------------------------
# Constants (update CLASS_LABELS after training)
# -------------------------------------------------------------
MAX_LENGTH = 50
CLASS_LABELS = ["negative", "neutral", "positive"]  # update this line using label_encoder.classes_

# -------------------------------------------------------------
# Cleaning Function
# -------------------------------------------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower().strip()
    return text

# -------------------------------------------------------------
# Prediction Function
# -------------------------------------------------------------
def predict_sentiment(text):
    text = clean_text(text.strip())
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post', truncating='post')

    preds = model.predict(padded)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    sentiment = CLASS_LABELS[class_idx].capitalize()
    return sentiment, confidence, preds

# -------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------
user_input = st.text_area("Enter a tweet or message below:", height=120)

if st.button("Predict Sentiment"):
    if user_input.strip():
        sentiment, confidence, preds = predict_sentiment(user_input)

        emoji_map = {
            "Positive": "😀",
            "Neutral": "😐",
            "Negative": "😡"
        }

        st.markdown(
            f"### {emoji_map[sentiment]}  **{sentiment}** sentiment detected\n"
            f"Confidence: `{confidence*100:.2f}%`"
        )
        st.write("Raw probabilities:", preds)
    else:
        st.warning("⚠️ Please enter some text for analysis.")

st.markdown("---")
st.caption("Built with 🧠 TensorFlow + Streamlit | Twitter Sentiment Analysis Project")
