# =============================================================
#  Twitter Sentiment Analysis App (BiLSTM + GloVe)
# =============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
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
TEXT_COLUMN_CANDIDATES = (
    "text",
    "tweet",
    "tweet_text",
    "full_text",
    "content",
    "message",
)

# -------------------------------------------------------------
# Cleaning Function
# -------------------------------------------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower().strip()
    return text


def find_text_column(dataframe):
    normalized = {column.lower().strip(): column for column in dataframe.columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    return None

# -------------------------------------------------------------
# Prediction Function
# -------------------------------------------------------------
def predict_sentiments(texts):
    cleaned_texts = [clean_text(str(text).strip()) for text in texts]
    seq = tokenizer.texts_to_sequences(cleaned_texts)
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post', truncating='post')

    preds = model.predict(padded)
    class_indices = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)
    sentiments = [CLASS_LABELS[index].capitalize() for index in class_indices]
    return sentiments, confidences, preds


def predict_sentiment(text):
    sentiments, confidences, preds = predict_sentiments([text])
    return sentiments[0], float(confidences[0]), preds

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

st.subheader("Batch CSV Prediction")
uploaded_file = st.file_uploader(
    "Upload a tweet CSV",
    type=["csv"],
    help="Supports text, tweet, tweet_text, full_text, content, or message columns.",
)

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    text_column = find_text_column(batch_df)

    if text_column is None:
        st.error("No tweet text column found. Add text, tweet, tweet_text, full_text, content, or message.")
    else:
        result_df = batch_df.dropna(subset=[text_column]).copy()
        if result_df.empty:
            st.warning("The selected text column has no rows to classify.")
        else:
            sentiments, confidences, _ = predict_sentiments(result_df[text_column])
            result_df["predicted_sentiment"] = sentiments
            result_df["prediction_confidence"] = [round(float(value), 4) for value in confidences]
            st.dataframe(result_df, use_container_width=True)
            st.download_button(
                "Download predictions",
                result_df.to_csv(index=False).encode("utf-8"),
                file_name="tweet_sentiment_predictions.csv",
                mime="text/csv",
            )

st.markdown("---")
st.caption("Built with 🧠 TensorFlow + Streamlit | Twitter Sentiment Analysis Project")
