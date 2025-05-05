# twitter_sentiment_app.py

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load
import numpy as np
import re

# Constants
MAX_LEN = 100  # must match training time

# Load trained components
model = tf.keras.models.load_model('sentiment_model.keras')
tokenizer = load('tokenizer.joblib')
label_encoder = load('label_encoder.joblib')

# Text preprocessing function
def clean_text(text):
    """Clean input text: remove punctuation and convert to lowercase."""
    return re.sub(r'[^a-zA-Z\s]', '', text.lower())

# Sentiment prediction function
def predict_sentiment(text):
    """Predict sentiment from raw input text."""
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    probs = model.predict(padded, verbose=0)[0]
    predicted_class = np.argmax(probs)
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label, probs

# Streamlit UI setup
st.set_page_config(page_title="Sentiment Analysis", page_icon="🔍")
st.title("🧠 Twitter Sentiment Analysis")
st.write("Enter a tweet-like sentence to analyze its **sentiment**.")

# Input area
text_input = st.text_area("💬 Your input here:")

# Predict button
if st.button("🔍 Predict Sentiment"):
    if text_input.strip():
        sentiment, probabilities = predict_sentiment(text_input)
        st.success(f"**Predicted Sentiment:** {str(sentiment).capitalize()}")

        st.markdown("### 🔢 Confidence Scores:")
        for label, score in zip(label_encoder.classes_, probabilities):
            st.write(f"- **{str(label).capitalize()}**: {score * 100:.2f}%")
    else:
        st.warning("⚠️ Please enter some text before prediction.")
