import streamlit as st
import numpy as np
import joblib

# Interface
st.markdown("## Iris Species Prediction")
# Text input for next character prediction
context_size = st.slider(
    "Number of k characters prediction", min_value=1, max_value=10, value=5
)
embedding_dim = st.slider("Embedding dimension", min_value=1, max_value=100, value=50)
text_input = st.text_input("Enter text for next character prediction")

# Predict button
if st.button("Predict"):
    model = joblib.load("iris_model.pkl")
    predictions = model.predict(text_input, context_size, embedding_dim)
