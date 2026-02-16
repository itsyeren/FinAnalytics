import streamlit as st
import random

st.title("Long Model")

st.sidebar.header("Controls")

ticker = st.sidebar.selectbox(
    "Select stock",
    ["AAPL", "MSFT", "NVDA"]
)

st.subheader("Selection")
st.write({"ticker": ticker})

st.subheader("Model Prediction")

# dummy prediction logic for demonstration purposes
score = random.uniform(0.4, 0.7)
direction = "UP" if score > 0.5 else "DOWN"
confidence = round(score * 100)

if direction == "UP":
    st.success(f"↑ UP  %{confidence}")
else:
    st.error(f"↓ DOWN %{confidence}")
