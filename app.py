import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    layout="wide"
)

# ---------------- LOAD MODEL (OFFLINE) ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

model = load_model()

# ---------------- HEADER ----------------
st.markdown(
    "<h1 style='text-align:center;'>🐦 Twitter Sentiment Analysis Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Offline sentiment analysis using a pretrained Transformer model</p>",
    unsafe_allow_html=True
)

st.divider()

# ---------------- LOAD CSV ----------------
st.subheader("📁 Twitter Dataset")

try:
    df = pd.read_csv("twitter_sentiment_500_FINAL.csv")
    st.success(f"Dataset loaded successfully: {len(df)} tweets")
    st.dataframe(df.head())
except:
    st.error("CSV file not found. Please keep CSV in the same folder as app.py")
    st.stop()

st.divider()

# ---------------- ANALYZE BUTTON ----------------
if st.button("🔍 Analyze Sentiment"):
    st.info("Analyzing tweets... please wait")

    df["Predicted_Sentiment"] = df["tweet"].apply(
        lambda text: model(text)[0]["label"]
    )

    st.success("Sentiment analysis completed!")

    # ---------------- RESULTS ----------------
    st.subheader("📊 Analysis Results")
    st.dataframe(df.head())

    # ---------------- SENTIMENT COUNTS ----------------
    counts = df["Predicted_Sentiment"].value_counts()

    col1, col2, col3 = st.columns(3)
    col1.metric("😊 Positive Tweets", counts.get("POSITIVE", 0))
    col2.metric("😠 Negative Tweets", counts.get("NEGATIVE", 0))
    col3.metric("📦 Total Tweets", len(df))

    # ---------------- BAR CHART ----------------
    st.subheader("📈 Sentiment Distribution")

    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Tweets")
    st.pyplot(fig)

    # ---------------- DOWNLOAD RESULT ----------------
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Analyzed CSV",
        data=csv,
        file_name="twitter_sentiment_results.csv",
        mime="text/csv"
    )
