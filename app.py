import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

st.set_page_config(page_title="Twitter Sentiment Analysis", layout="wide")

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

model = load_model()

st.title("🐦 Twitter Sentiment Analysis Dashboard")
st.write("Offline sentiment analysis using a pretrained Transformer model")

# ✅ CSV UPLOAD (FIX)
uploaded_file = st.file_uploader(
    "Upload twitter_sentiment_500_FINAL.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload the CSV file to continue")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success(f"Loaded {len(df)} tweets")
st.dataframe(df.head())

if st.button("Analyze Sentiment"):
    df["Predicted_Sentiment"] = df["tweet"].apply(
        lambda x: model(str(x))[0]["label"]
    )

    st.success("Analysis completed")
    st.dataframe(df.head())

    counts = df["Predicted_Sentiment"].value_counts()

    st.subheader("📊 Sentiment Distribution")
    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Tweets")
    st.pyplot(fig)


   
