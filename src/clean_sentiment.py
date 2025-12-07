import pandas as pd
import re
import torch
from transformers import pipeline

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def add_sentiment():
    df = pd.read_csv("data/news_with_stock.csv")
    df["clean_headline"] = df["headline"].apply(clean_text)

    device = 0 if torch.cuda.is_available() else -1
    model = pipeline("sentiment-analysis",
                     model="yiyanghkust/finbert-tone",
                     tokenizer="yiyanghkust/finbert-tone",
                     device=device)

    sentiments = model(df["headline"].tolist(), truncation=True)
    df["sentiment"] = [s["label"].lower() for s in sentiments]

    df.to_csv("data/processed_dataset.csv", index=False)
    print("Sentiment applied:", df.shape)