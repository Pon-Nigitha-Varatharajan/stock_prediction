import os
import re
import pandas as pd
import torch
from transformers import pipeline


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def add_sentiment(input_csv="data/news_with_stock.csv", output_csv="data/processed_dataset.csv", batch_size=32, model_name="yiyanghkust/finbert-tone"):
    """Reads merged news+stock CSV, computes sentiment on cleaned headlines and writes processed CSV.

    Uses HF pipeline with batching to avoid OOM or single-call limits.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "headline" not in df.columns:
        raise KeyError("'headline' column not found in input CSV")

    df["clean_headline"] = df["headline"].apply(clean_text)

    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        device=device,
    )

    sentiments = []
    texts = df["clean_headline"].fillna("").tolist()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # each model call returns a list of dicts like {'label': 'positive', 'score': 0.99}
        out = sentiment_pipe(batch, truncation=True)
        sentiments.extend(out)

    # align lengths
    if len(sentiments) != len(df):
        raise RuntimeError("Sentiment pipeline returned unexpected number of results")

    df["sentiment_label"] = [s.get("label", "neutral").lower() for s in sentiments]
    df["sentiment_score"] = [s.get("score", 0.0) for s in sentiments]

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print("✅ Sentiment added and saved:", output_csv)
    return df


if __name__ == "__main__":
    add_sentiment()
