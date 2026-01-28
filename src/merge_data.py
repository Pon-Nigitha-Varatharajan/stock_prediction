import os
import pandas as pd
import numpy as np


def merge_news_and_stock(
    news_csv="data/finnhub_general_news.csv",
    stock_csv="data/all_stock_data.csv",
    output_csv="data/news_with_stock.csv",
    return_threshold=0.0,  
):
    print("🔧 Loading news and stock data...")

    if not os.path.exists(news_csv) or not os.path.exists(stock_csv):
        raise FileNotFoundError("News or stock CSV missing.")

    df_news = pd.read_csv(news_csv)
    df_stock = pd.read_csv(stock_csv)

    # ----------------------------
    # News preprocessing
    # ----------------------------
    if "datetime" not in df_news.columns:
        raise KeyError("'datetime' not found in news CSV")
    if "symbol" not in df_news.columns:
        raise KeyError("'symbol' missing in news CSV")

    df_news["datetime"] = pd.to_datetime(df_news["datetime"], errors="coerce")
    df_news = df_news.dropna(subset=["datetime"])
    df_news["date"] = df_news["datetime"].dt.date

    # ----------------------------
    # Stock preprocessing
    # ----------------------------
    if "date" not in df_stock.columns or "symbol" not in df_stock.columns:
        raise KeyError("'symbol' or 'date' missing in stock CSV")

    df_stock["date"] = pd.to_datetime(df_stock["date"], errors="coerce").dt.date
    df_stock = df_stock.dropna(subset=["date"])

    # ✅ Force numeric (fixes your error)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df_stock.columns:
            # handle possible commas like "1,234.56"
            df_stock[col] = (
                df_stock[col]
                .astype(str)
                .str.replace(",", "", regex=False)
            )
            df_stock[col] = pd.to_numeric(df_stock[col], errors="coerce")

    df_stock = df_stock.dropna(subset=["close"]).reset_index(drop=True)
    df_stock = df_stock.sort_values(["symbol", "date"]).reset_index(drop=True)

    # ----------------------------
    # Compute next-day movement on stock table
    # ----------------------------
    df_stock["next_close"] = df_stock.groupby("symbol")["close"].shift(-1)
    df_stock["next_return"] = (df_stock["next_close"] - df_stock["close"]) / df_stock["close"]
    df_stock["movement"] = (df_stock["next_return"] > return_threshold).astype(int)

    df_stock = df_stock.dropna(subset=["next_close"]).reset_index(drop=True)

    print("📊 Stock movement distribution:")
    print(df_stock["movement"].value_counts(dropna=False))

    # ----------------------------
    # Merge
    # ----------------------------
    merged = pd.merge(df_news, df_stock, on=["symbol", "date"], how="inner")
    print("✅ Final merged shape:", merged.shape)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    merged.to_csv(output_csv, index=False)
    print("✅ Merged news+stock saved:", output_csv)

    return merged


if __name__ == "__main__":
    merge_news_and_stock()