import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from time import sleep

# Prefer environment variable. If not set, replace below (not recommended for production).
FINNHUB_API_KEY = "d1tjq09r01qth6pll2r0d1tjq09r01qth6pll2rg"

# A conservative sample of symbols. Use same list in fetch_stocks.py for consistency.
SYMBOLS = [
    "AAPL","MSFT","GOOG","AMZN","META","NVDA","TSLA",
    "JPM","BAC","WFC","KO","PEP","MCD","NKE","PG",
    "BRK-B","DIS","NFLX","ORCL","CRM"
]


def fetch_news(symbols=SYMBOLS, days_back=365, save_csv="data/finnhub_general_news.csv"):
    os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)

    if not FINNHUB_API_KEY:
        raise EnvironmentError("FINNHUB_API_KEY not set. Export it as an environment variable.")

    to_date = datetime.utcnow().date()
    from_date = to_date - timedelta(days=days_back)

    all_news = []
    for sym in symbols:
        print(f"Fetching news for {sym}...")
        url = (
            f"https://finnhub.io/api/v1/company-news?symbol={sym}&from={from_date}&to={to_date}&token={FINNHUB_API_KEY}"
        )
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            items = resp.json()
            for article in items:
                dt = article.get("datetime")
                headline = article.get("headline")
                if dt and headline:
                    all_news.append({
                        "symbol": sym,
                        "datetime": pd.to_datetime(dt, unit="s"),
                        "headline": headline,
                        "source": article.get("source", ""),
                        "url": article.get("url", ""),
                    })
        except Exception as e:
            print(f"Error fetching {sym}: {e}")

        sleep(0.2)  # be polite to API

    df = pd.DataFrame(all_news)
    if df.empty:
        print("⚠️ No news fetched")
    else:
        df.drop_duplicates(subset=["symbol", "datetime", "headline"], inplace=True)
        df["date"] = df["datetime"].dt.date
        os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
        df.to_csv(save_csv, index=False)
        print("✅ News saved:", df.shape)

    return df


if __name__ == "__main__":
    fetch_news()
