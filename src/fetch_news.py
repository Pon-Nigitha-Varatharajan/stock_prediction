import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from time import sleep

FINNHUB_API_KEY = "d1tjq09r01qth6pll2r0d1tjq09r01qth6pll2rg"

symbols = [
    "AAPL","MSFT","GOOG","AMZN","META","NVDA","TSLA",
    "JPM","BAC","WFC","KO","PEP","MCD","NKE","PG",
    "BRK.B","DIS","NFLX","ORCL","CRM"
]

def fetch_news():
    os.makedirs("data", exist_ok=True)
    to_date = datetime.today().date()
    from_date = to_date - timedelta(days=7)

    all_news = []

    for sym in symbols:
        print(f"Fetching news for {sym}...")
        url = f"https://finnhub.io/api/v1/company-news?symbol={sym}&from={from_date}&to={to_date}&token={FINNHUB_API_KEY}"

        try:
            resp = requests.get(url)
            resp.raise_for_status()
            data = resp.json()

            for article in data:
                dt = article.get("datetime")
                headline = article.get("headline")

                if dt and headline:
                    all_news.append({
                        "symbol": sym,
                        "datetime": pd.to_datetime(dt, unit="s"),
                        "headline": headline,
                        "source": article.get("source", ""),
                        "url": article.get("url", "")
                    })

        except Exception as e:
            print(f"Error fetching {sym}: {e}")

        sleep(0.2)

    df = pd.DataFrame(all_news)
    df.drop_duplicates(subset=["symbol","datetime","headline"], inplace=True)
    df.to_csv("data/finnhub_general_news.csv", index=False)
    print("News saved:", df.shape)