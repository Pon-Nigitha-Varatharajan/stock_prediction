import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

symbols = [
    "AAPL","MSFT","GOOG","AMZN","META","NVDA","TSLA",
    "JPM","BAC","WFC","KO","PEP","MCD","NKE","PG",
    "BRK.B","DIS","NFLX","ORCL","CRM"
]

def fetch_stocks():
    os.makedirs("data", exist_ok=True)

    to_date = datetime.today().date()
    from_date = to_date - timedelta(days=7)

    all_data = {}

    for sym in symbols:
        print(f"Fetching {sym}...")
        df = yf.download(sym, start=str(from_date), end=str(to_date), interval="1d")

        if df.empty:
            print(f"Skipping {sym}")
            continue

        df.reset_index(inplace=True)
        df = df.rename(columns={col: f"{col}_{sym}" for col in ["Open","High","Low","Close","Volume"]})
        df["symbol"] = sym

        df.to_csv(f"data/{sym}_stock.csv", index=False)
        all_data[sym] = df

    if all_data:
        combined = pd.concat(all_data.values(), ignore_index=True)
        combined["date"] = pd.to_datetime(combined["Date"]).dt.date
        combined.to_csv("data/all_stock_data.csv", index=False)
        print("Combined stock data saved:", combined.shape)