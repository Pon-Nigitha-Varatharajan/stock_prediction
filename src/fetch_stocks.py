import os
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

# Use same symbol list as fetch_news.py
SYMBOLS = [
    "AAPL","MSFT","GOOG","AMZN","META","NVDA","TSLA",
    "JPM","BAC","WFC","KO","PEP","MCD","NKE","PG",
    "BRK-B","DIS","NFLX","ORCL","CRM"
]


def fetch_stocks(tickers=None, days_back=365, save_dir="data"):
    os.makedirs(save_dir, exist_ok=True)
    if tickers is None:
        tickers = SYMBOLS

    to_date = datetime.utcnow().date()
    from_date = to_date - timedelta(days=days_back)

    all_rows = []

    for sym in tickers:
        print(f"Downloading {sym} from {from_date} to {to_date}...")
        try:
            df = yf.download(sym, start=str(from_date), end=str(to_date + timedelta(days=1)), interval="1d", progress=False)
        except Exception as e:
            print(f"Error downloading {sym}: {e}")
            continue

        if df is None or df.empty:
            print(f"No data for {sym}, skipping.")
            continue

        df = df.reset_index()
        # Standardize column names and long format
        df.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
        df["symbol"] = sym
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # Keep only needed columns
        subset = df[["symbol", "date", "open", "high", "low", "close", "volume"]]
        all_rows.append(subset)

        # save per-symbol optionally
        subset.to_csv(os.path.join(save_dir, f"{sym}_stock.csv"), index=False)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(os.path.join(save_dir, "all_stock_data.csv"), index=False)
        print("✅ Combined stock data saved:", combined.shape)
        return combined
    else:
        print("❌ No stock data downloaded")
        return pd.DataFrame()


if __name__ == "__main__":
    fetch_stocks()
