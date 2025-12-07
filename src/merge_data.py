# src/merge_data.py
import pandas as pd
import numpy as np

def merge_news_and_stock(stock_csv="data/all_stock_data.csv", news_csv="data/finnhub_general_news.csv", output_csv="data/news_with_stock.csv"):
    """
    Merge news data with stock data, compute next_close and movement labels,
    and save the final dataset.
    """
    print("Starting: Merge News + Stock Data ...")

    # -------------------------------
    # 1️⃣ Load stock data and news
    # -------------------------------
    combined_stocks = pd.read_csv(stock_csv)
    df_news = pd.read_csv(news_csv)

    # -------------------------------
    # 2️⃣ Ensure date columns exist
    # -------------------------------
    # Stocks
    if 'Date' not in combined_stocks.columns:
        possible_date_cols = [c for c in combined_stocks.columns if 'date' in c.lower()]
        if possible_date_cols:
            combined_stocks.rename(columns={possible_date_cols[0]: 'Date'}, inplace=True)
        else:
            raise KeyError("No date column found in stock data!")

    combined_stocks['Date'] = pd.to_datetime(combined_stocks['Date'])
    combined_stocks['date'] = combined_stocks['Date'].dt.date

    # News
    if 'datetime' not in df_news.columns:
        possible_dt_cols = [c for c in df_news.columns if 'date' in c.lower()]
        if possible_dt_cols:
            df_news.rename(columns={possible_dt_cols[0]: 'datetime'}, inplace=True)
        else:
            raise KeyError("No datetime column found in news data!")

    df_news['datetime'] = pd.to_datetime(df_news['datetime'])
    df_news['date'] = df_news['datetime'].dt.date

    # -------------------------------
    # 3️⃣ Ensure symbol columns exist
    # -------------------------------
    if 'symbol' not in combined_stocks.columns:
        possible_symbol_cols = [c for c in combined_stocks.columns if 'symbol' in c.lower()]
        if possible_symbol_cols:
            combined_stocks.rename(columns={possible_symbol_cols[0]: 'symbol'}, inplace=True)
        else:
            raise KeyError("No symbol column found in stock data!")

    # -------------------------------
    # 4️⃣ Merge news with stock data
    # -------------------------------
    merged = pd.merge(
        df_news,
        combined_stocks,
        left_on=['symbol','date'],
        right_on=['symbol','date'],
        how='inner'
    )

    print(f"✅ Merged shape: {merged.shape}")

    # -------------------------------
    # 5️⃣ Compute next_close per symbol
    # -------------------------------
    symbols = merged['symbol'].unique()
    for sym in symbols:
        close_col = f"Close_{sym}"
        next_col = f"{close_col}_next"
        if close_col in merged.columns:
            merged[next_col] = merged.groupby("symbol")[close_col].shift(-1)
        else:
            print(f"⚠ Warning: {close_col} not found in merged data.")

    # -------------------------------
    # 6️⃣ Compute movement per row
    # -------------------------------
    def compute_movement(row):
        sym = row['symbol']
        close_col = f"Close_{sym}"
        next_col = f"{close_col}_next"

        close = row.get(close_col)
        next_close = row.get(next_col)

        if pd.isna(close) or pd.isna(next_close):
            return np.nan
        return int(next_close > close)

    merged['movement'] = merged.apply(compute_movement, axis=1)
    merged.dropna(subset=['movement'], inplace=True)

    # -------------------------------
    # 7️⃣ Check label distribution
    # -------------------------------
    print("\n🎯 Movement Label Distribution:")
    print(merged['movement'].value_counts())

    # -------------------------------
    # 8️⃣ Save merged dataset
    # -------------------------------
    merged.to_csv(output_csv, index=False)
    print(f"✅ Merged news + stock data saved to {output_csv}")

    return merged