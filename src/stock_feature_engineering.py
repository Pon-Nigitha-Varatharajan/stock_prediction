import os
import numpy as np
import pandas as pd


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        "_".join(map(str, c)).lower() if isinstance(c, tuple) else str(c).lower()
        for c in df.columns
    ]
    return df


def _clean_numeric(series: pd.Series) -> pd.Series:
    """
    Robust numeric cleaner:
    - strips spaces
    - removes commas
    - removes non-numeric junk (keeps digits, dot, minus)
    """
    s = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace(r"[^0-9\.\-]+", "", regex=True)
    )
    return pd.to_numeric(s, errors="coerce")


def stock_feature_engineering(
    stock_csv="data/all_stock_data.csv",
    output_csv="data/stock_features.csv",
    return_threshold=0.002,     # 0.2% threshold for "up"
    min_rows_per_symbol=30      # drop tickers with too few rows
):
    """
    Creates a stock-only training dataset.

    Required input columns (case-insensitive):
    symbol, date, open, high, low, close, volume

    Outputs:
    - engineered numeric features
    - labels: movement (1 if next-day return > threshold else 0)
    """

    if not os.path.exists(stock_csv):
        raise FileNotFoundError(f"Missing stock CSV: {stock_csv}")

    df = pd.read_csv(stock_csv)
    df = _normalize_cols(df)

    required = {"symbol", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Stock CSV missing required columns: {missing}")

    # Clean basic columns
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "symbol"]).copy()

    # Robust numeric conversion
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = _clean_numeric(df[col])

    # Drop rows without close (cannot compute returns)
    df = df.dropna(subset=["close"]).copy()

    # Fill other missing OHLC with close; volume with 0
    df["open"] = df["open"].fillna(df["close"])
    df["high"] = df["high"].fillna(df["close"])
    df["low"] = df["low"].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0)

    # Drop tickers with too few rows
    counts = df.groupby("symbol")["close"].transform("count")
    df = df[counts >= min_rows_per_symbol].copy()

    # Sort for time series features
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # ---------------------------
    # Labels (next-day movement)
    # ---------------------------
    df["next_close"] = df.groupby("symbol")["close"].shift(-1)
    df["return_next"] = (df["next_close"] - df["close"]) / df["close"].replace({0: np.nan})
    df["movement"] = (df["return_next"] > return_threshold).astype(int)

    # Remove last row per symbol (no next_close)
    df = df.dropna(subset=["next_close"]).copy()

    # ---------------------------
    # Feature engineering
    # ---------------------------
    # returns
    df["return_1d"] = df.groupby("symbol")["close"].pct_change().fillna(0)
    df["return_1d_lag1"] = df.groupby("symbol")["return_1d"].shift(1).fillna(0)
    df["return_1d_lag2"] = df.groupby("symbol")["return_1d"].shift(2).fillna(0)

    # rolling stats (use transform so index stays aligned)
    df["sma_5"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["sma_10"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(10, min_periods=1).mean())
    df["volatility_5"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(5, min_periods=1).std()).fillna(0)

    # ranges
    df["range_pct"] = (df["high"] - df["low"]) / df["open"].replace({0: np.nan})
    df["close_open_pct"] = (df["close"] - df["open"]) / df["open"].replace({0: np.nan})

    # volume features
    df["volume_sma_10"] = df.groupby("symbol")["volume"].transform(lambda x: x.rolling(10, min_periods=1).mean()).fillna(0)

    # EMA / MACD
    df["ema_12"] = df.groupby("symbol")["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    df["ema_26"] = df.groupby("symbol")["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df["macd"] = df["ema_12"] - df["ema_26"]

    # RSI(14)
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.rolling(period, min_periods=1).mean()
        roll_down = down.rolling(period, min_periods=1).mean()
        rs = roll_up / roll_down.replace({0: np.nan})
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(0)

    df["rsi_14"] = df.groupby("symbol")["close"].transform(_rsi)

    # Final cleanup
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Save
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("✅ Stock features saved:", output_csv)
    print("📊 Label distribution:")
    print(df["movement"].value_counts())

    return df


if __name__ == "__main__":
    stock_feature_engineering()