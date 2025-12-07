import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib
import os

# ---------------------------------------------------------
# 1. ADD TECHNICAL INDICATORS FOR ALL COMPANIES
# ---------------------------------------------------------
def add_technical_indicators(df):
    df = df.copy()

    # Detect companies dynamically (AAPL, MSFT, AMZN, ...)
    companies = sorted({col.split("_")[1] for col in df.columns 
                        if col.startswith("Close_")})

    for comp in companies:
        open_col = f"Open_{comp}"
        high_col = f"High_{comp}"
        low_col = f"Low_{comp}"
        close_col = f"Close_{comp}"
        vol_col = f"Volume_{comp}"

        # Some companies may not exist for certain rows – skip safely
        if close_col not in df or vol_col not in df:
            continue

        close = df[close_col]
        volume = df[vol_col]

        # --- Basic returns ---
        df[f"return_1_{comp}"] = close.pct_change(fill_method=None)
        df[f"return_5_{comp}"] = close.pct_change(5, fill_method=None)
        df[f"return_10_{comp}"] = close.pct_change(10, fill_method=None)

        # --- Moving averages ---
        df[f"sma_5_{comp}"] = close.rolling(5).mean()
        df[f"sma_10_{comp}"] = close.rolling(10).mean()
        df[f"sma_20_{comp}"] = close.rolling(20).mean()

        # --- EMA ---
        df[f"ema_12_{comp}"] = close.ewm(span=12, adjust=False).mean()
        df[f"ema_26_{comp}"] = close.ewm(span=26, adjust=False).mean()

        # --- MACD ---
        df[f"macd_{comp}"] = df[f"ema_12_{comp}"] - df[f"ema_26_{comp}"]
        df[f"macd_signal_{comp}"] = df[f"macd_{comp}"].ewm(span=9, adjust=False).mean()

        # --- RSI ---
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        rs = roll_up / roll_down
        df[f"rsi_14_{comp}"] = 100 - (100 / (1 + rs))

        # --- Volatility ---
        df[f"rolling_std_10_{comp}"] = close.rolling(10).std()
        df[f"rolling_std_20_{comp}"] = close.rolling(20).std()

        # --- Volume indicators ---
        df[f"volume_sma_10_{comp}"] = volume.rolling(10).mean()
        df[f"volume_change_{comp}"] = volume.pct_change(fill_method=None)

        # --- Price ranges ---
        df[f"high_low_range_{comp}"] = df[high_col] - df[low_col]
        df[f"close_open_range_{comp}"] = close - df[open_col]

    # DO NOT DROP ROWS → This was causing dataset = 0 rows
    df = df.fillna(0)
    return df


# ---------------------------------------------------------
# 2. APPLY PCA
# ---------------------------------------------------------
def apply_pca():
    print("🔹 Loading feature-engineered dataset...")
    df = pd.read_csv("data/feature_engineered_dataset.csv")

    print("🔹 Adding technical indicators...")
    df = add_technical_indicators(df)

    if "movement" not in df:
        raise ValueError("❌ ERROR: 'movement' column missing from dataset")

    y = df["movement"]
    df = df.drop(columns=["movement"])

    print("🔹 Selecting numeric columns...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].fillna(0)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=10))
    ])

    print("🔹 Running PCA...")
    X_reduced = pipeline.fit_transform(X)

    os.makedirs("models", exist_ok=True)
    np.save("data/X_reduced.npy", X_reduced)
    np.save("data/y.npy", y.to_numpy())
    joblib.dump(pipeline, "models/pca_pipeline.pkl")

    print("✅ PCA reduced:", X_reduced.shape)


# ---------------------------------------------------------
# 3. RUN DIRECTLY
# ---------------------------------------------------------
if __name__ == "__main__":
    apply_pca()