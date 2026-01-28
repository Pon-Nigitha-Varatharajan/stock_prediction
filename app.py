# app.py
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# --- Make sure we can import from src/ ---
sys.path.append(os.path.abspath("src"))

from fetch_stocks import fetch_stocks
from fetch_news import fetch_news
from clean_sentiment import add_sentiment

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Stock Predictor + News Overlay", layout="wide")
st.title("📈 Stock Movement Predictor + News Sentiment Overlay")
st.caption("Models trained on stock-only data (MLP + XGBoost). News sentiment shown as an overlay (FinBERT).")
st.info(f"Last refresh: **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**")

# ----------------------------
# Helpers
# ----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert tuple/multiindex columns -> safe lowercase strings."""
    df = df.copy()
    df.columns = [
        "_".join(map(str, c)).lower() if isinstance(c, tuple) else str(c).lower()
        for c in df.columns
    ]
    return df

def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

def build_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features consistent with stock_feature_engineering.py (NO leakage).
    Expects: symbol, date, open, high, low, close, volume
    """
    df = normalize_columns(df)

    required = {"symbol", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in stock data: {missing}")

    # Make symbols clean strings
    df["symbol"] = df["symbol"].astype(str).str.strip()

    # Parse date safely
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    # Convert numeric columns safely
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where close is missing (but DO NOT drop entire tickers)
    df = df.dropna(subset=["close"]).copy()

    # Fill other price columns if missing
    df["open"] = df["open"].fillna(df["close"])
    df["high"] = df["high"].fillna(df["close"])
    df["low"] = df["low"].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0)

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Features
    df["return_1d"] = df.groupby("symbol")["close"].pct_change().fillna(0)

    df["sma_5"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["sma_10"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(10, min_periods=1).mean())
    df["volatility_5"] = df.groupby("symbol")["close"].transform(lambda x: x.rolling(5, min_periods=1).std()).fillna(0)

    df["range_pct"] = (df["high"] - df["low"]) / df["open"].replace({0: np.nan})
    df["close_open_pct"] = (df["close"] - df["open"]) / df["open"].replace({0: np.nan})

    df["return_1d_lag1"] = df.groupby("symbol")["return_1d"].shift(1).fillna(0)
    df["return_1d_lag2"] = df.groupby("symbol")["return_1d"].shift(2).fillna(0)

    return df.fillna(0)

# ----------------------------
# Load models (stock-only)
# ----------------------------
MODEL_DIR = "models"
try:
    mlp_model = joblib.load(os.path.join(MODEL_DIR, "mlp_stock.pkl"))
    xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_stock.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_stock.pkl"))
except Exception as e:
    st.error("❌ Could not load stock models. Make sure you ran: `python main.py`")
    st.exception(e)
    st.stop()

if not hasattr(scaler, "feature_names_in_"):
    st.error("❌ scaler_stock.pkl has no feature_names_in_. Retrain with scaler fit on a DataFrame.")
    st.stop()

EXPECTED_FEATURES = list(scaler.feature_names_in_)

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("⚙️ Controls")
days_back_stocks = st.sidebar.slider("Stock history window (days)", 30, 365, 180, step=30)
days_back_news = st.sidebar.slider("News window (days)", 1, 10, 3)

use_sentiment_overlay = st.sidebar.toggle("Use sentiment overlay in final score", value=True)
overlay_weight = st.sidebar.slider("Sentiment overlay weight", 0.0, 0.5, 0.15, step=0.05)

refresh = st.sidebar.button("🔄 Refresh data now")

# ----------------------------
# Step 1: Load stock data ONLY from CSV
# ----------------------------
st.subheader("📥 Stock Data (data/all_stock_data.csv)")

if refresh or not os.path.exists("data/all_stock_data.csv"):
    with st.spinner("Downloading / refreshing stock data..."):
        try:
            try:
                fetch_stocks(days_back=days_back_stocks)
            except TypeError:
                fetch_stocks()
        except Exception as e:
            st.warning("fetch_stocks failed. Using existing CSV if present.")
            st.caption(str(e))

stock_raw = safe_read_csv("data/all_stock_data.csv")
if stock_raw.empty:
    st.error("No stock data found in data/all_stock_data.csv. Run `python main.py` once.")
    st.stop()

stock_raw = normalize_columns(stock_raw)

required_cols = {"symbol", "date", "open", "high", "low", "close", "volume"}
missing = required_cols - set(stock_raw.columns)
if missing:
    st.error(f"Stock CSV missing required columns: {missing}")
    st.stop()

stock_raw["symbol"] = stock_raw["symbol"].astype(str).str.strip()

tickers = sorted(stock_raw["symbol"].dropna().unique().tolist())
selected_ticker = st.sidebar.selectbox("Select a ticker", tickers, index=0)

st.write(f"✅ Loaded stock rows: **{len(stock_raw)}** across **{len(tickers)}** tickers")

with st.expander("🔎 Debug: ticker row counts (raw CSV)"):
    st.write(stock_raw["symbol"].value_counts().head(30))

# ----------------------------
# Step 2: Feature building
# ----------------------------
st.subheader("⚙️ Feature Generation (Stock-only)")

try:
    stock_feat_all = build_stock_features(stock_raw)
except Exception as e:
    st.error("Feature build failed for stock data.")
    st.exception(e)
    st.stop()

df_t = stock_feat_all[stock_feat_all["symbol"] == selected_ticker].copy()
df_t = df_t.sort_values("date").reset_index(drop=True)

if df_t.empty:
    raw_rows = int((stock_raw["symbol"] == selected_ticker).sum())
    st.error(f"No rows for **{selected_ticker}** after feature building.")
    st.caption(f"Raw CSV rows for {selected_ticker}: {raw_rows}. Likely close/open columns are NaN for this ticker.")
    st.stop()

with st.expander("🔎 Debug: sample rows after feature build"):
    st.write(df_t.head(5))

# ----------------------------
# Step 3: Predictions over time
# ----------------------------
st.subheader("🤖 Price-based Predictions (MLP + XGBoost)")

X_hist = df_t.select_dtypes(include=["number"]).copy()
X_hist_aligned = X_hist.reindex(columns=EXPECTED_FEATURES, fill_value=0)
X_hist_scaled = scaler.transform(X_hist_aligned)

df_t["mlp_prob_up"] = mlp_model.predict_proba(X_hist_scaled)[:, 1]
df_t["xgb_prob_up"] = xgb_model.predict_proba(X_hist_scaled)[:, 1]

latest_row = df_t.tail(1).iloc[0]

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("MLP Prob Up", f"{latest_row['mlp_prob_up']:.3f}")
with c2:
    st.metric("XGB Prob Up", f"{latest_row['xgb_prob_up']:.3f}")
with c3:
    st.metric("Latest Close", f"{latest_row['close']:.2f}")

# ----------------------------
# Step 4: News + Sentiment
# ----------------------------
st.subheader("📰 News Sentiment Overlay (FinBERT)")

news_df = pd.DataFrame()
sentiment_score = 0.0

with st.spinner("Fetching news + computing sentiment..."):
    try:
        fetch_news(symbols=[selected_ticker], days_back=days_back_news)
        news_df = add_sentiment(
            input_csv="data/finnhub_general_news.csv",
            output_csv="data/news_with_sentiment.csv"
        )
    except Exception as e:
        st.warning("News fetch/sentiment failed (check API key / rate limits). Overlay set to 0.")
        st.caption(str(e))
        news_df = pd.DataFrame()

if not news_df.empty:
    news_df = normalize_columns(news_df)
    if "symbol" in news_df.columns:
        news_df["symbol"] = news_df["symbol"].astype(str).str.strip()
        news_df = news_df[news_df["symbol"] == selected_ticker].copy()

    if "sentiment_score" in news_df.columns and len(news_df) > 0:
        sentiment_score = float(pd.to_numeric(news_df["sentiment_score"], errors="coerce").fillna(0).mean())
    else:
        sentiment_score = 0.0

    st.write(f"Average sentiment score (last {days_back_news} days): **{sentiment_score:.3f}**")

    if "sentiment_label" in news_df.columns:
        st.bar_chart(news_df["sentiment_label"].fillna("unknown").value_counts())

    show_cols = [c for c in ["datetime", "headline", "source", "sentiment_label", "sentiment_score", "url"]
                 if c in news_df.columns]
    if show_cols:
        df_show = news_df[show_cols].copy()
        if "url" in df_show.columns:
            df_show["url"] = df_show["url"].apply(lambda u: f"[link]({u})" if isinstance(u, str) and u else "")
        st.dataframe(df_show.head(10), use_container_width=True)
else:
    st.info("No news available. Overlay = 0.")

# ----------------------------
# Step 5: Final score (Price + Sentiment)
# ----------------------------
st.subheader("✅ Final Signal (Price + Sentiment)")

base_prob = float(latest_row["xgb_prob_up"])
final_prob = base_prob

if use_sentiment_overlay:
    final_prob = float(np.clip(base_prob + overlay_weight * sentiment_score, 0, 1))

direction = "📈 UP" if final_prob >= 0.5 else "📉 DOWN"

c4, c5, c6 = st.columns(3)
with c4:
    st.metric("Base Prob (XGB)", f"{base_prob:.3f}")
with c5:
    st.metric("Sentiment Avg", f"{sentiment_score:.3f}")
with c6:
    st.metric("Final Prob", f"{final_prob:.3f}", direction)

# ----------------------------
# Visualizations
# ----------------------------
st.subheader("📊 Visualizations")

tab1, tab2, tab3 = st.tabs(["📈 Price", "🧠 Probabilities", "📰 News Sentiment Trend"])

with tab1:
    st.write("Close price trend")
    st.line_chart(df_t[["date", "close"]].set_index("date"))

    st.write("Indicators (SMA 5 & SMA 10)")
    st.line_chart(df_t[["date", "close", "sma_5", "sma_10"]].set_index("date"))

with tab2:
    st.write("Model probability trend (MLP vs XGB)")
    st.line_chart(df_t[["date", "mlp_prob_up", "xgb_prob_up"]].set_index("date"))

with tab3:
    if not news_df.empty and {"datetime", "sentiment_score"}.issubset(news_df.columns):
        tmp = news_df.copy()
        tmp["datetime"] = pd.to_datetime(tmp["datetime"], errors="coerce")
        tmp = tmp.dropna(subset=["datetime"]).sort_values("datetime")
        tmp = tmp[["datetime", "sentiment_score"]].set_index("datetime")
        st.write("Sentiment score trend")
        st.line_chart(tmp)
    else:
        st.info("No sentiment trend available.")

with st.expander("🔎 Debug info"):
    st.write("Expected model features:", len(EXPECTED_FEATURES))
    st.write("Sample expected features:", EXPECTED_FEATURES[:15])
    st.write("Rows for selected ticker:", len(df_t))
    st.write("Latest row date:", str(latest_row["date"]))

st.success("✅ App loaded successfully (price-model prediction + sentiment overlay).")