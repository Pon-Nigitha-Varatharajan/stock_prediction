import os
import sys

def run_step(name, func):
    print(f"\n🔹 Starting: {name} ...")
    try:
        func()
        print(f"✅ Finished: {name}")
    except Exception as e:
        print(f"❌ ERROR in {name}: {e}")
        sys.exit(1)


# -------------------------------------------------------
#  IMPORT FUNCTIONS
# -------------------------------------------------------
from src.fetch_news import fetch_news
from src.fetch_stocks import fetch_stocks
from src.merge_data import merge_news_and_stock   
from src.clean_sentiment import add_sentiment
from src.feature_engineering import feature_engineering as generate_features
from src.pca_reduce import apply_pca
from src.train_models import train_models


# -------------------------------------------------------
#  CHECK REQUIRED FOLDERS
# -------------------------------------------------------
REQUIRED_FOLDERS = ["data", "models", "src"]

for folder in REQUIRED_FOLDERS:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"📁 Created missing folder: {folder}")


# -------------------------------------------------------
#  MAIN PIPELINE
# -------------------------------------------------------
if __name__ == "__main__":

    print("\n🚀 Running Stock–News Prediction Pipeline")
    print("----------------------------------------")

    # 1️⃣ Fetch News
    run_step("News Fetching", fetch_news)

    # 2️⃣ Fetch Stock Prices
    run_step("Stock Price Fetching", fetch_stocks)

    # 3️⃣ Check if BOTH are available before merging
    if not os.path.exists("data/finnhub_general_news.csv"):
        print("❌ No news file found. Stopping pipeline.")
        sys.exit(1)

    run_step("Merge News + Stock Data", merge_news_and_stock)

    # 4️⃣ Check merge output before sentiment
    if not os.path.exists("data/news_with_stock.csv"):
        print("❌ Merged file not found. Cannot continue.")
        sys.exit(1)

    run_step("Sentiment Cleaning", add_sentiment)

    # 5️⃣ Feature Engineering
    run_step("Feature Engineering", generate_features)

    # 6️⃣ PCA
    run_step("PCA Dimensionality Reduction", apply_pca)

    # 7️⃣ Train Models
    run_step("Model Training", train_models)

    print("\n🎉 Pipeline completed successfully!")  