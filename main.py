# main.py

import os
import sys

# -------------------------------------------------------
# Utility to run steps safely
# -------------------------------------------------------
def run_step(name, func):
    print(f"\n🔹 Starting: {name} ...")
    try:
        func()
        print(f"✅ Finished: {name}")
    except Exception as e:
        print(f"❌ ERROR in {name}: {e}")
        sys.exit(1)

# -------------------------------------------------------
# Make sure src/ is in Python path (IMPORTANT)
# -------------------------------------------------------
sys.path.append(os.path.abspath("src"))

# -------------------------------------------------------
# IMPORT FUNCTIONS (STOCK-ONLY TRAINING PIPELINE)
# -------------------------------------------------------
from fetch_stocks import fetch_stocks
from fetch_news import fetch_news          # used later in Streamlit (not training)
from stock_feature_engineering import stock_feature_engineering
from train_stock_models import train_stock_models

# -------------------------------------------------------
# CHECK REQUIRED FOLDERS
# -------------------------------------------------------
REQUIRED_FOLDERS = ["data", "models", "src"]

for folder in REQUIRED_FOLDERS:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"📁 Created missing folder: {folder}")

# -------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------
if __name__ == "__main__":

    print("\n🚀 Running Stock–News Prediction Pipeline")
    print("----------------------------------------")

    # 1️⃣ Fetch historical stock prices (training data)
    run_step("Fetch Stock Prices", fetch_stocks)

    # 2️⃣ Fetch news (used only for Streamlit sentiment overlay)
    run_step("Fetch News", fetch_news)

    # 3️⃣ Stock-only feature engineering + labels
    run_step("Stock Feature Engineering", stock_feature_engineering)

    # 4️⃣ Train models (MLP + XGBoost)
    run_step("Train Stock Models", train_stock_models)

    print("\n🎉 Pipeline completed successfully!")