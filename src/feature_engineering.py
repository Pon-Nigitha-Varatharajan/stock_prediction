import pandas as pd
import numpy as np

def feature_engineering():
    df = pd.read_csv("data/processed_dataset.csv")

    def compute_row(row):
        sym = row["symbol"]
        try:
            op = row[f"Open_{sym}"]
            cl = row[f"Close_{sym}"]
            hi = row[f"High_{sym}"]
            lo = row[f"Low_{sym}"]

            return pd.Series([
                (cl - op) / op,
                (hi - lo) / op,
                np.std([op, cl, hi, lo])
            ])
        except:
            return pd.Series([np.nan, np.nan, np.nan])

    df[["daily_return","price_range","volatility"]] = df.apply(compute_row, axis=1)

    df.to_csv("data/feature_engineered_dataset.csv", index=False)
    print("Feature engineering done:", df.shape)