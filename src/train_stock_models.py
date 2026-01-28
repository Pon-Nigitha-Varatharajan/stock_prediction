import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb


def train_stock_models(
    feature_csv="data/stock_features.csv",
    model_dir="models",
):
    if not os.path.exists(feature_csv):
        raise FileNotFoundError(feature_csv)

    df = pd.read_csv(feature_csv)
    df.columns = [c.lower() for c in df.columns]

    if "movement" not in df.columns:
        raise KeyError("'movement' missing in stock features CSV")

    # Use only numeric columns for training
    X = df.select_dtypes(include=["number"]).drop(columns=["movement"], errors="ignore")
    y = df["movement"].astype(int)

    print("📊 Training label distribution:")
    print(y.value_counts())

    # Stratify only if possible
    min_class = int(y.value_counts().min())
    stratify = y if min_class >= 2 else None
    if stratify is None:
        print("⚠️ Not enough samples per class for stratify — using normal split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=True, stratify=stratify, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ----------------------------
    # MLP
    # ----------------------------
    print("\n🔹 Training MLP...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        learning_rate_init=1e-3,
        max_iter=600,
        random_state=42,
    )
    mlp.fit(X_train_scaled, y_train)

    y_pred_mlp = mlp.predict(X_test_scaled)
    print("MLP Accuracy:", accuracy_score(y_test, y_pred_mlp))
    print(classification_report(y_test, y_pred_mlp, zero_division=0))
    try:
        print("MLP ROC-AUC:", roc_auc_score(y_test, mlp.predict_proba(X_test_scaled)[:, 1]))
    except Exception:
        print("⚠️ MLP ROC-AUC not available")

    # ----------------------------
    # XGBoost (with imbalance support)
    # ----------------------------
    print("\n🔹 Training XGBoost...")

    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        raise ValueError(f"Only one class in y_train: {unique_classes}. Increase days_back or lower threshold.")

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(1, pos)

    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        base_score=0.5,
    )
    xgb_model.fit(X_train_scaled, y_train)

    y_pred_xgb = xgb_model.predict(X_test_scaled)
    print("XGB Accuracy:", accuracy_score(y_test, y_pred_xgb))
    print(classification_report(y_test, y_pred_xgb, zero_division=0))
    try:
        print("XGB ROC-AUC:", roc_auc_score(y_test, xgb_model.predict_proba(X_test_scaled)[:, 1]))
    except Exception:
        print("⚠️ XGB ROC-AUC not available")

    # Save
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(mlp, os.path.join(model_dir, "mlp_stock.pkl"))
    joblib.dump(xgb_model, os.path.join(model_dir, "xgb_stock.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler_stock.pkl"))

    print("\n✅ Stock models saved to:", model_dir)
    return mlp, xgb_model, scaler


if __name__ == "__main__":
    train_stock_models()