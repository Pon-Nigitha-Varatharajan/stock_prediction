import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import xgboost as xgb
import joblib
import os

def train_models():
    # -------------------------------
    # 1️⃣ Load dataset
    # -------------------------------
    df = pd.read_csv("data/feature_engineered_dataset.csv")
    
    # Features & target
    y = df["movement"].astype(int)
    X = df.drop(columns=["movement", "headline", "clean_headline", "sentiment"])  # drop text columns
    
    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number]).fillna(0)

    # -------------------------------
    # 2️⃣ Train-test split (time-series aware)
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    # -------------------------------
    # 3️⃣ Oversample minority class
    # -------------------------------
    smote = SMOTETomek(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # -------------------------------
    # 4️⃣ Train MLP Classifier
    # -------------------------------
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
        early_stopping=True,
        n_iter_no_change=50,
        verbose=True
    )
    mlp.fit(X_res, y_res)

    y_pred_mlp = mlp.predict(X_test)
    print("\n🔹 MLP Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_mlp))
    print(classification_report(y_test, y_pred_mlp, zero_division=0))
    try:
        print("ROC-AUC:", roc_auc_score(y_test, mlp.predict_proba(X_test)[:, 1]))
    except:
        print("ROC-AUC: cannot compute (single class predicted)")

    # -------------------------------
    # 5️⃣ Train XGBoost Classifier
    # -------------------------------
    scale_pos_weight = max(1, (y_train == 0).sum() / max(1, (y_train == 1).sum()))
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    xgb_model.fit(
        X_res, y_res,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=50,
        verbose=50
    )

    y_pred_xgb = xgb_model.predict(X_test)
    print("\n🔹 XGBoost Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
    print(classification_report(y_test, y_pred_xgb, zero_division=0))
    print("ROC-AUC:", roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]))

    # -------------------------------
    # 6️⃣ Save models
    # -------------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(mlp, "models/mlp_model.pkl")
    joblib.dump(xgb_model, "models/xgb_model.pkl")
    print("\n✅ Models saved at 'models/mlp_model.pkl' and 'models/xgb_model.pkl'")

if __name__ == "__main__":
    train_models()