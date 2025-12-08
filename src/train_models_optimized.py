# src/train_models_optimized.py

import os
import joblib
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb


def tune_mlp(feature_path="data/feature_engineered_dataset.csv", n_trials=50):
    """Hyperparameter tuning for MLP using Optuna."""
    df = pd.read_csv(feature_path)
    if "movement" not in df.columns:
        raise RuntimeError("❌ 'movement' column missing in dataset!")

    y = df["movement"].astype(int)
    X = df.drop(columns=["movement"]).select_dtypes(include=[np.number]).fillna(0)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, shuffle=False)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # SMOTE
    minority_count = np.sum(y_train == 1)
    k_neighbors = min(5, minority_count - 1) if minority_count > 1 else 1
    sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    def objective(trial):
        hidden1 = trial.suggest_int("hidden1", 32, 256)
        hidden2 = trial.suggest_int("hidden2", 16, 128)
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)

        mlp = MLPClassifier(
            hidden_layer_sizes=(hidden1, hidden2),
            learning_rate_init=lr,
            max_iter=500,
            random_state=42
        )
        mlp.fit(X_train_res, y_train_res)
        y_pred = mlp.predict(X_val_scaled)
        return 1 - accuracy_score(y_val, y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("✅ Best MLP params:", study.best_params)
    return study.best_params


def tune_xgb(feature_path="data/feature_engineered_dataset.csv", n_trials=50):
    """Hyperparameter tuning for XGBoost using Optuna."""
    df = pd.read_csv(feature_path)
    if "movement" not in df.columns:
        raise RuntimeError("❌ 'movement' column missing in dataset!")

    y = df["movement"].astype(int)
    X = df.drop(columns=["movement"]).select_dtypes(include=[np.number]).fillna(0)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, shuffle=False)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    def objective(trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": max(1, (y_train == 0).sum() / max(1, (y_train == 1).sum())),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "random_state": 42
        }
        model = xgb.XGBClassifier(**param)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        return 1 - accuracy_score(y_val, y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("✅ Best XGBoost params:", study.best_params)
    return study.best_params


def train_models(feature_path="data/feature_engineered_dataset.csv",
                 save_dir="models",
                 use_smote=True,
                 n_trials=50):
    """Train MLP and XGBoost models with optional hyperparameter tuning."""
    print("🔹 Loading feature-engineered dataset...")
    df = pd.read_csv(feature_path)

    if "movement" not in df.columns:
        raise RuntimeError("❌ 'movement' column missing in dataset!")

    y = df["movement"].astype(int)
    X = df.drop(columns=["movement"]).select_dtypes(include=[np.number]).fillna(0)

    # ----------------------------
    # Train/Test Split
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    # ----------------------------
    # Scale numerical features
    # ----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ----------------------------
    # SMOTE for class imbalance
    # ----------------------------
    if use_smote:
        minority_count = np.sum(y_train == 1)
        k_neighbors = min(5, minority_count - 1) if minority_count > 1 else 1
        sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_scaled, y_train = sm.fit_resample(X_train_scaled, y_train)

    # ----------------------------
    # Hyperparameter tuning
    # ----------------------------
    print("\n🔹 Tuning MLP...")
    mlp_params = tune_mlp(feature_path=feature_path, n_trials=n_trials)

    print("\n🔹 Tuning XGBoost...")
    xgb_params = tune_xgb(feature_path=feature_path, n_trials=n_trials)

    # ----------------------------
    # Train final models with optimized params
    # ----------------------------
    print("\n🔹 Training final MLP with optimized params...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(mlp_params["hidden1"], mlp_params["hidden2"]),
        learning_rate_init=mlp_params["learning_rate"],
        max_iter=800,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    y_pred_mlp = mlp.predict(X_test_scaled)
    print("\n🔹 MLP Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_mlp))
    print(classification_report(y_test, y_pred_mlp, zero_division=0))
    print("ROC-AUC:", roc_auc_score(y_test, mlp.predict_proba(X_test_scaled)[:, 1]))

    print("\n🔹 Training final XGBoost with optimized params...")
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train_scaled, y_train)
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    print("\n🔹 XGBoost Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
    print(classification_report(y_test, y_pred_xgb, zero_division=0))
    print("ROC-AUC:", roc_auc_score(y_test, xgb_model.predict_proba(X_test_scaled)[:, 1]))

    # ----------------------------
    # Save models and scaler
    # ----------------------------
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(mlp, os.path.join(save_dir, "mlp_model.pkl"))
    joblib.dump(xgb_model, os.path.join(save_dir, "xgb_model.pkl"))
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

    print("\n✅ Models saved successfully!")
    print(f"📁 MLP: {os.path.join(save_dir, 'mlp_model.pkl')}")
    print(f"📁 XGB: {os.path.join(save_dir, 'xgb_model.pkl')}")
    print(f"📁 Scaler: {os.path.join(save_dir, 'scaler.pkl')}")
    print("\n🎉 Training complete!")


if __name__ == "__main__":
    train_models(n_trials=25)  # Adjust trials as needed