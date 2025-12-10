#!/usr/bin/env python3
"""
evaluate_models.py

Evaluate saved CHF prediction models on a reproducible test split.

- Loads CHF dataset from chf_database.csv
- Reconstructs the same train/val/test split as in train_models.py
- Loads:
    - preprocessing/imputer.pkl
    - trained_models/xgboost_final.pkl
- Applies the imputer to the test features
- Computes RMSE, MAE, and R² on the held-out test set
- Writes metrics to logs/test_metrics.csv
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# Global config (must match train_models.py)
# -----------------------------
RANDOM_STATE = 42

FEATURE_COLUMNS = ["ks", "P", "Tsat", "Ra", "kf"]
TARGET_COLUMN = "CHF"


DATA_PATH = "data/chf_database.csv"
MODEL_DIR = "trained_models"
PREPROC_DIR = "preprocessing"
LOG_DIR = "logs"


def ensure_dirs() -> None:
    """Ensure that the logs directory exists."""
    os.makedirs(LOG_DIR, exist_ok=True)


def load_data(path: str = DATA_PATH):
    """Load CHF dataset and return (X, y)."""
    df = pd.read_csv(path)

    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {path}: {missing_cols}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(float).copy()

    return X, y


def split_data_for_test(X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_STATE):
    """
    Replicate the same splitting logic used in train_models.py:

      - 80% train (train+val pool), 20% test
      - From the 80% train pool, 15% → validation (≈ 12% overall)

    For evaluation we only need X_test, y_test, but we recreate the full split
    to guarantee the test set is identical.
    """
    from sklearn.model_selection import train_test_split

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
    )

    # The train/val split is not used downstream here, but we keep it
    # to mirror train_models.py exactly.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.15,
        random_state=random_state,
    )

    return X_train_full, X_tr, X_val, X_test, y_train_full, y_tr, y_val, y_test


def load_imputer_and_model():
    """Load the saved imputer and XGBoost model from disk."""
    imputer_path = os.path.join(PREPROC_DIR, "imputer.pkl")
    model_path = os.path.join(MODEL_DIR, "xgboost_final.pkl")

    if not os.path.exists(imputer_path):
        raise FileNotFoundError(f"Imputer file not found: {imputer_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(imputer_path, "rb") as f:
        imputer = pickle.load(f)

    with open(model_path, "rb") as f:
        xgb_model = pickle.load(f)

    return imputer, xgb_model


def evaluate_model(name: str,
                   model,
                   imputer,
                   X_test: pd.DataFrame,
                   y_test: pd.Series):
    """
    Apply the given imputer to X_test, predict with model, and compute
    RMSE, MAE, and R². Returns a (rmse, mae, r2) tuple.
    """
    # Transform test features using the fitted imputer
    X_test_imp = imputer.transform(X_test[FEATURE_COLUMNS])

    # Predict
    y_pred = model.predict(X_test_imp)

    # Compute metrics (RMSE computed explicitly for compatibility)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    print(f"\n=== {name} performance on held-out TEST ===")
    print(f"RMSE: {rmse:.3f} kW/m²")
    print(f"MAE : {mae:.3f} kW/m²")
    print(f"R²  : {r2:.4f}")

    return rmse, mae, r2


def main():
    ensure_dirs()

    print(f"Loading data from: {DATA_PATH}")
    X, y = load_data(DATA_PATH)

    (
        X_train_full,
        X_tr,
        X_val,
        X_test,
        y_train_full,
        y_tr,
        y_val,
        y_test,
    ) = split_data_for_test(X, y, random_state=RANDOM_STATE)

    # Load artifacts
    imputer, xgb_model = load_imputer_and_model()

    # Evaluate XGBoost
    rmse, mae, r2 = evaluate_model(
        name="XGBoost (final)",
        model=xgb_model,
        imputer=imputer,
        X_test=X_test,
        y_test=y_test,
    )

    # Save metrics to CSV for reproducibility
    metrics_path = os.path.join(LOG_DIR, "test_metrics_xgb.csv")
    metrics_df = pd.DataFrame(
        [{
            "model": "xgboost_final",
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }]
    )
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved test metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
