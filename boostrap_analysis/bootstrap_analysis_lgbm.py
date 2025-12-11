"""
Bootstrap Uncertainty Quantification for final LightGBM model

- Loads CHF dataset
- Recreates the same 80/20 train–test split (SEED=42)
- Loads preprocessing/imputer.pkl
- Loads trained_models/lightgbm_final.pkl
- Computes baseline RMSE, MAE, R² on the test set
- Performs bootstrap resampling to obtain 95% CIs for RMSE, MAE, R²
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# Global config & paths
# -----------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)

DATA_PATH          = "data/chf_database.csv"
ROOT              = Path(".").resolve()
PREPROCESSING_DIR = ROOT / "preprocessing"
TRAINED_MODELS_DIR = ROOT / "trained_models"

FEATURE_COLUMNS = ["ks", "kf", "P", "Tsat", "Ra"]
TARGET_COLUMN   = "CHF"


# -----------------------------
# Helper functions
# -----------------------------
def load_data(path):
    df = pd.read_csv(path)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(float).copy()
    return X, y


def recreate_train_test_split(X, y, seed=SEED):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )
    return X_train, X_test, y_train, y_test


def load_imputer():
    imp_path = PREPROCESSING_DIR / "imputer_lgbm.pkl"
    if not imp_path.exists():
        raise FileNotFoundError(f"Imputer not found at: {imp_path}")
    return joblib.load(imp_path)


def load_lgbm_model():
    path = TRAINED_MODELS_DIR / "lightgbm_final.pkl"
    if not path.exists():
        raise FileNotFoundError(f"LightGBM model not found at: {path}")
    return joblib.load(path)


# -----------------------------
# Baseline evaluation
# -----------------------------
print("[INFO] Loading data...")
X, y = load_data(DATA_PATH)

print("[INFO] Recreating 80/20 train–test split...")
_, X_test, _, y_test = recreate_train_test_split(X, y, SEED)

print("[INFO] Loading imputer and model...")
imputer = load_imputer()
model   = load_lgbm_model()

# Make sure feature order matches imputer/model
if hasattr(imputer, "feature_names_in_"):
    cols = list(imputer.feature_names_in_)
    X_test = X_test[cols]
else:
    X_test = X_test[FEATURE_COLUMNS]

X_test_imp = imputer.transform(X_test)

y_pred = model.predict(X_test_imp)

rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae  = float(mean_absolute_error(y_test, y_pred))
r2   = float(r2_score(y_test, y_pred))

print("\n=== Baseline Test Performance (LightGBM) ===")
print(f"RMSE: {rmse:.3f}")
print(f"MAE : {mae:.3f}")
print(f"R²  : {r2:.4f}")


# =======================================================
# Bootstrap Uncertainty Quantification (Final LightGBM)
# =======================================================
n_boot = 1000
rng = np.random.default_rng(SEED)

y_test_arr = np.asarray(y_test)
y_pred_arr = np.asarray(y_pred)

n_test = len(y_test_arr)

rmse_boot = []
mae_boot  = []
r2_boot   = []

for _ in range(n_boot):
    # sample indices with replacement
    idx = rng.integers(0, n_test, size=n_test)

    y_t = y_test_arr[idx]
    y_p = y_pred_arr[idx]

    # RMSE (no squared=False here)
    rmse_boot.append(np.sqrt(mean_squared_error(y_t, y_p)))
    mae_boot.append(mean_absolute_error(y_t, y_p))
    r2_boot.append(r2_score(y_t, y_p))

rmse_ci = np.percentile(rmse_boot, [2.5, 97.5])
mae_ci  = np.percentile(mae_boot,  [2.5, 97.5])
r2_ci   = np.percentile(r2_boot,   [2.5, 97.5])

print("\n=== Bootstrap Confidence Intervals (Final LightGBM) ===")
print(f"RMSE: {rmse:.3f} | 95% CI: [{rmse_ci[0]:.3f}, {rmse_ci[1]:.3f}]")
print(f"MAE:  {mae:.3f} | 95% CI: [{mae_ci[0]:.3f}, {mae_ci[1]:.3f}]")
print(f"R²:   {r2:.4f} | 95% CI: [{r2_ci[0]:.4f}, {r2_ci[1]:.4f}]")
