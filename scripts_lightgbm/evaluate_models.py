"""
evaluate_models.py  (COLAB VERSION, LightGBM only)

- Loads CHF data
- Recreates the same 80/20 train–test split
- Loads preprocessing/imputer.pkl
- Loads trained_models/lightgbm_final.pkl
- Evaluates on the test set and saves metrics to results/evaluation_metrics.csv
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# [EV1] Global config & paths (COLAB-SAFE)
# ============================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)

DATA_PATH         = "/content/drive/MyDrive/chf_database.csv"
ROOT              = Path(".").resolve()
PREPROCESSING_DIR = ROOT / "preprocessing"
TRAINED_MODELS_DIR = ROOT / "trained_models"
RESULTS_DIR        = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = ["ks", "kf", "P", "Tsat", "Ra"]
TARGET_COLUMN   = "CHF"


# ============================================================
# [EV2] Data loading & split
# ============================================================
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


# ============================================================
# [EV3] Load imputer & model
# ============================================================
def load_imputer():
    imp_path = PREPROCESSING_DIR / "imputer.pkl"
    if not imp_path.exists():
        raise FileNotFoundError(f"Imputer file not found at: {imp_path}")
    return joblib.load(imp_path)


def load_model(path: Path, name: str):
    if not path.exists():
        print(f"[WARN] {name} model not found, skipping: {path}")
        return None
    return joblib.load(path)


# ============================================================
# [EV4] Evaluate helper
# ============================================================
def evaluate_model(name, model, X_test_imp, y_test):
    y_pred = model.predict(X_test_imp)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    print(f"\n[RESULT] {name}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE : {mae:.3f}")
    print(f"  R²  : {r2:.4f}")

    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


# ============================================================
# [EV5] Main
# ============================================================
def main():
    print("[INFO] Loading data...")
    X, y = load_data(DATA_PATH)

    print("[INFO] Recreating 80/20 train–test split...")
    _, X_test, _, y_test = recreate_train_test_split(X, y, SEED)

    print("[INFO] Loading imputer...")
    imputer = load_imputer()

    # Match feature order used during fit
    if hasattr(imputer, "feature_names_in_"):
        cols = list(imputer.feature_names_in_)
        X_test = X_test[cols]
    else:
        X_test = X_test[FEATURE_COLUMNS]

    X_test_imp = imputer.transform(X_test)

    metrics = []

    # LightGBM only
    lgb_path = TRAINED_MODELS_DIR / "lightgbm_final.pkl"
    lgb_model = load_model(lgb_path, "LightGBM")
    if lgb_model is not None:
        metrics.append(evaluate_model("LightGBM", lgb_model, X_test_imp, y_test))

    if metrics:
        df_metrics = pd.DataFrame(metrics)
        out_path = RESULTS_DIR / "evaluation_metrics.csv"
        df_metrics.to_csv(out_path, index=False)
        print(f"\n[INFO] Saved evaluation metrics → {out_path}")
    else:
        print("[WARN] No models evaluated.")


if __name__ == "__main__":
    main()
