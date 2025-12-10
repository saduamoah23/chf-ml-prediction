#!/usr/bin/env python3
"""
shap_analysis.py

SHAP-based interpretability for the final XGBoost CHF model.

- Loads CHF dataset from chf_database.csv
- Reconstructs the same train/val/test split as in train_models.py
- Loads:
    - preprocessing/imputer.pkl
    - trained_models/xgboost_final.pkl
- Applies the imputer to the test features
- Computes SHAP values on the test set
- Generates and saves:
    - Global SHAP bar plot
    - SHAP beeswarm summary plot
    - SHAP dependence plots for the top 3 features
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# Global config (must match train_models.py)
# -----------------------------
RANDOM_STATE = 42

# IMPORTANT: these must match chf_database.csv column names
FEATURE_COLUMNS = ["ks", "P", "Tsat", "Ra", "kf"]
TARGET_COLUMN = "CHF"

DATA_PATH = "/content/drive/MyDrive/chf_database.csv"

#DATA_PATH = "chf_database.csv"
MODEL_DIR = "trained_models"
PREPROC_DIR = "preprocessing"
FIG_DIR = "figures"


def ensure_dirs() -> None:
    """Ensure the figures directory exists."""
    os.makedirs(FIG_DIR, exist_ok=True)


def load_data(path: str = DATA_PATH):
    """Load CHF dataset and return (X, y)."""
    df = pd.read_csv(path)

    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {path}: {missing_cols}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(float).copy()
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_STATE):
    """
    Replicate the data split from train_models.py:

      - 80% train (train+val pool), 20% test
      - From the 80% train pool, 15% → validation

    For SHAP we mainly use X_test, y_test.
    """
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
    )

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


def compute_test_metrics(model, imputer, X_test: pd.DataFrame, y_test: pd.Series):
    """Compute RMSE, MAE, R² on the held-out test set and return (X_test_imp, y_pred, metrics)."""
    X_test_imp = imputer.transform(X_test[FEATURE_COLUMNS])
    y_pred = model.predict(X_test_imp)

    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    print("MODEL PERFORMANCE SUMMARY (for SHAP report)")
    print(f"  RMSE : {rmse:.3f} kW/m²")
    print(f"  MAE  : {mae:.3f} kW/m²")
    print(f"  R²   : {r2:.4f}")
    print("-" * 60)

    return X_test_imp, y_pred, (rmse, mae, r2)


def run_shap_analysis(model, X_test_imp: np.ndarray):
    """Run SHAP analysis and save bar, beeswarm, and top-3 dependence plots."""
    # 0) Ensure SHAP is available
    try:
        import shap
    except ImportError:
        raise SystemExit("Please install shap first: pip install shap")

    # 1) Build explainer (TreeExplainer for XGBoost if possible)
    try:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    except Exception:
        # Fallback to generic Explainer if TreeExplainer is unavailable
        explainer = shap.Explainer(model, X_test_imp)

    # 2) Compute SHAP values
    shap_vals = explainer(X_test_imp)

    # Handle SHAP versions that return Explanation vs raw array/list
    if hasattr(shap_vals, "values"):
        sv = shap_vals.values
    else:
        sv = shap_vals
    if isinstance(sv, list):  # e.g., multiclass; for regression usually single array
        sv = sv[0]

    # ------------------------------------------------------
    # 5) Global importance — SHAP bar chart
    # ------------------------------------------------------
    shap.summary_plot(
        sv,
        X_test_imp,
        feature_names=FEATURE_COLUMNS,
        plot_type="bar",
        show=False,
    )
    plt.gcf().set_size_inches(7, 4)
    plt.title("SHAP Global Importance (Bar)")
    plt.tight_layout()

    bar_path = os.path.join(FIG_DIR, "shap_global_importance_bar_xgb.jpg")
    plt.savefig(bar_path, format="jpg", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {bar_path}")

    # ------------------------------------------------------
    # 6) Global distribution — SHAP beeswarm
    # ------------------------------------------------------
    shap.summary_plot(
        sv,
        X_test_imp,
        feature_names=FEATURE_COLUMNS,
        show=False,
    )
    plt.gcf().set_size_inches(7, 4)
    plt.title("SHAP Beeswarm (feature impact distribution)")
    plt.tight_layout()

    beeswarm_path = os.path.join(FIG_DIR, "shap_beeswarm_xgb.jpg")
    plt.savefig(beeswarm_path, format="jpg", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {beeswarm_path}")

    # ------------------------------------------------------
    # 7) SHAP Dependence Plots for Top 3 Features
    # ------------------------------------------------------
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(-mean_abs)[:3]
    top_feats = [FEATURE_COLUMNS[i] for i in top_idx]

    print("Top SHAP features:", top_feats)

    for f in top_feats:
        plt.figure(figsize=(6, 4))
        shap.dependence_plot(
            f,
            sv,
            X_test_imp,
            feature_names=FEATURE_COLUMNS,
            show=False,
        )
        plt.title(f"SHAP Dependence: {f}")
        plt.tight_layout()

        # Safe filename: replace any problematic characters
        fname = f"SHAP_Dependence_{f}.jpg".replace("/", "_").replace(" ", "_")
        out_path = os.path.join(FIG_DIR, fname)

        plt.savefig(out_path, format="jpg", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {out_path}")


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
    ) = split_data(X, y, random_state=RANDOM_STATE)

    imputer, xgb_model = load_imputer_and_model()

    # Compute metrics and get imputed test matrix
    X_test_imp, y_pred, metrics = compute_test_metrics(
        model=xgb_model,
        imputer=imputer,
        X_test=X_test,
        y_test=y_test,
    )

    # Run SHAP and generate plots
    run_shap_analysis(xgb_model, X_test_imp)


if __name__ == "__main__":
    main()
