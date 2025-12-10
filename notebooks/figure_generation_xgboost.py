#!/usr/bin/env python3
"""
figure_generation.py

Generates ALL non-SHAP figures for the thesis:

- Predicted vs Actual CHF
- Residuals vs Predicted
- Residuals vs each feature
- Histogram of residuals
- QQ plot of residuals
- (Optional) dataset pairplots or correlation heatmaps

This script loads:
    - chf_database.csv
    - preprocessing/imputer.pkl
    - trained_models/xgboost_final.pkl
Reconstructs the same train/val/test split as train_models.py
Computes predictions and residuals on the held-out test set
Saves all figures to /figures directory as 300 DPI JPEGs
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# CONFIG (must match train_models.py)
# -----------------------------
RANDOM_STATE = 42
FEATURE_COLUMNS = ["ks", "kf", "P", "Tsat", "Ra"]
TARGET_COLUMN = "CHF"

DATA_PATH = "data/chf_database.csv"
MODEL_DIR = "trained_models"
PREPROC_DIR = "preprocessing"
FIG_DIR = "figures"


def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)

    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(float).copy()
    return X, y


def split_data(X, y):
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=RANDOM_STATE
    )

    return X_train_full, X_tr, X_val, X_test, y_train_full, y_tr, y_val, y_test


def load_imputer_and_model():
    imputer_path = os.path.join(PREPROC_DIR, "imputer.pkl")
    model_path = os.path.join(MODEL_DIR, "xgboost_final.pkl")

    with open(imputer_path, "rb") as f:
        imputer = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return imputer, model


def generate_figures(X_test, y_test, y_pred):
    """Generate all standard non-SHAP figures."""

    residuals = y_test.values - y_pred

    # ------------------------
    # 1. Predicted vs Actual
    # ------------------------
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=y_test, y=y_pred, edgecolor='k', alpha=0.75)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', linewidth=1)
    plt.xlabel("Actual CHF (kW/m²)")
    plt.ylabel("Predicted CHF (kW/m²)")
    plt.title("Predicted vs Actual CHF")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pred_vs_actual.jpg"), dpi=300)
    plt.close()

    # ------------------------
    # 2. Residuals vs Predicted
    # ------------------------
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.7, edgecolor='k')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted CHF (kW/m²)")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residuals vs Predicted")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "residuals_vs_predicted.jpg"), dpi=300)
    plt.close()

    # ------------------------
    # 3. Residuals vs each feature
    # ------------------------
    plot_df = X_test.copy()
    plot_df["residuals"] = residuals

    for col in FEATURE_COLUMNS:
        plt.figure(figsize=(6, 4))
        plt.scatter(plot_df[col], plot_df["residuals"], alpha=0.7, edgecolor='k')
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel(col)
        plt.ylabel("Residuals")
        plt.title(f"Residuals vs {col}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        fname = f"residuals_vs_{col}.jpg"
        plt.savefig(os.path.join(FIG_DIR, fname), dpi=300)
        plt.close()

    # ------------------------
    # 4. Histogram of residuals
    # ------------------------
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, bins=20, kde=True, color='gray')
    plt.title("Residuals Histogram")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "residual_histogram.jpg"), dpi=300)
    plt.close()

    # ------------------------
    # 5. QQ plot
    # ------------------------
    plt.figure(figsize=(6, 4))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("QQ Plot of Residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "residuals_qqplot.jpg"), dpi=300)
    plt.close()

    print("All non-SHAP figures saved.")


def main():
    ensure_dirs()

    X, y = load_data()
    _, X_tr, X_val, X_test, y_train_full, y_tr, y_val, y_test = split_data(X, y)
    imputer, model = load_imputer_and_model()

    # # Impute and predict
    # X_test_imp = imputer.transform(X_test[FEATURE_COLUMNS])
    # y_pred = model.predict(X_test_imp)

        # Ensure columns are in the same order the imputer/model expect
    cols_expected = getattr(imputer, "feature_names_in_", None)
    if cols_expected is not None:
        try:
            X_to_transform = X_test.loc[:, cols_expected]
        except KeyError:
            # Expected feature names not all present in X_test; fall back to configured feature list
            X_to_transform = X_test[FEATURE_COLUMNS]
    else:
        X_to_transform = X_test[FEATURE_COLUMNS]
    
    # Impute and predict
    X_test_imp = imputer.transform(X_to_transform)
    y_pred = model.predict(X_test_imp)

    generate_figures(X_test, y_test, y_pred)


if __name__ == "__main__":
    main()
