# ============================================================
# [SHAP1] Imports & global configuration
# ============================================================
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

# Reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)

# In Colab, you should first %cd into your repo root:
# %cd /content/chf-ml-prediction
ROOT = Path(".").resolve()

# Path to your CSV in Google Drive
DATA_PATH          = "/content/drive/MyDrive/chf_database.csv"
PREPROCESSING_DIR  = ROOT / "preprocessing"
TRAINED_MODELS_DIR = ROOT / "trained_models"
FIG_DIR            = ROOT / "figures"

FIG_DIR.mkdir(exist_ok=True)

# Must match your chf_database.csv header names
FEATURE_COLUMNS = ["ks", "kf", "P", "Tsat", "Ra"]
TARGET_COLUMN   = "CHF"


# ============================================================
# [SHAP2] Data loading & train/test split
# ============================================================
def load_data(path: str):
    """Load CHF dataset and return feature matrix X and target y."""
    df = pd.read_csv(path)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(float).copy()
    return X, y


def recreate_train_test_split(X, y, seed: int = SEED):
    """
    Recreate the same 80/20 split used in train_models.py:

        train_test_split(X, y, test_size=0.20, random_state=SEED)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
    )
    return X_train, X_test, y_train, y_test


# ============================================================
# [SHAP3] Load imputer & trained LightGBM model
# ============================================================
def load_imputer():
    """Load the fitted median SimpleImputer from preprocessing/imputer.pkl."""
    imp_path = PREPROCESSING_DIR / "imputer.pkl"
    if not imp_path.exists():
        raise FileNotFoundError(f"Imputer file not found at: {imp_path}")
    imputer = joblib.load(imp_path)
    return imputer


def load_lightgbm_model():
    """Load the trained LightGBM model from trained_models/lightgbm_final.pkl."""
    model_path = TRAINED_MODELS_DIR / "lightgbm_final.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"LightGBM model file not found at: {model_path}")
    model = joblib.load(model_path)
    if not isinstance(model, LGBMRegressor):
        print("[WARN] Loaded object is not an LGBMRegressor; SHAP may still work but check compatibility.")
    return model


# ============================================================
# [SHAP4] SHAP plots (bar, beeswarm, dependence)
# ============================================================
def generate_shap_plots(model, X_test_imp_df: pd.DataFrame, feature_cols):
    """
    Compute SHAP values and produce:
    - SHAP global importance bar plot
    - SHAP beeswarm plot
    - SHAP dependence plots for top 3 features
    """
    print("[INFO] Computing SHAP values (this may take a moment)...")

    # TreeExplainer is optimized for tree-based models (like LightGBM)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_imp_df)

    # Handle possible list output (older SHAP versions for regression sometimes return [array])
    if isinstance(shap_values, list):
        sv = np.array(shap_values[0])
    else:
        sv = np.array(shap_values)

    print(f"[INFO] SHAP values shape: {sv.shape}")

    # -----------------------------
    # (1) SHAP Global Importance — Bar
    # -----------------------------
    plt.figure(figsize=(7, 4))
    shap.summary_plot(
        sv,
        X_test_imp_df,
        plot_type="bar",
        show=False,
    )
    plt.title("SHAP Global Importance (LightGBM)")
    plt.tight_layout()

    out_bar = FIG_DIR / "SHAP_Global_Importance_bar_LightGBM.jpg"
    plt.savefig(
        out_bar,
        format="jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"[SAVED] {out_bar}")

    # -----------------------------
    # (2) SHAP Beeswarm — Feature Impact Distribution
    # -----------------------------
    plt.figure(figsize=(7, 4))
    shap.summary_plot(
        sv,
        X_test_imp_df,
        show=False,
    )
    plt.title("SHAP Beeswarm (feature impact distribution)")
    plt.tight_layout()

    out_bee = FIG_DIR / "SHAP_Beeswarm_LightGBM.jpg"
    plt.savefig(
        out_bee,
        format="jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"[SAVED] {out_bee}")

    # -----------------------------
    # (3) SHAP Dependence Plots — Top 3 Features
    # -----------------------------
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(-mean_abs)[:3]
    top_feats = [feature_cols[i] for i in top_idx]

    print("[INFO] Top 3 SHAP features:", top_feats)

    for f in top_feats:
        plt.figure(figsize=(6, 4))
        shap.dependence_plot(
            f,
            sv,
            X_test_imp_df,
            show=False,
        )
        plt.title(f"SHAP Dependence: {f}")
        plt.tight_layout()

        safe_name = f.replace("/", "_")
        out_dep = FIG_DIR / f"SHAP_Dependence_{safe_name}.jpg"
        plt.savefig(
            out_dep,
            format="jpg",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"[SAVED] {out_dep}")


# ============================================================
# [SHAP5] Main SHAP workflow
# ============================================================
def main():
    print("[INFO] Loading dataset...")
    X, y = load_data(DATA_PATH)

    print("[INFO] Recreating 80/20 train–test split...")
    X_train, X_test, y_train, y_test = recreate_train_test_split(X, y, SEED)

    print("[INFO] Loading imputer and LightGBM model...")
    imputer = load_imputer()
    model   = load_lightgbm_model()

    # Ensure column order matches what the imputer/model expect
    if hasattr(imputer, "feature_names_in_"):
        feature_cols = list(imputer.feature_names_in_)
        X_test = X_test[feature_cols]
    else:
        feature_cols = FEATURE_COLUMNS
        X_test = X_test[feature_cols]

    # Apply the same imputation used during training
    X_test_imp = imputer.transform(X_test)
    X_test_imp_df = pd.DataFrame(X_test_imp, columns=feature_cols)

    # At this point, you *could* compute metrics, but we keep this script
    # strictly focused on SHAP interpretability as requested.

    print("[INFO] Generating SHAP interpretability plots...")
    generate_shap_plots(model, X_test_imp_df, feature_cols)

    print("[INFO] SHAP analysis completed. All SHAP figures saved in 'figures/'.")


if __name__ == "__main__":
    main()
