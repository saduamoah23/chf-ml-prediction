"""
train_models.py  (COLAB VERSION, USING ks/kf/P/Tsat/Ra)

End-to-end training script for the LightGBM CHF prediction model.

- Loads CHF dataset from chf_database.csv
- Uses features:  ks, kf, P, Tsat, Ra
- Target:         CHF
- 80/20 train-test split (SEED = 42)
- 15% of training set used as validation for early stopping
- Hyperparameter tuning with RandomizedSearchCV (5-fold CV)
- Saves:
    preprocessing/imputer.pkl
    trained_models/lightgbm_final.pkl
    hyperparameter_logs/lgbm_random_search.log
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from scipy.stats import randint, loguniform


# ============================================================
# [TM1] Global configuration & paths (COLAB-SAFE)
# ============================================================
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)

# In Colab, cd into repo first, e.g.:
# %cd /content/chf-ml-prediction

DATA_PATH   = "/content/drive/MyDrive/chf_database.csv"
MODEL_DIR   = Path("trained_models")
PREPROC_DIR = Path("preprocessing")
HP_LOG_DIR  = Path("hyperparameter_logs")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
PREPROC_DIR.mkdir(exist_ok=True)
HP_LOG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = ["ks", "kf", "P", "Tsat", "Ra"]
TARGET_COLUMN   = "CHF"


# ============================================================
# [TM2] Load dataset + create splits
# ============================================================
def load_dataset(path: str):
    """Load dataset and enforce feature order."""
    df = pd.read_csv(path)

    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(float).copy()
    return X, y


def make_splits(X, y):
    """80/20 train-test, then 15% of train as validation."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=SEED
    )

    return X_train, X_test, y_train, y_test, X_tr, X_val, y_tr, y_val


# ============================================================
# [TM3] Hyperparameter tuning (sklearn API)
# ============================================================
def tune_lightgbm(X_train: pd.DataFrame, y_train: pd.Series):
    """RandomizedSearchCV on a Pipeline[imputer + LGBMRegressor]."""

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LGBMRegressor(
            objective="regression",
            boosting_type="gbdt",
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )),
    ])

    param_distributions = {
        "model__learning_rate":     [0.08, 0.10, 0.12, 0.14, 0.17, 0.20, 0.25],
        "model__n_estimators":      [350, 400, 600, 800, 1000, 1200, 1400, 1600, 2000, 3000],
        "model__num_leaves":        randint(15, 80),
        "model__min_child_samples": [3, 5, 7, 10, 13, 20, 30, 40],
        "model__min_split_gain":    [0.0, 0.01, 0.05, 0.10],
        "model__feature_fraction":  [0.6, 0.8, 0.9, 1.0],
        "model__bagging_fraction":  [0.6, 0.8, 0.9, 1.0],
        "model__bagging_freq":      [0, 1],
        "model__reg_alpha":         loguniform(1e-4, 1.0),
        "model__reg_lambda":        loguniform(1e-3, 10.0),
        "model__max_bin":           [63, 127, 255, 355],
        "model__boost_from_average":[True, False],
        "model__max_depth":         [-1, 5, 7, 9, 11, 13, 15, 19],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    print("[INFO] Running LightGBM hyperparameter search...")

    gs = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=100,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        random_state=SEED,
        verbose=1,
        refit=True,
    )

    gs.fit(X_train, y_train)

    # Save search log
    log_path = HP_LOG_DIR / "lgbm_random_search.log"
    with open(log_path, "w") as f:
        f.write("BEST PARAMS:\n")
        f.write(str(gs.best_params_))
        f.write("\n\nBEST SCORE (neg RMSE):\n")
        f.write(str(gs.best_score_))

    print(f"[INFO] Saved hyperparameter log → {log_path}")
    print("Best params:", gs.best_params_)
    return gs


# ============================================================
# [TM4] Fit + save imputer
# ============================================================
def fit_save_imputer(X_train, X_tr, X_val, X_test):
    """Fit median imputer on X_train, save it, and return imputed splits."""
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)

    imp_path = PREPROC_DIR / "imputer.pkl"
    joblib.dump(imputer, imp_path)
    print(f"[INFO] Saved imputer → {imp_path}")

    X_train_imp = pd.DataFrame(imputer.transform(X_train), columns=FEATURE_COLUMNS)
    X_tr_imp    = pd.DataFrame(imputer.transform(X_tr),    columns=FEATURE_COLUMNS)
    X_val_imp   = pd.DataFrame(imputer.transform(X_val),   columns=FEATURE_COLUMNS)
    X_test_imp  = pd.DataFrame(imputer.transform(X_test),  columns=FEATURE_COLUMNS)

    return X_train_imp, X_tr_imp, X_val_imp, X_test_imp


# ============================================================
# [TM5] Native LightGBM training with early stopping
# ============================================================
def train_final_model(gs,
                      X_train_imp, X_tr_imp, X_val_imp,
                      y_train,    y_tr,    y_val):
    """
    Use best params from RandomizedSearchCV, run native lgb.train with
    early stopping on (X_tr_imp, X_val_imp), then refit a clean
    sklearn LGBMRegressor on the full imputed training set with
    n_estimators = best_iteration.
    """

    # Strip 'model__' prefix
    bp = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}

    # Avoid passing n_estimators twice
    n_estimators_from_search = int(bp.pop("n_estimators"))

    # Optional params with defaults
    max_depth      = int(bp.get("max_depth", -1))
    min_split_gain = float(bp.get("min_split_gain", 0.0))
    max_bin        = int(bp.get("max_bin", 255))
    bfa            = bool(bp.get("boost_from_average", True))

    # --- Native LightGBM for early stopping ---
    train_set = lgb.Dataset(X_tr_imp, label=y_tr, feature_name=FEATURE_COLUMNS)
    val_set   = lgb.Dataset(X_val_imp, label=y_val, feature_name=FEATURE_COLUMNS)

    params = {
        "objective":          "regression",
        "metric":             "rmse",
        "boosting_type":      "gbdt",
        "learning_rate":      float(bp["learning_rate"]),
        "num_leaves":         int(bp["num_leaves"]),
        "min_child_samples":  int(bp["min_child_samples"]),
        "feature_fraction":   float(bp["feature_fraction"]),
        "bagging_fraction":   float(bp["bagging_fraction"]),
        "bagging_freq":       int(bp["bagging_freq"]),
        "reg_alpha":          float(bp["reg_alpha"]),
        "reg_lambda":         float(bp["reg_lambda"]),
        "min_split_gain":     float(min_split_gain),
        "max_bin":            int(max_bin),
        "boost_from_average": bfa,
        "max_depth":          int(max_depth),
        "verbosity":          -1,
        "seed":               SEED,
    }

    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=5000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=120, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    best_n = booster.best_iteration or n_estimators_from_search
    print(f"[INFO] Best iteration (early stopping): {best_n}")

    # --- Final sklearn LightGBM Refit ---
    final_params = {
        "objective":          "regression",
        "boosting_type":      "gbdt",
        "random_state":       SEED,
        "n_jobs":             -1,
        "verbose":            -1,
        "learning_rate":      float(bp["learning_rate"]),
        "num_leaves":         int(bp["num_leaves"]),
        "min_child_samples":  int(bp["min_child_samples"]),
        "feature_fraction":   float(bp["feature_fraction"]),
        "bagging_fraction":   float(bp["bagging_fraction"]),
        "bagging_freq":       int(bp["bagging_freq"]),
        "reg_alpha":          float(bp["reg_alpha"]),
        "reg_lambda":         float(bp["reg_lambda"]),
        "min_split_gain":     float(min_split_gain),
        "max_bin":            int(max_bin),
        "boost_from_average": bfa,
        "max_depth":          int(max_depth),
        "n_estimators":       int(best_n),
    }

    model = LGBMRegressor(**final_params)
    model.fit(X_train_imp, y_train)

    out_path = MODEL_DIR / "lightgbm_final.pkl"
    joblib.dump(model, out_path)
    print(f"[INFO] Saved final LightGBM model → {out_path}")

    return model


# ============================================================
# [TM6] MAIN EXECUTION
# ============================================================
def main():
    print("[INFO] Loading dataset...")
    X, y = load_dataset(DATA_PATH)

    print("[INFO] Making train/val/test splits...")
    X_train, X_test, y_train, y_test, X_tr, X_val, y_tr, y_val = make_splits(X, y)

    print("[INFO] Hyperparameter tuning...")
    gs = tune_lightgbm(X_train, y_train)

    print("[INFO] Fitting + saving imputer...")
    X_train_imp, X_tr_imp, X_val_imp, X_test_imp = fit_save_imputer(
        X_train, X_tr, X_val, X_test
    )

    print("[INFO] Training final LightGBM model with early stopping...")
    model = train_final_model(
        gs,
        X_train_imp, X_tr_imp, X_val_imp,
        y_train,    y_tr,    y_val,
    )

    # Quick test-set evaluation
    print("[INFO] Evaluating on hold-out test set...")
    y_pred = model.predict(X_test_imp)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    print(f"[RESULT] LightGBM Test Performance:")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE : {mae:.3f}")
    print(f"  R²  : {r2:.4f}")
    print("[INFO] Training completed.")


if __name__ == "__main__":
    main()
