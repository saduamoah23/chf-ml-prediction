#!/usr/bin/env python3
"""
train_models.py

End-to-end training script for the XGBoost CHF model.

- Loads CHF dataset from chf_database.csv
- Splits into train/val/test
- Runs RandomizedSearchCV with a pipeline (SimpleImputer + XGBRegressor)
- Uses early stopping with xgb.train to determine optimal n_estimators
- Refits final XGBRegressor on imputed training data
- Evaluates on the held-out test set
- Saves:
    - preprocessing/imputer.pkl
    - trained_models/xgboost_final.pkl
    - logs/xgb_random_search_results.csv
"""

import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# -----------------------------
# Global config
# -----------------------------
RANDOM_STATE = 42

# For the GitHub dataset:
# chf_database.csv must have columns:
#   ks, kf, P, Tsat, Ra, CHF, study_id
FEATURE_COLUMNS = ["ks", "P", "Tsat", "Ra", "kf"]
TARGET_COLUMN = "CHF"

DATA_PATH = "/content/drive/MyDrive/chf_database.csv"

#DATA_PATH = "data/chf_database.csv"
MODEL_DIR = "trained_models"
PREPROC_DIR = "preprocessing"
LOG_DIR = "logs"


def ensure_dirs() -> None:
    """Create required output directories if they don't exist."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PREPROC_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def load_data(path: str = DATA_PATH):
    """Load CHF dataset and return (X, y) as DataFrames/Series."""
    df = pd.read_csv(path)

    # Basic sanity check
    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {path}: {missing_cols}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(float).copy()

    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_STATE):
    """
    Two-level split:
      - 80% train (train+val pool), 20% test
      - from the 80% train pool, 15% → validation (≈ 12% overall)
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


def run_xgb_random_search(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = RANDOM_STATE):
    """
    RandomizedSearchCV over XGBRegressor with a SimpleImputer inside a Pipeline.
    Returns:
      - gs: fitted RandomizedSearchCV object
    """
    base_xgb = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", base_xgb),
    ])

    param_distributions = {
        "model__learning_rate":    [0.04, 0.08, 0.10, 0.12, 0.14, 0.17, 0.20],
        "model__n_estimators":     [300, 450, 600, 650, 700, 800, 1000, 1200, 1400, 1600, 2000],
        "model__max_depth":        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        "model__min_child_weight": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        "model__colsample_bytree": [0.8, 0.9, 1.0],
        "model__subsample":        [0.8, 0.9, 1.0],
        "model__reg_alpha":        [0.0, 0.2, 0.5, 0.6, 1.0],
        "model__reg_lambda":       [0.0, 0.1, 0.5, 0.6, 1.0],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    gs = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=60,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
        refit=True,
    )

    print("\n=== Running XGBoost RandomizedSearchCV ===")
    gs.fit(X_train, y_train)

    print("\nBest params from RandomizedSearchCV:")
    print(gs.best_params_)
    print(f"Best CV score (neg RMSE): {gs.best_score_:.4f}")

    # Save raw CV results as a log for reproducibility
    cvres = gs.cv_results_
    results_df = pd.DataFrame({
        "mean_test_score": cvres["mean_test_score"],
        "std_test_score":  cvres["std_test_score"],
        "param_model__learning_rate":    cvres["param_model__learning_rate"].data,
        "param_model__n_estimators":     cvres["param_model__n_estimators"].data,
        "param_model__max_depth":        cvres["param_model__max_depth"].data,
        "param_model__min_child_weight": cvres["param_model__min_child_weight"].data,
        "param_model__colsample_bytree": cvres["param_model__colsample_bytree"].data,
        "param_model__subsample":        cvres["param_model__subsample"].data,
        "param_model__reg_alpha":        cvres["param_model__reg_alpha"].data,
        "param_model__reg_lambda":       cvres["param_model__reg_lambda"].data,
    })
    results_df.to_csv(os.path.join(LOG_DIR, "xgb_random_search_results.csv"), index=False)
    print(f"\nSaved RandomizedSearchCV results to {os.path.join(LOG_DIR, 'xgb_random_search_results.csv')}")

    return gs


def train_final_xgb_with_es(
    gs: RandomizedSearchCV,
    X_train_full: pd.DataFrame,
    X_tr: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_full: pd.Series,
    y_tr: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
):
    """
    Use best hyperparameters from RandomizedSearchCV:
      1) Fit a median imputer on X_train_full.
      2) Use xgb.train with early stopping (train vs val) to get best num_boost_round.
      3) Refit an XGBRegressor with those hyperparameters and best_n on all training data.
      4) Evaluate on the held-out test set.
      5) Save imputer and final model to disk.
    """

    # -------------------------------------------------
    # 1) Extract best params (strip "model__")
    # -------------------------------------------------
    bp = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}

    # Sanity check on required keys
    needed = ["learning_rate", "max_depth", "min_child_weight",
              "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]
    missing = [k for k in needed if k not in bp]
    if missing:
        raise RuntimeError(f"Missing keys in best_params_: {missing}. best_params_ = {gs.best_params_}")

    # -------------------------------------------------
    # 2) Fit imputer on full training pool
    # -------------------------------------------------
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train_full[FEATURE_COLUMNS])

    # Transform subsets consistently
    X_tr_imp   = imputer.transform(X_tr[FEATURE_COLUMNS])
    X_val_imp  = imputer.transform(X_val[FEATURE_COLUMNS])
    X_test_imp = imputer.transform(X_test[FEATURE_COLUMNS])
    X_train_full_imp = imputer.transform(X_train_full[FEATURE_COLUMNS])

    # -------------------------------------------------
    # 3) Early stopping with xgb.train
    # -------------------------------------------------
    params = {
        "objective":        "reg:squarederror",
        "eval_metric":      "rmse",
        "tree_method":      "hist",
        "eta":              bp["learning_rate"],
        "max_depth":        int(bp["max_depth"]),
        "min_child_weight": bp["min_child_weight"],
        "subsample":        bp["subsample"],
        "colsample_bytree": bp["colsample_bytree"],
        "alpha":            bp["reg_alpha"],
        "lambda":           bp["reg_lambda"],
        "seed":             RANDOM_STATE,
    }

    dtrain = xgb.DMatrix(X_tr_imp,  label=y_tr.values,  feature_names=FEATURE_COLUMNS)
    dvalid = xgb.DMatrix(X_val_imp, label=y_val.values, feature_names=FEATURE_COLUMNS)

    print("\n=== Running xgb.train with early stopping ===")
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=120,
        verbose_eval=False,
    )

    best_n = booster.best_iteration + 1
    print(f"Best iteration from early stopping: {best_n}")

    # -------------------------------------------------
    # 4) Refit sklearn XGBRegressor on all training data
    # -------------------------------------------------
    xgb_final = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        learning_rate=bp["learning_rate"],
        max_depth=int(bp["max_depth"]),
        min_child_weight=bp["min_child_weight"],
        subsample=bp["subsample"],
        colsample_bytree=bp["colsample_bytree"],
        reg_alpha=bp["reg_alpha"],
        reg_lambda=bp["reg_lambda"],
        n_estimators=best_n,
    )

    xgb_final.fit(X_train_full_imp, y_train_full.values)

        # -------------------------------------------------
    # 5) Evaluate on the untouched test set
    # -------------------------------------------------
    y_pred_test = xgb_final.predict(X_test_imp)

    # Some sklearn versions (or shadowed imports) do not accept 'squared' kwarg.
    # Compute RMSE manually for maximum compatibility.
    mse  = mean_squared_error(y_test, y_pred_test)  # default squared=True
    rmse = float(np.sqrt(mse))
    mae  = float(mean_absolute_error(y_test, y_pred_test))
    r2   = float(r2_score(y_test, y_pred_test))

    print("\n=== Final XGBoost performance on held-out TEST ===")
    print(f"RMSE: {rmse:.3f} kW/m²")
    print(f"MAE : {mae:.3f} kW/m²")
    print(f"R²  : {r2:.4f}")


    # -------------------------------------------------
    # 6) Save imputer and model
    # -------------------------------------------------
    imputer_path = os.path.join(PREPROC_DIR, "imputer.pkl")
    model_path   = os.path.join(MODEL_DIR, "xgboost_final.pkl")

    with open(imputer_path, "wb") as f:
        pickle.dump(imputer, f)

    with open(model_path, "wb") as f:
        pickle.dump(xgb_final, f)

    print(f"\nSaved median imputer to: {imputer_path}")
    print(f"Saved final XGBoost model to: {model_path}")

    return xgb_final, imputer, (rmse, mae, r2)


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

    # 1) Hyperparameter search (with Pipeline-imputer)
    gs = run_xgb_random_search(X_train_full, y_train_full, random_state=RANDOM_STATE)

    # 2) Early-stopped final model + save imputer/model
    train_final_xgb_with_es(
        gs=gs,
        X_train_full=X_train_full,
        X_tr=X_tr,
        X_val=X_val,
        X_test=X_test,
        y_train_full=y_train_full,
        y_tr=y_tr,
        y_val=y_val,
        y_test=y_test,
    )


if __name__ == "__main__":
    main()
