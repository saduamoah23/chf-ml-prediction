# ============================================================
# NOTEBOOK: figure_generation.ipynb
# Reproduces ALL non-SHAP Chapter 4 Figures
# ============================================================

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

# -----------------------------------
# Configuration
# -----------------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)

DATA_PATH          = "/content/drive/MyDrive/chf_database.csv"
ROOT               = Path(".")
PREPROCESSING_DIR  = ROOT / "preprocessing"
TRAINED_MODELS_DIR = ROOT / "trained_models"
FIG_DIR            = ROOT / "figures"

FIG_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = ["ks", "kf", "P", "Tsat", "Ra"]
TARGET_COLUMN   = "CHF"

sns.set(style="whitegrid")

# -----------------------------------
# LOAD DATA + SPLIT
# -----------------------------------
df = pd.read_csv(DATA_PATH)
X  = df[FEATURE_COLUMNS].copy()
y  = df[TARGET_COLUMN].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED
)

# -----------------------------------
# LOAD IMPUTER + MODEL
# -----------------------------------
imputer = joblib.load(PREPROCESSING_DIR / "imputer.pkl")
model   = joblib.load(TRAINED_MODELS_DIR / "lightgbm_final.pkl")

# Match feature order
if hasattr(imputer, "feature_names_in_"):
    feature_cols = list(imputer.feature_names_in_)
else:
    feature_cols = FEATURE_COLUMNS

X_test = X_test[feature_cols]
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)

# -----------------------------------
# PREDICTION + METRICS
# -----------------------------------
y_pred = model.predict(X_test_imp)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("Evaluation Metrics:")
print(f"  RMSE : {rmse:.3f}")
print(f"  MAE  : {mae:.3f}")
print(f"  R²   : {r2:.4f}")

# ============================================================
# FIGURE 1 — Predicted vs Actual CHF
# ============================================================
plt.figure(figsize=(5, 5))
plt.scatter(
    y_test, y_pred,
    color="darkorange",
    alpha=0.7, edgecolor="k", linewidth=0.5,
)

mn = min(y_test.min(), y_pred.min())
mx = max(y_test.max(), y_pred.max())
plt.plot([mn, mx], [mn, mx], "r--", lw=1)

plt.xlabel("Actual qCHF (kW/m²)")
plt.ylabel("Predicted qCHF (kW/m²)")
plt.title("Predicted vs Actual CHF")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(FIG_DIR / "Predicted_vs_Actual_CHF.jpg", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# FIGURE 2 — Residuals vs Predicted
# ============================================================
residuals = y_test - y_pred

plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(0.0, color="red", linestyle="--", linewidth=1)

plt.xlabel("Predicted q''_CHF (kW/m²)")
plt.ylabel("Residuals (Actual − Predicted)")
plt.title("Residuals vs Predicted CHF")
plt.tight_layout()

plt.savefig(FIG_DIR / "Residuals_vs_Predicted_CHF.jpg", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# FIGURE 3 — Residuals-by-feature grid
# ============================================================
error = (y_test - y_pred) / y_test  # relative error

plot_df = X_test_imp.copy()
plot_df["error"]     = error.values
plot_df["Predicted"] = y_pred
plot_df["Actual"]    = y_test.values

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    have_lowess = True
except:
    have_lowess = False

cols = 3
n = len(feature_cols)
rows = int(np.ceil(n / cols))

fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3.8*rows), squeeze=False)
axes = axes.ravel()

for i, f in enumerate(feature_cols):
    ax = axes[i]
    x = plot_df[f].values
    y = plot_df["error"].values

    ax.scatter(x, y, s=18, alpha=0.6)
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1)

    ax.set_title(f"error vs {f}")
    ax.set_xlabel(f)
    ax.set_ylabel("(Actual − Predicted)/Actual")

    if have_lowess and np.unique(x).size > 10:
        lo = lowess(y, x, frac=0.6, return_sorted=True)
        ax.plot(lo[:, 0], lo[:, 1], color="black", lw=2)

for j in range(i+1, rows*cols):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(FIG_DIR / "Residuals_by_Feature_Grid.jpg", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# FIGURE 4 — Residual Histogram + Q-Q Plot
# ============================================================
err = error.values

fig, ax = plt.subplots(1, 2, figsize=(10,4))

# Histogram
ax[0].hist(err, bins=20, alpha=0.8)
ax[0].set_title("Residual Error Histogram")
ax[0].set_xlabel("(Actual − Predicted)/Actual")

# Q-Q
try:
    import scipy.stats as st
    (osm, osr), (slope, intercept, _) = st.probplot(err, dist="norm", plot=None)
    ax[1].scatter(osm, osr, s=12, alpha=0.7)
    ax[1].plot(osm, slope*osm + intercept, color="red", lw=1)
    ax[1].set_title("Residual Error Q–Q Plot")
except:
    ax[1].text(0.5,0.5,"scipy not available",ha="center",va="center")
    ax[1].set_axis_off()

plt.tight_layout()
plt.savefig(FIG_DIR / "Residual_Error_Hist_QQ.jpg", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# FIGURE 5 — Feature Importance (LightGBM)
# ============================================================
importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": model.feature_importances_,
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(6,4))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()

plt.title("Feature Importance (LightGBM)")
plt.xlabel("Importance Score")
plt.tight_layout()

plt.savefig(FIG_DIR / "Feature_Importance_LightGBM.jpg", dpi=300, bbox_inches="tight")
plt.show()

print("\nAll Chapter 4 non-SHAP figures saved successfully!")
