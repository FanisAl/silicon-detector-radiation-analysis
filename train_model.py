"""
train_model.py
--------------
Trains a physics-normalised Random Forest Regressor to predict radiation
fluence from I-V curve features of silicon sensors.

Normalisation pipeline (applied before feature extraction)
----------------------------------------------------------
1. Temperature correction (Arrhenius, E_eff = 1.21 eV):
       I_norm = I_meas * (T_ref/T_meas)^2 * exp(-E_eff/(2*k_B)*(1/T_ref - 1/T_meas))
   Scales all currents to their equivalent at T_ref = 20 deg C.

2. Volume normalisation:
       rho = I_norm / V_sensor   [A/cm^3]
   Converts total current to volumetric current density, removing
   sensor-size dependence.

Features used
-------------
    norm_density_max_V  : current density at 100 V [A/cm^3]
    norm_density_mid_V  : current density at  50 V [A/cm^3]
    norm_slope          : (density_max - density_mid) / (100 - 50) [A/(cm^3·V)]

Target
------
    radiation_fluence   : neutron-equivalent fluence [neq/cm^2]

Reference for E_eff
-------------------
Chilingarov, A. (2013). Temperature dependence of the current generated
in Si bulk. JINST 8 P10003.  -> recommends E_eff = 1.21 eV.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import joblib

# ---------------------------------------------------------------------------
# Temperature normalisation (Arrhenius)
# E_eff = 1.21 eV — Chilingarov (2013), used consistently in all scripts.
# ---------------------------------------------------------------------------
E_EFF = 1.21
K_B   = 8.617e-5
T_REF = 273.15 + 20.0


def normalize_to_20C(current, temp_C, E_eff=E_EFF):
    """Scale measured current to its equivalent at 20 deg C."""
    T = 273.15 + temp_C
    factor = ((T_REF / T) ** 2
              * np.exp((-E_eff / (2.0 * K_B)) * (1.0 / T_REF - 1.0 / T)))
    return current * factor


# ---------------------------------------------------------------------------
# Step 1: Load data and extract normalised features
# ---------------------------------------------------------------------------
print("--- Step 1: Loading data and extracting features ---")

df = pd.read_csv("data/ml_ready_detector_data_v2.csv")
features_list = []

for dev_id, group in df.groupby("device_id"):
    fluence = group["radiation_fluence"].iloc[0]
    temp    = group["temperature_C"].iloc[0]
    vol     = group["volume_cm3"].iloc[0]

    max_voltage = group["voltage"].max()
    i_max = group.loc[group["voltage"] == max_voltage, "current_Amperes"].values[0]

    # Use 50 V as mid-point; fall back to nearest available voltage
    if 50 in group["voltage"].values:
        mid_voltage = 50
    else:
        idx = (group["voltage"] - 50).abs().idxmin()
        mid_voltage = group.loc[idx, "voltage"]
    i_mid = group.loc[group["voltage"] == mid_voltage, "current_Amperes"].values[0]

    # 1. Temperature correction
    i_max_norm = normalize_to_20C(i_max, temp)
    i_mid_norm = normalize_to_20C(i_mid, temp)

    # 2. Volume normalisation -> current density [A/cm^3]
    rho_max = i_max_norm / vol
    rho_mid = i_mid_norm / vol

    # 3. Normalised slope [A/(cm^3·V)]
    norm_slope = (rho_max - rho_mid) / (max_voltage - mid_voltage)

    features_list.append({
        "device_id":         dev_id,
        "norm_density_max_V": rho_max,
        "norm_density_mid_V": rho_mid,
        "norm_slope":         norm_slope,
        "target_fluence":     fluence,
    })

df_features = pd.DataFrame(features_list)

print(f"  Total devices : {len(df_features)}")
print(f"  Density at max V : {df_features['norm_density_max_V'].min():.3e}"
      f" to {df_features['norm_density_max_V'].max():.3e} A/cm^3")

# ---------------------------------------------------------------------------
# Step 2: Train / test split
# ---------------------------------------------------------------------------
print("\n--- Step 2: Train / test split (80/20) ---")

X = df_features[["norm_density_max_V", "norm_density_mid_V", "norm_slope"]]
y = df_features["target_fluence"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"  Training : {len(X_train)} devices")
print(f"  Test     : {len(X_test)} devices")

# ---------------------------------------------------------------------------
# Step 3: Train Random Forest
# ---------------------------------------------------------------------------
print("\n--- Step 3: Training Random Forest Regressor ---")

model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# Step 4: Evaluation
# ---------------------------------------------------------------------------
print("\n--- Step 4: Evaluation ---")

preds = model.predict(X_test)
r2   = r2_score(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"  R^2  : {r2:.4f}")
print(f"  RMSE : {rmse:.3e} neq/cm^2")
print()
print("  NOTE: The high R^2 reflects that the augmentation encodes a")
print("  deterministic ΔI ∝ Φ relation; the model inverts it. Performance")
print("  on real experimental data requires retraining on measured datasets.")

print("\n  Feature importances:")
for feat, imp in zip(X.columns, model.feature_importances_):
    print(f"    {feat:<26} {imp:.4f}")

# ---------------------------------------------------------------------------
# Step 5: Plots
# ---------------------------------------------------------------------------
print("\n--- Step 5: Saving evaluation plots ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Actual vs predicted
ax = axes[0]
ax.scatter(y_test, preds, alpha=0.5, color="steelblue",
           edgecolors="k", linewidths=0.3, label="Predictions")
ax.plot([y.min(), y.max()], [y.min(), y.max()],
        "r--", lw=2, label="Perfect prediction")
ax.set_title("Random Forest: actual vs predicted fluence", fontsize=13)
ax.set_xlabel(r"Actual fluence (neq/cm$^2$)", fontsize=11)
ax.set_ylabel(r"Predicted fluence (neq/cm$^2$)", fontsize=11)
ax.text(0.05, 0.92, f"$R^2$ = {r2:.4f}", transform=ax.transAxes,
        fontsize=11, color="darkred")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.6)

# Feature importances
ax2 = axes[1]
feat_labels = ["Density at max V", "Density at mid V", "Norm. slope"]
bars = ax2.barh(feat_labels, model.feature_importances_,
                color="steelblue", edgecolor="k", linewidth=0.5)
for bar, val in zip(bars, model.feature_importances_):
    ax2.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=10)
ax2.set_xlabel("Importance", fontsize=11)
ax2.set_title("Feature importances", fontsize=13)
ax2.set_xlim(0, 1)
ax2.grid(True, axis="x", linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("rf_prediction_results.png", dpi=300, bbox_inches="tight")
print("  Plot saved as rf_prediction_results.png")

# ---------------------------------------------------------------------------
# Step 6: Save model
# ---------------------------------------------------------------------------
joblib.dump(model, "rf_detector_brain_normalized.pkl")
print("  Model saved as rf_detector_brain_normalized.pkl")
