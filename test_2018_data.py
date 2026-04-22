"""
test_2018_data.py  —  Real-data validation: Scharf (2018)
---------------------------------------------------------
Tests the trained model against data from:
    Scharf, C. et al. (2018). Radiation hardness studies of silicon
    sensors for the CMS Phase-2 Inner Tracker upgrade.

Measurement temperature: -30 deg C
Expected fluence: 8.87e+14 neq/cm^2

Unit convention in the source data
-----------------------------------
The input CSV contains current density in A/m^2 (log-log axis from paper).
Conversion chain: A/m^2 -> A/cm^2 (÷ 10000) -> A/cm^3 (÷ thickness [cm]).

Features are extracted at 100 V to stay below the avalanche breakdown
region that begins above ~300 V in this dataset.

Normalisation parameters
------------------------
E_eff  = 1.21 eV   (Chilingarov 2013 — consistent with train_model.py)
"""

import pandas as pd
import numpy as np
import joblib

E_EFF = 1.21
K_B   = 8.617e-5
T_REF = 273.15 + 20.0


def normalize_to_20C(current, temp_C):
    """Arrhenius correction to 20 deg C."""
    T = 273.15 + temp_C
    factor = ((T_REF / T) ** 2
              * np.exp((-E_EFF / (2.0 * K_B)) * (1.0 / T_REF - 1.0 / T)))
    return current * factor


print("--- Real-data validation: Scharf (2018) ---")

# Load extracted I-V data (current density in A/m^2, digitised from paper)
df = pd.read_csv("data/Default Dataset_2018.csv", sep=";", decimal=",",
                 header=None, names=["Voltage", "J_A_per_m2"])
df = df.sort_values("Voltage").reset_index(drop=True)

# Extract features at 100 V and 50 V (below avalanche onset)
closest_100 = df.iloc[(df["Voltage"] - 100).abs().argsort()[:1]]
max_voltage = closest_100["Voltage"].values[0]
J_max_m2    = closest_100["J_A_per_m2"].values[0]

closest_50  = df.iloc[(df["Voltage"] - 50).abs().argsort()[:1]]
mid_voltage = closest_50["Voltage"].values[0]
J_mid_m2    = closest_50["J_A_per_m2"].values[0]

# Unit conversion: A/m^2 -> A/cm^2 -> A/cm^3
THICKNESS_CM = 0.0285   # 285 µm -> cm (Scharf 2018)
J_max_cm3 = (J_max_m2 / 10000.0) / THICKNESS_CM
J_mid_cm3 = (J_mid_m2 / 10000.0) / THICKNESS_CM

# Temperature normalisation
T_MEAS = -30.0   # deg C
rho_max = normalize_to_20C(J_max_cm3, T_MEAS)
rho_mid = normalize_to_20C(J_mid_cm3, T_MEAS)
slope   = (rho_max - rho_mid) / (max_voltage - mid_voltage)

print(f"Thickness        : {THICKNESS_CM*1e4:.0f} µm")
print(f"Physics features (normalised to 20 deg C):")
print(f"  Density at max V : {rho_max:.4e} A/cm^3")
print(f"  Density at mid V : {rho_mid:.4e} A/cm^3")
print(f"  Slope            : {slope:.4e} A/(cm^3·V)")

# Prediction
try:
    model = joblib.load("rf_detector_brain_normalized.pkl")
    features = pd.DataFrame([{
        "norm_density_max_V": rho_max,
        "norm_density_mid_V": rho_mid,
        "norm_slope":         slope,
    }])
    pred         = model.predict(features)[0]
    real_fluence = 8.87e14

    print(f"\nModel prediction : {pred:.2e} neq/cm^2")
    print(f"Paper value      : {real_fluence:.2e} neq/cm^2")
    error = abs(pred - real_fluence) / real_fluence * 100
    print(f"Relative error   : {error:.1f}%")

except FileNotFoundError:
    print("[ERROR] rf_detector_brain_normalized.pkl not found.")
    print("  Please run train_model.py first.")
