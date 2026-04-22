"""
test.py  —  Real-data validation: Riedler (1998)
-------------------------------------------------
Tests the trained model against a real irradiated silicon diode measured
by Riedler et al. (1998) at T = -10 deg C.

Expected fluence from paper: 1.80e+14 neq/cm^2

Normalisation parameters
------------------------
E_eff  = 1.21 eV   (Chilingarov 2013 — consistent with train_model.py)
T_meas = -10 deg C
Volume = 0.016 cm^3 (estimated from diode geometry in the paper)
"""

import pandas as pd
import numpy as np
import joblib

E_EFF  = 1.21   # eV — Chilingarov (2013), consistent with training
K_B    = 8.617e-5
T_REF  = 273.15 + 20.0


def normalize_to_20C(current, temp_C):
    """Arrhenius correction to 20 deg C."""
    T = 273.15 + temp_C
    factor = ((T_REF / T) ** 2
              * np.exp((-E_EFF / (2.0 * K_B)) * (1.0 / T_REF - 1.0 / T)))
    return current * factor


print("--- Real-data validation: Riedler (1998) ---")

# Load extracted I-V data (digitised from paper)
df = pd.read_csv("data/Default Dataset_1998.csv", sep=";", decimal=",",
                 header=None, names=["Voltage", "LeakageCurrent"])
df = df.sort_values("Voltage").reset_index(drop=True)

# Extract features at 100 V and 50 V
max_voltage  = df["Voltage"].max()
i_max_raw    = df.loc[df["Voltage"] == max_voltage, "LeakageCurrent"].values[0]

closest_50   = df.iloc[(df["Voltage"] - 50).abs().argsort()[:1]]
mid_voltage  = closest_50["Voltage"].values[0]
i_mid_raw    = closest_50["LeakageCurrent"].values[0]

# Convert µA to A
i_max_A = i_max_raw * 1e-6
i_mid_A = i_mid_raw * 1e-6

# Experimental conditions (Riedler 1998)
T_MEAS   = -10.0    # deg C
VOL_CM3  = 0.016    # cm^3

# Temperature normalisation
i_max_norm = normalize_to_20C(i_max_A, T_MEAS)
i_mid_norm = normalize_to_20C(i_mid_A, T_MEAS)

# Volume normalisation -> current density [A/cm^3]
rho_max = i_max_norm / VOL_CM3
rho_mid = i_mid_norm / VOL_CM3
slope   = (rho_max - rho_mid) / (max_voltage - mid_voltage)

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
    real_fluence = 1.80e14

    print(f"\nModel prediction : {pred:.2e} neq/cm^2")
    print(f"Paper value      : {real_fluence:.2e} neq/cm^2")
    error = abs(pred - real_fluence) / real_fluence * 100
    print(f"Relative error   : {error:.1f}%")

except FileNotFoundError:
    print("[ERROR] rf_detector_brain_normalized.pkl not found.")
    print("  Please run train_model.py first.")
