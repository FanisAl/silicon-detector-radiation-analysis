"""
test_2016_data.py  —  Real-data validation: Chen, UCSC (2016)
-------------------------------------------------------------
Tests the trained model against sensor W8-SD4-D6 from:
    Chen, S. (2016). Silicon Sensor R&D for the CMS Tracker Upgrade.
    PhD Thesis, UC Santa Cruz.

Sensor geometry (Table 1 of thesis):
    Area      = 1.028 mm x 1.028 mm
    Thickness = 50 µm
    Volume    = 5.29e-6 cm^3

Measurement temperature: -20 deg C
Expected fluence: 1.00e+15 neq/cm^2

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


print("--- Real-data validation: Chen, UCSC (2016) ---")

# Load extracted I-V data (digitised from thesis)
df = pd.read_csv("data/Default Dataset_2016.csv", sep=";", decimal=",",
                 header=None, names=["Voltage", "Current_A"])
df = df.sort_values("Voltage").reset_index(drop=True)

# Features at 100 V and 50 V (well below breakdown)
closest_100 = df.iloc[(df["Voltage"] - 100).abs().argsort()[:1]]
max_voltage = closest_100["Voltage"].values[0]
i_max_A     = closest_100["Current_A"].values[0]

closest_50  = df.iloc[(df["Voltage"] - 50).abs().argsort()[:1]]
mid_voltage = closest_50["Voltage"].values[0]
i_mid_A     = closest_50["Current_A"].values[0]

# Sensor volume from Table 1 of Chen (2016)
AREA_CM2  = (1.028 / 10) ** 2    # mm -> cm, then squared
THICK_CM  = 0.005                  # 50 µm -> cm
VOL_CM3   = AREA_CM2 * THICK_CM
T_MEAS    = -20.0                  # deg C

# Temperature normalisation
i_max_norm = normalize_to_20C(i_max_A, T_MEAS)
i_mid_norm = normalize_to_20C(i_mid_A, T_MEAS)

# Volume normalisation -> current density [A/cm^3]
rho_max = i_max_norm / VOL_CM3
rho_mid = i_mid_norm / VOL_CM3
slope   = (rho_max - rho_mid) / (max_voltage - mid_voltage)

print(f"Sensor volume    : {VOL_CM3:.4e} cm^3")
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
    real_fluence = 1.00e15

    print(f"\nModel prediction : {pred:.2e} neq/cm^2")
    print(f"Paper value      : {real_fluence:.2e} neq/cm^2")
    error = abs(pred - real_fluence) / real_fluence * 100
    print(f"Relative error   : {error:.1f}%")

except FileNotFoundError:
    print("[ERROR] rf_detector_brain_normalized.pkl not found.")
    print("  Please run train_model.py first.")
