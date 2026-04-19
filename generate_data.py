"""
generate_data.py
----------------
Physics-informed data augmentation for silicon radiation detector simulation.

Baseline data
-------------
Bhatt et al. (2024). Elimination of dead layer in silicon particle detectors
via induced electric field based charge collection.
Zenodo. https://doi.org/10.5281/zenodo.13683410  (CC BY 4.0)

Augmentation model
------------------
Radiation damage is modelled via the Hamburg model:

    Delta_I(20C) = alpha * Phi * V_sensor

    alpha   = 4.0e-17 A/cm   (NIEL damage rate constant)
    Phi     = neutron-equivalent fluence [neq/cm^2]
    V_sensor = sensor volume [cm^3]

Temperature dependence follows the Arrhenius relation:

    I(T) = I(20C) * (T/T_ref)^2 * exp(-E_g / (2*k_B) * (1/T - 1/T_ref))

    E_g = 1.12 eV  (silicon bandgap)

5000 synthetic sensors are generated spanning:
    Fluence      : 0 to 1e15 neq/cm^2
    Temperature  : -20 to +25 deg C
    Sensor volume: 0.01 to 0.2 cm^3

References
----------
Moll, M. (1999). Radiation Damage in Silicon Particle Detectors.
    PhD Thesis, Universitat Hamburg.
Chilingarov, A. (2013). Temperature dependence of the current generated
    in Si bulk. JINST 8 P10003.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
ALPHA_20C      = 4.0e-17   # NIEL damage rate constant [A/cm]
E_G            = 1.21      # Silicon bandgap [eV]
K_B            = 8.617e-5  # Boltzmann constant [eV/K]
T_REF_K        = 273.15 + 20.0
NOISE_FRACTION = 0.05
NUM_DETECTORS  = 5000


def temperature_scaling(T_celsius):
    """Arrhenius scaling of leakage current relative to 20 deg C."""
    T = 273.15 + T_celsius
    return ((T / T_REF_K) ** 2
            * np.exp((-E_G / (2.0 * K_B)) * (1.0 / T - 1.0 / T_REF_K)))


# ---------------------------------------------------------------------------
# Step 1: Load baseline
# ---------------------------------------------------------------------------
print("Loading baseline I-V data...")

df_real = pd.read_csv("data/Figure2bIV.csv", sep=";")
df_real = df_real.iloc[1:].reset_index(drop=True)
df_real["Reverse Voltage"]         = pd.to_numeric(df_real["Reverse Voltage"])
df_real["Leakage current density"] = pd.to_numeric(
    df_real["Leakage current density"]) * 1e-9   # nA/cm^2 -> A/cm^2

print(f"  {len(df_real)} voltage points, "
      f"V = {df_real['Reverse Voltage'].min():.0f}"
      f"–{df_real['Reverse Voltage'].max():.0f} V")

# ---------------------------------------------------------------------------
# Step 2: Augmentation
# ---------------------------------------------------------------------------
print(f"Generating {NUM_DETECTORS} synthetic sensors...")

np.random.seed(42)
rows = []

for dev_id in range(NUM_DETECTORS):
    fluence   = np.random.uniform(0.0, 1e15)
    temp_C    = np.random.uniform(-20.0, 25.0)
    vol_cm3   = np.random.uniform(0.01, 0.2)
    t_factor  = temperature_scaling(temp_C)
    rad_20C   = ALPHA_20C * fluence * vol_cm3   # [A]

    for _, row in df_real.iterrows():
        v      = row["Reverse Voltage"]
        base_i = row["Leakage current density"] * vol_cm3   # A/cm^2 * cm^3 = A
        i_20C  = base_i + rad_20C * (1.0 + 0.002 * v)
        i_meas = i_20C * t_factor
        noise  = np.random.normal(0, NOISE_FRACTION * abs(i_meas))

        rows.append({
            "device_id":         dev_id,
            "temperature_C":     temp_C,
            "volume_cm3":        vol_cm3,
            "radiation_fluence": fluence,
            "voltage":           v,
            "current_Amperes":   i_meas + noise,
        })

df_ml = pd.DataFrame(rows)
df_ml.to_csv("data/ml_ready_detector_data_v2.csv", index=False)
print(f"  Saved {len(df_ml):,} rows to data/ml_ready_detector_data_v2.csv")

# ---------------------------------------------------------------------------
# Step 3: Diagnostic plot  (T = 20 C, V = 0.1 cm^3 for clarity)
# ---------------------------------------------------------------------------
print("Saving diagnostic plot...")

T_PLOT, V_PLOT = 20.0, 0.1
t_plot = temperature_scaling(T_PLOT)

fig, ax = plt.subplots(figsize=(9, 5))
colors = plt.cm.plasma(np.linspace(0.1, 0.9, 5))

for fluence, color in zip([0, 2.5e14, 5e14, 7.5e14, 1e15], colors):
    rad = ALPHA_20C * fluence * V_PLOT
    i_nA = [(row["Leakage current density"] * V_PLOT
             + rad * (1.0 + 0.002 * row["Reverse Voltage"])) * t_plot * 1e9
            for _, row in df_real.iterrows()]
    ax.plot(df_real["Reverse Voltage"], i_nA, color=color,
            label=rf"$\Phi$ = {fluence:.1e} neq/cm$^2$", linewidth=1.8)

ax.set_xlabel("Reverse voltage (V)", fontsize=11)
ax.set_ylabel("Leakage current (nA)", fontsize=11)
ax.set_title("Simulated I-V curves at selected radiation fluences\n"
             r"($T$ = 20 °C, $V_{\rm sensor}$ = 0.1 cm$^3$)", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("iv_curves_sample.png", dpi=300, bbox_inches="tight")
print("  Plot saved as iv_curves_sample.png")
