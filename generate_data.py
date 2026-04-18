"""
generate_data.py
----------------
Physics-informed data augmentation for silicon radiation detector simulation.

Baseline
--------
Real un-irradiated I-V data digitised from a published silicon sensor
measurement (Zenodo repository). The baseline describes the leakage current
density [nA/cm^2] as a function of reverse bias voltage [V].

Augmentation model
------------------
Radiation damage increases leakage current according to:

    ΔI = α · Φ · V_sensor                              (Hamburg model)

where
    α       = 4.0e-17 A·cm   (NIEL damage rate constant)
    Φ       = neutron-equivalent fluence [neq/cm^2]
    V_sensor = sensor volume [cm^3]

A linear voltage dependence and Gaussian instrument noise are added to
produce realistic I-V curves across 500 synthetic devices spanning
fluences from 0 to 1×10^15 neq/cm^2.

Reference
---------
Moll, M. (1999). Radiation Damage in Silicon Particle Detectors.
PhD Thesis, Universität Hamburg.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Step 1: Load real baseline I-V data
# ---------------------------------------------------------------------------
df_real = pd.read_csv("data/Figure2bIV.csv", sep=';')
df_real = df_real.iloc[1:].reset_index(drop=True)   # drop units row
df_real['Reverse Voltage']          = pd.to_numeric(df_real['Reverse Voltage'])
df_real['Leakage current density']  = pd.to_numeric(df_real['Leakage current density'])

print(f"Baseline loaded: {len(df_real)} voltage points, "
      f"V = {df_real['Reverse Voltage'].min():.0f}–"
      f"{df_real['Reverse Voltage'].max():.0f} V")

# ---------------------------------------------------------------------------
# Step 2: Physics-informed augmentation
# ---------------------------------------------------------------------------
np.random.seed(42)

NUM_DETECTORS = 500
ALPHA  = 4.0e-17   # damage rate constant [A·cm]
VOLUME = 0.1       # sensor volume [cm^3]

augmented_data = []

for device_id in range(NUM_DETECTORS):
    fluence = np.random.uniform(0, 1e15)   # neq/cm^2

    # Current increase from radiation damage [nA]
    radiation_current_increase = ALPHA * fluence * VOLUME * 1e9

    for _, row in df_real.iterrows():
        v      = row['Reverse Voltage']
        base_i = row['Leakage current density']

        # Linear voltage dependence + Gaussian instrument noise (2 % of signal)
        signal  = base_i + radiation_current_increase * (1 + 0.002 * v)
        noise   = np.random.normal(0, 0.02 * max(signal, 1e-3))
        sim_i   = signal + noise

        augmented_data.append({
            'device_id':        device_id,
            'radiation_fluence': fluence,
            'voltage':          v,
            'current':          sim_i
        })

# ---------------------------------------------------------------------------
# Step 3: Save dataset
# ---------------------------------------------------------------------------
df_ml_ready = pd.DataFrame(augmented_data)
df_ml_ready.to_csv('data/ml_ready_detector_data.csv', index=False)

print(f"Dataset saved: {len(df_ml_ready):,} data points "
      f"({NUM_DETECTORS} devices × {len(df_real)} voltage steps)")
print(f"Fluence range : 0 – 1e15 neq/cm^2")
print(f"Current range : {df_ml_ready['current'].min():.3e}"
      f" – {df_ml_ready['current'].max():.3e} nA")

# ---------------------------------------------------------------------------
# Step 4: Diagnostic plot — sample I-V curves at different fluences
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

sample_fluences = [0, 2.5e14, 5e14, 7.5e14, 1e15]
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(sample_fluences)))

for fluence, color in zip(sample_fluences, colors):
    sample_device = df_ml_ready[
        np.abs(df_ml_ready['radiation_fluence'] - fluence) ==
        np.abs(df_ml_ready['radiation_fluence'] - fluence).min()
    ]
    ax.plot(sample_device['voltage'], sample_device['current'],
            color=color, label=f'Φ = {fluence:.1e} neq/cm²', linewidth=1.5)

ax.set_xlabel('Reverse voltage (V)', fontsize=11)
ax.set_ylabel('Leakage current (nA)', fontsize=11)
ax.set_title('Simulated I-V Curves at Selected Radiation Fluences', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('iv_curves_sample.png', dpi=300, bbox_inches='tight')
print("Diagnostic plot saved as 'iv_curves_sample.png'")
