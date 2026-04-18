"""
live_monitor.py
---------------
Simulates a real-time monitoring system for silicon radiation detectors.

For each incoming "measurement", the script:
  1. Generates physically consistent I-V features from a realistic fluence value.
  2. Feeds the features to the trained Random Forest model.
  3. Raises a maintenance alert if predicted fluence exceeds the critical limit.

Physical basis
--------------
Features are derived from the same ΔI ∝ α·Φ relation used in data generation,
with added Gaussian instrument noise to simulate realistic sensor readings.

  alpha  = 4.0e-17 A·cm  (NIEL damage constant)
  volume = 0.1 cm^3

  I(V)   = I_baseline(V) + alpha * Phi * volume * (1 + 0.002*V)  [nA]
"""

import joblib
import pandas as pd
import numpy as np
import time

# ---------------------------------------------------------------------------
# Physical parameters — must match generate_data.py
# ---------------------------------------------------------------------------
ALPHA  = 4.0e-17   # damage rate constant [A·cm]
VOLUME = 0.1       # sensor volume [cm^3]

I_BASELINE_MAX_V = 4.35  # baseline leakage at 100 V [nA]
I_BASELINE_MID_V = 4.07  # baseline leakage at  50 V [nA]

MAX_V = 100.0
MID_V = 50.0

NOISE_FRACTION = 0.02   # 2% Gaussian instrument noise

# Detector health threshold
CRITICAL_FLUENCE_LIMIT = 8.0e14  # neq/cm^2

# ---------------------------------------------------------------------------
# Helper: generate physically consistent features for a given fluence
# ---------------------------------------------------------------------------
def simulate_iv_features(fluence: float) -> dict:
    """
    Derive I-V features from fluence using the radiation damage model.
    Adds realistic instrument noise.
    """
    rad_increase = ALPHA * fluence * VOLUME * 1e9  # [nA]

    i_max = I_BASELINE_MAX_V + rad_increase * (1 + 0.002 * MAX_V)
    i_mid = I_BASELINE_MID_V + rad_increase * (1 + 0.002 * MID_V)

    # Gaussian noise proportional to signal magnitude
    i_max += np.random.normal(0, NOISE_FRACTION * max(i_max, 1e-3))
    i_mid += np.random.normal(0, NOISE_FRACTION * max(i_mid, 1e-3))

    slope = (i_max - i_mid) / (MAX_V - MID_V)

    return {
        'current_at_max_V': i_max,
        'current_at_mid_V': i_mid,
        'curve_slope':      slope
    }

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("=" * 55)
print("  Silicon Detector Real-Time Monitoring System")
print("=" * 55)
print("Loading trained model...")

try:
    model = joblib.load('rf_detector_brain.pkl')
except FileNotFoundError:
    print("\n[ERROR] 'rf_detector_brain.pkl' not found.")
    print("  Please run train_model.py first.")
    exit(1)

print("Model loaded successfully.\n")
print(f"Critical fluence threshold : {CRITICAL_FLUENCE_LIMIT:.1e} neq/cm^2")
print("-" * 55)

# ---------------------------------------------------------------------------
# Simulated live feed — 8 detector readings
# ---------------------------------------------------------------------------
np.random.seed()  # fresh seed for each run

print("\nWaiting for live measurements...\n")
time.sleep(1)

n_measurements = 8

for i in range(1, n_measurements + 1):
    time.sleep(1.5)

    # Draw a realistic fluence (some detectors healthy, some degraded)
    true_fluence = np.random.uniform(0, 1.1e15)

    # Derive physically consistent I-V features + noise
    features = simulate_iv_features(true_fluence)
    df_live   = pd.DataFrame([features])
    detector_id = np.random.randint(1000, 9999)

    # Model prediction
    predicted_fluence = model.predict(df_live)[0]

    status = "ALARM" if predicted_fluence > CRITICAL_FLUENCE_LIMIT else "OK"
    tag    = "[!!]" if status == "ALARM" else "[ OK ]"

    print(f"[{i}/{n_measurements}] Detector #{detector_id}")
    print(f"  I at 100 V   : {features['current_at_max_V']:.3e} nA")
    print(f"  I at  50 V   : {features['current_at_mid_V']:.3e} nA")
    print(f"  I-V slope    : {features['curve_slope']:.3e} nA/V")
    print(f"  Predicted    : {predicted_fluence:.2e} neq/cm^2")
    print(f"  Status {tag} : {status}")

    if status == "ALARM":
        print("  >>> Predictive maintenance recommended.")
    print()

print("=" * 55)
print("  Monitoring session complete.")
print("=" * 55)
