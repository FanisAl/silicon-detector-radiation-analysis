"""
train_model.py
--------------
Trains a Random Forest Regressor to predict radiation fluence
from I-V curve features of silicon detectors.

Features used:
  - current_at_max_V  : leakage current at maximum reverse voltage (100 V)
  - current_at_mid_V  : leakage current at mid voltage (50 V)
  - curve_slope       : slope of the I-V curve between 50 V and 100 V

Target:
  - radiation_fluence : neutron-equivalent fluence [neq/cm^2]
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import joblib

# ---------------------------------------------------------------------------
# STEP 1: Load augmented dataset and extract per-device features
# ---------------------------------------------------------------------------
print("--- Step 1: Loading data and extracting features ---")

df = pd.read_csv('data/ml_ready_detector_data.csv')

features_list = []

for dev_id, group in df.groupby('device_id'):
    fluence = group['radiation_fluence'].iloc[0]

    max_voltage = group['voltage'].max()   # 100 V
    mid_voltage = 50.0                     # reference mid-point voltage

    current_at_max_v = group.loc[
        group['voltage'] == max_voltage, 'current'
    ].values[0]

    # Use closest available voltage if 50 V is not present
    if mid_voltage not in group['voltage'].values:
        idx = (group['voltage'] - mid_voltage).abs().idxmin()
        mid_voltage = group.loc[idx, 'voltage']

    current_at_mid_v = group.loc[
        group['voltage'] == mid_voltage, 'current'
    ].values[0]

    # FIX: correct slope formula — delta_I / delta_V
    slope = (current_at_max_v - current_at_mid_v) / (max_voltage - mid_voltage)

    features_list.append({
        'device_id':      dev_id,
        'current_at_max_V': current_at_max_v,
        'current_at_mid_V': current_at_mid_v,
        'curve_slope':      slope,
        'target_fluence':   fluence
    })

df_features = pd.DataFrame(features_list)

print(df_features.head())
print(f"\nTotal devices for training: {len(df_features)}")
print(f"\nFeature ranges (physically expected):")
print(f"  current_at_max_V : {df_features['current_at_max_V'].min():.3e}"
      f" – {df_features['current_at_max_V'].max():.3e} nA")
print(f"  current_at_mid_V : {df_features['current_at_mid_V'].min():.3e}"
      f" – {df_features['current_at_mid_V'].max():.3e} nA")
print(f"  curve_slope      : {df_features['curve_slope'].min():.3e}"
      f" – {df_features['curve_slope'].max():.3e} nA/V")

# ---------------------------------------------------------------------------
# STEP 2: Train / test split
# ---------------------------------------------------------------------------
print("\n--- Step 2: Train / test split ---")

X = df_features[['current_at_max_V', 'current_at_mid_V', 'curve_slope']]
y = df_features['target_fluence']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set : {len(X_train)} devices")
print(f"Test set     : {len(X_test)}  devices")

# ---------------------------------------------------------------------------
# STEP 3: Train Random Forest
# ---------------------------------------------------------------------------
print("\n--- Step 3: Training Random Forest Regressor ---")

rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# STEP 4: Evaluation
# ---------------------------------------------------------------------------
print("\n--- Step 4: Evaluation ---")

predictions = rf_model.predict(X_test)
r2  = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print(f"R^2  score : {r2:.4f}")
print(f"RMSE       : {rmse:.3e} neq/cm^2")
print()
print("NOTE: The high R^2 reflects that the augmentation embeds a linear")
print("  Delta_I ~ alpha * Phi relation, which the model successfully")
print("  inverts. This validates the methodology, not a new physical")
print("  discovery. Performance on real experimental data would differ.")

# ---------------------------------------------------------------------------
# STEP 5: Feature importance
# ---------------------------------------------------------------------------
importances = rf_model.feature_importances_
print("\nFeature importances:")
for feat, imp in zip(X.columns, importances):
    print(f"  {feat:<22} {imp:.4f}")

# ---------------------------------------------------------------------------
# STEP 6: Plots
# ---------------------------------------------------------------------------
print("\n--- Step 5: Generating plots ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Actual vs Predicted ---
ax = axes[0]
ax.scatter(y_test, predictions, alpha=0.6, color='steelblue',
           edgecolors='k', linewidths=0.4, label='Predictions')
ax.plot([y.min(), y.max()], [y.min(), y.max()],
        'r--', lw=2, label='Perfect prediction')
ax.set_title('Random Forest: Actual vs Predicted Fluence', fontsize=13)
ax.set_xlabel(r'Actual fluence (neq/cm$^2$)', fontsize=11)
ax.set_ylabel(r'Predicted fluence (neq/cm$^2$)', fontsize=11)
ax.legend()
ax.grid(True, linestyle=':', alpha=0.6)
ax.text(0.05, 0.92, f'$R^2$ = {r2:.4f}', transform=ax.transAxes,
        fontsize=11, color='darkred')

# --- Feature Importance ---
ax2 = axes[1]
feat_names = ['I at max V', 'I at mid V', 'I-V slope']
bars = ax2.barh(feat_names, importances, color='steelblue', edgecolor='k',
                linewidth=0.5)
ax2.set_xlabel('Importance', fontsize=11)
ax2.set_title('Feature Importances', fontsize=13)
ax2.set_xlim(0, 1)
for bar, val in zip(bars, importances):
    ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=10)
ax2.grid(True, axis='x', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('rf_prediction_results.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'rf_prediction_results.png'")

# ---------------------------------------------------------------------------
# STEP 7: Save model
# ---------------------------------------------------------------------------
joblib.dump(rf_model, 'rf_detector_brain.pkl')
print("Model saved as 'rf_detector_brain.pkl'")
