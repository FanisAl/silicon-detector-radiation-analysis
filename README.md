# Silicon Detector Radiation Damage Analysis

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

Silicon tracking detectors in high-luminosity particle accelerators (HL-LHC at CERN)
are exposed to extreme radiation levels over their operational lifetime. Radiation
damage increases leakage current, degrades charge collection efficiency, and
ultimately limits detector performance and lifespan.

This project analyses Current-Voltage (I-V) characteristics of silicon sensors
and applies a Machine Learning approach to predict cumulative radiation exposure
(fluence) from measurable electrical features — a step towards automated predictive
maintenance of detector systems.

## Methodology

### 1. Baseline data

A real I-V curve from an un-irradiated silicon sensor was used as the physical
baseline (source: Zenodo). The dataset provides leakage current density [nA/cm²]
as a function of reverse bias voltage (0–100 V).

### 2. Physics-informed data augmentation

Radiation damage was modelled using the Hamburg model:

```
ΔI = α · Φ · V_sensor
```

where `α = 4.0 × 10⁻¹⁷ A·cm` is the NIEL damage rate constant,
`Φ` is the neutron-equivalent fluence [neq/cm²], and `V_sensor` is the
sensor volume. A linear voltage dependence and Gaussian instrument noise (2 %)
were added to generate 500 synthetic I-V curves spanning fluences
from 0 to 1 × 10¹⁵ neq/cm².

### 3. Feature extraction

Three physically motivated features were extracted per device:

| Feature | Description |
|---|---|
| `current_at_max_V` | Leakage current at maximum reverse voltage (100 V) |
| `current_at_mid_V` | Leakage current at mid-range voltage (50 V) |
| `curve_slope` | Average I-V slope: ΔI / ΔV between 50 V and 100 V |

### 4. Machine learning model

A `RandomForestRegressor` (100 estimators, max depth 5) was trained on
80 % of the devices and evaluated on the remaining 20 %.

### 5. Live monitoring simulation

A simulated real-time monitor (`live_monitor.py`) generates physically
consistent sensor readings, queries the trained model for fluence prediction,
and raises a predictive maintenance alert when the estimated fluence exceeds
the critical threshold of 8 × 10¹⁴ neq/cm².

## Results

| Metric | Value |
|---|---|
| R² score | 0.997 |
| RMSE | ~1.5 × 10¹³ neq/cm² |

**Note on model performance:** The high R² reflects that the data augmentation
embeds a deterministic linear relation (`ΔI ∝ Φ`), which the Random Forest
successfully inverts. This validates the end-to-end methodology and monitoring
pipeline. Performance on real experimental data — where device geometry,
temperature, trap dynamics, and non-uniform irradiation introduce additional
variance — would differ and require retraining on measured datasets.

<img width="4161" height="1759" alt="rf_prediction_results" src="https://github.com/user-attachments/assets/454cabd9-876f-4cbe-89ee-064a370e5e75" />
<img width="2370" height="1466" alt="iv_curves_sample" src="https://github.com/user-attachments/assets/0c53ba51-5728-4b16-a160-0eee59e2d702" />


## Repository structure

```
silicon-detector-analysis/
├── data/
│   ├── Figure2bIV.csv             # Real baseline I-V data (Zenodo)
│   └── ml_ready_detector_data.csv # Augmented synthetic dataset
├── generate_data.py               # Physics-informed data augmentation
├── train_model.py                 # Feature extraction, model training, evaluation
├── live_monitor.py                # Simulated real-time monitoring system
├── rf_detector_brain.pkl          # Trained Random Forest model
├── rf_prediction_results.png      # Actual vs predicted fluence plot
├── iv_curves_sample.png           # Sample I-V curves at selected fluences
├── requirements.txt
└── README.md
```

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Generate the synthetic dataset
python generate_data.py

# 2. Train the Random Forest and evaluate
python train_model.py

# 3. Start the live monitoring simulation
python live_monitor.py
```

## Physical background

- **Radiation damage in silicon:** Incident particles displace lattice atoms,
  creating defect clusters that act as generation-recombination centres,
  increasing bulk leakage current.
- **Hamburg model:** The linear relation `ΔI ∝ α·Φ` is well established
  for fluences up to ~10¹⁶ neq/cm² at room temperature.
- **HL-LHC context:** Inner tracker layers of CMS and ATLAS are expected
  to accumulate fluences of 10¹⁵–10¹⁶ neq/cm² over the full HL-LHC run,
  making radiation hardness assessment a critical engineering challenge.

## References

1. Moll, M. (1999). *Radiation Damage in Silicon Particle Detectors*.
   PhD Thesis, Universität Hamburg.
2. Leroy, C. & Rancoita, P.-G. (2007). Particle interaction and displacement
   damage in silicon devices operated in radiation environments.
   *Rep. Prog. Phys.*, 70, 493.
3. CERN RD50 Collaboration. Development of radiation hard semiconductor
   devices for very high luminosity colliders. https://cern.ch/rd50
4. Setälä, O. et al. (2024). Elimination of dead layer in silicon particle detectors via induced electric field based charge collection [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.13683410

## Author

Fanis Alexakis  
MSc Microsystems & Nanodevices, NTUA  
BSc Physics, NKUA  
fanisalexakis64@gmail.com
