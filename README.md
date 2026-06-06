# CGM Forecast Monitor

A real time glucose forecasting and anomaly detection system for Tandem t:slim X2 insulin pump users. An LSTM model trained on historical pump data predicts the next 2 hours of CGM readings every 10 minutes, and flags when live sensor values diverge significantly from the forecast.

---

## How It Works

1. **Training:** `train.py` ingests Tandem Source CSV exports, merges CGM, basal, and bolus data into aligned 5-minute windows, fits a preprocessing pipeline, and trains the LSTM.
2. **Forecasting:** `simulate.py` connects to the Tandem Source API, fetches the last 6 hours of live pump events, runs the model, and prints a detection report with a Dexcom-styled chart.
3. **Anomaly detection:** a weighted composite score (value error + slope error + rising slope penalty) flags readings that diverge from the forecast at 2σ or 3σ.

---

## Project Structure

```
├── data_processing.py      # Merge & feature engineer CGM/basal/bolus DataFrames
├── dataset.py              # PyTorch sliding window Dataset
├── model.py                # CgmLstm: 3 layer LSTM → FC(24)
├── model_preprocessing.py  # Sklearn imputation + MinMax scaling pipeline
├── simulate.py             # Live forecast, anomaly detection, console report
├── train.py                # Training script
├── visualizer.py           # Matplotlib charts
├── Training/               # Place zip reports here (not committed)
├── model_weights.pth       # Saved model weights (not committed)
├── preprocessor.joblib     # Fitted sklearn pipeline (not committed)
├── detection_stats.joblib  # Fitted anomaly detection benchmarks (not committed)
└── .env                    # Tandem credentials (not committed — see Setup)
```

---

## Model

| Component | Detail |
|---|---|
| Architecture | 3 layer LSTM + linear head |
| Input | 7 features × 48 steps (4 hours) |
| Output | 24 steps (2-hour forecast) |
| Loss | 1 × value MSE + 0.15 × slope MSE + 0.05 × curvature MSE |
| Optimizer | Adam, lr=0.001 |

**Input features** (per 5-minute step):

- CGM reading (normalised)
- Commanded basal dose
- Insulin delivered (bolus)
- Carbohydrates
- Insulin on board (linear decay model, 4-hour duration of insulin action)
- Time of day with sin/cos encoding

---

## Requirements

```
torch
numpy
pandas
scikit-learn
matplotlib
joblib
tconnectsync
python-dotenv
```

Install with:

```bash
pip install torch numpy pandas scikit-learn matplotlib joblib tconnectsync python-dotenv
```

A CUDA capable GPU is expected for model training. Forecasting runs on CPU if needed (change `"cuda"` to `"cpu"` in `model.py`)

---

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/ThudSoundEffect/cgm-forecast-monitor.git
cd cgm-forecast-monitor
```

**2. Create a `.env` file** with your Tandem Source credentials:

```
TANDEM_EMAIL=your@email.com
TANDEM_PASSWORD=yourpassword
TANDEM_PUMP_ID=1234567
```

**3. Add training data**

Place one or more `.zip` reports exported from Tandem Source in the `Training/` directory. Each zip will contain CSVs named `*-CGM.csv`, `*-Basal-doses.csv`, and `*-Bolus.csv`.

---

## Training

```bash
python train.py
```

This script will:
- Extract and merge all zip archives in `Training/`
- Fit the preprocessing pipeline and save `preprocessor.joblib`
- Aggregates anomaly detection metrics and save `detection_stats.joblib`
- Train the LSTM for 10 epochs and save `model_weights.pth`

All three saved files are necessary for running the simulator.

---

## Running the Monitor

```bash
python simulate.py
```

Fetches most recent available data from Tandem Source, makes forecasts, displays a forecast chart, and prints an anomaly detection report every 10 minutes:

```
────────────────────────────────────────────────────
  CGM Anomaly Detection Report  [2025-06-01 14:35]
────────────────────────────────────────────────────
  Detection score   :    612.3   (thresholds: 3800 / 5200)
  No anomaly detected
```

The chart shows color coded glucose readings (green = in range, orange = low/high, red = hypo/hyper) and dashed 2-hour forecast.

<img width="1400" height="500" alt="image" src="https://github.com/user-attachments/assets/a73ce3ec-4845-453b-b1fc-e5252483fc31" />

---

## Anomaly Detection

The detection score combines three error terms:

| Term | Weight | Purpose |
|---|---|---|
| Value MSE | 1.0 | Overall glucose level accuracy |
| Slope L1 | 1.5 | Rate of change accuracy |
| Rising slope MSE | 3.0 | Penalises unforecasted rising trends in real time data |

A detectection score above mean + 2σ triggers a mild anomaly warning; above mean + 3σ triggers a strong warning. 

---

## Disclaimer

This project is not a medical device and is not intended for clinical use. Do not make treatment decisions based on its output.
