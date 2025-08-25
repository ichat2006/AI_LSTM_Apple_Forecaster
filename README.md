# AAPL 2-Week Return Forecasting — Leakage-Safe LSTM + Baselines

Hi! This project is my end-to-end exploration of forecasting **10 trading-day (≈2-week) log returns** for AAPL.  
It’s designed to be educational, reproducible, and easy to tweak.

**What this does**
- Pulls price data **directly from APIs** in the notebook (no local CSV required).  
  If a price table isn’t already in memory, the notebook **falls back to Stooq AAPL daily** so it still runs.
- Engineers leakage-safe features (momentum, realized volatility, volume dynamics, optional fundamentals/news).
- Builds a **volatility-normalized** 2-week target so training isn’t dominated by volatility spikes.
- Evaluates with **purged, embargoed walk-forward splits** (expanding train; test blocked; gap ≥ horizon) to avoid label overlap leakage.
- Trains an **LSTM** with constant-shape windows (no TF retracing spam).  
  When TensorFlow isn’t installed, it **falls back to a Ridge baseline** so you can still run the pipeline.
- Produces an out-of-sample chart plus RMSE & directional accuracy.

> Research/education only — **not** financial advice.

---

## Why I built it this way

**Leakage-safe evaluation.** Overlapping forward returns make it easy to “cheat” accidentally. I use expanding walk-forward splits and an embargo gap (≥ horizon) so any label information near the test period is **purged** from training.

**Stabilized target.** I predict  
\[
\tilde r_t=\frac{\log(P_{t+10}/P_t)}{\hat\sigma_t\sqrt{10}},
\]
where \(\hat\sigma_t\) is a rolling std of daily log returns. This reduces heteroskedasticity; Huber loss then handles the remaining tails.

**Constant-shape inputs.** I generate windows with `keras.utils.timeseries_dataset_from_array`, which keeps shapes fixed and prevents TensorFlow from retracing graphs every step.

**Honest baselines.** I include Ridge (and optionally XGBoost) under the *same* splits so I can judge whether the LSTM actually adds value.

---

## What’s in here

- **Notebook cells** that:
  1. Import dependencies & set config (horizon, model choice, etc.)
  2. Use your API-fetched prices (auto-detects your DataFrame even if it isn’t named `px`).
  3. Build features: momentum (MA gaps), realized vol, volume dynamics, plus **optional** `fund` & `sent` tables if you provide them.
  4. Create the **vol-normalized** 2-week target.
  5. Generate **walk-forward** splits with an **embargo** gap.
  6. Train **LSTM / Ridge / XGBoost** and log per-fold metrics.
  7. Plot out-of-sample predictions vs. actuals and print RMSE & directional accuracy.

- **Auto-fallback**: If no price table is found in memory, the notebook fetches **AAPL daily** from Stooq so you can run end-to-end without setup.

---

## Installation

Use Python 3.9–3.12 and a virtual environment.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
