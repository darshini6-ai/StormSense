# StormSense 🌩️

### Hyper-local Severe Convective Weather Nowcasting

StormSense is an AI-powered weather nowcasting prototype designed to predict the short-term evolution of severe convective storms and provide an interactive decision-support visualization.

## Problem

Severe convective storms can develop and move rapidly, making hyper-local warning lead time difficult to achieve. StormSense explores AI-based spatiotemporal forecasting to predict the future evolution of storm activity and visualize potential affected regions.

## Solution

StormSense uses the SEVIR dataset and a Residual ConvLSTM deep learning model to forecast future VIL (Vertically Integrated Liquid), a radar-derived storm intensity field.

The system pipeline is:

SEVIR → Preprocessing → Residual ConvLSTM V3 → Future VIL → Storm Detection → Storm Tracking → Multi-Hazard Risk Visualization

## Current Operational System

The StormSense workstation supports:

- SEVIR VIL data loading (Event-disjoint evaluation)
- Temporal preprocessing (12 past frames → 12 future forecast frames)
- Residual ConvLSTM V3 autoregressive forecasting (+5m to +60m)
- Storm-cell core detection & centroid tracking
- Multi-hazard fusion with observed GOES-16 GLM lightning detections
- Global forecast playback controller (Play, Pause, Reset, 1x/2x/4x, timeline slider)
- Interactive tactical radar visualization with contour overlays and vector trajectories
- Side-by-side ground truth comparison and live error telemetry

## Model

The primary model architecture is **StormSense ConvLSTM V3 Residual** (`StormSenseConvLSTMv3Residual`).

Configuration:

- Input: 12 frames (-60 to 0 min)
- Forecast: 12 frames (+5 to +60 min)
- Temporal resolution: 5 minutes per frame
- Input history: 60 minutes
- Validated forecast horizon: 60 minutes
- Spatial resolution: 128 × 128

## Dataset

StormSense is trained and evaluated on the NOAA/MIT SEVIR (Spatiotemporal Environmental Radar and Cellular Intelligence) dataset, incorporating calibrated Vertically Integrated Liquid (VIL) and complementary Geostationary Lightning Mapper (GLM) observations.

The dataset itself is not stored directly in this repository due to size.

## Scientific Validation

StormSense Residual V3 was rigorously evaluated using an **event-disjoint** protocol where complete storm events are strictly segregated between train, validation, and test splits (595 train events, 128 validation events, 128 held-out test events).

### Official Validated Benchmark Results

- **Test Events:** 128
- **Held-Out Test Windows:** 3,328
- **Validated Forecast Horizon:** 60 minutes (12 frames @ 5-min intervals)
- **Evaluation Protocol:** Strict Event-Disjoint Split
- **Residual V3 MSE:** 0.054802
- **Persistence Baseline MSE:** 0.083375
- **Relative MSE Improvement:** +34.27%

StormSense ConvLSTM V3 Residual outperforms the operational persistence baseline across all evaluated lead times from +5 to +60 minutes.

## Experimental 2-Hour Outlook

StormSense also includes an experimental 2-hour recursive rollout extending the forecast horizon from 60 minutes to 120 minutes (24 frames total).

- **Validated Forecast (T+00 → T+60):** Governed by the official event-disjoint benchmark results above.
- **Experimental Extended Outlook (T+65 → T+120):** Generated via unvalidated recursive rollout. The T+65 to T+120 portion is **NOT** part of the validated scientific benchmark and must not be interpreted as validated 2-hour model performance.

## Technology Stack

- Python 3.10+
- PyTorch (MPS / CUDA / CPU)
- ConvLSTM Spatiotemporal Modeling
- NumPy, Pandas, SciPy, h5py
- OpenCV, scikit-image
- Plotly, Streamlit, Streamlit Autorefresh
- NOAA/MIT SEVIR & GOES-16 GLM

## Project Structure

```text
StormSense/
├── app/
│   ├── main.py                  # Streamlit entrypoint & workstation routing
│   ├── playback.py              # Global playback timeline controller
│   ├── experimental_rollout.py  # 2-hour experimental recursive rollout
│   ├── page_forecast.py         # Primary Command Console hero view
│   ├── page_forecast_analysis.py# Forecast diagnostics & progression matrix
│   ├── page_storm_cells.py      # Convective core detection & tracking
│   ├── page_risk_analysis.py    # Threat mask & area coverage analysis
│   ├── page_backtest.py         # Multi-horizon backtest vs persistence
│   ├── page_metrics.py          # Official event-disjoint benchmark charts
│   ├── storm_tracker.py         # Connected-component core detection
│   ├── risk_engine.py           # Multi-hazard risk assessment
│   └── multihazard_fusion.py    # VIL + GLM lightning fusion
├── models/
│   └── convlstm_v3_residual.py  # StormSense ConvLSTM V3 Residual architecture
├── data/
│   └── sevir/                   # Local SEVIR VIL & GLM data directories
├── tests/
└── requirements.txt
```
