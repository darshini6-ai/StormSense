# StormSense 🌩️

### Hyper-local Severe Convective Weather Nowcasting

StormSense is an AI-powered weather nowcasting prototype designed to predict the short-term evolution of severe convective storms and provide an interactive decision-support visualization.

## Problem

Severe convective storms can develop and move rapidly, making hyper-local warning lead time difficult to achieve. StormSense explores AI-based spatiotemporal forecasting to predict the future evolution of storm activity and visualize potential affected regions.

## Solution

StormSense uses the SEVIR dataset and a ConvLSTM-based deep learning model to forecast future VIL (Vertically Integrated Liquid), a radar-derived storm intensity field.

The system pipeline is:

SEVIR → Preprocessing → ConvLSTM → Future VIL → Storm Detection → Storm Tracking → Risk Visualization

## Current Prototype

The current MVP supports:

- SEVIR VIL data loading
- Temporal preprocessing
- 12-frame historical input
- 12-frame autoregressive forecasting
- ConvLSTM-based prediction
- Storm-cell detection
- Storm-cell tracking
- Interactive VIL visualization
- Forecast time slider
- Predicted vs actual comparison
- Storm trajectory visualization

## Model

The primary experimental model is StormSense ConvLSTM V3.

Current prototype configuration:

- Input: 12 frames
- Forecast: 12 frames
- Temporal resolution: 5 minutes per frame
- Input history: 60 minutes
- Forecast horizon: 60 minutes
- Spatial resolution used for training: 128 × 128

## Dataset

StormSense uses the NOAA/MIT SEVIR dataset.

The dataset itself is not included in this repository because of its size.

## Evaluation

The current experimental evaluation compares StormSense against a persistence baseline.

On the current held-out window evaluation:

- StormSense V3 MSE: 0.047284
- Persistence baseline MSE: 0.058341
- Relative MSE improvement: 18.95%

This evaluation is an experimental prototype result and is not yet an event-disjoint scientific benchmark.

## Technology Stack

- Python
- PyTorch
- ConvLSTM
- NumPy
- Pandas
- SciPy
- h5py
- OpenCV
- scikit-image
- Plotly
- Streamlit
- SEVIR

## Project Structure

```text
StormSense/
├── app/
├── models/
├── data/
├── notebooks/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore

