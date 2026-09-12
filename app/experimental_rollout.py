# ============================================================
# StormSense - Experimental 2-Hour Forecast Rollout Module
# UNVALIDATED RECURSIVE ROLLOUT EXTENSION
# ============================================================
# IMPORTANT SCIENTIFIC NOTICE:
# The validated StormSense ConvLSTM V3 model produces a 60-minute
# forecast (12 future frames x 5 minutes).
# This module implements an experimental recursive rollout to produce
# 24 future frames (120 minutes).
#
# This extended rollout is UNVALIDATED and EXPERIMENTAL.
# It does NOT alter the official 60-minute benchmark results.
# ============================================================

import os
import numpy as np
import torch
import torch.nn.functional as F
import h5py
import streamlit as st


def generate_recursive_rollout_2hr(model, past_tensor, device):
    """
    Generate 24 future frames (120 minutes) using 2-stage recursive rollout.

    Stage 1: Past 12 frames -> Validated 12 forecast frames (+5m to +60m)
    Stage 2: Forecast frames 1..12 -> Experimental 12 forecast frames (+65m to +120m)

    Parameters:
        model: Trained StormSenseConvLSTMv3Residual model
        past_tensor: Tensor of shape (12, 1, 128, 128)
        device: Torch device

    Returns:
        pred_24_np: Numpy array of shape (24, 128, 128)
    """
    model_input_stage1 = past_tensor.unsqueeze(0).to(device)  # (1, 12, 1, 128, 128)

    with torch.no_grad():
        # Stage 1: Validated 60-minute window (frames 1..12)
        stage1_pred = model(
            model_input_stage1,
            future_steps=12,
            teacher_forcing_ratio=0.0
        )  # (1, 12, 1, 128, 128)

        # Stage 2: Recursive rollout for extended window (+65m to +120m, frames 13..24)
        stage2_pred = model(
            stage1_pred,
            future_steps=12,
            teacher_forcing_ratio=0.0
        )  # (1, 12, 1, 128, 128)

        # Concatenate 12 + 12 = 24 future frames
        combined_pred = torch.cat([stage1_pred, stage2_pred], dim=1)  # (1, 24, 1, 128, 128)

    pred_24_np = combined_pred.cpu().squeeze(0)[:, 0].numpy()  # (24, 128, 128)
    return pred_24_np


def get_actual_frames_120(file_path, sequence_index, image_size=128):
    """
    Retrieve ground truth observations up to 24 future frames (+120 min) if available in SEVIR.
    SEVIR VIL episodes contain 49 frames (12 past + up to 37 future).
    """
    if not os.path.exists(file_path):
        return None

    try:
        with h5py.File(file_path, "r") as f:
            num_events = f["vil"].shape[0]
            idx = max(0, min(sequence_index, num_events - 1))
            vil_data = f["vil"][idx]  # (384, 384, 49)

        vil_tensor = torch.tensor(vil_data, dtype=torch.float32) / 255.0
        vil_tensor = vil_tensor.permute(2, 0, 1).unsqueeze(1)  # (49, 1, 384, 384)

        vil_resized = F.interpolate(
            vil_tensor,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False
        )

        # 12 past frames + 24 future frames = 36 frames total
        actual_24 = vil_resized[12:36, 0].numpy()  # (24, 128, 128)
        return actual_24
    except Exception:
        return None


def render_experimental_badge(current_minute):
    """
    Renders visual indicators clearly distinguishing validated (0-60m)
    from experimental (65-120m) forecast frames.
    """
    if current_minute <= 60:
        st.markdown(
            """
            <div style="display: inline-flex; align-items: center; gap: 6px;
                        background: rgba(52, 211, 153, 0.12);
                        border: 1px solid rgba(52, 211, 153, 0.35);
                        border-radius: 4px; padding: 3px 8px;
                        font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; color: #34d399;">
                <span style="width: 6px; height: 6px; border-radius: 50%; background: #34d399;"></span>
                <span>VALIDATED FORECAST HORIZON (T+00 → T+60)</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="display: inline-flex; align-items: center; gap: 6px;
                        background: rgba(245, 158, 11, 0.15);
                        border: 1px solid rgba(245, 158, 11, 0.45);
                        border-radius: 4px; padding: 3px 8px;
                        font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; color: #fbbf24;">
                <span style="width: 6px; height: 6px; border-radius: 50%; background: #fbbf24;"></span>
                <span>⚠️ EXPERIMENTAL EXTENDED OUTLOOK (T+65 → T+120) — UNVALIDATED RECURSIVE ROLLOUT</span>
            </div>
            """,
            unsafe_allow_html=True
        )
