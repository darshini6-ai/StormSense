import os
import sys
import textwrap
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from app.map_utils import create_vil_map
from app.experimental_rollout import render_experimental_badge


def render_risk_analysis_page(pred_frames, contrast_mode=False):
    """
    Renders the Threat Alerts & Risk Analysis Page synchronized with global timeline.
    """
    st.markdown(
        textwrap.dedent("""
        <div style="margin-bottom: 8px;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #00f0ff; font-family: 'JetBrains Mono', monospace; text-transform: uppercase;">
                THREAT ALERTS & RISK ANALYSIS
            </div>
            <div style="font-size: 0.70rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
                Spatial extent, thresholded convective signatures, and area coverage derived from predicted VIL fields.
            </div>
        </div>
        """).strip(),
        unsafe_allow_html=True
    )

    num_pred = len(pred_frames)
    lead_options = [(s + 1) * 5 for s in range(num_pred)]

    global_mins = int(st.session_state.get("timeline_minutes", 15))
    default_val = global_mins if global_mins in lead_options else (15 if 15 in lead_options else lead_options[0])

    r_col1, r_col2 = st.columns([1, 1])
    with r_col1:
        risk_horizon = st.select_slider(
            "Select Forecast Horizon",
            options=lead_options,
            value=default_val,
            format_func=lambda x: f"+{x} min" if x <= 60 else f"+{x} min [EXP]",
            key="ra_horizon_select_slider"
        )
        if risk_horizon != st.session_state.get("timeline_minutes", 0):
            st.session_state.timeline_minutes = risk_horizon
            st.session_state.global_timeline = risk_horizon

        render_experimental_badge(risk_horizon)

    with r_col2:
        threshold_val = st.slider(
            "Prototype VIL Threshold",
            min_value=0.02,
            max_value=0.50,
            value=0.15,
            step=0.01,
            format="%.2f"
        )

    r_idx = (risk_horizon // 5) - 1
    r_frame = pred_frames[r_idx]

    max_v = float(r_frame.max())
    mean_v = float(r_frame.mean())
    coverage_px = int((r_frame >= threshold_val).sum())
    total_px = r_frame.size
    coverage_pct = (coverage_px / total_px) * 100.0

    if max_v >= 0.70:
        sig_label = "HIGH CONVECTIVE SIGNAL"
        sig_color = "#f87171"
    elif max_v >= 0.35:
        sig_label = "MODERATE CONVECTIVE SIGNAL"
        sig_color = "#fbbf24"
    elif max_v >= 0.05:
        sig_label = "LOW CONVECTIVE SIGNAL"
        sig_color = "#34d399"
    else:
        sig_label = "MINIMAL SIGNAL"
        sig_color = "#64748b"

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.markdown(
            textwrap.dedent(f"""
            <div class="stitch-card" style="text-align: center;">
                <div style="font-size: 0.58rem; color: #94a3b8; text-transform: uppercase; font-family: 'JetBrains Mono', monospace;">Maximum VIL</div>
                <div style="font-size: 1.35rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #c084fc;">{max_v:.3f}</div>
                <div style="font-size: 0.58rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">Horizon: +{risk_horizon}m</div>
            </div>
            """).strip(),
            unsafe_allow_html=True
        )
    with kpi_cols[1]:
        st.markdown(
            textwrap.dedent(f"""
            <div class="stitch-card" style="text-align: center;">
                <div style="font-size: 0.58rem; color: #94a3b8; text-transform: uppercase; font-family: 'JetBrains Mono', monospace;">Mean VIL</div>
                <div style="font-size: 1.35rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #34d399;">{mean_v:.3f}</div>
                <div style="font-size: 0.58rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">Domain Average</div>
            </div>
            """).strip(),
            unsafe_allow_html=True
        )
    with kpi_cols[2]:
        st.markdown(
            textwrap.dedent(f"""
            <div class="stitch-card" style="text-align: center;">
                <div style="font-size: 0.58rem; color: #94a3b8; text-transform: uppercase; font-family: 'JetBrains Mono', monospace;">Threshold Coverage</div>
                <div style="font-size: 1.35rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #00f0ff;">{coverage_pct:.2f}%</div>
                <div style="font-size: 0.58rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">{coverage_px} of {total_px} px</div>
            </div>
            """).strip(),
            unsafe_allow_html=True
        )
    with kpi_cols[3]:
        st.markdown(
            textwrap.dedent(f"""
            <div class="stitch-card" style="text-align: center;">
                <div style="font-size: 0.58rem; color: #94a3b8; text-transform: uppercase; font-family: 'JetBrains Mono', monospace;">Classification</div>
                <div style="font-size: 0.85rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: {sig_color}; padding-top: 4px;">{sig_label}</div>
                <div style="font-size: 0.58rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">Derived Indicator</div>
            </div>
            """).strip(),
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
    title_suffix = f"at +{risk_horizon}m" + (" [EXPERIMENTAL]" if risk_horizon > 60 else "")
    map_fig = create_vil_map(
        r_frame,
        title=f"Thresholded Threat Mask (VIL >= {threshold_val:.2f}) {title_suffix}",
        contrast_enhance=contrast_mode
    )
    st.plotly_chart(map_fig, use_container_width=True)
