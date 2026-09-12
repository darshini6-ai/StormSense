import os
import sys
import textwrap
import numpy as np
import streamlit as st

from app.map_utils import create_vil_map, find_relative_storm_center
from app.storm_tracker import detect_storm_cells
from app.ui_theme import (
    frame_to_base64_png,
    render_cell_diagnostics_panel,
    render_benchmark_performance_panel,
    render_scientific_integrity_panel
)
from app.experimental_rollout import render_experimental_badge


def render_forecast_page(event_id, past_frames, pred_frames, actual_future_frames, contrast_mode=False):
    """
    Renders the Primary Command Console Hero Page synchronized with the global timeline.
    """
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "OVERLAY"
    if "show_obs_layer" not in st.session_state:
        st.session_state.show_obs_layer = True
    if "show_pred_layer" not in st.session_state:
        st.session_state.show_pred_layer = True
    if "show_cores_layer" not in st.session_state:
        st.session_state.show_cores_layer = True
    if "show_vectors_layer" not in st.session_state:
        st.session_state.show_vectors_layer = True

    sel_mins = int(st.session_state.get("timeline_minutes", 0))
    total_pred = len(pred_frames)
    max_mins = total_pred * 5

    # Determine current displayed frame
    observed_t0 = past_frames[-1]
    if sel_mins == 0:
        current_frame = observed_t0
        frame_title = f"Observed SEVIR VIL — Event {event_id} (t0 / NOW)"
    else:
        frame_idx = min((sel_mins // 5) - 1, total_pred - 1)
        current_frame = pred_frames[frame_idx]
        if sel_mins <= 60:
            frame_title = f"Residual V3 Forecast (+{sel_mins}m)"
        else:
            frame_title = f"Experimental Recursive Rollout (+{sel_mins}m) [UNVALIDATED]"

    # Compute trajectory from relative centroid tracking across horizons
    trajectory = []
    past_center = find_relative_storm_center(observed_t0, percentile=90)
    if past_center is not None:
        trajectory.append(past_center)

    if sel_mins > 0:
        num_steps = min(sel_mins // 5, total_pred)
        for s in range(num_steps):
            c = find_relative_storm_center(pred_frames[s], percentile=90)
            if c is not None:
                trajectory.append(c)

    # Detect convective cores using backend tracker
    detected_cells = detect_storm_cells(current_frame, threshold=0.15, min_area=8)
    if not detected_cells:
        detected_cells = detect_storm_cells(current_frame, threshold=0.04, min_area=6)

    # Actual ground truth frame for comparison if horizon > 0
    if sel_mins > 0:
        val_idx = min((sel_mins // 5) - 1, len(actual_future_frames) - 1)
        actual_comp_frame = actual_future_frames[val_idx]
    else:
        actual_comp_frame = observed_t0

    # ------------------------------------------------------------
    # MAIN 2-COLUMN OPERATIONAL WORKSPACE (RADAR AS HERO ~70%)
    # ------------------------------------------------------------
    col_radar, col_intel = st.columns([1.95, 1.05], gap="small")

    # ============================================================
    # LEFT / CENTER: HERO RADAR WORKSTATION
    # ============================================================
    with col_radar:
        # Layer Legend Toolbar
        tb_col1, tb_col2 = st.columns([2.1, 0.9])

        with tb_col1:
            st.markdown(
                textwrap.dedent(f"""
                <div style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center; background: var(--surface-low); padding: 5px 10px; border-radius: 4px; border: 1px solid var(--border-outline); font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;">
                    <span style="display: flex; align-items: center; gap: 4px; color: #00f0ff; font-weight: 600;">
                        <span style="width: 6px; height: 6px; border-radius: 50%; background: #00f0ff;"></span>
                        OBSERVED (t0)
                    </span>
                    <span style="color: var(--surface-highest);">|</span>
                    <span style="display: flex; align-items: center; gap: 4px; color: #c084fc; font-weight: 600;">
                        <span style="width: 6px; height: 6px; border-radius: 50%; background: #c084fc;"></span>
                        MODEL FORECAST (+{sel_mins}m)
                    </span>
                    <span style="color: var(--surface-highest);">|</span>
                    <span style="display: flex; align-items: center; gap: 4px; color: #34d399; font-weight: 600;">
                        <span style="width: 6px; height: 6px; border-radius: 50%; background: #34d399;"></span>
                        ACTUAL FUTURE
                    </span>
                    <span style="color: var(--surface-highest);">|</span>
                    <span style="display: flex; align-items: center; gap: 4px; color: #f87171;">
                        <span style="width: 6px; height: 6px; border-radius: 50%; background: #f87171;"></span>
                        CELL CORES
                    </span>
                </div>
                """).strip(),
                unsafe_allow_html=True
            )

        with tb_col2:
            v_mode = st.radio(
                "View Mode",
                options=["OVERLAY", "SPLIT", "RESIDUAL"],
                index=["OVERLAY", "SPLIT", "RESIDUAL"].index(st.session_state.view_mode),
                horizontal=True,
                label_visibility="collapsed"
            )
            st.session_state.view_mode = v_mode

        # Plotly Tactical Radar Map
        fig_radar = create_vil_map(
            current_frame,
            title=frame_title,
            trajectory=trajectory if len(trajectory) > 1 else None,
            storm_cells=detected_cells,
            contrast_enhance=contrast_mode,
            observed_frame=observed_t0,
            view_mode=st.session_state.view_mode,
            show_obs=st.session_state.show_obs_layer,
            show_pred=st.session_state.show_pred_layer,
            show_cores=st.session_state.show_cores_layer,
            show_vectors=st.session_state.show_vectors_layer
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})

        # Floating Tactical Reticle Telemetry Summary & Normalized VIL Legend
        primary_cell = detected_cells[0] if detected_cells else None
        if primary_cell:
            rel_x = primary_cell["centroid_x"] - 64.0
            rel_y = 64.0 - primary_cell["centroid_y"]
            dist_grid = float(np.sqrt(rel_x**2 + rel_y**2))
            angle_deg = (float(np.degrees(np.arctan2(rel_x, rel_y))) + 360) % 360
            coord_str = f"X: {rel_x:+.1f}, Y: {rel_y:+.1f}"
            range_str = f"Dist: {dist_grid:.1f} px @ {angle_deg:03.0f}°"
        else:
            coord_str = "X: 0.0, Y: 0.0"
            range_str = "Dist: 0.0 px @ 000°"

        st.markdown(
            textwrap.dedent(f"""
            <div style="display: grid; grid-template-columns: 1.4fr 1fr; gap: 8px; margin-top: -12px; margin-bottom: 8px;">
                <div style="background: var(--surface-low); border: 1px solid var(--border-outline); border-radius: 4px; padding: 5px 10px; font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: var(--text-outline);">RELATIVE GRID COORDS:</span>
                        <strong style="color: #00f0ff;">{coord_str}</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 2px;">
                        <span style="color: var(--text-outline);">OFFSET & AZIMUTH:</span>
                        <strong style="color: #e2e8f0;">{range_str}</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 2px;">
                        <span style="color: var(--text-outline);">ESTIMATED DISPLACEMENT:</span>
                        <span style="color: #34d399;">Derived from model forecast</span>
                    </div>
                </div>
                <div style="background: var(--surface-low); border: 1px solid var(--border-outline); border-radius: 4px; padding: 5px 10px; font-family: 'JetBrains Mono', monospace;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.58rem; color: var(--text-outline);">
                        <span>NORMALIZED VIL</span>
                        <span style="color: #00f0ff; font-weight: bold;">Normalized VIL [0,1]</span>
                    </div>
                    <div class="stitch-progress-bar" style="height: 6px; background: linear-gradient(90deg, #0b0e14 0%, #06b6d4 20%, #22c55e 40%, #eab308 60%, #f97316 70%, #ef4444 80%, #ec4899 90%, #ffffff 100%); margin: 3px 0;"></div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.55rem; color: var(--text-muted);">
                        <span>0.0</span><span>0.2</span><span>0.4</span><span>0.6</span><span>0.8</span><span style="color: #00f0ff; font-weight: bold;">1.0</span>
                    </div>
                </div>
            </div>
            """).strip(),
            unsafe_allow_html=True
        )

        # ------------------------------------------------------------
        # Validation / Horizon Status Indicator
        # ------------------------------------------------------------
        render_experimental_badge(sel_mins)

        # ------------------------------------------------------------
        # Side-by-Side Validation Strip: Observed vs Predicted vs Actual
        # ------------------------------------------------------------
        val_mins = sel_mins if sel_mins > 0 else 15
        val_idx = min((val_mins // 5) - 1, total_pred - 1)
        pred_val_frame = pred_frames[val_idx]
        actual_val_frame = actual_future_frames[val_idx] if len(actual_future_frames) > val_idx else actual_future_frames[-1]

        event_horizon_mse = float(np.mean((pred_val_frame - actual_val_frame) ** 2))
        event_horizon_mae = float(np.mean(np.abs(pred_val_frame - actual_val_frame)))

        st.markdown(
            textwrap.dedent(f"""
            <div style="background: var(--surface-low); border: 1px solid var(--border-outline); border-radius: 4px; padding: 6px 10px; margin-top: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #00f0ff; font-weight: bold; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;">
                        🔍 GROUND TRUTH COMPARISON (+{val_mins}m)
                    </span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #34d399; font-weight: bold;">
                        Sample Live MSE: {event_horizon_mse:.6f} | MAE: {event_horizon_mae:.6f}
                    </span>
                </div>
            </div>
            """).strip(),
            unsafe_allow_html=True
        )

        hv_col1, hv_col2, hv_col3 = st.columns(3, gap="small")
        with hv_col1:
            st.markdown(
                textwrap.dedent(f"""
                <div style="text-align: center; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #00f0ff; margin-bottom: 2px;">
                    Observed Context (t0)
                </div>
                <img src="{frame_to_base64_png(observed_t0, 'turbo', contrast=contrast_mode)}" style="width: 100%; border-radius: 3px; border: 1px solid var(--border-outline);" />
                """).strip(),
                unsafe_allow_html=True
            )
        with hv_col2:
            st.markdown(
                textwrap.dedent(f"""
                <div style="text-align: center; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #c084fc; margin-bottom: 2px;">
                    Forecast Output (+{val_mins}m)
                </div>
                <img src="{frame_to_base64_png(pred_val_frame, 'turbo', contrast=contrast_mode)}" style="width: 100%; border-radius: 3px; border: 1px solid #c084fc;" />
                """).strip(),
                unsafe_allow_html=True
            )
        with hv_col3:
            st.markdown(
                textwrap.dedent(f"""
                <div style="text-align: center; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #34d399; margin-bottom: 2px;">
                    Actual SEVIR Observation (+{val_mins}m)
                </div>
                <img src="{frame_to_base64_png(actual_val_frame, 'turbo', contrast=contrast_mode)}" style="width: 100%; border-radius: 3px; border: 1px solid #34d399;" />
                """).strip(),
                unsafe_allow_html=True
            )

    # ============================================================
    # RIGHT: TACTICAL INTELLIGENCE & TELEMETRY PANEL
    # ============================================================
    with col_intel:
        render_cell_diagnostics_panel(
            cells=detected_cells,
            pred_frame=current_frame,
            actual_frame=actual_comp_frame,
            selected_minutes=sel_mins
        )

        render_benchmark_performance_panel()

        render_scientific_integrity_panel()
