import os
import sys
import textwrap
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from app.map_utils import create_vil_map, find_relative_storm_center
from app.storm_tracker import detect_storm_cells
from app.multihazard_fusion import build_multihazard_summary
from app.ui_theme import frame_to_base64_png


def render_forecast_analysis_page(
    past_frames,
    pred_frames,
    actual_future_frames,
    event_id=None,
    contrast_mode=False
):
    """
    Renders the Analytical Forecast Analysis Workspace.
    """
    st.markdown(
        textwrap.dedent("""
        <div style="margin-bottom: 8px;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #00f0ff; font-family: 'JetBrains Mono', monospace; text-transform: uppercase;">
                FORECAST ANALYSIS WORKSPACE
            </div>
            <div style="font-size: 0.70rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
                Historical context (-60m to 0m) → Autoregressive prediction (+5m to +60m) → Multi-horizon progression matrix.
            </div>
        </div>
        """).strip(),
        unsafe_allow_html=True
    )

    fa_col_ctrl, fa_col_map = st.columns([1.1, 1.9], gap="small")

    with fa_col_ctrl:
        selected_lead = st.select_slider(
            "Forecast Horizon",
            options=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
            value=15,
            format_func=lambda x: f"+{x} min"
        )
        step_idx = (selected_lead // 5) - 1
        sel_pred_frame = pred_frames[step_idx]
        sel_actual_frame = actual_future_frames[step_idx]

        mse_val = float(np.mean((sel_pred_frame - sel_actual_frame) ** 2))
        mae_val = float(np.mean(np.abs(sel_pred_frame - sel_actual_frame)))
        peak_pred = float(sel_pred_frame.max())
        mean_pred = float(sel_pred_frame.mean())
        peak_act = float(sel_actual_frame.max())

        st.markdown(
            textwrap.dedent(f"""
            <div class="stitch-card">
                <div class="stitch-card-title">
                    <span>HORIZON DIAGNOSTICS (+{selected_lead}m)</span>
                    <span style="color: #c084fc;">STEP {step_idx + 1}/12</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;">
                    <div class="core-stat">
                        <span class="core-stat-lbl">SAMPLE MSE</span>
                        <div class="core-stat-val" style="color: #00f0ff;">{mse_val:.6f}</div>
                    </div>
                    <div class="core-stat">
                        <span class="core-stat-lbl">SAMPLE MAE</span>
                        <div class="core-stat-val" style="color: #34d399;">{mae_val:.6f}</div>
                    </div>
                    <div class="core-stat">
                        <span class="core-stat-lbl">PEAK FORECAST</span>
                        <div class="core-stat-val" style="color: #c084fc;">{peak_pred:.3f}</div>
                    </div>
                    <div class="core-stat">
                        <span class="core-stat-lbl">PEAK ACTUAL</span>
                        <div class="core-stat-val" style="color: #f1f5f9;">{peak_act:.3f}</div>
                    </div>
                </div>
                <div style="margin-top: 6px; font-size: 0.60rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
                    Domain Mean Intensity: <strong style="color: #00f0ff;">{mean_pred:.4f}</strong>
                </div>
            </div>
            """).strip(),
            unsafe_allow_html=True
        )

        # Historical Context Preview (Last 3 observed frames)
        st.markdown(
            textwrap.dedent("""
            <div style="font-size: 0.65rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace; margin-bottom: 4px; text-transform: uppercase;">
                Recent Observed Context
            </div>
            """).strip(),
            unsafe_allow_html=True
        )
        ctx_cols = st.columns(3)
        for i, past_idx in enumerate([9, 10, 11]):
            mins_ago = (11 - past_idx) * 5
            lbl = "t0 (NOW)" if mins_ago == 0 else f"-{mins_ago}m"
            with ctx_cols[i]:
                st.markdown(
                    textwrap.dedent(f"""
                    <div style="text-align: center; font-size: 0.58rem; color: #00f0ff; font-family: 'JetBrains Mono', monospace; margin-bottom: 2px;">
                        {lbl}
                    </div>
                    <img src="{frame_to_base64_png(past_frames[past_idx], 'turbo', contrast=contrast_mode)}" style="width: 100%; border-radius: 3px; border: 1px solid var(--border-outline);" />
                    """).strip(),
                    unsafe_allow_html=True
                )

    with fa_col_map:
        cells_at_lead = detect_storm_cells(sel_pred_frame, threshold=0.10, min_area=6)
        if not cells_at_lead:
            cells_at_lead = detect_storm_cells(sel_pred_frame, threshold=0.03, min_area=4)

        map_fig = create_vil_map(
            sel_pred_frame,
            title=f"Residual V3 Nowcast (+{selected_lead} min horizon)",
            storm_cells=cells_at_lead,
            contrast_enhance=contrast_mode,
            observed_frame=past_frames[-1],
            view_mode="OVERLAY"
        )
        st.plotly_chart(map_fig, use_container_width=True)

    # ------------------------------------------------------------
    # Complete 12-Frame Forecast Progression Matrix (+5m to +60m)
    # ------------------------------------------------------------
    st.markdown(
        textwrap.dedent("""
        <div style="margin-top: 8px; margin-bottom: 6px; font-size: 0.72rem; font-weight: bold; color: #00f0ff; font-family: 'JetBrains Mono', monospace; text-transform: uppercase;">
            Autoregressive 60-Minute Evolution Progression
        </div>
        """).strip(),
        unsafe_allow_html=True
    )

    prog_cols_top = st.columns(6)
    for s in range(6):
        lead_m = (s + 1) * 5
        with prog_cols_top[s]:
            st.markdown(
                textwrap.dedent(f"""
                <div style="text-align: center; font-size: 0.60rem; color: #c084fc; font-family: 'JetBrains Mono', monospace; margin-bottom: 2px;">
                    +{lead_m} MIN
                </div>
                <img src="{frame_to_base64_png(pred_frames[s], 'turbo', contrast=contrast_mode)}" style="width: 100%; border-radius: 3px; border: 1px solid var(--border-outline);" />
                """).strip(),
                unsafe_allow_html=True
            )

    prog_cols_bot = st.columns(6)
    for s in range(6, 12):
        lead_m = (s + 1) * 5
        with prog_cols_bot[s - 6]:
            st.markdown(
                textwrap.dedent(f"""
                <div style="text-align: center; font-size: 0.60rem; color: #c084fc; font-family: 'JetBrains Mono', monospace; margin-bottom: 2px;">
                    +{lead_m} MIN
                </div>
                <img src="{frame_to_base64_png(pred_frames[s], 'turbo', contrast=contrast_mode)}" style="width: 100%; border-radius: 3px; border: 1px solid var(--border-outline);" />
                """).strip(),
                unsafe_allow_html=True
            )

    # ------------------------------------------------------------
    # Observed GLM Lightning Context
    # ------------------------------------------------------------
    if event_id:
        try:
            lightning_summary = build_multihazard_summary(event_id)

            total_lightning = int(
                lightning_summary["total_lightning_events"]
            )
            active_lightning = int(
                lightning_summary["active_lightning_frames"]
            )
            peak_lightning = int(
                lightning_summary["peak_lightning_frame_count"]
            )
            lightning_counts = lightning_summary["lightning_counts"]

            st.markdown(
                textwrap.dedent("""
                <div style="margin-top: 18px; margin-bottom: 6px;
                            font-size: 0.72rem; font-weight: bold;
                            color: #00f0ff;
                            font-family: 'JetBrains Mono', monospace;
                            text-transform: uppercase;">
                    Observed GLM Lightning Context
                </div>
                <div style="margin-bottom: 10px;
                            font-size: 0.60rem; color: #94a3b8;
                            font-family: 'JetBrains Mono', monospace;">
                    Complementary observed lightning activity.
                    Lightning is not predicted by the current model.
                </div>
                """).strip(),
                unsafe_allow_html=True
            )

            lc1, lc2, lc3 = st.columns(3)

            with lc1:
                st.metric(
                    "TOTAL LIGHTNING EVENTS",
                    total_lightning
                )

            with lc2:
                st.metric(
                    "ACTIVE 5-MIN FRAMES",
                    active_lightning
                )

            with lc3:
                st.metric(
                    "PEAK / 5 MIN",
                    peak_lightning
                )

            # Keep observed lightning context aligned with the
            # current StormSense 60-minute forecast horizon.
            lightning_counts_60 = lightning_counts[:13]

            timeline_labels = [
                f"+{i * 5}m"
                for i in range(len(lightning_counts_60))
            ]

            fig_lightning = go.Figure()

            fig_lightning.add_trace(
                go.Scatter(
                    x=timeline_labels,
                    y=lightning_counts_60,
                    mode="lines+markers",
                    name="Observed GLM Lightning",
                    line=dict(width=2),
                    marker=dict(size=5)
                )
            )

            fig_lightning.update_layout(
                height=260,
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10
                ),
                xaxis_title="Time",
                yaxis_title="Lightning Events / 5 min",
                template="plotly_dark",
                showlegend=False
            )

            st.plotly_chart(
                fig_lightning,
                use_container_width=True
            )

            # --------------------------------------------------------
            # Geographic Lightning Detection
            # --------------------------------------------------------
            st.markdown(
                textwrap.dedent("""
                <div style="margin-top: 16px; margin-bottom: 6px;
                            font-size: 0.72rem; font-weight: bold;
                            color: #00f0ff;
                            font-family: 'JetBrains Mono', monospace;
                            text-transform: uppercase;">
                    Lightning Detection Locations
                </div>
                <div style="margin-bottom: 10px;
                            font-size: 0.60rem; color: #94a3b8;
                            font-family: 'JetBrains Mono', monospace;">
                    Geographic locations of observed GLM lightning events
                    for the selected 60-minute analysis window.
                </div>
                """).strip(),
                unsafe_allow_html=True
            )

            from app.lightning_sevir import load_lightning_event

            lightning_result = load_lightning_event(event_id)
            lightning_points = lightning_result["lightning"]

            # Display only the same 0–60 minute window as the VIL forecast.
            lightning_points_60 = lightning_points[
                (lightning_points[:, 0] >= 0)
                & (lightning_points[:, 0] <= 3600)
            ]

            if len(lightning_points_60) > 0:
                lightning_df = pd.DataFrame({
                    "time_min": lightning_points_60[:, 0] / 60.0,
                    "latitude": lightning_points_60[:, 1],
                    "longitude": lightning_points_60[:, 2],
                })

                fig_geo_lightning = go.Figure()

                fig_geo_lightning.add_trace(
                    go.Scattergeo(
                        lon=lightning_df["longitude"],
                        lat=lightning_df["latitude"],
                        mode="markers",
                        marker=dict(
                            size=7,
                            opacity=0.75
                        ),
                        text=[
                            f"+{t:.1f} min"
                            for t in lightning_df["time_min"]
                        ],
                        hovertemplate=(
                            "Lightning detection"
                            "<br>Time: %{text}"
                            "<br>Lat: %{lat:.4f}"
                            "<br>Lon: %{lon:.4f}"
                            "<extra></extra>"
                        ),
                        name="Observed GLM Lightning"
                    )
                )

                fig_geo_lightning.update_geos(
                    showcountries=True,
                    showcoastlines=True,
                    showland=True,
                    fitbounds="locations"
                )

                fig_geo_lightning.update_layout(
                    height=430,
                    margin=dict(
                        l=10,
                        r=10,
                        t=20,
                        b=10
                    ),
                    template="plotly_dark",
                    showlegend=False
                )

                st.plotly_chart(
                    fig_geo_lightning,
                    use_container_width=True
                )

                st.caption(
                    f"{len(lightning_points_60)} observed GLM "
                    f"lightning detections displayed."
                )
            else:
                st.info(
                    "No observed GLM lightning detections were "
                    "recorded in the 0–60 minute analysis window."
                )

        except Exception as exc:
            st.info(
                f"Observed lightning context unavailable "
                f"for event {event_id}: {exc}"
            )
