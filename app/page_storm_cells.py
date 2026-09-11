import os
import sys
import textwrap
import numpy as np
import pandas as pd
import streamlit as st

from app.map_utils import create_vil_map, find_relative_storm_center
from app.storm_tracker import detect_storm_cells


def render_storm_cells_page(pred_frames, contrast_mode=False):
    """Renders the dedicated Storm Cells analytical workspace."""
    st.markdown(
        textwrap.dedent("""
        <div style="margin-bottom: 8px;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #00f0ff; font-family: 'JetBrains Mono', monospace; text-transform: uppercase;">
                STORM CELL DIAGNOSTICS & CENTROID TRACKING
            </div>
            <div style="font-size: 0.70rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
                Connected-component convective core detection, relative centroid migration, and core progression table.
            </div>
        </div>
        """).strip(),
        unsafe_allow_html=True
    )

    cell_horizon = st.slider(
        "Forecast Horizon for Cell Diagnostics",
        min_value=5,
        max_value=60,
        value=15,
        step=5,
        format="+%d min"
    )

    step_idx = (cell_horizon // 5) - 1
    analyzed_frame = pred_frames[step_idx]

    detected = detect_storm_cells(analyzed_frame, threshold=0.10, min_area=6)
    if not detected:
        detected = detect_storm_cells(analyzed_frame, threshold=0.03, min_area=4)

    c_left, c_right = st.columns([1.35, 1.45], gap="small")

    with c_left:
        history_rows = []
        full_trajectory = []
        for s in range(12):
            m = (s + 1) * 5
            f = pred_frames[s]
            cells_s = detect_storm_cells(f, threshold=0.04, min_area=6)
            center = find_relative_storm_center(f, percentile=90)
            if center is not None:
                full_trajectory.append(center)

            top_cell = cells_s[0] if len(cells_s) > 0 else None
            rel_cx = f"({center[0] - 64.0:+.1f}, {64.0 - center[1]:+.1f})" if center else "N/A"
            history_rows.append({
                "Horizon": f"+{m}m",
                "Cores": len(cells_s),
                "Centroid (X, Y)": rel_cx,
                "Peak VIL": f"{float(f.max()):.3f}",
                "Mean VIL": f"{float(f.mean()):.3f}",
                "Core Area (px)": top_cell["area"] if top_cell else 0
            })

        st.markdown(f"**Detected Cores at +{cell_horizon} min:**")
        if detected:
            for i, c in enumerate(detected[:3]):
                rel_x = c["centroid_x"] - 64.0
                rel_y = 64.0 - c["centroid_y"]
                st.markdown(
                    textwrap.dedent(f"""
                    <div class="core-box">
                        <div class="core-header">
                            <span style="color: #00f0ff; font-family: 'JetBrains Mono', monospace; font-weight: bold; font-size: 0.74rem;">
                                CONVECTIVE CORE #{i + 1}
                            </span>
                            <span style="background: rgba(0, 240, 255, 0.15); color: #00f0ff; font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; padding: 2px 6px; border-radius: 2px;">
                                Area: {c['area']} px
                            </span>
                        </div>
                        <div class="core-grid-2x2">
                            <div class="core-stat">
                                <span class="core-stat-lbl">SEVIR GRID POS</span>
                                <div class="core-stat-val" style="font-size: 0.74rem;">(X: {rel_x:+.1f}, Y: {rel_y:+.1f})</div>
                            </div>
                            <div class="core-stat">
                                <span class="core-stat-lbl">PEAK VIL</span>
                                <div class="core-stat-val" style="color: #c084fc; font-size: 0.74rem;">{c['max_vil']:.3f}</div>
                            </div>
                            <div class="core-stat">
                                <span class="core-stat-lbl">MEAN VIL</span>
                                <div class="core-stat-val" style="color: #34d399; font-size: 0.74rem;">{c['mean_vil']:.3f}</div>
                            </div>
                            <div class="core-stat">
                                <span class="core-stat-lbl">ECHO TOP HEIGHT</span>
                                <div class="core-stat-val" style="color: #64748b; font-size: 0.68rem;">Not available</div>
                            </div>
                        </div>
                    </div>
                    """).strip(),
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                textwrap.dedent("""
                <div class="core-box" style="text-align: center; padding: 16px;">
                    <div style="font-size: 0.72rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">
                        No storm cells crossed the convective threshold.
                    </div>
                </div>
                """).strip(),
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        st.markdown("**Complete 60-Minute Forecast Progression:**")
        df_hist = pd.DataFrame(history_rows)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

    with c_right:
        cell_fig = create_vil_map(
            analyzed_frame,
            title=f"Detected Cells & Trajectory at +{cell_horizon}m",
            trajectory=full_trajectory,
            storm_cells=detected,
            contrast_enhance=contrast_mode
        )
        st.plotly_chart(cell_fig, use_container_width=True)
