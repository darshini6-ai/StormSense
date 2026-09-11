import os
import sys
import textwrap
import numpy as np
import pandas as pd
import streamlit as st
from app.ui_theme import frame_to_base64_png


def render_backtest_page(past_frames, pred_frames, actual_future_frames, bench_data, contrast_mode=False):
    """Renders the Backtest & Persistence Comparison validation page."""
    st.markdown(
        textwrap.dedent("""
        <div style="margin-bottom: 8px;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #00f0ff; font-family: 'JetBrains Mono', monospace; text-transform: uppercase;">
                MULTI-HORIZON BACKTEST & BENCHMARK VALIDATION
            </div>
            <div style="font-size: 0.70rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
                Comparative evaluation against ground-truth SEVIR observations and persistence baseline across 12 lead times.
            </div>
        </div>
        """).strip(),
        unsafe_allow_html=True
    )

    st.markdown(
        textwrap.dedent("""
        <div class="stitch-card" style="padding: 8px 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="background: rgba(192, 132, 252, 0.15); border: 1px solid #c084fc; color: #c084fc; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; font-weight: bold; padding: 2px 6px; border-radius: 3px;">
                        EVENT-DISJOINT EVALUATION
                    </span>
                    <span style="font-weight: 600; color: #e2e8f0; font-size: 0.78rem; font-family: 'JetBrains Mono', monospace;">
                        595 train | 128 validation | 128 test | 3,328 test windows
                    </span>
                </div>
                <div style="font-size: 0.70rem; color: #34d399; font-weight: bold; font-family: 'JetBrains Mono', monospace;">
                    ✓ Residual V3 outperforms persistence across all lead times (+5m to +60m)
                </div>
            </div>
        </div>
        """).strip(),
        unsafe_allow_html=True
    )

    b_kpis = st.columns(3)
    with b_kpis[0]:
        st.markdown(
            textwrap.dedent(f"""
            <div class="stitch-card" style="text-align: center;">
                <div style="font-size: 0.58rem; color: #c084fc; text-transform: uppercase; font-family: 'JetBrains Mono', monospace; font-weight: bold;">
                    Residual V3 MSE
                </div>
                <div style="font-size: 1.5rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #c084fc;">
                    {bench_data['overall_v3']:.6f}
                </div>
                <div style="font-size: 0.58rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
                    3,328 Held-Out Test Windows
                </div>
            </div>
            """).strip(),
            unsafe_allow_html=True
        )
    with b_kpis[1]:
        st.markdown(
            textwrap.dedent(f"""
            <div class="stitch-card" style="text-align: center;">
                <div style="font-size: 0.58rem; color: #64748b; text-transform: uppercase; font-family: 'JetBrains Mono', monospace; font-weight: bold;">
                    Persistence MSE
                </div>
                <div style="font-size: 1.5rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #94a3b8;">
                    {bench_data['overall_persistence']:.6f}
                </div>
                <div style="font-size: 0.58rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">
                    Meteorological Baseline
                </div>
            </div>
            """).strip(),
            unsafe_allow_html=True
        )
    with b_kpis[2]:
        st.markdown(
            textwrap.dedent(f"""
            <div class="stitch-card" style="text-align: center;">
                <div style="font-size: 0.58rem; color: #34d399; text-transform: uppercase; font-family: 'JetBrains Mono', monospace; font-weight: bold;">
                    Overall Improvement
                </div>
                <div style="font-size: 1.5rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #34d399;">
                    +{bench_data['improvement']:.2f}%
                </div>
                <div style="font-size: 0.58rem; color: #34d399; font-family: 'JetBrains Mono', monospace; font-weight: bold;">
                    Lower Mean Squared Error
                </div>
            </div>
            """).strip(),
            unsafe_allow_html=True
        )

    b_horizon = st.select_slider(
        "Select Lead Time Horizon to Inspect",
        options=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        value=15,
        format_func=lambda x: f"+{x} min"
    )

    b_idx = (b_horizon // 5) - 1
    p_frame = pred_frames[b_idx]
    a_frame = actual_future_frames[b_idx]
    persistence_frame = past_frames[-1]

    sample_mse = float(np.mean((p_frame - a_frame) ** 2))
    pers_sample_mse = float(np.mean((persistence_frame - a_frame) ** 2))

    v3_h_mse = bench_data["v3_mse"][b_idx]
    pers_h_mse = bench_data["persistence_mse"][b_idx]
    h_gain = ((pers_h_mse - v3_h_mse) / pers_h_mse) * 100.0

    st.markdown(
        textwrap.dedent(f"""
        <div style="margin: 4px 0 8px 0; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; color: #94a3b8;">
            At <strong>+{b_horizon} min</strong> horizon:
            Benchmark V3 MSE = <span style="color: #c084fc; font-weight: bold;">{v3_h_mse:.6f}</span> vs
            Persistence = <span style="color: #64748b; font-weight: bold;">{pers_h_mse:.6f}</span>
            (<span style="color: #34d399; font-weight: bold;">+{h_gain:.2f}% gain</span>) |
            Event Live MSE: <span style="color: #c084fc;">{sample_mse:.6f}</span> vs Persistence: <span style="color: #64748b;">{pers_sample_mse:.6f}</span>
        </div>
        """).strip(),
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            textwrap.dedent(f"""
            <div style="text-align: center; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #94a3b8; margin-bottom: 2px;">
                Persistence Baseline (t0)
            </div>
            <img src="{frame_to_base64_png(persistence_frame, 'turbo', contrast=contrast_mode)}" style="width: 100%; border-radius: 3px; border: 1px solid var(--border-outline);" />
            """).strip(),
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            textwrap.dedent(f"""
            <div style="text-align: center; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #c084fc; margin-bottom: 2px;">
                Residual V3 Forecast (+{b_horizon}m)
            </div>
            <img src="{frame_to_base64_png(p_frame, 'turbo', contrast=contrast_mode)}" style="width: 100%; border-radius: 3px; border: 1px solid #c084fc;" />
            """).strip(),
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            textwrap.dedent(f"""
            <div style="text-align: center; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #34d399; margin-bottom: 2px;">
                Actual SEVIR Observation (+{b_horizon}m)
            </div>
            <img src="{frame_to_base64_png(a_frame, 'turbo', contrast=contrast_mode)}" style="width: 100%; border-radius: 3px; border: 1px solid #34d399;" />
            """).strip(),
            unsafe_allow_html=True
        )
