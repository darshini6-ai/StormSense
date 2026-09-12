# ============================================================
# StormSense - Global Forecast Playback Controller
# Synchronized Multi-Page Temporal Navigation
# ============================================================

import streamlit as st
from streamlit_autorefresh import st_autorefresh


# ============================================================
# Playback Configuration
# ============================================================

SPEED_INTERVALS = {
    1: 4000,   # 1x (4.0s per 5-min step)
    2: 2000,   # 2x (2.0s per 5-min step)
    4: 1000,   # 4x (1.0s per 5-min step)
}


# ============================================================
# Initialize Global Playback State
# ============================================================

def initialize_playback():
    """
    Initialize global StormSense timeline and playback session state.

    Timeline:
        T+00 = Observed ground truth at t0
        T+05 = +5 min prediction (Frame 1)
        ...
        T+60 = +60 min prediction (Frame 12)
        [Optional Experimental Rollout]:
        T+65 = +65 min prediction (Frame 13)
        ...
        T+120 = +120 min prediction (Frame 24)
    """
    if "timeline_minutes" not in st.session_state:
        st.session_state.timeline_minutes = 0

    if "playback_active" not in st.session_state:
        st.session_state.playback_active = False

    if "playback_speed" not in st.session_state:
        st.session_state.playback_speed = 1

    if "experimental_2hr_mode" not in st.session_state:
        st.session_state.experimental_2hr_mode = False

    if "global_timeline" not in st.session_state:
        st.session_state.global_timeline = st.session_state.timeline_minutes


# ============================================================
# Playback Helper Controls
# ============================================================

def get_max_minutes():
    """Return the current maximum horizon (60 or 120 mins)."""
    return 120 if st.session_state.get("experimental_2hr_mode", False) else 60


def timeline_changed():
    """Callback when global timeline slider is adjusted."""
    st.session_state.timeline_minutes = int(st.session_state.global_timeline)
    st.session_state.playback_active = False


def toggle_playback():
    """Toggle PLAY / PAUSE state."""
    st.session_state.playback_active = not st.session_state.playback_active


def reset_playback():
    """Reset playback to observed T+00 frame."""
    st.session_state.timeline_minutes = 0
    st.session_state.global_timeline = 0
    st.session_state.playback_active = False


def advance_frame():
    """Advance timeline by exactly one 5-minute timestep."""
    max_min = get_max_minutes()
    current = int(st.session_state.timeline_minutes)
    if current >= max_min:
        next_time = 0
    else:
        next_time = current + 5

    st.session_state.timeline_minutes = next_time
    st.session_state.global_timeline = next_time


# ============================================================
# Render Global Playback Controller
# ============================================================

def render_global_playback():
    """
    Renders the unified global StormSense forecast playback controller.
    """
    initialize_playback()
    max_min = get_max_minutes()

    # Ensure timeline stays in valid range
    if st.session_state.timeline_minutes > max_min:
        st.session_state.timeline_minutes = max_min

    # --------------------------------------------------------
    # Autorefresh loop when playback is active
    # --------------------------------------------------------
    if st.session_state.playback_active:
        st_autorefresh(
            interval=SPEED_INTERVALS.get(st.session_state.playback_speed, 2000),
            key="stormsense_global_autorefresh"
        )
        advance_frame()

    current_mins = int(st.session_state.timeline_minutes)
    current_frame = current_mins // 5
    total_frames = max_min // 5

    # --------------------------------------------------------
    # Container UI Layout
    # --------------------------------------------------------
    st.markdown(
        """
        <div class="stormsense-playback-container">
        """,
        unsafe_allow_html=True
    )

    col_btn_play, col_btn_reset, col_btn_prev, col_btn_next, col_speed, col_telemetry = st.columns(
        [1.1, 1.0, 0.8, 0.8, 1.1, 2.0],
        gap="small"
    )

    with col_btn_play:
        btn_label = "⏸ PAUSE" if st.session_state.playback_active else "▶ PLAY"
        btn_type = "primary" if st.session_state.playback_active else "secondary"
        if st.button(btn_label, use_container_width=True, key="btn_global_play", type=btn_type):
            toggle_playback()
            st.rerun()

    with col_btn_reset:
        if st.button("↺ RESET", use_container_width=True, key="btn_global_reset"):
            reset_playback()
            st.rerun()

    with col_btn_prev:
        if st.button("◀ -5m", use_container_width=True, key="btn_global_prev"):
            st.session_state.timeline_minutes = max(0, current_mins - 5)
            st.session_state.playback_active = False
            st.rerun()

    with col_btn_next:
        if st.button("+5m ▶", use_container_width=True, key="btn_global_next"):
            st.session_state.timeline_minutes = min(max_min, current_mins + 5)
            st.session_state.playback_active = False
            st.rerun()

    with col_speed:
        selected_speed = st.selectbox(
            "Playback Speed",
            options=[1, 2, 4],
            index=[1, 2, 4].index(st.session_state.playback_speed),
            format_func=lambda s: f"{s}× Speed",
            key="select_playback_speed",
            label_visibility="collapsed"
        )
        if selected_speed != st.session_state.playback_speed:
            st.session_state.playback_speed = selected_speed

    with col_telemetry:
        status_color = "#34d399" if st.session_state.playback_active else "#94a3b8"
        status_text = "PLAYING" if st.session_state.playback_active else "PAUSED"
        horizon_type = "VALIDATED (+60m)" if current_mins <= 60 else "⚠️ EXPERIMENTAL (+120m)"
        horizon_color = "#00f0ff" if current_mins <= 60 else "#fbbf24"

        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center;
                        background: var(--surface-low); border: 1px solid var(--border-outline);
                        border-radius: 4px; padding: 4px 10px; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;">
                <div>
                    <span style="color: var(--text-outline);">STATUS:</span>
                    <strong style="color: {status_color};">{status_text}</strong>
                </div>
                <div>
                    <span style="color: var(--text-outline);">HORIZON:</span>
                    <strong style="color: {horizon_color};">T+{current_mins:02d}m</strong>
                    <span style="color: var(--text-muted);">({current_frame}/{total_frames})</span>
                </div>
                <div>
                    <span style="color: {horizon_color}; font-size: 0.60rem;">{horizon_type}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # Timeline Slider
    # --------------------------------------------------------
    st.session_state.global_timeline = current_mins
    st.slider(
        "Global Forecast Timeline",
        min_value=0,
        max_value=max_min,
        step=5,
        value=current_mins,
        format="T+%d min",
        key="global_timeline",
        on_change=timeline_changed,
        label_visibility="collapsed"
    )

    # --------------------------------------------------------
    # Milestone Quick-Jump Markers
    # --------------------------------------------------------
    if max_min == 60:
        milestones = [
            ("T+00 (OBS)", 0),
            ("T+15m", 15),
            ("T+30m", 30),
            ("T+45m", 45),
            ("T+60m (HORIZON)", 60)
        ]
    else:
        milestones = [
            ("T+00", 0),
            ("T+30m", 30),
            ("T+60m [VAL]", 60),
            ("T+90m [EXP]", 90),
            ("T+120m [EXP]", 120)
        ]

    m_cols = st.columns(len(milestones))
    for idx, (label, m_val) in enumerate(milestones):
        with m_cols[idx]:
            is_active = (current_mins == m_val)
            if st.button(
                label,
                key=f"ms_btn_{m_val}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.timeline_minutes = m_val
                st.session_state.playback_active = False
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
