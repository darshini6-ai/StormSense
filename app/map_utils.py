import numpy as np
import plotly.graph_objects as go


def create_vil_map(
    frame,
    title="StormSense VIL Forecast",
    bbox=None,
    trajectory=None,
    storm_cells=None,
    contrast_enhance=False,
    observed_frame=None,
    view_mode="OVERLAY",
    show_obs=True,
    show_pred=True,
    show_cores=True,
    show_vectors=True
):
    """
    Create a tactical radar canvas for normalized SEVIR VIL fields.

    Coordinates use relative SEVIR model grid coordinates [-64 to +64].
    """
    height, width = frame.shape
    disp_frame = np.clip(frame, 0.0, 1.0)
    if contrast_enhance:
        disp_frame = np.sqrt(disp_frame)

    # Relative SEVIR grid coordinates centered at (64, 64)
    x = np.arange(width) - 64.0
    y = 64.0 - np.arange(height)

    fig = go.Figure()

    # Base Heatmap / Differential
    if view_mode == "RESIDUAL" and observed_frame is not None:
        obs_disp = np.clip(observed_frame, 0.0, 1.0)
        diff = disp_frame - obs_disp
        fig.add_trace(
            go.Heatmap(
                z=diff,
                x=x,
                y=y,
                colorscale="RdBu_r",
                zmid=0,
                zmin=-0.5,
                zmax=0.5,
                colorbar=dict(
                    title=dict(
                        text="Δ VIL (Pred - Obs)",
                        font=dict(family="JetBrains Mono", size=10, color="#849495")
                    ),
                    thickness=12,
                    len=0.75,
                    tickfont=dict(family="JetBrains Mono", size=10, color="#849495")
                ),
                hovertemplate="Grid X: %{x:+.1f}<br>Grid Y: %{y:+.1f}<br>Δ VIL: %{z:+.3f}<extra></extra>"
            )
        )
    else:
        fig.add_trace(
            go.Heatmap(
                z=disp_frame,
                x=x,
                y=y,
                colorscale="Turbo",
                zmin=0.0,
                zmax=1.0,
                colorbar=dict(
                    title=dict(
                        text="VIL Intensity [0-1]",
                        font=dict(family="JetBrains Mono", size=10, color="#849495")
                    ),
                    thickness=12,
                    len=0.75,
                    tickfont=dict(family="JetBrains Mono", size=10, color="#849495")
                ),
                hovertemplate="Grid X: %{x:+.1f}<br>Grid Y: %{y:+.1f}<br>VIL: %{z:.3f}<extra></extra>"
            )
        )

    # Tactical concentric range rings across relative domain
    ring_radii = [16, 32, 48, 64]
    for r in ring_radii:
        theta = np.linspace(0, 2 * np.pi, 120)
        rx = r * np.cos(theta)
        ry = r * np.sin(theta)
        fig.add_trace(
            go.Scatter(
                x=rx,
                y=ry,
                mode="lines",
                line=dict(color="rgba(0, 240, 255, 0.22)", width=1, dash="dot"),
                hoverinfo="skip",
                showlegend=False
            )
        )

    # Tactical Crosshairs (N-S, E-W axes)
    fig.add_shape(type="line", x0=-64, y0=0, x1=64, y1=0, line=dict(color="rgba(59, 73, 75, 0.8)", width=1, dash="dash"))
    fig.add_shape(type="line", x0=0, y0=-64, x1=0, y1=64, line=dict(color="rgba(59, 73, 75, 0.8)", width=1, dash="dash"))

    # Cardinal direction indicators
    fig.add_annotation(x=0, y=61, text="N (000°)", showarrow=False, font=dict(color="#00f0ff", size=10, family="JetBrains Mono"))
    fig.add_annotation(x=61, y=0, text="E (090°)", showarrow=False, font=dict(color="#849495", size=9, family="JetBrains Mono"))
    fig.add_annotation(x=0, y=-61, text="S (180°)", showarrow=False, font=dict(color="#849495", size=9, family="JetBrains Mono"))
    fig.add_annotation(x=-61, y=0, text="W (270°)", showarrow=False, font=dict(color="#849495", size=9, family="JetBrains Mono"))

    # Storm trajectory overlay
    if trajectory and show_vectors and len(trajectory) > 1:
        traj_arr = np.asarray(trajectory)
        tx = traj_arr[:, 0] - 64.0
        ty = 64.0 - traj_arr[:, 1]
        fig.add_trace(
            go.Scatter(
                x=tx,
                y=ty,
                mode="lines+markers",
                name="Centroid Migration",
                line=dict(color="#ffb95f", width=2.5),
                marker=dict(size=6, color="#ffb95f", symbol="circle"),
                hovertemplate="Migrated Step: (X: %{x:+.1f}, Y: %{y:+.1f})<extra></extra>"
            )
        )

    # Detected convective storm cells overlay
    if storm_cells and show_cores:
        for idx, cell in enumerate(storm_cells):
            cx = cell["centroid_x"] - 64.0
            cy = 64.0 - cell["centroid_y"]
            r_box = max(3.0, np.sqrt(cell.get("area", 16)) / 1.5)

            # Bounding circular marker for convective core
            theta = np.linspace(0, 2 * np.pi, 30)
            cx_ring = cx + r_box * np.cos(theta)
            cy_ring = cy + r_box * np.sin(theta)
            fig.add_trace(
                go.Scatter(
                    x=cx_ring,
                    y=cy_ring,
                    mode="lines",
                    line=dict(color="#ef4444" if idx == 0 else "#00f0ff", width=1.5),
                    fill="toself",
                    fillcolor="rgba(239, 68, 68, 0.15)" if idx == 0 else "rgba(0, 240, 255, 0.10)",
                    name=f"Core #{cell.get('id', idx+1)}",
                    hoverinfo="skip",
                    showlegend=False
                )
            )
            # Centroid point marker
            fig.add_trace(
                go.Scatter(
                    x=[cx],
                    y=[cy],
                    mode="markers+text",
                    marker=dict(size=7, color="#ef4444" if idx == 0 else "#00f0ff", symbol="cross"),
                    text=[f"#{cell.get('id', idx+1)}"],
                    textposition="top right",
                    textfont=dict(color="#e1e2ec", size=10, family="JetBrains Mono"),
                    name=f"Cell #{cell.get('id', idx+1)}",
                    hovertemplate=f"<b>Core #{cell.get('id', idx+1)}</b><br>Grid: (X: %{{x:+.1f}}, Y: %{{y:+.1f}})<br>Peak VIL: {cell.get('max_vil', 0):.3f}<br>Area: {cell.get('area', 0)} px<extra></extra>"
                )
            )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(family="JetBrains Mono, sans-serif", size=13, color="#e1e2ec"),
            x=0.02,
            y=0.96
        ),
        paper_bgcolor="#0b0e15",
        plot_bgcolor="#0b0e15",
        font=dict(family="JetBrains Mono, Inter, sans-serif", color="#e1e2ec"),
        xaxis=dict(
            title="SEVIR Grid X (relative)",
            gridcolor="#191b23",
            zerolinecolor="#3b494b",
            range=[-64, 64],
            constrain="domain",
            tickfont=dict(size=10, family="JetBrains Mono")
        ),
        yaxis=dict(
            title="SEVIR Grid Y (relative)",
            gridcolor="#191b23",
            zerolinecolor="#3b494b",
            range=[-64, 64],
            scaleanchor="x",
            scaleratio=1,
            tickfont=dict(size=10, family="JetBrains Mono")
        ),
        height=540,
        margin=dict(l=25, r=25, t=45, b=25),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=10, color="#849495", family="JetBrains Mono"),
            bgcolor="rgba(11, 14, 21, 0.75)"
        )
    )

    return fig


def find_relative_storm_center(frame, percentile=90):
    """
    Find the center of the strongest relative spatial region.
    This is a diagnostic metric derived from available model data.
    """
    threshold = np.percentile(frame, percentile)
    mask = frame >= threshold
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())
