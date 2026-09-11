import numpy as np
import plotly.graph_objects as go


def create_vil_map(
    frame,
    title="StormSense VIL Forecast",
    bbox=None,
    trajectory=None
):
    """
    Create an interactive VIL map.

    frame:
        2D numpy array containing VIL.

    bbox:
        Optional geographic bounding box:
        (min_lat, min_lon, max_lat, max_lon)

    trajectory:
        Optional list of (x, y) grid coordinates.
    """

    height, width = frame.shape

    # --------------------------------------------------------
    # Grid coordinates
    # --------------------------------------------------------

    if bbox is not None:

        min_lat, min_lon, max_lat, max_lon = bbox

        x = np.linspace(
            min_lon,
            max_lon,
            width
        )

        y = np.linspace(
            min_lat,
            max_lat,
            height
        )

        x_title = "Longitude"
        y_title = "Latitude"

    else:

        x = np.arange(width)
        y = np.arange(height)

        x_title = "Grid X"
        y_title = "Grid Y"

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=frame,
            x=x,
            y=y,
            colorscale="Turbo",
            zmin=0,
            zmax=1,
            colorbar=dict(
                title="Normalized VIL"
            ),
            hovertemplate=(
                "X/Longitude: %{x:.3f}<br>"
                "Y/Latitude: %{y:.3f}<br>"
                "VIL: %{z:.3f}"
                "<extra></extra>"
            )
        )
    )

    # --------------------------------------------------------
    # Storm trajectory
    # --------------------------------------------------------

    if trajectory:

        trajectory = np.asarray(
            trajectory
        )

        if trajectory.ndim == 2 and len(trajectory) > 0:

            tx = trajectory[:, 0]
            ty = trajectory[:, 1]

            if bbox is not None:

                tx = (
                    min_lon
                    + tx / (width - 1)
                    * (max_lon - min_lon)
                )

                ty = (
                    min_lat
                    + ty / (height - 1)
                    * (max_lat - min_lat)
                )

            fig.add_trace(
                go.Scatter(
                    x=tx,
                    y=ty,
                    mode="lines+markers",
                    name="Storm trajectory",
                    line=dict(
                        width=4
                    ),
                    marker=dict(
                        size=9
                    ),
                    hovertemplate=(
                        "Position: "
                        "(%{x:.3f}, %{y:.3f})"
                        "<extra></extra>"
                    )
                )
            )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=650,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    return fig


def find_relative_storm_center(
    frame,
    percentile=90
):
    """
    Find the center of the strongest relative
    spatial region.

    This is a visualization diagnostic and is
    NOT a calibrated severe-weather threshold.
    """

    threshold = np.percentile(
        frame,
        percentile
    )

    mask = frame >= threshold

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    center_x = float(xs.mean())
    center_y = float(ys.mean())

    return center_x, center_y
