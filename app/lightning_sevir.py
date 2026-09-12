from __future__ import annotations

import os
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CATALOG_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "sevir",
    "CATALOG.csv",
)


def load_catalog() -> pd.DataFrame:
    """Load the SEVIR catalog."""

    if not os.path.exists(CATALOG_FILE):
        raise FileNotFoundError(
            f"SEVIR catalog not found: {CATALOG_FILE}"
        )

    return pd.read_csv(
        CATALOG_FILE,
        low_memory=False,
    )


def find_lightning_record(
    event_id: str,
) -> dict:
    """
    Find the lightning record and spatial metadata
    for one SEVIR event.
    """

    catalog = load_catalog()

    rows = catalog[
        (catalog["id"].astype(str) == str(event_id))
        & (
            catalog["img_type"]
            .astype(str)
            .str.lower()
            == "lght"
        )
    ]

    if rows.empty:
        raise KeyError(
            f"No lightning record found for event {event_id}"
        )

    row = rows.iloc[0]

    return {
        "event_id": str(row["id"]),
        "file_name": str(row["file_name"]),
        "file_index": int(row["file_index"]),
        "time_utc": str(row["time_utc"]),
        "projection": str(row["proj"]),
        "width_m": float(row["width_m"]),
        "height_m": float(row["height_m"]),
        "llcrnrlat": float(row["llcrnrlat"]),
        "llcrnrlon": float(row["llcrnrlon"]),
        "urcrnrlat": float(row["urcrnrlat"]),
        "urcrnrlon": float(row["urcrnrlon"]),
    }


def load_lightning_event(
    event_id: str,
) -> Dict:
    """
    Load raw SEVIR GLM lightning data.

    Lightning columns:
        0 = time offset in seconds
        1 = latitude
        2 = longitude
        3 = x coordinate supplied by SEVIR
        4 = y coordinate supplied by SEVIR
    """

    record = find_lightning_record(
        event_id
    )

    file_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "sevir",
        record["file_name"],
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Lightning file not found: {file_path}"
        )

    with h5py.File(file_path, "r") as f:

        if record["event_id"] not in f:
            raise KeyError(
                f"Event {record['event_id']} not found in "
                f"{file_path}"
            )

        lightning = np.asarray(
            f[record["event_id"]][:]
        ).astype(np.float32)

    if lightning.ndim != 2 or lightning.shape[1] != 5:
        raise ValueError(
            f"Unexpected lightning shape for "
            f"{event_id}: {lightning.shape}"
        )

    return {
        **record,
        "lightning": lightning,
    }


def bin_lightning_to_5min(
    lightning: np.ndarray,
    num_frames: int = 49,
    frame_seconds: int = 300,
) -> List[Dict]:
    """
    Convert raw lightning events into 5-minute bins.
    """

    if lightning.ndim != 2 or lightning.shape[1] != 5:
        raise ValueError(
            f"Expected lightning shape (N,5), "
            f"got {lightning.shape}"
        )

    frames = []

    for frame in range(num_frames):

        start = frame * frame_seconds
        end = start + frame_seconds

        mask = (
            (lightning[:, 0] >= start)
            & (lightning[:, 0] < end)
        )

        points = lightning[mask]

        frames.append(
            {
                "frame": frame,
                "start_seconds": start,
                "end_seconds": end,
                "count": int(len(points)),
                "points": points,
            }
        )

    return frames


def lightning_to_model_grid(
    lightning: np.ndarray,
    projection: str,
    llcrnrlat: float,
    llcrnrlon: float,
    urcrnrlat: float,
    urcrnrlon: float,
    model_size: int = 128,
) -> np.ndarray:
    """
    Project SEVIR lightning latitude/longitude points
    into the 128x128 StormSense model grid.

    Output columns:
        0 = model pixel x
        1 = model pixel y
        2 = relative x
        3 = relative y
    """

    if lightning.ndim != 2 or lightning.shape[1] != 5:
        raise ValueError(
            f"Expected lightning shape (N, 5), "
            f"got {lightning.shape}"
        )

    if model_size <= 1:
        raise ValueError(
            "model_size must be greater than 1."
        )

    crs = CRS.from_proj4(projection)

    transformer = Transformer.from_crs(
        "EPSG:4326",
        crs,
        always_xy=True,
    )

    # --------------------------------------------------------
    # Project lightning points
    # --------------------------------------------------------

    lon = lightning[:, 2]
    lat = lightning[:, 1]

    x_lightning, y_lightning = transformer.transform(
        lon,
        lat,
    )

    # --------------------------------------------------------
    # Project all four event-domain corners
    #
    # We must normalize against the projected event bounds,
    # not against the global projection origin.
    # --------------------------------------------------------

    corner_lon = np.array(
        [
            llcrnrlon,
            llcrnrlon,
            urcrnrlon,
            urcrnrlon,
        ],
        dtype=np.float64,
    )

    corner_lat = np.array(
        [
            llcrnrlat,
            urcrnrlat,
            llcrnrlat,
            urcrnrlat,
        ],
        dtype=np.float64,
    )

    x_corners, y_corners = transformer.transform(
        corner_lon,
        corner_lat,
    )

    x_min = float(np.min(x_corners))
    x_max = float(np.max(x_corners))
    y_min = float(np.min(y_corners))
    y_max = float(np.max(y_corners))

    if x_max <= x_min:
        raise ValueError(
            "Invalid projected longitude bounds."
        )

    if y_max <= y_min:
        raise ValueError(
            "Invalid projected latitude bounds."
        )

    # --------------------------------------------------------
    # Normalize projected coordinates to [0, 1]
    # --------------------------------------------------------

    x_norm = (
        x_lightning - x_min
    ) / (
        x_max - x_min
    )

    # Image coordinates increase downward,
    # so projected north is mapped to smaller pixel y.
    y_norm = (
        y_max - y_lightning
    ) / (
        y_max - y_min
    )

    # --------------------------------------------------------
    # Convert to 128x128 model pixel coordinates
    # --------------------------------------------------------

    x_model = (
        x_norm
        * (model_size - 1)
    )

    y_model = (
        y_norm
        * (model_size - 1)
    )

    # --------------------------------------------------------
    # Match existing app/map_utils.py convention:
    #
    # x = pixel_x - 64
    # y = 64 - pixel_y
    # --------------------------------------------------------

    relative_x = (
        x_model
        - model_size / 2.0
    )

    relative_y = (
        model_size / 2.0
        - y_model
    )

    valid = (
        (x_model >= 0.0)
        & (x_model <= model_size - 1)
        & (y_model >= 0.0)
        & (y_model <= model_size - 1)
    )

    return np.column_stack(
        [
            x_model,
            y_model,
            relative_x,
            relative_y,
            valid,
        ]
    )

def lightning_summary(
    event_id: str,
) -> Dict:
    """
    Create a 49-frame, 5-minute lightning summary.
    """

    result = load_lightning_event(
        event_id
    )

    frames = bin_lightning_to_5min(
        result["lightning"]
    )

    counts = np.array(
        [
            frame["count"]
            for frame in frames
        ],
        dtype=np.int32,
    )

    return {
        **result,
        "frames": frames,
        "counts": counts,
        "total_flashes": int(
            len(result["lightning"])
        ),
        "active_frames": int(
            np.count_nonzero(counts)
        ),
        "max_frame_count": int(
            counts.max()
        ),
        "mean_active_frame_count": (
            float(
                counts[counts > 0].mean()
            )
            if np.any(counts > 0)
            else 0.0
        ),
    }


if __name__ == "__main__":

    EVENT_ID = "S793011"

    result = lightning_summary(
        EVENT_ID
    )

    print("=" * 70)
    print("STORMSENSE SEVIR LIGHTNING TEST")
    print("=" * 70)

    print("Event ID:", result["event_id"])
    print("File:", result["file_name"])
    print("Catalog index:", result["file_index"])
    print("Time UTC:", result["time_utc"])

    print()
    print(
        "Projection:",
        result["projection"],
    )

    print(
        "Domain:",
        result["width_m"],
        "x",
        result["height_m"],
        "meters",
    )

    print()
    print(
        "Raw lightning events:",
        result["total_flashes"],
    )

    print(
        "Active 5-min frames:",
        result["active_frames"],
    )

    print(
        "Maximum events in one 5-min frame:",
        result["max_frame_count"],
    )

    print(
        "Mean events in active frame:",
        f"{result['mean_active_frame_count']:.2f}",
    )

    print()
    print("5-minute lightning timeline:")
    print("-" * 70)

    for frame in result["frames"]:

        if frame["count"] > 0:

            print(
                f"+{frame['start_seconds'] // 60:02d} min | "
                f"count={frame['count']}"
            )
