from __future__ import annotations

import numpy as np

from app.lightning_sevir import lightning_summary


def point_in_cell(
    x: float,
    y: float,
    bbox: tuple[float, float, float, float],
) -> bool:
    """
    Check whether a lightning point lies inside a cell bounding box.

    bbox:
        (x_min, y_min, x_max, y_max)
    """
    x_min, y_min, x_max, y_max = bbox

    return (
        x_min <= x <= x_max
        and y_min <= y <= y_max
    )


def associate_lightning_with_cells(
    cells: list[dict],
    lightning_points: np.ndarray,
) -> list[dict]:
    """
    Associate lightning points with detected storm cells.

    Expected lightning columns:
        0 = time
        1 = latitude
        2 = longitude
        3 = x grid
        4 = y grid

    Each cell should optionally provide:
        x_min
        y_min
        x_max
        y_max
    """

    results = []

    for cell_index, cell in enumerate(cells, start=1):

        if not all(
            key in cell
            for key in (
                "x_min",
                "y_min",
                "x_max",
                "y_max",
            )
        ):
            results.append(
                {
                    **cell,
                    "cell_id": cell_index,
                    "lightning_count": 0,
                    "lightning_active": False,
                }
            )
            continue

        count = 0

        for point in lightning_points:

            x = float(point[3])
            y = float(point[4])

            if point_in_cell(
                x,
                y,
                (
                    float(cell["x_min"]),
                    float(cell["y_min"]),
                    float(cell["x_max"]),
                    float(cell["y_max"]),
                ),
            ):
                count += 1

        results.append(
            {
                **cell,
                "cell_id": cell_index,
                "lightning_count": count,
                "lightning_active": count > 0,
            }
        )

    return results


def build_multihazard_summary(
    event_id: str,
) -> dict:
    """
    Create an observed multi-hazard summary for one SEVIR event.

    This does NOT predict lightning.
    It combines:
        - observed lightning activity
        - lightning timeline
        - existing storm-cell analysis later

    The VIL forecast remains produced independently
    by Residual V3.
    """

    lightning = lightning_summary(
        event_id
    )

    counts = lightning["counts"]

    active_frames = np.where(
        counts > 0
    )[0]

    peak_count = int(
        counts.max()
    )

    return {
        "event_id": event_id,
        "total_lightning_events": lightning[
            "total_flashes"
        ],
        "active_lightning_frames": lightning[
            "active_frames"
        ],
        "peak_lightning_frame_count": peak_count,
        "lightning_counts": counts,
        "active_frame_indices": active_frames,
        "has_observed_lightning": bool(
            peak_count > 0
        ),
        "hazard_mode": (
            "LIGHTNING-ACTIVE"
            if peak_count > 0
            else "NO-LIGHTNING-DETECTED"
        ),
    }


if __name__ == "__main__":

    EVENT_ID = "S793011"

    result = build_multihazard_summary(
        EVENT_ID
    )

    print("=" * 70)
    print("STORMSENSE MULTI-HAZARD SUMMARY")
    print("=" * 70)

    print("Event ID:", result["event_id"])

    print(
        "Total lightning events:",
        result["total_lightning_events"],
    )

    print(
        "Active lightning frames:",
        result["active_lightning_frames"],
    )

    print(
        "Peak 5-min lightning count:",
        result["peak_lightning_frame_count"],
    )

    print(
        "Hazard mode:",
        result["hazard_mode"],
    )

    print()
    print("Lightning activity:")
    print("-" * 70)

    for frame, count in enumerate(
        result["lightning_counts"]
    ):

        if count > 0:

            print(
                f"+{frame * 5:02d} min | "
                f"{int(count):4d} lightning events"
            )