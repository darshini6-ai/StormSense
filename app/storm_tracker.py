import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


def detect_storm_cells(vil_frame, threshold=0.05, min_area=20):
    """
    Detect connected storm cells from a VIL frame.

    Parameters
    ----------
    vil_frame : numpy.ndarray
        2D normalized VIL frame, values approximately 0-1.
    threshold : float
        Minimum VIL value used to define a storm region.
    min_area : int
        Minimum number of pixels required for a storm cell.

    Returns
    -------
    list of dict
        Detected storm cells with location and intensity information.
    """

    mask = vil_frame >= threshold

    labeled, num_features = ndimage.label(mask)

    cells = []

    for label_id in range(1, num_features + 1):
        ys, xs = np.where(labeled == label_id)

        area = len(xs)

        if area < min_area:
            continue

        cell_values = vil_frame[ys, xs]

        cells.append({
            "id": len(cells),
            "area": area,
            "centroid_x": float(np.mean(xs)),
            "centroid_y": float(np.mean(ys)),
            "max_vil": float(np.max(cell_values)),
            "mean_vil": float(np.mean(cell_values))
        })

    return cells


def match_cells(cells_a, cells_b, max_distance=15):
    """
    Match storm cells between two consecutive frames.

    Uses Hungarian assignment for one-to-one matching and
    rejects matches that move farther than max_distance pixels.

    Returns
    -------
    list of dict
        Matched cell pairs.
    """

    if not cells_a or not cells_b:
        return []

    cost_matrix = np.zeros((len(cells_a), len(cells_b)))

    for i, cell_a in enumerate(cells_a):
        for j, cell_b in enumerate(cells_b):

            dx = cell_b["centroid_x"] - cell_a["centroid_x"]
            dy = cell_b["centroid_y"] - cell_a["centroid_y"]

            distance = np.sqrt(dx**2 + dy**2)

            cost_matrix[i, j] = distance

    rows, cols = linear_sum_assignment(cost_matrix)

    matches = []

    for row, col in zip(rows, cols):

        distance = cost_matrix[row, col]

        if distance > max_distance:
            continue

        cell_a = cells_a[row]
        cell_b = cells_b[col]

        dx = cell_b["centroid_x"] - cell_a["centroid_x"]
        dy = cell_b["centroid_y"] - cell_a["centroid_y"]

        matches.append({
            "from_id": cell_a["id"],
            "to_id": cell_b["id"],
            "from_x": cell_a["centroid_x"],
            "from_y": cell_a["centroid_y"],
            "to_x": cell_b["centroid_x"],
            "to_y": cell_b["centroid_y"],
            "dx": float(dx),
            "dy": float(dy),
            "distance": float(distance),
            "from_max_vil": cell_a["max_vil"],
            "to_max_vil": cell_b["max_vil"],
            "from_area": cell_a["area"],
            "to_area": cell_b["area"]
        })

    return matches


def track_cells(
    frame_a,
    frame_b,
    threshold=0.05,
    min_area=20,
    max_distance=15
):
    """
    Detect and track storm cells between two consecutive frames.
    """

    cells_a = detect_storm_cells(
        frame_a,
        threshold=threshold,
        min_area=min_area
    )

    cells_b = detect_storm_cells(
        frame_b,
        threshold=threshold,
        min_area=min_area
    )

    matches = match_cells(
        cells_a,
        cells_b,
        max_distance=max_distance
    )

    return {
        "frame_a_cells": cells_a,
        "frame_b_cells": cells_b,
        "matches": matches
    }
