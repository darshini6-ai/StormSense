import numpy as np
from scipy import ndimage


def create_risk_mask(vil_frame, threshold=0.15):
    """
    Convert a normalized VIL frame into a binary storm-risk mask.

    Parameters
    ----------
    vil_frame : numpy.ndarray
        Normalized VIL frame with values approximately in [0, 1].

    threshold : float
        VIL threshold used to identify significant storm regions.

    Returns
    -------
    numpy.ndarray
        Binary mask where:
        1 = storm-risk region
        0 = background
    """

    risk_mask = vil_frame >= threshold

    return risk_mask.astype(np.uint8)


def detect_storm_cells(risk_mask, min_area=20):
    """
    Detect connected storm cells from a binary risk mask.

    Parameters
    ----------
    risk_mask : numpy.ndarray
        Binary storm-risk mask.

    min_area : int
        Minimum number of pixels required for a storm cell.

    Returns
    -------
    list
        Detected storm cells with area and centroid.
    """

    labeled, num_features = ndimage.label(risk_mask)

    cells = []

    for label_id in range(1, num_features + 1):

        component = labeled == label_id
        area = int(component.sum())

        if area < min_area:
            continue

        y_coords, x_coords = np.where(component)

        centroid_y = float(y_coords.mean())
        centroid_x = float(x_coords.mean())

        cells.append({
            "id": len(cells) + 1,
            "area": area,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y
        })

    return cells


def calculate_risk_level(vil_frame):
    """
    Estimate an overall storm risk level from a VIL frame.
    """

    max_vil = float(np.max(vil_frame))
    mean_vil = float(np.mean(vil_frame))

    if max_vil >= 0.70:
        level = "HIGH"
    elif max_vil >= 0.40:
        level = "MODERATE"
    elif max_vil >= 0.15:
        level = "LOW"
    else:
        level = "NONE"

    return {
        "level": level,
        "max_vil": max_vil,
        "mean_vil": mean_vil
    }
