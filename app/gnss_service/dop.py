from dataclasses import dataclass, field
import math

import numpy as np


@dataclass(frozen=True)
class DopResult:
    visible: bool
    satellite_count: int
    gdop: float | None = None
    pdop: float | None = None
    hdop: float | None = None
    vdop: float | None = None
    tdop: float | None = None
    visible_satellites: tuple[str, ...] = field(default_factory=tuple)


def dop_from_directions(directions) -> DopResult:
    """Calculate DOPs from ENU unit vectors.

    Each direction is ``(east, north, up)``. A fourth one-clock column is used,
    so at least four independent satellite equations are required.
    """
    vectors = [np.asarray(item, dtype=float) for item in directions]
    if len(vectors) < 4:
        return DopResult(False, len(vectors))
    matrix = np.ones((len(vectors), 4), dtype=float)
    for index, vector in enumerate(vectors):
        norm = np.linalg.norm(vector)
        if norm <= 0:
            return DopResult(False, len(vectors))
        matrix[index, 0:3] = vector / norm
    try:
        covariance = np.linalg.inv(matrix.T @ matrix)
    except np.linalg.LinAlgError:
        return DopResult(False, len(vectors))
    # Guard against numerical rank failure.
    if np.any(np.diag(covariance) <= 0):
        return DopResult(False, len(vectors))
    hdop2 = float(covariance[0, 0] + covariance[1, 1])
    vdop2 = float(covariance[2, 2])
    tdop2 = float(covariance[3, 3])
    pdop2 = hdop2 + vdop2
    gdop2 = pdop2 + tdop2
    if min(hdop2, vdop2, tdop2, pdop2, gdop2) < -1e-9:
        return DopResult(False, len(vectors))
    return DopResult(
        True,
        len(vectors),
        math.sqrt(max(0.0, gdop2)),
        math.sqrt(max(0.0, pdop2)),
        math.sqrt(max(0.0, hdop2)),
        math.sqrt(max(0.0, vdop2)),
        math.sqrt(max(0.0, tdop2)),
    )
