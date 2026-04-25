"""Lane-line geometry: slope filtering, averaging, and extrapolation.

The notebook had three latent bugs we fix here:

1. Division by zero when a Hough segment is exactly vertical
   (``x2 == x1``). The list comprehension blew up at import-of-frame
   time on certain inputs.
2. ``[sum(y)/len(y) for y in zip(*lines)]`` collapses silently to ``[]``
   when no lines are detected on a side, then later code did
   ``len(left_lane_detection) > 0`` which was OK — but the *next*
   division did ``(y3-y1)/(x2-x0)`` and could blow up again. We
   short-circuit cleanly.
3. The slope cut ``-0.9 < slope < -0.4`` rejected steep lanes near the
   bottom of the image. Widened slightly and made it a config knob.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LineParams:
    """Slope/intercept representation of a single lane line."""

    slope: float
    intercept: float

    def x_at(self, y: float) -> int:
        """Return the integer x for a given y on this line."""
        return int(round((y - self.intercept) / self.slope))


def _segment_slope_intercept(x1: float, y1: float, x2: float, y2: float) -> LineParams | None:
    """Convert a Hough segment to slope/intercept, or ``None`` if vertical."""
    if x2 == x1:
        return None
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return LineParams(slope=slope, intercept=intercept)


def split_left_right(
    segments: Iterable[tuple[int, int, int, int]],
    *,
    min_abs_slope: float = 0.4,
    max_abs_slope: float = 1.0,
) -> tuple[list[LineParams], list[LineParams]]:
    """Partition Hough segments into left- and right-lane candidates.

    Args:
        segments: Iterable of ``(x1, y1, x2, y2)`` tuples in image
            coordinates (y grows downward).
        min_abs_slope: Reject segments flatter than this — they're
            usually horizon/cracks.
        max_abs_slope: Reject segments steeper than this — usually
            shadows or noise.

    Returns:
        ``(left, right)`` where each entry is a list of ``LineParams``.
        In image coordinates the *left* lane has *negative* slope (going
        from bottom-left to top-right inside the ROI) and the right lane
        has positive slope.
    """
    left: list[LineParams] = []
    right: list[LineParams] = []
    for x1, y1, x2, y2 in segments:
        params = _segment_slope_intercept(float(x1), float(y1), float(x2), float(y2))
        if params is None:
            continue
        abs_slope = abs(params.slope)
        if abs_slope < min_abs_slope or abs_slope > max_abs_slope:
            continue
        if params.slope < 0:
            left.append(params)
        else:
            right.append(params)
    return left, right


def average_line(lines: list[LineParams]) -> LineParams | None:
    """Average a bag of lines in slope/intercept space."""
    if not lines:
        return None
    slope = float(np.mean([line.slope for line in lines]))
    intercept = float(np.mean([line.intercept for line in lines]))
    return LineParams(slope=slope, intercept=intercept)


def extrapolate(
    line: LineParams, image_height: int, *, top_y_ratio: float = 0.6
) -> tuple[int, int, int, int]:
    """Project a line to a full ``(x1, y1, x2, y2)`` segment.

    The bottom y is the image height; the top y is ``top_y_ratio *
    image_height`` (a bit below the horizon).
    """
    y1 = image_height
    y2 = int(round(image_height * top_y_ratio))
    return line.x_at(y1), y1, line.x_at(y2), y2
