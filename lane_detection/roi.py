"""Region-of-interest masking.

The original notebook hard-coded ROI vertices to a single 960x540
resolution. This module derives a trapezoidal ROI from the image
dimensions so the pipeline works on arbitrary inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TrapezoidROI:
    """Relative trapezoidal region of interest.

    Coordinates are fractions of width/height so the same ROI works
    across resolutions.
    """

    bottom_left_x: float = 0.0
    bottom_right_x: float = 1.0
    top_left_x: float = 0.48
    top_right_x: float = 0.52
    top_y: float = 0.58

    def vertices(self, height: int, width: int) -> np.ndarray:
        """Materialize integer pixel vertices for an image of (H, W)."""
        return np.array(
            [
                [
                    (int(self.bottom_left_x * width), height),
                    (int(self.top_left_x * width), int(self.top_y * height)),
                    (int(self.top_right_x * width), int(self.top_y * height)),
                    (int(self.bottom_right_x * width), height),
                ]
            ],
            dtype=np.int32,
        )


def region_of_interest(image: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Mask everything outside the polygon defined by ``vertices``.

    Works on both single-channel and multi-channel images.
    """
    mask = np.zeros_like(image)
    if image.ndim > 2:
        ignore_color = (255,) * image.shape[2]
    else:
        ignore_color = 255
    cv2.fillPoly(mask, vertices, ignore_color)
    return cv2.bitwise_and(image, mask)
