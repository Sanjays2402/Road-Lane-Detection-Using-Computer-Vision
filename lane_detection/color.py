"""Color-space helpers for lane segmentation.

The plain Canny+Hough pipeline does well on bright, well-lit roads but
falls apart on the Udacity *challenge* clip, where shadows and pavement
color changes wash out edges. Filtering for the actual lane-marking
colors (white + yellow) in HLS first, then doing edge detection on the
combined mask, makes the pipeline noticeably more robust.

This module deliberately avoids any global state and operates on
``numpy.ndarray`` images in RGB order (matching ``matplotlib.image`` and
``moviepy``), converting internally.
"""

from __future__ import annotations

import cv2
import numpy as np

# HLS thresholds tuned on the Udacity test footage. White lanes are
# high-lightness regardless of hue; yellow lanes sit in a narrow hue band
# with moderate saturation. These ranges are intentionally wide to keep
# the recall high — false positives are pruned later by ROI + Hough.
_WHITE_LOWER = np.array([0, 200, 0], dtype=np.uint8)
_WHITE_UPPER = np.array([180, 255, 255], dtype=np.uint8)
_YELLOW_LOWER = np.array([15, 38, 115], dtype=np.uint8)
_YELLOW_UPPER = np.array([35, 204, 255], dtype=np.uint8)


def hls_color_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Return a uint8 mask isolating white and yellow regions of the image.

    Args:
        image_rgb: ``(H, W, 3)`` array in RGB color order, ``uint8``.

    Returns:
        ``(H, W)`` ``uint8`` mask with values in ``{0, 255}``.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(
            f"hls_color_mask expects an (H, W, 3) RGB image, got shape {image_rgb.shape}"
        )
    hls = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HLS)
    white_mask = cv2.inRange(hls, _WHITE_LOWER, _WHITE_UPPER)
    yellow_mask = cv2.inRange(hls, _YELLOW_LOWER, _YELLOW_UPPER)
    return cv2.bitwise_or(white_mask, yellow_mask)


def apply_color_filter(image_rgb: np.ndarray) -> np.ndarray:
    """Zero out pixels that are not white-or-yellow in HLS space.

    Returns an RGB image of the same shape and dtype as the input.
    """
    mask = hls_color_mask(image_rgb)
    return cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
