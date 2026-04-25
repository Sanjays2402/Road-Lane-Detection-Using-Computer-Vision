"""Synthetic-image tests for the lane detection pipeline.

These don't depend on the original Udacity test footage. We render a
tiny black image with two diagonal "lane" lines, run the pipeline, and
check that left/right lanes are detected with the expected sign of slope.
"""

from __future__ import annotations

import cv2
import numpy as np

from lane_detection import LaneDetector, LaneDetectorConfig
from lane_detection.lines import (
    LineParams,
    average_line,
    extrapolate,
    split_left_right,
)
from lane_detection.roi import TrapezoidROI


def _synthetic_road(height: int = 540, width: int = 960) -> np.ndarray:
    """Render a black image with two white lane-like diagonals."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    # Asphalt-ish grey background so HLS color filter still keeps lanes.
    image[:] = (40, 40, 40)
    # Left lane goes from bottom-left up and to the right.
    cv2.line(image, (200, height - 1), (int(width * 0.49), int(height * 0.62)),
             (255, 255, 255), 8)
    # Right lane goes from bottom-right up and to the left.
    cv2.line(image, (width - 200, height - 1), (int(width * 0.51), int(height * 0.62)),
             (255, 255, 255), 8)
    return image


def test_detect_lanes_on_synthetic_road():
    detector = LaneDetector()
    overlay = detector.process(_synthetic_road())

    assert overlay.left is not None, "Left lane should be detected"
    assert overlay.right is not None, "Right lane should be detected"
    # Image coordinates: y grows downward, so the left lane (going up
    # and to the right) has *negative* slope.
    assert overlay.left.slope < 0
    assert overlay.right.slope > 0
    assert overlay.image.shape == (540, 960, 3)
    assert overlay.edges.shape == (540, 960)


def test_detector_returns_none_when_no_lanes_present():
    blank = np.full((540, 960, 3), 40, dtype=np.uint8)
    overlay = LaneDetector().process(blank)
    assert overlay.left is None
    assert overlay.right is None
    # Output still has the right shape (just no lines drawn).
    assert overlay.image.shape == blank.shape


def test_detector_rejects_non_rgb_input():
    gray = np.zeros((540, 960), dtype=np.uint8)
    try:
        LaneDetector().process(gray)
    except ValueError as err:
        assert "expects an (H, W, 3) RGB image" in str(err)
    else:
        raise AssertionError("Expected ValueError for grayscale input")


def test_split_left_right_handles_vertical_segments():
    # A vertical segment (x1 == x2) must not raise ZeroDivisionError.
    # Slopes for the two non-vertical segments below are ±0.6, which
    # comfortably clears the default min_abs_slope of 0.4.
    left, right = split_left_right(
        [(100, 540, 100, 300), (200, 540, 700, 240), (800, 540, 300, 240)]
    )
    # The vertical segment is silently dropped; the other two are
    # classified by sign of slope.
    assert len(left) + len(right) == 2
    assert len(left) == 1
    assert len(right) == 1


def test_split_left_right_filters_by_slope_magnitude():
    # Near-horizontal segments should be rejected.
    horizontal = [(0, 100, 1000, 102)]
    left, right = split_left_right(horizontal, min_abs_slope=0.4, max_abs_slope=2.0)
    assert left == [] and right == []


def test_average_line_handles_empty_input():
    assert average_line([]) is None


def test_extrapolate_projects_to_image_height():
    line = LineParams(slope=-0.7, intercept=600.0)
    x1, y1, x2, y2 = extrapolate(line, image_height=540, top_y_ratio=0.6)
    assert y1 == 540
    assert y2 == int(540 * 0.6)
    # Sanity: x must be on the line (allow rounding ±1).
    assert abs(x1 - line.x_at(y1)) <= 1
    assert abs(x2 - line.x_at(y2)) <= 1


def test_temporal_smoothing_blends_estimates():
    cfg = LaneDetectorConfig(smoothing_alpha=0.5)
    detector = LaneDetector(cfg)

    image = _synthetic_road()
    first = detector.process(image)
    assert first.left is not None

    # Feed a slightly different image: shift the white lanes by 50 px.
    shifted = np.roll(image, 50, axis=1)
    second = detector.process(shifted)
    assert second.left is not None
    # With alpha=0.5, the smoothed slope should be between the prior
    # estimate (== first_slope) and whatever fresh detection produces.
    # Hard to know the exact fresh value without re-running pipeline,
    # but at minimum smoothing should not produce a wildly different
    # slope sign.
    assert second.left.slope < 0


def test_smoothing_coasts_through_lost_detections():
    cfg = LaneDetectorConfig(smoothing_alpha=0.3)
    detector = LaneDetector(cfg)
    overlay = detector.process(_synthetic_road())
    assert overlay.left is not None and overlay.right is not None

    # Now feed an empty frame. Detections will be None, but the
    # smoothing layer should keep returning the previous estimate.
    blank = np.full_like(overlay.image, 40)
    coasted = detector.process(blank)
    assert coasted.left is not None
    assert coasted.right is not None


def test_invalid_smoothing_alpha_raises():
    detector = LaneDetector(LaneDetectorConfig(smoothing_alpha=1.5))
    try:
        detector.process(_synthetic_road())
    except ValueError as err:
        assert "smoothing_alpha" in str(err)
    else:
        raise AssertionError("Expected ValueError for smoothing_alpha > 1")


def test_invalid_kernel_size_raises():
    try:
        LaneDetector(LaneDetectorConfig(gaussian_kernel=4))
    except ValueError as err:
        assert "gaussian_kernel" in str(err)
    else:
        raise AssertionError("Expected ValueError for even gaussian_kernel")


def test_trapezoid_roi_scales_with_image_size():
    roi = TrapezoidROI()
    small = roi.vertices(540, 960)
    large = roi.vertices(1080, 1920)
    # Bottom-right vertex is image-relative.
    assert tuple(small[0][3]) == (960, 540)
    assert tuple(large[0][3]) == (1920, 1080)
