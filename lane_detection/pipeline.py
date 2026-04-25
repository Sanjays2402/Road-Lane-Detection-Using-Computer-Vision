"""End-to-end lane detection pipeline.

Public entry points:

    >>> from lane_detection import LaneDetector
    >>> detector = LaneDetector()
    >>> overlay = detector.process(image_rgb)

For one-shot use without temporal smoothing, ``detect_lanes(image)`` is
a thin functional wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .color import apply_color_filter
from .lines import LineParams, average_line, extrapolate, split_left_right
from .roi import TrapezoidROI, region_of_interest


@dataclass
class LaneDetectorConfig:
    """Knobs for the lane detection pipeline.

    Defaults are tuned for the Udacity 960x540 test footage but degrade
    gracefully on other resolutions because the ROI is fractional.
    """

    # Pre-processing
    use_color_filter: bool = True
    gaussian_kernel: int = 5
    canny_low: int = 50
    canny_high: int = 150

    # Region of interest
    roi: TrapezoidROI = field(default_factory=TrapezoidROI)

    # Hough transform
    hough_rho: float = 1.0
    hough_theta: float = np.pi / 180
    hough_threshold: int = 20
    hough_min_line_len: int = 20
    hough_max_line_gap: int = 300

    # Slope filtering
    min_abs_slope: float = 0.4
    max_abs_slope: float = 2.0

    # Output rendering
    line_color: tuple[int, int, int] = (255, 0, 0)  # red in RGB
    line_thickness: int = 10
    overlay_alpha: float = 0.8
    overlay_beta: float = 1.0
    top_y_ratio: float = 0.6

    # Temporal smoothing for video. ``None`` disables smoothing (image
    # mode). With a value in (0, 1], the new frame's line params are
    # blended with the running estimate as
    # ``new = (1 - alpha) * prev + alpha * detected``. Lower alpha =
    # smoother but laggier.
    smoothing_alpha: Optional[float] = None


@dataclass(frozen=True)
class LaneOverlay:
    """Result of one pipeline pass.

    Attributes:
        image: Input RGB image with lane lines drawn on top.
        left: Slope/intercept of the left lane (or ``None`` if not
            detected this frame).
        right: Slope/intercept of the right lane.
        edges: The Canny edge map after ROI masking — useful for
            debugging.
    """

    image: np.ndarray
    left: Optional[LineParams]
    right: Optional[LineParams]
    edges: np.ndarray


class LaneDetector:
    """Stateful lane detector.

    When ``config.smoothing_alpha`` is set, the detector keeps a running
    estimate of left/right lane parameters across frames and blends in
    each new detection. This eliminates most of the visible jitter on
    dashcam video at no detection-quality cost.

    Instances are not thread-safe.
    """

    def __init__(self, config: LaneDetectorConfig | None = None) -> None:
        self.config = config or LaneDetectorConfig()
        if self.config.gaussian_kernel % 2 == 0 or self.config.gaussian_kernel < 1:
            raise ValueError(
                f"gaussian_kernel must be a positive odd integer, "
                f"got {self.config.gaussian_kernel}"
            )
        self._left_estimate: Optional[LineParams] = None
        self._right_estimate: Optional[LineParams] = None

    def reset(self) -> None:
        """Forget the running smoothing estimate."""
        self._left_estimate = None
        self._right_estimate = None

    def process(self, image_rgb: np.ndarray) -> LaneOverlay:
        """Run the pipeline on a single RGB frame."""
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(
                f"LaneDetector.process expects an (H, W, 3) RGB image, "
                f"got shape {image_rgb.shape}"
            )

        height, width = image_rgb.shape[:2]
        cfg = self.config

        # 1. Color filtering — keeps only white/yellow lane markings.
        # Notebook bug: it called gaussian_blur on the RGB image, then
        # Canny on the still-color image. We do it correctly here:
        # color filter -> gray -> blur -> Canny.
        prepared = apply_color_filter(image_rgb) if cfg.use_color_filter else image_rgb
        gray = cv2.cvtColor(prepared, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (cfg.gaussian_kernel, cfg.gaussian_kernel), 0)
        edges = cv2.Canny(blurred, cfg.canny_low, cfg.canny_high)

        # 2. Mask to ROI.
        vertices = cfg.roi.vertices(height, width)
        masked_edges = region_of_interest(edges, vertices)

        # 3. Hough.
        hough_segments = cv2.HoughLinesP(
            masked_edges,
            cfg.hough_rho,
            cfg.hough_theta,
            cfg.hough_threshold,
            np.array([]),
            minLineLength=cfg.hough_min_line_len,
            maxLineGap=cfg.hough_max_line_gap,
        )

        # 4. Geometry.
        if hough_segments is None:
            left_params, right_params = None, None
        else:
            segments = [tuple(seg[0]) for seg in hough_segments]
            left_lines, right_lines = split_left_right(
                segments,
                min_abs_slope=cfg.min_abs_slope,
                max_abs_slope=cfg.max_abs_slope,
            )
            left_params = average_line(left_lines)
            right_params = average_line(right_lines)

        # 5. Temporal smoothing (video).
        left_params = self._blend(self._left_estimate, left_params)
        right_params = self._blend(self._right_estimate, right_params)
        if cfg.smoothing_alpha is not None:
            self._left_estimate = left_params
            self._right_estimate = right_params

        # 6. Render.
        overlay = self._render(image_rgb, left_params, right_params)
        return LaneOverlay(
            image=overlay,
            left=left_params,
            right=right_params,
            edges=masked_edges,
        )

    # --- internals ---------------------------------------------------

    def _blend(
        self, prev: Optional[LineParams], detected: Optional[LineParams]
    ) -> Optional[LineParams]:
        alpha = self.config.smoothing_alpha
        if alpha is None:
            return detected
        if not 0.0 < alpha <= 1.0:
            raise ValueError(
                f"smoothing_alpha must be in (0, 1], got {alpha}"
            )
        if detected is None:
            # Lost detection this frame — coast on previous estimate.
            return prev
        if prev is None:
            return detected
        return LineParams(
            slope=(1 - alpha) * prev.slope + alpha * detected.slope,
            intercept=(1 - alpha) * prev.intercept + alpha * detected.intercept,
        )

    def _render(
        self,
        image_rgb: np.ndarray,
        left: Optional[LineParams],
        right: Optional[LineParams],
    ) -> np.ndarray:
        cfg = self.config
        line_layer = np.zeros_like(image_rgb)
        height = image_rgb.shape[0]
        for line in (left, right):
            if line is None:
                continue
            try:
                x1, y1, x2, y2 = extrapolate(line, height, top_y_ratio=cfg.top_y_ratio)
            except (ZeroDivisionError, OverflowError):
                # Pathological slope — skip rather than crash.
                continue
            cv2.line(line_layer, (x1, y1), (x2, y2), cfg.line_color, cfg.line_thickness)
        return cv2.addWeighted(image_rgb, cfg.overlay_alpha, line_layer, cfg.overlay_beta, 0.0)


def detect_lanes(
    image_rgb: np.ndarray, config: LaneDetectorConfig | None = None
) -> LaneOverlay:
    """One-shot helper for single-image use."""
    return LaneDetector(config).process(image_rgb)
