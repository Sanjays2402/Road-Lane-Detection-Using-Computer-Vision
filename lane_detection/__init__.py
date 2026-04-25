"""Road lane detection pipeline.

A small, well-tested classical computer-vision pipeline for finding lane
lines in road images and dashcam video. Built around OpenCV with optional
HLS color filtering and exponential temporal smoothing for video.

The original implementation lived in a single Udacity Self-Driving Car
Nanodegree notebook. This package refactors it into something you can
``pip install``, import, test, and run from the command line.
"""

from .pipeline import (
    LaneDetector,
    LaneDetectorConfig,
    LaneOverlay,
    detect_lanes,
)

__all__ = [
    "LaneDetector",
    "LaneDetectorConfig",
    "LaneOverlay",
    "detect_lanes",
]

__version__ = "0.2.0"
