"""Command-line interface for the lane detector.

Examples:

    # Process a single image
    lane-detect image input.jpg --output out.jpg

    # Process all images in a folder
    lane-detect image input_dir/ --output output_dir/

    # Process a video with temporal smoothing
    lane-detect video drive.mp4 --output drive_lanes.mp4 --smoothing 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .pipeline import LaneDetector, LaneDetectorConfig

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _load_image(path: Path) -> np.ndarray:
    """Read an image as RGB (matplotlib/moviepy convention)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _save_image(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise IOError(f"Could not write image: {path}")


def _iter_image_paths(input_path: Path) -> Iterable[Path]:
    if input_path.is_dir():
        for child in sorted(input_path.iterdir()):
            if child.suffix.lower() in _IMAGE_EXTS:
                yield child
    else:
        yield input_path


def _build_config(args: argparse.Namespace) -> LaneDetectorConfig:
    cfg = LaneDetectorConfig()
    if args.smoothing is not None:
        cfg.smoothing_alpha = args.smoothing
    if args.no_color_filter:
        cfg.use_color_filter = False
    return cfg


def _process_images(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    detector = LaneDetector(cfg)

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    paths = list(_iter_image_paths(input_path))
    if not paths:
        print(f"No images found at {input_path}", file=sys.stderr)
        return 1

    for image_path in paths:
        image = _load_image(image_path)
        result = detector.process(image)
        if output_path is None:
            target = image_path.with_name(f"{image_path.stem}_lanes{image_path.suffix}")
        elif output_path.suffix.lower() in _IMAGE_EXTS:
            target = output_path
        else:
            target = output_path / image_path.name
        _save_image(target, result.image)
        print(f"{image_path} -> {target}")
    return 0


def _process_video(args: argparse.Namespace) -> int:
    try:
        from moviepy.editor import VideoFileClip
    except ImportError as err:  # pragma: no cover - exercised only when moviepy missing
        print(
            f"Video processing requires moviepy. Install with: pip install moviepy. ({err})",
            file=sys.stderr,
        )
        return 2

    cfg = _build_config(args)
    if cfg.smoothing_alpha is None:
        cfg.smoothing_alpha = 0.2  # Sensible default for video.
    detector = LaneDetector(cfg)

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(
        f"{input_path.stem}_lanes{input_path.suffix}"
    )

    clip = VideoFileClip(str(input_path))
    detector.reset()
    processed = clip.fl_image(lambda frame: detector.process(frame).image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.write_videofile(str(output_path), audio=False, logger=None)
    print(f"{input_path} -> {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lane-detect",
        description="Detect lane lines in road images and dashcam video.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    image_parser = sub.add_parser("image", help="Process a single image or a directory of images.")
    image_parser.add_argument("input", help="Path to image file or directory.")
    image_parser.add_argument(
        "--output",
        "-o",
        help="Output file or directory. If omitted, writes alongside input as <name>_lanes.<ext>.",
    )
    image_parser.add_argument(
        "--smoothing", type=float, default=None,
        help="Optional smoothing alpha in (0, 1]. Mostly useful for video; on images it just "
             "lags the detection between successive calls.",
    )
    image_parser.add_argument(
        "--no-color-filter", action="store_true",
        help="Disable HLS white/yellow filtering. Useful for benchmarking.",
    )
    image_parser.set_defaults(func=_process_images)

    video_parser = sub.add_parser("video", help="Process a video file.")
    video_parser.add_argument("input", help="Path to input video.")
    video_parser.add_argument(
        "--output", "-o", help="Output video path. Defaults to <input>_lanes.<ext>.",
    )
    video_parser.add_argument(
        "--smoothing", type=float, default=None,
        help="Smoothing alpha in (0, 1]. Lower = smoother but laggier. Defaults to 0.2.",
    )
    video_parser.add_argument(
        "--no-color-filter", action="store_true",
        help="Disable HLS white/yellow filtering.",
    )
    video_parser.set_defaults(func=_process_video)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
