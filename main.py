"""
main.py - Entry point for the Python OpenCV Air Canvas application.
Run:
    python main.py
    python main.py --help
"""

import argparse
from air_pen import (
    AirPenApp,
    DEFAULT_CAMERA_INDEX,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
)


def main():
    parser = argparse.ArgumentParser(
        description="Air Canvas - Mid-air drawing canvas with OpenCV & MediaPipe"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help=f"Webcam device index (default: {DEFAULT_CAMERA_INDEX})",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Camera capture width (default: {DEFAULT_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Camera capture height (default: {DEFAULT_HEIGHT})",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional path to a video file to play instead of webcam stream",
    )

    args = parser.parse_args()

    app = AirPenApp(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        video_path=args.video,
    )
    app.run()


if __name__ == "__main__":
    main()
