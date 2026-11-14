"""
config.py - Configuration options for Air Canvas.
Modify these defaults to customize camera, dimensions, styling, and gesture sensitivity.
"""

# Camera and display settings
CAMERA_INDEX = 0
WIDTH = 1280
HEIGHT = 720

# Drawing presets
DEFAULT_COLOR = "Ice Cyan"
PEN_THICKNESS = 6
ERASER_THICKNESS = 35

# Pinch-to-draw hysteresis thresholds (ratio of pinch distance to hand scale)
# Start threshold: pinch ratio must drop below this to begin drawing
PINCH_START_THRESHOLD = 0.18
# Stop threshold: pinch ratio must rise above this to release drawing
PINCH_STOP_THRESHOLD = 0.28

