"""
air_pen.py - Python OpenCV Air Canvas
Features:
- Adaptive 1-Euro Filter for smoothed, low-jitter cursor tracking
- Midpoint Quadratic Bézier Curve interpolation for fluid digital calligraphy
- Ergonomic Pinch-to-Draw gesture (eliminates hand fatigue and restlessness)
- Physical Color Marker / Stylus tracking mode (hold any real pen or marker)
- Spacebar / Key Trigger mode (draw with relaxed hand)
- Real-time bloom neon glow
- 3 Background modes: Blackboard, Webcam AR, Whiteboard
- Session video recording (key 'R')
- High-res PNG snapshot export (key 'S')
"""

import os
import sys
import time
import datetime
import math
import cv2
import numpy as np

# Suppress verbose library logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from hand_tracker import HandTracker, PointFilter2D

# Try loading configuration defaults from config.py if available
try:
    import config
    DEFAULT_CAMERA_INDEX = getattr(config, "CAMERA_INDEX", 0)
    DEFAULT_WIDTH = getattr(config, "WIDTH", 1280)
    DEFAULT_HEIGHT = getattr(config, "HEIGHT", 720)
    DEFAULT_COLOR_NAME = getattr(config, "DEFAULT_COLOR", "Ice Cyan")
    DEFAULT_PEN_THICKNESS = getattr(config, "PEN_THICKNESS", 6)
    DEFAULT_ERASER_THICKNESS = getattr(config, "ERASER_THICKNESS", 35)
    DEFAULT_PINCH_START = getattr(config, "PINCH_START_THRESHOLD", 0.18)
    DEFAULT_PINCH_STOP = getattr(config, "PINCH_STOP_THRESHOLD", 0.28)
except ImportError:
    DEFAULT_CAMERA_INDEX = 0
    DEFAULT_WIDTH = 1280
    DEFAULT_HEIGHT = 720
    DEFAULT_COLOR_NAME = "Ice Cyan"
    DEFAULT_PEN_THICKNESS = 6
    DEFAULT_ERASER_THICKNESS = 35
    DEFAULT_PINCH_START = 0.18
    DEFAULT_PINCH_STOP = 0.28

# Color presets (BGR for OpenCV)
# Original web app color: rgb(176, 225, 255) -> BGR: (255, 225, 176)
COLORS = [
    {"name": "Ice Cyan",   "bgr": (255, 225, 176), "rgb": (176, 225, 255)}, # Default Air Canvas color
    {"name": "Neon Green", "bgr": (57, 255, 20),   "rgb": (20, 255, 57)},
    {"name": "Hot Pink",   "bgr": (180, 20, 255),  "rgb": (255, 20, 180)},
    {"name": "Amber Gold", "bgr": (0, 215, 255),   "rgb": (255, 215, 0)},
    {"name": "Orange Red", "bgr": (34, 126, 255),  "rgb": (255, 126, 34)},
    {"name": "Pure White", "bgr": (255, 255, 255), "rgb": (255, 255, 255)},
]

BACKGROUND_MODES = ["Blackboard", "Webcam AR", "Whiteboard"]
INPUT_MODES = [
    "Pinch-to-Draw 🤏",         # Ergonomic, natural pen grip
    "Physical Marker 🖊️",       # Hold any colored pen or marker cap
    "Key Hold (Space/D) ⌨️",     # Relaxed hand, press key to write
    "Index Finger ☝️",          # Legacy pointing
]


def compute_bezier_curve(p_start, p_control, p_end, steps=6):
    """
    Computes sampled points along a quadratic Bézier curve:
    B(t) = (1-t)^2 * p_start + 2*(1-t)*t * p_control + t^2 * p_end
    """
    curve_pts = []
    for s in range(steps + 1):
        t = s / float(steps)
        bx = int((1.0 - t) ** 2 * p_start[0] + 2.0 * (1.0 - t) * t * p_control[0] + t ** 2 * p_end[0])
        by = int((1.0 - t) ** 2 * p_start[1] + 2.0 * (1.0 - t) * t * p_control[1] + t ** 2 * p_end[1])
        curve_pts.append((bx, by))
    return curve_pts


class Stroke:
    """Represents a smooth continuous stroke with quadratic Bézier interpolation."""

    def __init__(self, color_bgr, thickness, is_eraser=False):
        self.points = []
        self.color_bgr = color_bgr
        self.thickness = thickness
        self.is_eraser = is_eraser

    def add_point(self, pt):
        self.points.append(pt)

    def draw_latest_curve(self, canvas):
        """Draws the newest smooth curve segment onto the canvas."""
        n = len(self.points)
        if n < 2:
            return
        elif n == 2:
            # First segment: straight line
            color = (0, 0, 0) if self.is_eraser else self.color_bgr
            cv2.line(canvas, self.points[0], self.points[1], color, self.thickness, cv2.LINE_AA)
        else:
            # Midpoint quadratic Bézier interpolation for smooth curve
            p0 = self.points[n - 3]
            p1 = self.points[n - 2]
            p2 = self.points[n - 1]

            m_prev = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
            m_curr = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

            color = (0, 0, 0) if self.is_eraser else self.color_bgr
            curve_pts = compute_bezier_curve(m_prev, p1, m_curr)
            for i in range(len(curve_pts) - 1):
                cv2.line(canvas, curve_pts[i], curve_pts[i + 1], color, self.thickness, cv2.LINE_AA)

    def draw_full(self, canvas):
        """Renders the entire smoothed stroke onto the canvas."""
        n = len(self.points)
        if n == 0:
            return
        elif n == 1:
            color = (0, 0, 0) if self.is_eraser else self.color_bgr
            cv2.circle(canvas, self.points[0], self.thickness // 2, color, -1, cv2.LINE_AA)
            return
        elif n == 2:
            color = (0, 0, 0) if self.is_eraser else self.color_bgr
            cv2.line(canvas, self.points[0], self.points[1], color, self.thickness, cv2.LINE_AA)
            return

        color = (0, 0, 0) if self.is_eraser else self.color_bgr
        # Draw start segment
        m_start = ((self.points[0][0] + self.points[1][0]) // 2, (self.points[0][1] + self.points[1][1]) // 2)
        cv2.line(canvas, self.points[0], m_start, color, self.thickness, cv2.LINE_AA)

        # Draw smooth midpoint curves
        for i in range(1, n - 1):
            p0 = self.points[i - 1]
            p1 = self.points[i]
            p2 = self.points[i + 1]

            m_prev = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
            m_curr = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

            curve_pts = compute_bezier_curve(m_prev, p1, m_curr)
            for k in range(len(curve_pts) - 1):
                cv2.line(canvas, curve_pts[k], curve_pts[k + 1], color, self.thickness, cv2.LINE_AA)

        # Draw end segment
        m_end = ((self.points[n - 2][0] + self.points[n - 1][0]) // 2, (self.points[n - 2][1] + self.points[n - 1][1]) // 2)
        cv2.line(canvas, m_end, self.points[n - 1], color, self.thickness, cv2.LINE_AA)


class AirPenApp:
    """Main Air Pen application controller."""

    def __init__(
        self,
        camera_index=DEFAULT_CAMERA_INDEX,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        video_path=None,
        pen_thickness=DEFAULT_PEN_THICKNESS,
        eraser_thickness=DEFAULT_ERASER_THICKNESS,
        pinch_start_thresh=DEFAULT_PINCH_START,
        pinch_stop_thresh=DEFAULT_PINCH_STOP,
    ):
        self.cam_width = width
        self.cam_height = height
        self.camera_index = camera_index
        self.video_path = video_path
        self.is_video_file = video_path is not None
        self.video_fps = 0.0

        # Video capture source
        if self.is_video_file:
            self.cap = cv2.VideoCapture(video_path)
            if self.cap.isOpened():
                self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        elif sys.platform.startswith("win"):
            self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(camera_index)
        else:
            self.cap = cv2.VideoCapture(camera_index)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)

        ret, frame = self.cap.read()
        if ret:
            self.cam_height, self.cam_width = frame.shape[:2]
        else:
            print("[AirCanvas] Warning: Camera not immediately readable, using default dimensions.")

        # Canvas & strokes
        self.drawing_canvas = np.zeros((self.cam_height, self.cam_width, 3), dtype=np.uint8)
        self.strokes = []
        self.current_stroke = None
        self.prev_pt = None

        # Tools & parameters
        self.color_idx = 0
        for i, c in enumerate(COLORS):
            if c["name"].lower() == DEFAULT_COLOR_NAME.lower():
                self.color_idx = i
                break

        self.pen_thickness = pen_thickness
        self.eraser_thickness = eraser_thickness
        self.is_eraser = False

        # Input & background modes
        self.input_mode_idx = 0  # 0: Pinch-to-Draw (Ergonomic), 1: Physical Marker, 2: Key Hold, 3: Index Finger
        self.bg_mode_idx = 0     # 0: Blackboard, 1: Webcam AR, 2: Whiteboard
        self.key_drawing = False # State for Key Mode and Marker Mode (Space/D)
        self.show_hud = True
        self.fullscreen = False

        # Physical Marker / Stylus tracking HSV range (default neon green / bright marker)
        self.marker_hsv_lower = np.array([35, 80, 80])
        self.marker_hsv_upper = np.array([85, 255, 255])
        self.marker_filter = PointFilter2D(min_cutoff=0.8, beta=0.02)
        self.calibrating_marker = False

        # Video recording
        self.is_recording = False
        self.video_writer = None
        self.recording_start_time = 0

        # Notifications & toast banner
        self.toast_msg = "Pinch thumb & index to write! Relax your hand."
        self.toast_time = time.time() + 4.5

        # Toolbar layout
        self._init_toolbar()

        # Hand tracker parameters (lazy initialization)
        self.pinch_start_thresh = pinch_start_thresh
        self.pinch_stop_thresh = pinch_stop_thresh
        self.tracker = None

        # Window setup
        self.window_name = "Air Canvas - OpenCV & MediaPipe"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.cam_width, self.cam_height)

        # Performance monitoring
        self.fps = 0.0
        self._fps_counter = 0
        self._fps_timer = time.time()

    def get_tracker(self):
        """Lazily initializes and returns the HandTracker instance on first hand-tracking use."""
        if self.tracker is None:
            self.tracker = HandTracker(
                max_hands=1,
                min_detection_conf=0.6,
                min_tracking_conf=0.6,
                pinch_start_thresh=self.pinch_start_thresh,
                pinch_stop_thresh=self.pinch_stop_thresh,
            )
        return self.tracker

    def _init_toolbar(self):
        """Precomputes layout of top interactive palette buttons."""
        self.buttons = []
        top = 10
        btn_h = 44
        btn_w = 88
        spacing = 8
        start_x = 15

        # Color buttons
        for i, col in enumerate(COLORS):
            bx = start_x + i * (btn_w + spacing)
            self.buttons.append({
                "type": "color",
                "idx": i,
                "name": col["name"],
                "color_bgr": col["bgr"],
                "rect": (bx, top, btn_w, btn_h),
            })

        # Eraser button
        bx = start_x + len(COLORS) * (btn_w + spacing)
        self.buttons.append({
            "type": "eraser",
            "name": "Eraser",
            "color_bgr": (40, 40, 40),
            "rect": (bx, top, btn_w, btn_h),
        })

        # Clear button
        bx += btn_w + spacing
        self.buttons.append({
            "type": "clear",
            "name": "Clear",
            "color_bgr": (30, 30, 90),
            "rect": (bx, top, btn_w, btn_h),
        })

        # Mode button
        bx += btn_w + spacing
        self.buttons.append({
            "type": "input_mode",
            "name": "Mode [G]",
            "color_bgr": (50, 40, 80),
            "rect": (bx, top, 110, btn_h),
        })

        self._hover_btn_idx = None
        self._hover_start_time = 0

    def show_toast(self, text, duration=2.5):
        """Displays temporary notification message on screen."""
        self.toast_msg = text
        self.toast_time = time.time() + duration

    def set_color(self, idx):
        """Selects drawing color."""
        self.color_idx = idx % len(COLORS)
        self.is_eraser = False
        self.show_toast(f"Color: {COLORS[self.color_idx]['name']}")

    def toggle_eraser(self):
        """Toggles between pen and eraser."""
        self.is_eraser = not self.is_eraser
        status = "Eraser active" if self.is_eraser else f"Pen active ({COLORS[self.color_idx]['name']})"
        self.show_toast(status)

    def cycle_input_mode(self):
        """Cycles between Pinch-to-Draw, Physical Marker, Key Hold, and Index Finger."""
        self.input_mode_idx = (self.input_mode_idx + 1) % len(INPUT_MODES)
        self.show_toast(f"Mode: {INPUT_MODES[self.input_mode_idx]}", duration=3.0)

    def clear_canvas(self):
        """Clears all strokes and resets canvas."""
        self.strokes.clear()
        self.current_stroke = None
        self.prev_pt = None
        self.drawing_canvas[:] = 0
        self.show_toast("Canvas cleared")

    def undo_stroke(self):
        """Removes the last stroke drawn."""
        if self.strokes:
            self.strokes.pop()
            self._redraw_canvas()
            self.show_toast("Undo last stroke")
        else:
            self.show_toast("Nothing to undo")

    def _redraw_canvas(self):
        """Redraws all strokes with smooth quadratic Bézier interpolation."""
        self.drawing_canvas[:] = 0
        for stroke in self.strokes:
            stroke.draw_full(self.drawing_canvas)

    def toggle_recording(self):
        """Starts or stops session video recording."""
        if not self.is_recording:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.avi"
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            fps = self.fps if self.fps > 10.0 else 25.0
            self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (self.cam_width, self.cam_height))
            self.is_recording = True
            self.recording_start_time = time.time()
            self.show_toast(f"Recording started -> {filename}", duration=3.0)
        else:
            self.is_recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.show_toast("Recording saved!", duration=3.0)

    def compose_frame(self, frame=None):
        """
        Composites strokes onto the active background mode (Blackboard, Webcam AR, Whiteboard).
        Returns the composited BGR image.
        """
        glowed = self._render_glow(self.drawing_canvas)

        if self.bg_mode_idx == 0:  # Blackboard
            return glowed
        elif self.bg_mode_idx == 1:  # Webcam AR overlay
            if frame is None and self.cap.isOpened():
                ret, raw_frame = self.cap.read()
                if ret:
                    frame = cv2.flip(raw_frame, 1)
            if frame is not None:
                return self._blend_ar(frame, glowed)
            return glowed
        else:  # Whiteboard
            shape = frame.shape if frame is not None else self.drawing_canvas.shape
            output = np.full(shape, 255, dtype=np.uint8)
            gray = cv2.cvtColor(glowed, cv2.COLOR_BGR2GRAY)
            mask = gray > 10
            output[mask] = glowed[mask]
            return output

    def save_image(self):
        """Saves current artwork to PNG."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"air_canvas_{timestamp}.png"
        output = self.compose_frame()
        cv2.imwrite(filename, output)
        self.show_toast(f"Saved to {filename}!")
        print(f"[AirCanvas] Saved artwork to: {os.path.abspath(filename)}")

    def _render_glow(self, canvas):
        """
        Fast real-time bloom glow effect.
        Matches HTML5 Canvas shadowBlur: 15 shadowColor.
        """
        h, w = canvas.shape[:2]
        small = cv2.resize(canvas, (w // 4, h // 4), interpolation=cv2.INTER_LINEAR)
        blurred = cv2.GaussianBlur(small, (15, 15), 0)
        glow = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.addWeighted(canvas, 1.0, glow, 1.35, 0)

    def _blend_ar(self, background, glowed_canvas):
        """Blends neon strokes onto webcam video background."""
        gray = cv2.cvtColor(glowed_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        bg_part = cv2.bitwise_and(background, background, mask=mask_inv)
        fg_part = cv2.bitwise_and(glowed_canvas, glowed_canvas, mask=mask)
        return cv2.add(bg_part, fg_part)

    def _track_physical_marker(self, frame):
        """Tracks the tip of any real colored marker or pen cap."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.marker_hsv_lower, self.marker_hsv_upper)
        # Morphological noise filtering
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 60:
                ((cx, cy), radius) = cv2.minEnclosingCircle(c)
                now = time.time()
                sx, sy = self.marker_filter.filter(cx, cy, now)
                return True, (int(sx), int(sy)), int(radius)

        self.marker_filter.reset()
        return False, None, 0

    def _calibrate_marker_color(self, frame):
        """Samples HSV color from center reticle box."""
        h, w = frame.shape[:2]
        box_size = 40
        cx, cy = w // 2, h // 2
        roi = frame[cy - box_size // 2 : cy + box_size // 2, cx - box_size // 2 : cx + box_size // 2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mean_h = np.median(hsv_roi[:, :, 0])
        mean_s = np.median(hsv_roi[:, :, 1])
        mean_v = np.median(hsv_roi[:, :, 2])

        self.marker_hsv_lower = np.array([max(0, mean_h - 15), max(50, mean_s - 60), max(50, mean_v - 60)])
        self.marker_hsv_upper = np.array([min(179, mean_h + 15), min(255, mean_s + 60), min(255, mean_v + 60)])
        self.calibrating_marker = False
        self.show_toast("Marker color calibrated!", duration=2.5)

    def _check_toolbar_selection(self, x, y):
        """Checks if cursor is hovering over a toolbar button with dwell time."""
        now = time.time()
        for i, btn in enumerate(self.buttons):
            bx, by, bw, bh = btn["rect"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                if self._hover_btn_idx != i:
                    self._hover_btn_idx = i
                    self._hover_start_time = now
                elif (now - self._hover_start_time) > 0.35:  # 350ms dwell
                    if btn["type"] == "color":
                        self.set_color(btn["idx"])
                    elif btn["type"] == "eraser":
                        self.toggle_eraser()
                    elif btn["type"] == "clear":
                        self.clear_canvas()
                    elif btn["type"] == "input_mode":
                        self.cycle_input_mode()
                    self._hover_btn_idx = None
                    self._hover_start_time = now + 0.8
                return i
        self._hover_btn_idx = None
        return None

    def _draw_toolbar(self, frame):
        """Renders header toolbar."""
        if not self.show_hud:
            return

        h, w = frame.shape[:2]
        header_h = 65
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), (18, 18, 22), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        for i, btn in enumerate(self.buttons):
            bx, by, bw, bh = btn["rect"]
            b_type = btn["type"]

            is_selected = False
            if b_type == "color" and not self.is_eraser and self.color_idx == btn["idx"]:
                is_selected = True
            elif b_type == "eraser" and self.is_eraser:
                is_selected = True

            is_hovered = (self._hover_btn_idx == i)

            if b_type == "color":
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), btn["color_bgr"], -1)
            else:
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), btn["color_bgr"], -1)

            border_color = (255, 255, 255) if (is_selected or is_hovered) else (70, 70, 80)
            thickness = 3 if is_selected else (2 if is_hovered else 1)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), border_color, thickness)

            text = btn["name"]
            text_color = (15, 15, 15) if (b_type == "color" and btn["name"] in ["Ice Cyan", "Pure White", "Neon Green", "Amber Gold"]) else (240, 240, 240)
            font_scale = 0.40
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            tx = bx + (bw - tw) // 2
            ty = by + (bh + th) // 2
            cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)

    def _draw_hud(self, frame):
        """Renders footer bar, status, and recording indicator."""
        if not self.show_hud:
            return

        h, w = frame.shape[:2]

        mode_str = f"Input: {INPUT_MODES[self.input_mode_idx]} [G]"
        bg_str = f"Bg: {BACKGROUND_MODES[self.bg_mode_idx]} [B]"
        tool_str = "Eraser" if self.is_eraser else f"{COLORS[self.color_idx]['name']}"
        size_str = f"{self.eraser_thickness if self.is_eraser else self.pen_thickness}px [+/-]"
        shortcuts = "[Space] Clear | [U] Undo | [S] Save | [R] Record | [F] Fullscreen | [Q] Quit"

        info_text = f"{mode_str}  |  {bg_str}  |  Tool: {tool_str}  |  Size: {size_str}  |  FPS: {self.fps:.0f}"

        (info_w, info_h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)
        (sc_w, _), _ = cv2.getTextSize(shortcuts, cv2.FONT_HERSHEY_SIMPLEX, 0.39, 1)

        margin = 15
        spacing = 20
        two_lines = (info_w + sc_w + margin * 2 + spacing > w)

        bar_h = 52 if two_lines else 34
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (14, 14, 18), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        if two_lines:
            cv2.putText(frame, info_text, (margin, h - bar_h + info_h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 210, 210), 1, cv2.LINE_AA)
            cv2.putText(frame, shortcuts, (margin, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (150, 150, 160), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, info_text, (margin, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 210, 210), 1, cv2.LINE_AA)
            cv2.putText(frame, shortcuts, (w - sc_w - margin, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (150, 150, 160), 1, cv2.LINE_AA)

        # Recording pulse indicator
        if self.is_recording:
            dur = int(time.time() - self.recording_start_time)
            rec_str = f"REC {dur // 60:02d}:{dur % 60:02d}"
            # Blink red dot every second
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(frame, (w - 110, 32), 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, rec_str, (w - 95, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2, cv2.LINE_AA)

        # Marker calibration reticle
        if self.calibrating_marker:
            cx, cy = w // 2, h // 2
            cv2.rectangle(frame, (cx - 25, cy - 25), (cx + 25, cy + 25), (0, 255, 255), 2)
            cv2.putText(frame, "Place marker tip in box & press T", (cx - 130, cy + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

        # Toast notification
        if time.time() < self.toast_time:
            (tw, th), _ = cv2.getTextSize(self.toast_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            px, py = (w - tw) // 2, 95
            pad = 8
            cv2.rectangle(frame, (px - pad - 6, py - th - pad), (px + tw + pad + 6, py + pad), (25, 25, 30), -1)
            cv2.rectangle(frame, (px - pad - 6, py - th - pad), (px + tw + pad + 6, py + pad), (100, 200, 255), 1)
            cv2.putText(frame, self.toast_msg, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_cursor(self, frame, x, y, is_drawing, is_hovering, pinch_ratio=None):
        """Draws responsive, glowing cursor showing pen contact state."""
        color = COLORS[self.color_idx]["bgr"]
        radius = self.eraser_thickness // 2 if self.is_eraser else self.pen_thickness

        if self.is_eraser:
            cv2.circle(frame, (x, y), radius, (220, 220, 220), 2, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
        elif is_drawing:
            # Drawing: solid white core with glowing colored halo
            cv2.circle(frame, (x, y), max(radius + 5, 9), color, 2, cv2.LINE_AA)
            cv2.circle(frame, (x, y), max(radius, 5), (255, 255, 255), -1, cv2.LINE_AA)
        else:
            # Hover: open ring with colored center dot
            cv2.circle(frame, (x, y), max(radius + 6, 10), color, 1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), max(radius - 1, 4), color, -1, cv2.LINE_AA)

        # When in Pinch mode, show pinch readiness indicator
        if self.input_mode_idx == 0 and pinch_ratio is not None and not is_drawing:
            p_ring = int(max(radius + 6, 10) + min(30, max(0, pinch_ratio * 40)))
            cv2.circle(frame, (x, y), p_ring, (120, 120, 140), 1, cv2.LINE_AA)

    def run(self):
        """Main application loop."""
        print("[AirCanvas] Starting Air Canvas loop. Press 'q' or 'Esc' to quit.")

        wait_delay = max(1, int(1000 / self.video_fps)) if self.is_video_file and self.video_fps > 0 else 1

        while self.cap.isOpened():
            success, raw_frame = self.cap.read()
            if not success:
                if self.is_video_file:
                    print("[AirCanvas] End of video file reached.")
                else:
                    print("[AirCanvas] Camera capture failed.")
                break

            # Mirror frame horizontally for natural interaction
            frame = cv2.flip(raw_frame, 1)

            # Update FPS counter
            self._fps_counter += 1
            now = time.time()
            if now - self._fps_timer >= 1.0:
                self.fps = self._fps_counter / (now - self._fps_timer)
                self._fps_counter = 0
                self._fps_timer = now

            cursor_x, cursor_y = None, None
            is_drawing = False
            is_hovering = False
            pinch_ratio = None

            # Input Mode 1: Physical Marker Tracking
            if self.input_mode_idx == 1:
                tracked, pt, _ = self._track_physical_marker(frame)
                if tracked:
                    cursor_x, cursor_y = pt
                    if not self.calibrating_marker:
                        is_drawing = self.key_drawing
                        is_hovering = not self.key_drawing
            else:
                # Hand Tracking Modes (Pinch, Key, Index) - lazily initializes HandTracker
                tracker = self.get_tracker()
                hand_data = tracker.find_hands(frame)
                if hand_data["detected"]:
                    cursor_x, cursor_y = hand_data["pen_tip"]
                    pinch_ratio = hand_data["pinch_ratio"]

                    # Determine drawing state based on selected input mode
                    if self.input_mode_idx == 0:  # Pinch-to-Draw (Ergonomic)
                        is_drawing = hand_data["is_pinched"]
                        is_hovering = not is_drawing
                    elif self.input_mode_idx == 2:  # Key Hold (Space / D)
                        is_drawing = self.key_drawing
                        is_hovering = not self.key_drawing
                    elif self.input_mode_idx == 3:  # Legacy Index finger
                        is_drawing = hand_data["is_index_only"]
                        is_hovering = hand_data["is_two_fingers"] or hand_data["is_open_palm"]

            # Toolbar interaction when hovering
            if is_hovering and cursor_x is not None and cursor_y is not None:
                if cursor_y < 75:
                    self._check_toolbar_selection(cursor_x, cursor_y)

            # Draw stroke if in drawing state and outside toolbar area
            if is_drawing and cursor_x is not None and cursor_y is not None and cursor_y >= 70:
                current_pt = (cursor_x, cursor_y)
                color = (0, 0, 0) if self.is_eraser else COLORS[self.color_idx]["bgr"]
                thickness = self.eraser_thickness if self.is_eraser else self.pen_thickness

                if self.current_stroke is None:
                    self.current_stroke = Stroke(color, thickness, self.is_eraser)
                    self.strokes.append(self.current_stroke)
                    self.current_stroke.add_point(current_pt)
                    self.prev_pt = current_pt
                else:
                    # Minimum distance filter (prevents stationary ink blob jitter)
                    dist = math.hypot(current_pt[0] - self.prev_pt[0], current_pt[1] - self.prev_pt[1])
                    if dist >= 2.5:
                        self.current_stroke.add_point(current_pt)
                        self.current_stroke.draw_latest_curve(self.drawing_canvas)
                        self.prev_pt = current_pt
            else:
                # Finger or marker lifted
                self.current_stroke = None
                self.prev_pt = None

            # Compose display based on background mode
            display = self.compose_frame(frame)

            # Render cursor
            if cursor_x is not None and cursor_y is not None:
                self._draw_cursor(display, cursor_x, cursor_y, is_drawing, is_hovering, pinch_ratio)

            # Render toolbar and HUD
            self._draw_toolbar(display)
            self._draw_hud(display)

            # Write frame if recording
            if self.is_recording and self.video_writer is not None:
                self.video_writer.write(display)

            # Show window
            cv2.imshow(self.window_name, display)

            # Keyboard controls
            key = cv2.waitKey(wait_delay) & 0xFF
            if key in [ord("q"), 27]:  # 'q' or Esc to quit
                break
            elif key == ord(" "):  # Space: toggle pen in Key/Marker mode, or clear canvas
                if self.input_mode_idx in (1, 2):
                    self.key_drawing = not self.key_drawing
                    self.show_toast(f"Pen {'Down' if self.key_drawing else 'Up'}")
                else:
                    self.clear_canvas()
            elif key in [ord("u"), ord("U")]:  # 'u' to undo
                self.undo_stroke()
            elif key in [ord("s"), ord("S")]:  # 's' to save PNG
                self.save_image()
            elif key in [ord("r"), ord("R")]:  # 'r' to toggle video recording
                self.toggle_recording()
            elif key in [ord("c"), ord("C")]:  # 'c' to cycle color
                self.set_color(self.color_idx + 1)
            elif key in [ord("e"), ord("E")]:  # 'e' for eraser
                self.toggle_eraser()
            elif key in [ord("b"), ord("B")]:  # 'b' to toggle background
                self.bg_mode_idx = (self.bg_mode_idx + 1) % len(BACKGROUND_MODES)
                self.show_toast(f"Background: {BACKGROUND_MODES[self.bg_mode_idx]}")
            elif key in [ord("g"), ord("G")]:  # 'g' to cycle input mode
                self.cycle_input_mode()
            elif key in [ord("d"), ord("D")]:  # 'd' to toggle pen in key / marker mode
                self.key_drawing = not self.key_drawing
                self.show_toast(f"Pen {'Down' if self.key_drawing else 'Up'}")
            elif key in [ord("t"), ord("T")]:  # 't' to calibrate physical marker color
                if not self.calibrating_marker:
                    self.calibrating_marker = True
                    self.show_toast("Place marker in box and press T", duration=4.0)
                else:
                    self._calibrate_marker_color(frame)
            elif key in [ord("+"), ord("=")]:  # '+' increase thickness
                if self.is_eraser:
                    self.eraser_thickness = min(80, self.eraser_thickness + 5)
                else:
                    self.pen_thickness = min(30, self.pen_thickness + 2)
                self.show_toast(f"Thickness: {self.eraser_thickness if self.is_eraser else self.pen_thickness}px")
            elif key in [ord("-"), ord("_")]:  # '-' decrease thickness
                if self.is_eraser:
                    self.eraser_thickness = max(10, self.eraser_thickness - 5)
                else:
                    self.pen_thickness = max(2, self.pen_thickness - 2)
                self.show_toast(f"Thickness: {self.eraser_thickness if self.is_eraser else self.pen_thickness}px")
            elif key in [ord("f"), ord("F")]:  # 'f' for fullscreen
                self.fullscreen = not self.fullscreen
                prop = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, prop)
            elif key in [ord("h"), ord("H")]:  # 'h' toggle HUD
                self.show_hud = not self.show_hud
            elif ord("1") <= key <= ord("6"):  # Number keys for color presets
                self.set_color(key - ord("1"))

        self.cleanup()

    def cleanup(self):
        """Releases camera, windows, and models."""
        print("[AirCanvas] Shutting down...")
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = AirPenApp()
    app.run()
