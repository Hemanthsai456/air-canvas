"""
hand_tracker.py - High-Precision Hand & Gesture Tracking
Features:
- Dual MediaPipe Engine (Modern Tasks Vision API 1.x+ & Legacy Solutions API)
- Adaptive 1-Euro Filter for smoothed, low-jitter cursor tracking
- Ergonomic Pinch-to-Draw gesture with scale-invariant hysteresis (relieves hand fatigue)
- Midpoint pen tracking (stabilized by thumb-index contact)
- Physical Color Marker / Stylus tracking mode
"""

import os
import sys
import time
import math
import hashlib
import urllib.request

# Suppress verbose TensorFlow / MediaPipe logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_FILENAME = "hand_landmarker.task"
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


class OneEuroFilter:
    """
    Adaptive 1-Euro Filter for human-computer interaction.
    Filters micro-tremors and low-speed jitter while maintaining responsiveness during fast movement.
    """

    def __init__(self, min_cutoff=0.8, beta=0.015, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, t):
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        dt = max(t - self.t_prev, 1e-4)
        self.t_prev = t

        # Estimate derivative of the signal
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        # Dynamically adapt cutoff frequency based on movement velocity
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat


class PointFilter2D:
    """Combines two 1-Euro filters for 2D (x, y) coordinates."""

    def __init__(self, min_cutoff=0.8, beta=0.02):
        self.filter_x = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.filter_y = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)

    def reset(self):
        self.filter_x.reset()
        self.filter_y.reset()

    def filter(self, x, y, timestamp):
        fx = self.filter_x.filter(float(x), timestamp)
        fy = self.filter_y.filter(float(y), timestamp)
        return fx, fy


class HandTracker:
    """Tracks hand landmarks and extracts ergonomic writing coordinates."""

    def __init__(
        self,
        model_path=None,
        max_hands=1,
        min_detection_conf=0.6,
        min_tracking_conf=0.6,
        pinch_start_thresh=0.18,
        pinch_stop_thresh=0.28,
    ):
        self.max_hands = max_hands
        self.min_detection_conf = min_detection_conf
        self.min_tracking_conf = min_tracking_conf
        self.pinch_start_thresh = pinch_start_thresh
        self.pinch_stop_thresh = pinch_stop_thresh
        self.mode = None
        self.detector = None
        self._start_time = time.time()

        # 1-Euro adaptive point filter
        self.filter_2d = PointFilter2D(min_cutoff=0.7, beta=0.02)

        # Pinch hysteresis state
        self.is_pinched = False

        # Model file resolution
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_path = os.path.join(script_dir, MODEL_FILENAME)
        else:
            self.model_path = model_path

        self._init_mediapipe()

    def _verify_model_hash(self):
        """Verifies the SHA-256 integrity of the hand landmarker model file."""
        hasher = hashlib.sha256()
        with open(self.model_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        actual_hash = hasher.hexdigest()
        if actual_hash != MODEL_SHA256:
            try:
                os.remove(self.model_path)
            except OSError:
                pass
            raise RuntimeError(
                f"Model file '{self.model_path}' failed SHA-256 verification.\n"
                f"Expected: {MODEL_SHA256}\n"
                f"Got:      {actual_hash}\n"
                f"The corrupted file was removed. Please restart to re-download."
            )

    def _ensure_model_exists(self):
        """Downloads hand_landmarker.task model if missing and verifies integrity."""
        if not os.path.exists(self.model_path):
            print(f"[HandTracker] Downloading MediaPipe model to '{self.model_path}'...")
            try:
                urllib.request.urlretrieve(MODEL_URL, self.model_path)
                print(f"[HandTracker] Model downloaded ({os.path.getsize(self.model_path)} bytes).")
            except Exception as e:
                print(f"[HandTracker] Error downloading model: {e}", file=sys.stderr)
                raise
        self._verify_model_hash()

    def _init_mediapipe(self):
        """Initializes either Tasks Vision API or Solutions Hands API."""
        import mediapipe as mp

        if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
            try:
                self._ensure_model_exists()
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                base_options = python.BaseOptions(model_asset_path=self.model_path)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=self.max_hands,
                    min_hand_detection_confidence=self.min_detection_conf,
                    min_tracking_confidence=self.min_tracking_conf,
                    running_mode=vision.RunningMode.VIDEO,
                )
                self.detector = vision.HandLandmarker.create_from_options(options)
                self.mode = "tasks"
                print("[HandTracker] Initialized with MediaPipe Tasks Vision API.")
                return
            except Exception as ex:
                print(f"[HandTracker] Failed to init Tasks API: {ex}, falling back to solutions...")

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.detector = mp.solutions.hands.Hands(
                max_num_hands=self.max_hands,
                min_detection_confidence=self.min_detection_conf,
                min_tracking_confidence=self.min_tracking_conf,
            )
            self.mode = "solutions"
            print("[HandTracker] Initialized with MediaPipe Solutions Hands API.")
            return

        raise RuntimeError("No compatible MediaPipe hand tracking interface found.")

    def reset_filters(self):
        """Resets coordinate filters when tracking is lost."""
        self.filter_2d.reset()
        self.is_pinched = False

    def find_hands(self, frame_bgr):
        """
        Processes a BGR video frame and extracts smooth pen coordinates and gestures.
        """
        import cv2
        import mediapipe as mp

        h, w, _ = frame_bgr.shape
        now = time.time()
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        raw_landmarks = None

        if self.mode == "tasks":
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((now - self._start_time) * 1000)
            result = self.detector.detect_for_video(mp_image, timestamp_ms)
            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                raw_landmarks = result.hand_landmarks[0]
        elif self.mode == "solutions":
            result = self.detector.process(rgb_frame)
            if result.multi_hand_landmarks and len(result.multi_hand_landmarks) > 0:
                raw_landmarks = result.multi_hand_landmarks[0].landmark

        if raw_landmarks is None:
            self.reset_filters()
            return {"detected": False}

        # Convert landmarks to pixel coordinates
        landmarks = []
        for lm in raw_landmarks:
            px, py = int(lm.x * w), int(lm.y * h)
            landmarks.append((px, py, lm.z))

        # Key landmark points
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_mcp = landmarks[9]
        middle_tip = landmarks[12]

        # Finger states [thumb, index, middle, ring, pinky]
        fingers = self._get_fingers_up(landmarks)

        # Hand scale reference (wrist to middle MCP joint distance)
        hand_scale = math.hypot(wrist[0] - middle_mcp[0], wrist[1] - middle_mcp[1])
        hand_scale = max(hand_scale, 20.0)

        # Distance between thumb tip and index tip
        pinch_dist = math.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
        pinch_ratio = pinch_dist / hand_scale

        # Hysteresis for pinch: requires closer contact to start, slightly wider to stop
        # This completely prevents fluttering or flickering!
        if self.is_pinched:
            self.is_pinched = (pinch_ratio < self.pinch_stop_thresh)
        else:
            self.is_pinched = (pinch_ratio < self.pinch_start_thresh)

        # Virtual pen tip position:
        # When pinched, the pen tip is the stabilized midpoint between thumb and index!
        # (This physical contact locks the coordinates firmly in place, removing tremor)
        if self.is_pinched:
            raw_pen_x = (thumb_tip[0] + index_tip[0]) / 2.0
            raw_pen_y = (thumb_tip[1] + index_tip[1]) / 2.0
        else:
            raw_pen_x = float(index_tip[0])
            raw_pen_y = float(index_tip[1])

        # Apply 1-Euro adaptive filter for coordinates
        smooth_x, smooth_y = self.filter_2d.filter(raw_pen_x, raw_pen_y, now)

        # Gestures
        # 1. Pinch gesture (ergonomic, natural pen grip)
        # 2. Legacy index-only gesture
        is_index_only = (fingers[1] and not fingers[2] and not fingers[3] and not fingers[4])
        # 3. Two fingers up (hover / selection)
        is_two_fingers = (fingers[1] and fingers[2] and not fingers[3] and not fingers[4])
        # 4. Open palm
        is_open_palm = all(fingers)

        return {
            "detected": True,
            "landmarks": landmarks,
            "pen_tip": (int(smooth_x), int(smooth_y)),
            "pen_tip_raw": (int(raw_pen_x), int(raw_pen_y)),
            "index_tip": (index_tip[0], index_tip[1]),
            "thumb_tip": (thumb_tip[0], thumb_tip[1]),
            "middle_tip": (middle_tip[0], middle_tip[1]),
            "is_pinched": self.is_pinched,
            "pinch_ratio": pinch_ratio,
            "fingers_up": fingers,
            "is_index_only": is_index_only,
            "is_two_fingers": is_two_fingers,
            "is_open_palm": is_open_palm,
        }

    def _get_fingers_up(self, landmarks):
        """Determines which fingers are extended."""
        fingers = []

        # Thumb: compare distance from thumb tip (4) to pinky base (17) vs thumb MCP (2) to pinky base (17)
        d_tip_pinky = math.hypot(landmarks[4][0] - landmarks[17][0], landmarks[4][1] - landmarks[17][1])
        d_mcp_pinky = math.hypot(landmarks[2][0] - landmarks[17][0], landmarks[2][1] - landmarks[17][1])
        fingers.append(d_tip_pinky > (d_mcp_pinky * 1.15))

        # 4 Fingers: tip Y vs PIP joint Y
        finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
        for tip_id, pip_id in finger_pairs:
            fingers.append(landmarks[tip_id][1] < landmarks[pip_id][1])

        return fingers

    def close(self):
        """Releases detector resources."""
        if self.detector is not None:
            if hasattr(self.detector, "close"):
                self.detector.close()
            self.detector = None
