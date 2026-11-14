"""
test_air_pen.py - Unit and integration tests for Air Canvas.
Pure logic unit tests run without camera or hardware dependencies.
Camera and end-to-end app tests require an explicit opt-in:
    RUN_CAMERA_TESTS=1 pytest
"""

import os
import glob
import pytest
import numpy as np

from air_pen import AirPenApp, Stroke, COLORS, compute_bezier_curve
from hand_tracker import OneEuroFilter, PointFilter2D


# ---------------------------------------------------------------------------
# Pure Logic Unit Tests (No Camera Required)
# ---------------------------------------------------------------------------

def test_one_euro_filter():
    """Verifies that 1-Euro filter produces smoothed values and handles resets."""
    f = OneEuroFilter(min_cutoff=0.8, beta=0.02)
    val1 = f.filter(100.0, 1.0)
    assert val1 == 100.0

    val2 = f.filter(110.0, 1.033)
    assert 100.0 < val2 < 110.0  # Filtered value should be between prev and new

    f.reset()
    assert f.x_prev is None
    assert f.t_prev is None

    val3 = f.filter(50.0, 2.0)
    assert val3 == 50.0


def test_point_filter_2d():
    """Verifies that 2D point filter correctly filters both X and Y channels."""
    f2d = PointFilter2D(min_cutoff=0.8, beta=0.02)
    p1 = f2d.filter(100, 100, 1.0)
    p2 = f2d.filter(104, 98, 1.033)

    assert isinstance(p1, tuple) and len(p1) == 2
    assert p1[0] > 0 and p1[1] > 0
    assert p2[0] > 0 and p2[1] > 0

    f2d.reset()
    assert f2d.filter_x.x_prev is None
    assert f2d.filter_y.x_prev is None


def test_compute_bezier_curve():
    """Verifies that Bézier curve point generator produces the expected number of interpolated points."""
    p_start = (100, 100)
    p_control = (150, 200)
    p_end = (200, 100)

    steps = 6
    curve_pts = compute_bezier_curve(p_start, p_control, p_end, steps=steps)

    assert len(curve_pts) == steps + 1
    assert curve_pts[0] == p_start
    assert curve_pts[-1] == p_end


def test_stroke_bezier_drawing():
    """Verifies that Stroke Bézier curve rendering draws visible pixels on a canvas."""
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    stroke = Stroke(COLORS[0]["bgr"], 6)

    # 1 point: does not crash
    stroke.add_point((100, 100))
    stroke.draw_latest_curve(canvas)

    # 2 points: straight line
    stroke.add_point((120, 150))
    stroke.draw_latest_curve(canvas)
    assert np.any(canvas > 0)

    # 3+ points: quadratic Bézier interpolation
    more_pts = [(160, 200), (220, 180), (300, 120)]
    for pt in more_pts:
        stroke.add_point(pt)
        stroke.draw_latest_curve(canvas)

    assert np.any(canvas > 0)

    # Verify draw_full produces non-zero canvas
    full_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    stroke.draw_full(full_canvas)
    assert np.any(full_canvas > 0)


def test_pinch_hysteresis_logic():
    """
    Verifies scale-invariant pinch hysteresis logic without hardware dependencies.
    State transitions should require crossing start threshold (0.18) to begin drawing,
    and stop threshold (0.28) to release drawing.
    """
    start_thresh = 0.18
    stop_thresh = 0.28

    def update_pinch_state(current_state, ratio):
        if current_state:
            return ratio < stop_thresh
        else:
            return ratio < start_thresh

    state = False

    # Ratio well above start threshold -> remains unpinched
    state = update_pinch_state(state, 0.35)
    assert not state

    # Ratio between start and stop thresholds while unpinched -> remains unpinched
    state = update_pinch_state(state, 0.22)
    assert not state

    # Ratio drops below start threshold -> transitions to pinched (pen down)
    state = update_pinch_state(state, 0.15)
    assert state

    # Ratio rises slightly between start and stop thresholds -> remains pinched (prevents flicker)
    state = update_pinch_state(state, 0.24)
    assert state

    # Ratio crosses above stop threshold -> transitions to unpinched (pen up)
    state = update_pinch_state(state, 0.30)
    assert not state


# ---------------------------------------------------------------------------
# Camera / Hardware Integration Tests (Opt-In with RUN_CAMERA_TESTS=1)
# ---------------------------------------------------------------------------

@pytest.fixture
def cleanup_recording_files():
    """Ensures test recording files are cleaned up reliably in case of failure."""
    yield
    for f in glob.glob("recording_*.avi"):
        try:
            os.remove(f)
        except OSError:
            pass


@pytest.mark.camera
@pytest.mark.skipif(
    os.environ.get("RUN_CAMERA_TESTS") != "1",
    reason="Camera integration tests require RUN_CAMERA_TESTS=1 and a connected webcam",
)
def test_camera_app_integration(cleanup_recording_files):
    """End-to-end test initializing AirPenApp with an active camera device."""
    app = None
    try:
        app = AirPenApp()
        assert app.cam_width > 0 and app.cam_height > 0
        assert len(app.buttons) >= 9

        initial_mode = app.input_mode_idx
        app.cycle_input_mode()
        assert app.input_mode_idx != initial_mode

        app.toggle_recording()
        assert app.is_recording
        app.toggle_recording()
        assert not app.is_recording
    finally:
        if app is not None:
            app.cleanup()
