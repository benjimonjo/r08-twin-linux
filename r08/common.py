"""
common.py - constants, theme, config file, tiny helpers.  (no heavy dependencies)

PROJECT CONVENTIONS (read this first)
  * Pose  = numpy array [X, Y, Z (inches), Xa, Ya, Za (degrees)] - exactly what the FANUC panel prints.
            Xa/Ya/Za are FANUC W/P/R.  Position is the tool-tip (TCP) in the FANUC WORLD frame.
  * Robot-base frame = metres, +Z up, origin on the J1 axis at the ground.
  * WORLD frame origin = on the J1 axis at the J2 height (see kinematics.TABLE_ORIGIN_IN_BASE_M).
  * Colours are hex strings in C; numpy/OpenCV code converts with hex_rgb().
  * r08_config.json (next to run.py) stores everything the user set up.  Old keys are kept
    so a config written by the previous r08_live_twin.py still works.
"""
import os, sys, json, math, time, ctypes
from types import SimpleNamespace

if sys.platform == "win32":          # one coordinate system for mss / pynput / Tk
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

import numpy as np
import threading

clock = time.monotonic
IN = 0.0254                                   # metres per inch
AXES = ["X", "Y", "Z", "Xa", "Ya", "Za"]
UNITS = ["in", "in", "in", "\u00b0", "\u00b0", "\u00b0"]
MINUS = "\u2212"

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PKG_DIR)
CONFIG_FILE = os.path.join(ROOT_DIR, "r08_config.json")
DIGIT_BANK_FILE = os.path.join(ROOT_DIR, "r08_digits.npz")
DEBUG_DIR = os.path.join(ROOT_DIR, "ocr_debug")
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"   # only used for the optional 'Step size' text


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def nice_btn(name):                           # "Xa-" -> "Xa\u2212" for display
    return name.replace("-", MINUS)


# ---- jog panel on the stream: (position step in, angle step in degrees) ----
STEP_LEVELS = [(36.0, 90.0), (12.0, 45.0), (5.0, 22.5), (1.0, 2.0), (0.25, 0.5), (0.05, 0.1)]
START_LEVEL = 2
POS_READ_RES = 0.1            # position read-out resolution (one decimal)
ANG_READ_RES = 1.0            # angle read-out resolution (whole degrees)

DEFAULT_SETTINGS = {
    "cursor_speed": 2400.0,   # px / second while the cursor glides to a button
    "settle": 0.8,            # pose must stay unchanged this long (s) before it is checked
    "motion_wait": 4.0,       # how long to wait for the robot to start moving after a click (s)
    "click_tol": 34,          # a click this close (px) to the target counts
    "pos_tol": 0.15,          # accepted position error (in)
    "ang_tol": 1.0,           # accepted angle error (deg)
    "verify_each": True,      # check the pose after every click
    "step_plus_larger": True, # does the panel's Step+ make the step bigger?
    "ocr_fps": 15,            # screen scans per second
    "viz_deg_s": 25.0,        # max speed of the DRAWN arm (real robot at low override is slow)
    "pose_smoothing": 0.28,   # visual pose interpolation strength
    "camera_pose_blend": 0.45, # camera-assisted pose contribution when calibrated
    "self_learn": True,       # digit reader adds confident new glyph samples to its bank
    "gpu": True,              # use the OpenGL renderer when moderngl is installed
    "show_axes": True,        # show world coordinate axes in the 3D view
    "show_perf": True,        # show the FPS/GPU usage counter
    "show_backend": True,     # show the renderer name in the 3D view
    "controller_enabled": False,
    "controller_input_mode": "plan",
    "controller_axis_invert": {},
    "controller_repeat_delay": 0.12,  # seconds between held-button/D-pad repeats
}

# ---- theme ----
C = SimpleNamespace(
    bg="#13151a", panel="#191c22", card="#1f232b", raised="#2a2f39", hover="#353b47", line="#303641",
    text="#e8e9ed", muted="#8d95a2", faint="#5d6573",
    yellow="#f2c230", yellow_d="#c99a0b", green="#4fbf7a", amber="#f0a020", red="#e5534b",
    blue="#6aa6ff", black="#17191d", grey="#343941", rob_red="#b83232", view_top="#1b1f27", view_bot="#0f1115",
)


class Config:
    """Tiny JSON key/value store. get/set/save."""

    def __init__(self, path):
        self.path, self.d = path, {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                self.d = json.load(fh)
        except Exception:
            self.d = {}

    def get(self, key, default=None):
        return self.d.get(key, default)

    def set(self, key, value):
        self.d[key] = value

    def save(self):
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.d, fh, indent=2)
            os.replace(tmp, self.path)
        except Exception as exc:
            print("Could not save config:", exc)


_capture_lock = threading.Lock()
_wayland_capture = None
_wayland_capture_source = "monitor"


def set_capture_source(source):
    """Choose the primary Wayland portal source for the next capture session."""
    global _wayland_capture, _wayland_capture_source
    source = str(source).lower()
    if source not in ("monitor", "window"):
        raise ValueError("Capture source must be 'monitor' or 'window'.")
    with _capture_lock:
        if _wayland_capture is not None and source != _wayland_capture_source:
            try: _wayland_capture.close()
            except Exception: pass
            _wayland_capture = None
        _wayland_capture_source = source


def new_mss(separate=False, source=None, restore_token=None):
    """Capture with PipeWire portal on Wayland; use mss on other desktops.

    ``separate=True`` opens an independent screen-share session/source.
    ``source`` selects ``monitor`` or ``window`` for Wayland portal capture.
    """
    global _wayland_capture
    if sys.platform.startswith("linux") and os.environ.get("WAYLAND_DISPLAY"):
        if separate:
            from .wayland_capture import WaylandCapture
            return WaylandCapture(source=source or "monitor", restore_token=restore_token)
        with _capture_lock:
            if _wayland_capture is None:
                from .wayland_capture import WaylandCapture
                _wayland_capture = WaylandCapture(
                    source=source or _wayland_capture_source, restore_token=restore_token)
            return _wayland_capture
    import mss
    return mss.MSS() if hasattr(mss, "MSS") else mss.mss()
