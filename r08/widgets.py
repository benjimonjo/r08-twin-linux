"""
widgets.py - small Tk helpers, fonts, key map, help text.  No application logic.
"""
import numpy as np
import tkinter as tk
from .common import C, AXES

FONT_UI = ("Segoe UI", 10)
FONT_B = ("Segoe UI Semibold", 10)
FONT_H = ("Segoe UI Semibold", 12)
FONT_BIG = ("Segoe UI Semibold", 16)
FONT_NUM = ("Consolas", 17, "bold")
FONT_MONO = ("Consolas", 9)
FONT_SM = ("Segoe UI", 9)

SETTING_DEFS = [   # key, label, low, high, step
    ("cursor_speed", "Cursor speed (pixels / second)", 600, 6000, 100),
    ("click_tol", "Click accuracy needed (pixels)", 10, 80, 2),
    ("settle", "Wait for the robot to be still (s)", 0.3, 3.0, 0.1),
    ("motion_wait", "Wait for it to start moving (s)", 1.0, 10.0, 0.5),
    ("pos_tol", "Position tolerance (in)", 0.05, 1.0, 0.05),
    ("ang_tol", "Angle tolerance (\u00b0)", 0.5, 5.0, 0.5),
    ("viz_deg_s", "Drawn arm speed limit (\u00b0/s)", 5, 120, 5),
]
KEYMAP = {"q": "X+", "a": "X-", "w": "Y+", "s": "Y-", "e": "Z+", "d": "Z-",
          "u": "Xa+", "j": "Xa-", "i": "Ya+", "k": "Ya-", "o": "Za+", "l": "Za-"}
NEEDED_BUTTONS = [f"{a}{s}" for a in AXES for s in "+-"] + ["Step+", "Step-", "Abort"]

HELP_TEXT = """Getting started
  1. Setup \u203a Run full setup.  Drag a box round the six numbers, then click the jog-panel buttons it asks for.
     The banner (IDLE / MOVING) and the \u2018Step size\u2019 text are optional but make checking more reliable.
  2. Digit reader: press \u2018Teach digits\u2019 once or twice (type what the stream shows).  It then learns your screen\u2019s font.
  3. Keep the stream and the jog panel visible \u2013 don\u2019t park this window on top of them.

Aiming and driving
  \u2022 Jog buttons / keys change the TARGET in \u2018Go to a pose\u2019 (the ghost robot previews it).  Press Go to send it.
  \u2022 Go works out the fewest panel clicks.  The pointer glides onto each button and a yellow ring appears.
    YOU click.  I wait for the robot to stop, check the numbers, then continue (re-planning if something differs).

Map objects
  New plate / New pipe: touch the tool tip to the corners / ends, press Set point each time.

Keys
  Jog: Q W E = X+ Y+ Z+ \u2002 A S D = X\u2212 Y\u2212 Z\u2212 \u2002 U I O = Xa+ Ya+ Za+ \u2002 J K L = Xa\u2212 Ya\u2212 Za\u2212
  1 \u2013 6 pick a step size \u2002 F follow the tool \u2002 T trail \u2002 Home reset the view
  Esc stop the job  \u2022  F9 pointer back on the target  \u2022  F10 park the pointer on the panel\u2019s Abort

3D view
  Left-drag orbit \u2022 right- or middle-drag (or Shift + left-drag) pan \u2022 wheel zoom \u2022 double-click resets

Safety
  Nothing is ever clicked for you.  Esc / F10 work from any window.
"""


def mk_button(parent, text, command, bg=None, fg=None, hover=None, font=FONT_B, padx=12, pady=6, **kw):
    bg, hover, fg = bg or C.raised, hover or C.hover, fg or C.text
    b = tk.Button(parent, text=text, command=command, bg=bg, fg=fg, activebackground=hover, activeforeground=fg,
                  relief="flat", bd=0, highlightthickness=0, cursor="hand2", font=font, padx=padx, pady=pady,
                  disabledforeground=C.faint, **kw)
    b._bg, b._hv = bg, hover
    b.bind("<Enter>", lambda e: b.config(bg=b._hv) if str(b["state"]) != "disabled" else None)
    b.bind("<Leave>", lambda e: b.config(bg=b._bg))
    return b


def restyle(b, bg, hover=None, fg=None):
    b._bg, b._hv = bg, hover or bg
    b.config(bg=bg, activebackground=b._hv)
    if fg:
        b.config(fg=fg, activeforeground=fg)


def fmt_pos(v):
    return f"{v + 0.0:+.1f}"


def fmt_ang(v):
    s = f"{v + 0.0:+.1f}"
    return s[:-2] if s.endswith(".0") else s


class JointBar:
    W, Hh = 170, 10

    def __init__(self, parent, lo, hi):
        self.lo, self.hi = lo, hi
        self.cv = tk.Canvas(parent, width=self.W, height=self.Hh, bg=C.card, highlightthickness=0, bd=0)
        self.cv.create_rectangle(0, 3, self.W, 7, fill=C.raised, outline="")
        z = (0 - lo) / (hi - lo) * self.W
        self.bar = self.cv.create_rectangle(z, 3, z, 7, fill=C.yellow_d, outline="")
        self.zero = z
        self.cv.create_line(z, 0, z, self.Hh, fill=C.faint)
        self.mark = self.cv.create_rectangle(0, 0, 3, self.Hh, fill=C.yellow, outline="")

    def set(self, deg):
        f = (deg - self.lo) / (self.hi - self.lo)
        x = float(np.clip(f, 0, 1)) * self.W
        col = C.red if (f < 0 or f > 1) else C.amber if (f < 0.04 or f > 0.96) else C.yellow
        self.cv.coords(self.bar, min(self.zero, x), 3, max(self.zero, x), 7)
        self.cv.coords(self.mark, x - 1.5, 0, x + 1.5, self.Hh)
        self.cv.itemconfig(self.mark, fill=col)
