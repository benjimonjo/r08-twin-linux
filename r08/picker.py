"""
picker.py - one-time setup overlay: drag a box round the six numbers, click a few jog-panel buttons.

    Picker(root, keys).run() -> {"region":..., "status":..., "step":..., "buttons": {name: [x, y]}} or None
    keys come from ALL_KEYS / BUTTON_KEYS (see STAGE_DEFS for what each step asks).
    compute_buttons() derives the 12 jog buttons from two corner clicks per block.
"""
import math
import tkinter as tk
from PIL import Image, ImageTk
from .common import C, nice_btn, new_mss
from .inputs import WAYLAND_SESSION

STAGE_DEFS = {   # key: (kind, title, hint, optional, short label)
    "region": ("rect", "Drag a box around the six numbers",
               "Just the 2 \u00d7 3 grid of values. Leave out the \u2018Position / Angle\u2019 and X Y Z labels, then press Enter.", False, "numbers"),
    "pos_tl": ("point", "Click the X+ button", "Top-left button of the Position block on the jog panel.", False, "X+"),
    "pos_br": ("point", "Click the Z\u2212 button", "Bottom-right button of the Position block.", False, "Z\u2212"),
    "ang_tl": ("point", "Click the Xa+ button", "Top-left button of the Angle block.", False, "Xa+"),
    "ang_br": ("point", "Click the Za\u2212 button", "Bottom-right button of the Angle block.", False, "Za\u2212"),
    "step_plus": ("point", "Click Step+", "The green button.", False, "Step+"),
    "step_minus": ("point", "Click Step\u2212", "The red button next to it.", False, "Step\u2212"),
    "abort": ("point", "Click Abort", "The red Abort button.", False, "Abort"),
    "review": ("review", "Check the button positions",
               "Drag any marker that isn\u2019t centred on its button, then press Enter.", False, ""),
    "status": ("rect", "Drag a box around the coloured IDLE / MOVING banner",
               "Optional. Lets me see exactly when the robot has stopped.  Esc skips this.", True, "banner"),
    "step_label": ("rect", "Drag a box around the \u2018Step size: \u2026\u2019 text",
                   "Optional. Lets me read the panel\u2019s real step size (needs Tesseract).  Esc skips this.", True, "step text"),
}
BUTTON_KEYS = ["pos_tl", "pos_br", "ang_tl", "ang_br", "step_plus", "step_minus", "abort", "review"]
ALL_KEYS = ["region"] + BUTTON_KEYS + ["status", "step_label"]


def compute_buttons(pts):
    out = {}

    def grid(a, b, top, bot):
        if a in pts and b in pts:
            (x0, y0), (x1, y1) = pts[a], pts[b]
            for c in range(3):
                x = x0 + (x1 - x0) * c / 2
                out[top[c]], out[bot[c]] = (x, y0), (x, y1)
    grid("pos_tl", "pos_br", ["X+", "Y+", "Z+"], ["X-", "Y-", "Z-"])
    grid("ang_tl", "ang_br", ["Xa+", "Ya+", "Za+"], ["Xa-", "Ya-", "Za-"])
    for k, n in (("step_plus", "Step+"), ("step_minus", "Step-"), ("abort", "Abort")):
        if k in pts:
            out[n] = pts[k]
    return {k: [float(v[0]), float(v[1])] for k, v in out.items()}


class Picker:
    """Full-screen overlay on a frozen screenshot."""

    def __init__(self, root, keys, mon_index=0):
        self.root, self.keys, self.i, self.mon_i = root, list(keys), 0, mon_index
        self.in_root = WAYLAND_SESSION
        self.root_geometry, self.root_title = root.geometry(), root.title()
        self.sct = new_mss()
        mons = list(self.sct.monitors)
        self.monitors = mons[1:] if len(mons) > 1 else mons
        self.mon_i = min(self.mon_i, len(self.monitors) - 1)
        self.pts, self.rects, self.markers = {}, {}, {}
        self.pending, self.drag, self.mdrag = None, None, None
        self.win = self.cv = self.photo = None
        self.banner_top, self.result = True, None
        self.done = tk.BooleanVar(master=root, value=False)

    def run(self):
        self._open()
        self.root.wait_variable(self.done)
        return self.result

    def _finish(self, ok):
        if ok:
            out = {}
            if "region" in self.rects:
                out["region"] = self.rects["region"]
            if "status" in self.rects:
                out["status"] = self.rects["status"]
            if "step_label" in self.rects:
                out["step"] = self.rects["step_label"]
            if self.markers:
                out["buttons"] = {k: [float(v[0]), float(v[1])] for k, v in self.markers.items()}
            elif any(k in self.pts for k in ("pos_tl", "step_plus", "abort")):
                out["buttons"] = compute_buttons(self.pts)
            self.result = out
        else:
            self.result = None
        try:
            if self.in_root:
                self.cv.destroy()
                self.root.title(self.root_title)
                self.root.geometry(self.root_geometry)
            else:
                self.win.destroy()
        except Exception:
            pass
        self.done.set(True)

    def _open(self):
        if self.win is not None:
            try:
                if self.in_root:
                    self.cv.destroy()
                else:
                    self.win.destroy()
            except Exception:
                pass
        mon = self.mon = self.monitors[self.mon_i]
        shot = self.sct.grab(mon)
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        # Do not cover the desktop if the capture backend failed or was denied.
        extrema = img.getextrema()
        if all(hi <= 2 for _, hi in extrema):
            raise RuntimeError(
                "Screen capture returned a black image. Check that screen sharing "
                "was allowed in the desktop permission dialog, then retry setup."
            )
        if self.in_root:
            w = self.win = self.root
            w.title("R08 setup — click this window to enable keyboard controls")
        else:
            w = self.win = tk.Toplevel(self.root)
            w.title("R08 setup — click this window to enable keyboard controls")
        w.geometry(f"{mon['width']}x{mon['height']}+{mon['left']}+{mon['top']}")
        self.cv = tk.Canvas(w, highlightthickness=0, bd=0, cursor="crosshair", bg="black")
        if self.in_root:
            self.cv.place(x=0, y=0, relwidth=1, relheight=1)
            w.deiconify()
        else:
            self.cv.pack(fill="both", expand=True)
        w.update()
        cw, ch = max(self.cv.winfo_width(), 2), max(self.cv.winfo_height(), 2)
        if (cw, ch) != img.size:
            img = img.resize((cw, ch))
        self.cw, self.ch = cw, ch
        self.sx, self.sy = mon["width"] / cw, mon["height"] / ch
        self.photo = ImageTk.PhotoImage(img)
        self.cv.create_image(0, 0, anchor="nw", image=self.photo)
        self.cv.bind("<ButtonPress-1>", self._press)
        self.cv.bind("<B1-Motion>", self._motion_drag)
        self.cv.bind("<ButtonRelease-1>", self._release)
        self.cv.bind("<Motion>", self._hover)
        if not self.in_root:
            w.bind("<KeyPress>", self._key)
        self.cv.bind("<KeyPress>", self._key)
        self.cv.focus_set()
        self._redraw()

    def _abs(self, ex, ey):
        return self.mon["left"] + ex * self.sx, self.mon["top"] + ey * self.sy

    def _cv(self, ax, ay):
        return (ax - self.mon["left"]) / self.sx, (ay - self.mon["top"]) / self.sy

    @property
    def key(self):
        return self.keys[self.i]

    @property
    def kind(self):
        return STAGE_DEFS[self.key][0]

    def _next(self):
        self.pending = None
        self.i += 1
        if self.i >= len(self.keys):
            self._finish(True)
            return
        if self.key == "review":
            self.markers = compute_buttons(self.pts)
            if not self.markers:
                self._next()
                return
        self._redraw()

    def _back(self):
        if self.i == 0:
            return
        self.i -= 1
        self.pending = None
        k = self.key
        self.pts.pop(k, None)
        self.rects.pop(k, None)
        if k == "review":
            self.markers = {}
        self._redraw()

    def _skip_buttons(self):
        while self.i < len(self.keys) and self.key in BUTTON_KEYS:
            self.pts.pop(self.key, None)
            self.i += 1
        self.markers, self.pending = {}, None
        if self.i >= len(self.keys):
            self._finish(True)
        else:
            self._redraw()

    def _key(self, e):
        if self.done.get() or not (0 <= self.i < len(self.keys)):
            return
        k = e.keysym
        if k in ("Return", "KP_Enter"):
            if self.kind == "rect" and self.pending:
                x1, y1, x2, y2 = self.pending
                ax1, ay1 = self._abs(x1, y1)
                ax2, ay2 = self._abs(x2, y2)
                self.rects[self.key] = {"left": int(ax1), "top": int(ay1), "width": int(ax2 - ax1), "height": int(ay2 - ay1)}
                self._next()
            elif self.kind == "review":
                self._next()
        elif k == "Escape":
            if STAGE_DEFS[self.key][3]:
                self._next()
            else:
                self._finish(False)
        elif k == "BackSpace":
            self._back()
        elif k in ("m", "M") and len(self.monitors) > 1:
            self.mon_i = (self.mon_i + 1) % len(self.monitors)
            self._open()
        elif k in ("s", "S") and self.key in BUTTON_KEYS:
            self._skip_buttons()

    def _hover(self, e):
        cv = self.cv
        if not cv.find_withtag("xh"):
            cv.create_line(0, 0, 0, 0, fill=C.yellow, width=1, tags=("xh",))
            cv.create_line(0, 0, 0, 0, fill=C.yellow, width=1, tags=("xh",))
        a, b = cv.find_withtag("xh")[:2]
        cv.coords(a, 0, e.y, self.cw, e.y)
        cv.coords(b, e.x, 0, e.x, self.ch)
        if self.drag is None and self.mdrag is None:
            bb = self._banner_box()
            if bb[0] - 30 <= e.x <= bb[2] + 30 and bb[1] - 30 <= e.y <= bb[3] + 30:      # dodge the cursor
                self.banner_top = not self.banner_top
                self._redraw()

    def _near_marker(self, x, y):
        best = None
        for name, (ax, ay) in self.markers.items():
            cx, cy = self._cv(ax, ay)
            d = math.hypot(cx - x, cy - y)
            if d < 22 and (best is None or d < best[0]):
                best = (d, name)
        return best[1] if best else None

    def _press(self, e):
        # Some Wayland window managers ignore focus requests for newly created
        # windows. A user click both focuses the app and keeps keys working on
        # subsequent setup stages.
        self.cv.focus_set()
        if self.kind == "point":
            self.pts[self.key] = self._abs(e.x, e.y)
            self._next()
        elif self.kind == "rect":
            self.drag = (e.x, e.y)
            self.pending = None
        elif self.kind == "review":
            self.mdrag = self._near_marker(e.x, e.y)

    def _motion_drag(self, e):
        if self.drag:
            x0, y0 = self.drag
            self.pending = (min(x0, e.x), min(y0, e.y), max(x0, e.x), max(y0, e.y))
            self._redraw()
        elif self.mdrag:
            self.markers[self.mdrag] = list(self._abs(e.x, e.y))
            self._redraw()

    def _release(self, e):
        if self.drag:
            self.drag = None
            if self.pending and (self.pending[2] - self.pending[0] < 12 or self.pending[3] - self.pending[1] < 8):
                self.pending = None
            self._redraw()
        self.mdrag = None

    def _banner_box(self):
        w, h = min(780, self.cw - 40), 112
        x0 = (self.cw - w) / 2
        y0 = 18 if self.banner_top else self.ch - 18 - h
        return x0, y0, x0 + w, y0 + h

    def _marker(self, ax, ay, label, col):
        x, y = self._cv(ax, ay)
        if not (0 <= x <= self.cw and 0 <= y <= self.ch):
            return
        c = self.cv
        c.create_oval(x - 9, y - 9, x + 9, y + 9, outline=col, width=2, tags="ov")
        c.create_line(x - 15, y, x + 15, y, fill=col, tags="ov")
        c.create_line(x, y - 15, x, y + 15, fill=col, tags="ov")
        t = c.create_text(x, y - 26, text=label, fill=C.text, font=("Segoe UI Semibold", 10), tags="ov")
        bb = c.bbox(t)
        r = c.create_rectangle(bb[0] - 5, bb[1] - 2, bb[2] + 5, bb[3] + 2, fill=C.bg, outline=col, tags="ov")
        c.tag_lower(r, t)

    def _rect(self, x1, y1, x2, y2, label, grid=False):
        c = self.cv
        c.create_rectangle(x1, y1, x2, y2, outline=C.yellow, width=2, tags="ov")
        if grid:
            for k in (1, 2):
                c.create_line(x1 + (x2 - x1) * k / 3, y1, x1 + (x2 - x1) * k / 3, y2, fill=C.yellow, dash=(4, 3), tags="ov")
            c.create_line(x1, (y1 + y2) / 2, x2, (y1 + y2) / 2, fill=C.yellow, dash=(4, 3), tags="ov")
        if label:
            c.create_text(x1 + 4, y1 - 10, text=label, fill=C.yellow, anchor="w", font=("Segoe UI Semibold", 10), tags="ov")

    def _redraw(self):
        c = self.cv
        c.delete("ov")
        for key, r in self.rects.items():
            x1, y1 = self._cv(r["left"], r["top"])
            x2, y2 = self._cv(r["left"] + r["width"], r["top"] + r["height"])
            self._rect(x1, y1, x2, y2, STAGE_DEFS[key][4], grid=(key == "region"))
        if self.pending:
            self._rect(*self.pending, "", grid=(self.key == "region"))
        for key, (ax, ay) in self.pts.items():
            self._marker(ax, ay, STAGE_DEFS[key][4], C.yellow)
        for name, (ax, ay) in self.markers.items():
            self._marker(ax, ay, nice_btn(name), C.green)
        x0, y0, x1, y1 = self._banner_box()
        _, title, hint, optional, _ = STAGE_DEFS[self.key]
        c.create_rectangle(x0, y0, x1, y1, fill=C.bg, outline=C.yellow, width=2, tags="ov")
        c.create_text(x0 + 18, y0 + 12, text=title, anchor="nw", fill=C.text, font=("Segoe UI Semibold", 16), tags="ov")
        c.create_text(x0 + 18, y0 + 42, text=hint, anchor="nw", fill=C.muted, width=x1 - x0 - 36, font=("Segoe UI", 10), tags="ov")
        keys = "Backspace goes back"
        if self.kind == "rect":
            keys = "Drag, then Enter   \u00b7   " + keys
        if self.key in BUTTON_KEYS:
            keys += "   \u00b7   S skips the panel buttons"
        if len(self.monitors) > 1:
            keys += f"   \u00b7   M switches screen ({self.mon_i + 1}/{len(self.monitors)})"
        keys += "   \u00b7   Esc " + ("skips" if optional else "cancels")
        c.create_text(x0 + 18, y1 - 12, text=f"{self.i + 1} of {len(self.keys)}    {keys}", anchor="sw",
                      fill=C.faint, font=("Segoe UI", 9), tags="ov")
