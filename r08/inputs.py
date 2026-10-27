"""
inputs.py - everything that touches the real mouse / keyboard / overlay windows.  NEVER clicks for the user.

    Cursor    glides the pointer onto a spot (stops if the user grabs the mouse)
    Clicks    global left-click listener (pynput) -> queue of (time, x, y)
    Hotkeys   global Esc (stop) / F9 (pointer to target) / F10 (park on Abort)
    Beacon    click-through ring + label drawn over the button the user should click
Without pynput the viewer still works; guided controls are disabled (HAVE_PYNPUT False).
"""
import sys, math, time, queue, threading, ctypes
import tkinter as tk
from .common import C, clock

try:
    from pynput import mouse as pn_mouse, keyboard as pn_keyboard
    HAVE_PYNPUT = True
except Exception:
    pn_mouse = pn_keyboard = None
    HAVE_PYNPUT = False


def make_click_through(win):
    """Windows only: let mouse clicks pass straight through a topmost overlay window."""
    if sys.platform != "win32":
        return
    try:
        win.update_idletasks()
        user32 = ctypes.windll.user32
        hwnd = user32.GetParent(win.winfo_id()) or win.winfo_id()
        ex = user32.GetWindowLongW(hwnd, -20)
        user32.SetWindowLongW(hwnd, -20, ex | 0x80000 | 0x20 | 0x80 | 0x08000000)
    except Exception:
        pass


def dark_titlebar(win):
    if sys.platform != "win32":
        return
    try:
        win.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(win.winfo_id()) or win.winfo_id()
        val = ctypes.c_int(1)
        for attr in (20, 19):
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(val), ctypes.sizeof(val))
    except Exception:
        pass


class Cursor:
    def __init__(self, settings):
        self.settings, self.gen, self.target = settings, 0, None
        self.ctl = pn_mouse.Controller() if HAVE_PYNPUT else None

    def glide(self, x, y):
        self.target = (int(x), int(y))
        if self.ctl is None:
            return
        self.gen += 1
        threading.Thread(target=self._run, args=(int(x), int(y), self.gen), daemon=True).start()

    def stop(self):
        self.gen += 1

    def resnap(self):
        if self.target:
            self.glide(*self.target)

    def _run(self, x, y, gen):
        try:
            x0, y0 = self.ctl.position
            dist = math.hypot(x - x0, y - y0)
            dur = min(0.9, max(0.14, dist / max(300.0, float(self.settings["cursor_speed"]))))
            last, t_start = (x0, y0), time.perf_counter()
            while True:
                if gen != self.gen:
                    return
                t = (time.perf_counter() - t_start) / dur
                if t >= 1.0:
                    break
                cur = self.ctl.position
                if math.hypot(cur[0] - last[0], cur[1] - last[1]) > 45:      # the user grabbed the mouse
                    return
                e = t * t * (3 - 2 * t)
                last = (int(x0 + (x - x0) * e), int(y0 + (y - y0) * e))
                self.ctl.position = last
                time.sleep(0.004)
            if gen == self.gen:
                self.ctl.position = (x, y)
        except Exception:
            pass


class Clicks:
    def __init__(self):
        self.q = queue.Queue()
        self.listener = None
        if HAVE_PYNPUT:
            try:
                self.listener = pn_mouse.Listener(on_click=self._on_click)
                self.listener.daemon = True
                self.listener.start()
            except Exception:
                self.listener = None

    def _on_click(self, x, y, button, pressed):
        if not pressed and button == pn_mouse.Button.left:      # act on release so the real click completes first
            self.q.put((clock(), x, y))

    def drain(self):
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out

    def stop(self):
        if self.listener:
            self.listener.stop()


class Hotkeys:
    def __init__(self):
        self.stop_flag, self.snap_flag, self.abort_flag = threading.Event(), threading.Event(), threading.Event()
        self.listener = None
        if HAVE_PYNPUT:
            try:
                self.listener = pn_keyboard.Listener(on_press=self._on_press)
                self.listener.daemon = True
                self.listener.start()
            except Exception:
                self.listener = None

    def _on_press(self, key):
        try:
            if key == pn_keyboard.Key.esc:
                self.stop_flag.set()
            elif key == pn_keyboard.Key.f9:
                self.snap_flag.set()
            elif key == pn_keyboard.Key.f10:
                self.abort_flag.set()
        except Exception:
            pass

    def stop(self):
        if self.listener:
            self.listener.stop()


class Beacon:
    """A ring + label that sits on the button you should click.  Clicks pass straight through it."""
    W, H, CX, CY = 280, 170, 140, 112
    KEY = "#010203"

    def __init__(self, root):
        self.win = tk.Toplevel(root)
        self.win.withdraw()
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        try:
            self.win.attributes("-transparentcolor", self.KEY)
        except tk.TclError:
            pass
        self.cv = tk.Canvas(self.win, width=self.W, height=self.H, bg=self.KEY, highlightthickness=0, bd=0)
        self.cv.pack()
        cx, cy = self.CX, self.CY
        self.outer = self.cv.create_oval(cx - 40, cy - 40, cx + 40, cy + 40, outline=C.yellow, width=2)
        self.ring = self.cv.create_oval(cx - 24, cy - 24, cx + 24, cy + 24, outline=C.yellow, width=4)
        self.pill = self.cv.create_rectangle(0, 0, 0, 0, fill=C.bg, outline=C.yellow, width=2)
        self.text = self.cv.create_text(cx, 28, text="", fill=C.text, font=("Segoe UI Semibold", 13))
        self.visible, self.mode, self.clickable_set = False, "aim", False
        self.t0 = 0.0

    def show(self, x, y, text, mode="aim"):
        self.cv.itemconfig(self.text, text=text)
        bb = self.cv.bbox(self.text)
        if bb:
            self.cv.coords(self.pill, bb[0] - 10, bb[1] - 5, bb[2] + 10, bb[3] + 5)
        self.win.geometry(f"{self.W}x{self.H}+{int(x) - self.CX}+{int(y) - self.CY}")
        self.set_mode(mode)
        self.win.deiconify()
        self.win.lift()
        self.win.attributes("-topmost", True)
        if not self.clickable_set:
            make_click_through(self.win)
            self.clickable_set = True
        self.visible, self.t0 = True, clock()

    def set_mode(self, mode):
        self.mode = mode
        col = {"aim": C.yellow, "check": C.blue, "ok": C.green, "abort": C.red}.get(mode, C.yellow)
        for it in (self.outer, self.ring, self.pill):
            self.cv.itemconfig(it, outline=col)

    def hide(self):
        if self.visible:
            self.win.withdraw()
        self.visible = False

    def tick(self, now):
        if not self.visible:
            return
        ph = (now - self.t0) * (2.4 if self.mode in ("aim", "abort") else 1.0)
        r = 34 + 9 * (0.5 + 0.5 * math.sin(ph * 2 * math.pi))
        cx, cy = self.CX, self.CY
        self.cv.coords(self.outer, cx - r, cy - r, cx + r, cy + r)

    def destroy(self):
        try:
            self.win.destroy()
        except Exception:
            pass
