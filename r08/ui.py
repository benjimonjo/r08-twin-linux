"""
ui.py - the Tk application (class App).  It wires everything together; heavy logic lives in the other modules.

Main loop: App.tick() every ~16 ms ->
    guide.tick  (guided clicks)  ->  read Shared.snapshot()  ->  smooth drawn joints (q_vis)  ->
    if dirty: App._render() builds a render.Frame and shows the image  ->  every 0.1 s refresh the side panel.

Where things are:
    _build_*_card       sidebar cards (pose, joints, jog, goto, objects, digit reader)
    jog_btn / go_pose   jog = edit the TARGET fields (ghost previews it); Go starts a guided job (guide.py)
    start_teach / set_point / ...   mapping plates and pipes by touching them with the tool tip
    teach_digits / save_ocr_debug   digit-reader calibration and diagnosis
    _render             builds the Frame for render.py
"""
import os, math, time, traceback
import numpy as np
import cv2
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk

from .common import (C, AXES, UNITS, MINUS, clock, wrap180, hex_rgb, Config, CONFIG_FILE, DIGIT_BANK_FILE, DEBUG_DIR,
                     DEFAULT_SETTINGS, STEP_LEVELS, START_LEVEL, new_mss)
from . import kinematics as K
from . import digits
from .model3d import Camera, RobotMesh, plate_solid, pipe_solid, obj_summary
from .render import make_renderer
from .gpu_usage import GpuUsageMonitor
from .object_points import parse_xyz_points
from .screen_reader import Shared, OcrWorker, StepReader, IKWorker, split_cells, grab_gray
from .planner import step_presses, jog_press, plan_presses, step_value, fmt_step, N_LEVELS
from .guide import Guide
from .inputs import HAVE_PYNPUT, WAYLAND_SESSION, LINUX_NO_PYNPUT, Cursor, Clicks, Hotkeys, Beacon, dark_titlebar
from .picker import Picker, ALL_KEYS, BUTTON_KEYS
from .widgets import (FONT_UI, FONT_B, FONT_H, FONT_BIG, FONT_NUM, FONT_MONO, FONT_SM, SETTING_DEFS, KEYMAP,
                      NEEDED_BUTTONS, HELP_TEXT, mk_button, restyle, fmt_pos, fmt_ang, JointBar)
from types import SimpleNamespace

PLATE_STEPS = ["corner 1 (everything is measured from it)", "corner 2 (along one edge)", "corner 3 (along the other edge)"]
PIPE_STEPS = ["top of the pipe at one end", "top of the pipe at the other end"]


class App:
    def __init__(self):
        self.cfg = Config(CONFIG_FILE)
        self.settings = dict(DEFAULT_SETTINGS)
        self.settings.update(self.cfg.get("settings", {}) or {})
        if self.cfg.get("tool"):
            K.TOOL.update(self.cfg.get("tool"))
        self.root = tk.Tk()
        self.root.title("R08 live twin \u2013 FANUC R-2000iA/200F")
        self.root.configure(bg=C.bg)
        self.root.geometry(self.cfg.get("window", "1500x920"))
        self.root.minsize(1180, 720)
        self.setup_open = False
        self.buttons = {k: (float(v[0]), float(v[1])) for k, v in (self.cfg.get("buttons", {}) or {}).items()}
        self.panel_level = int(np.clip(self.cfg.get("level", START_LEVEL), 0, N_LEVELS - 1))
        self.ui_level = self.panel_level
        self.waypoints = list(self.cfg.get("waypoints", []) or [])
        self.objects = list(self.cfg.get("objects", []) or [])
        self.sel_i, self.teach = -1, None
        self._obj_soup, self._labels, self._objs_dirty = None, [], True

        self.bank = digits.DigitBank(DIGIT_BANK_FILE)
        self.shared = Shared()
        self.ik = IKWorker(self.shared)
        self.ik.start()
        self.ocr = OcrWorker(self.shared, self.settings, self.ik, self.bank)
        self.ocr.start()
        self.step_reader = StepReader(self.shared)
        self.step_reader.start()
        self.cursor = Cursor(self.settings)
        self.clicks = Clicks()
        self.hot = Hotkeys()
        self.beacon = Beacon(self.root)
        self.guide = Guide(self)
        self.abort_pending, self.abort_t = False, 0.0

        self.renderer, note = make_renderer(bool(self.settings.get("gpu", True)))
        self.gpu_monitor = GpuUsageMonitor(); self.gpu_monitor.start()
        self.gpu_usage = None
        self.robot_mesh, self.ghost_mesh, self.cam = RobotMesh(), RobotMesh(), Camera()
        self.q_vis = np.radians([0.0, 35.0, 25.0, 0.0, 40.0, 0.0])
        self.q_target = self.q_vis.copy()
        self.live_t, self.last_seq = None, -1
        self.trail_list, self.trail_arr = [], None
        self.ghost, self.ghost_t, self.ghost_warned = None, None, False
        self.target_vec = None
        self.dirty, self.photo, self.photo_size, self.vw, self.vh = True, None, None, 900, 600
        self.render_ms, self.fps, self._fps_n, self._fps_t = 0.0, 0.0, 0, clock()
        self.drag_mode, self.drag_last = None, (0, 0)
        self.preview_job = None
        self._cache = {}
        self.last_tick, self.last_side = clock(), 0.0
        self._nrej_seen, self._rej_t = 0, 0.0

        self._build_ui()
        dark_titlebar(self.root)
        self._apply_regions()
        self._refresh_chips(self.shared.snapshot())
        self.banner("idle", "Ready", "Jog the target with the buttons, then press Go.", None)
        self.log(f"R08 live twin started.  3D: {self.renderer.name}.", "dim")
        if note:
            self.log(note, "warn")
        if not HAVE_PYNPUT:
            if LINUX_NO_PYNPUT:
                self.log("Linux global input hooks are disabled outside confirmed X11 sessions; setup keys work when this window has focus.", "warn")
            else:
                self.log("pynput is not installed \u2013 viewer only.  pip install pynput to enable the guided controls.", "warn")
        if not self.cfg.get("region"):
            self.log("First run: opening setup \u2013 drag a box around the six numbers, then click a few panel buttons.", "act")
            self.root.after(600, lambda: self.run_setup(ALL_KEYS))
        elif not self._controls_ready():
            self.log("Panel buttons aren\u2019t set up yet \u2013 Setup \u203a Panel buttons.", "warn")
        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)
        self.root.after(30, self.tick)

    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.shutdown()

    # ------------------------------------------------------------------ env API used by Guide
    def log(self, msg, tag="info"):
        t = self.logw
        t.config(state="normal")
        t.insert("end", time.strftime("%H:%M:%S") + "  ", "ts")
        t.insert("end", msg + "\n", tag)
        if int(t.index("end-1c").split(".")[0]) > 400:
            t.delete("1.0", "60.0")
        t.see("end")
        t.config(state="disabled")

    def banner(self, kind, title, detail, progress):
        col = {"idle": C.faint, "aim": C.yellow, "check": C.blue, "done": C.green, "fail": C.red}[kind]
        self.b_strip.config(bg=col)
        self.b_title.config(text=title, fg=C.text if kind == "idle" else col)
        self.b_detail.config(text=detail)
        self.b_prog.delete("all")
        if progress:
            i, n = progress
            w = self.b_prog.winfo_width() or 600
            self.b_prog.create_rectangle(0, 0, w * min(1.0, i / max(1, n)), 6, fill=col, outline="")

    def pop_stop(self):
        if self.hot.stop_flag.is_set():
            self.hot.stop_flag.clear()
            return True
        return False

    def point_in_own_window(self, x, y):
        r = self.root
        return r.winfo_rootx() <= x <= r.winfo_rootx() + r.winfo_width() and r.winfo_rooty() <= y <= r.winfo_rooty() + r.winfo_height()

    def hit_test(self, x, y):
        best = None
        for name, (bx, by) in self.buttons.items():
            d = math.hypot(x - bx, y - by)
            if d <= self.settings["click_tol"] and (best is None or d < best[0]):
                best = (d, name)
        return best[1] if best else None

    def passive_click(self, c):
        name = self.hit_test(c[1], c[2])
        if name is None:
            return
        if name == "Abort" and self.abort_pending:
            self.abort_pending = False
            self.beacon.hide()
            self.log("Panel Abort pressed.", "warn")
        elif name.startswith("Step") and not self.shared.snapshot().has_step:
            self.apply_step_click(name)

    def set_target(self, target):
        self.target_vec = np.array(target, float)
        snap = self.shared.snapshot()
        full = np.where(np.isfinite(self.target_vec), self.target_vec, snap.pose)
        self.ghost_t, self.ghost_warned = None, False
        self.ik.request("ghost", full)

    def job_finished(self, ok):
        with self.shared.lock:
            g = self.shared.ik.get("ghost")
        self.target_vec, self.ghost, self.ghost_t = None, None, (g.t if g is not None else None)
        self.dirty = True
        self._set_controls()

    def set_panel_level(self, level, why=None):
        level = int(np.clip(level, 0, N_LEVELS - 1))
        if level != self.panel_level:
            if why:
                self.log(f"Panel step size is {fmt_step(level)} ({why}).", "dim")
            self.panel_level = level
            self.cfg.set("level", level)
            self.cfg.save()
        self._paint_chips()

    def apply_step_click(self, name):
        larger = (name == "Step+") == bool(self.settings["step_plus_larger"])
        self.set_panel_level(self.panel_level + (-1 if larger else 1))

    def flip_step_direction(self):
        self.settings["step_plus_larger"] = not self.settings["step_plus_larger"]
        self._save_settings()
        self.log("Learned: on your panel Step+ makes the step " + ("larger." if self.settings["step_plus_larger"] else "smaller."), "act")

    # ------------------------------------------------------------------ config
    def _save_settings(self):
        self.cfg.set("settings", self.settings)
        self.cfg.save()

    def _apply_regions(self):
        self.shared.set_regions(region=self.cfg.get("region"), status=self.cfg.get("status"), step=self.cfg.get("step"))

    def _controls_ready(self):
        return all(n in self.buttons for n in NEEDED_BUTTONS)

    # ------------------------------------------------------------------ UI construction
    def _build_ui(self):
        root = self.root
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=0)
        root.grid_rowconfigure(1, weight=1)
        head = tk.Frame(root, bg=C.panel, height=48)
        head.grid(row=0, column=0, columnspan=2, sticky="ew")
        head.grid_propagate(False)
        tk.Label(head, text=" R08 ", bg=C.yellow, fg="#15130a", font=("Segoe UI Black", 11)).pack(side="left", padx=(14, 8), pady=11)
        tk.Label(head, text="live twin", bg=C.panel, fg=C.text, font=FONT_H).pack(side="left")
        tk.Label(head, text="FANUC R-2000iA/200F", bg=C.panel, fg=C.muted, font=FONT_UI).pack(side="left", padx=10)
        self.chip_read = tk.Label(head, text="", bg=C.panel, font=FONT_UI)
        self.chip_read.pack(side="left", padx=(24, 12))
        self.chip_ctl = tk.Label(head, text="", bg=C.panel, font=FONT_UI)
        self.chip_ctl.pack(side="left", padx=12)
        mk_button(head, "Help", self.open_help, bg=C.panel, hover=C.raised, fg=C.muted).pack(side="right", padx=(0, 10), pady=7)
        mk_button(head, "Settings", self.open_settings, bg=C.panel, hover=C.raised, fg=C.muted).pack(side="right", pady=7)
        self.setup_btn = mk_button(head, "Setup \u25be", self._setup_menu, bg=C.panel, hover=C.raised, fg=C.text)
        self.setup_btn.pack(side="right", pady=7)
        left = tk.Frame(root, bg=C.bg)
        left.grid(row=1, column=0, sticky="nsew")
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)
        bar = tk.Frame(left, bg=C.bg)
        bar.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 6))
        for name in ("iso", "front", "side", "top"):
            mk_button(bar, name.capitalize(), lambda n=name: self._view_preset(n), padx=10, pady=3, font=FONT_SM).pack(side="left", padx=(0, 4))
        self.follow = tk.BooleanVar(value=False)
        self.show_trail = tk.BooleanVar(value=True)
        for text, var in (("Follow tool", self.follow), ("Trail", self.show_trail)):
            tk.Checkbutton(bar, text=text, variable=var, bg=C.bg, fg=C.text, selectcolor=C.raised, activebackground=C.bg,
                           activeforeground=C.text, font=FONT_SM, bd=0, highlightthickness=0, command=self._mark_dirty).pack(side="left", padx=(12, 0))
        mk_button(bar, "Reset view", lambda: self._view_preset("reset"), padx=10, pady=3, font=FONT_SM).pack(side="left", padx=12)
        tk.Label(bar, text="drag to orbit \u00b7 right-drag to pan \u00b7 wheel to zoom", bg=C.bg, fg=C.faint, font=FONT_SM).pack(side="right")
        self.view = tk.Canvas(left, bg=C.view_bot, highlightthickness=1, highlightbackground=C.line, bd=0, cursor="fleur")
        self.view.grid(row=1, column=0, sticky="nsew", padx=10)
        self.img_item = self.view.create_image(0, 0, anchor="nw")
        self.view.bind("<Configure>", self._on_resize)
        for b in (1, 2, 3):
            self.view.bind(f"<ButtonPress-{b}>", self._drag_start)
            self.view.bind(f"<B{b}-Motion>", self._drag_move)
        self.view.bind("<MouseWheel>", self._wheel)
        self.view.bind("<Button-4>", lambda e: self._wheel(SimpleNamespace(delta=120)))      # Linux
        self.view.bind("<Button-5>", lambda e: self._wheel(SimpleNamespace(delta=-120)))
        self.view.bind("<Double-Button-1>", lambda e: self._view_preset("reset"))
        gb = tk.Frame(left, bg=C.card, highlightbackground=C.line, highlightthickness=1)
        gb.grid(row=2, column=0, sticky="ew", padx=10, pady=(8, 0))
        self.b_strip = tk.Frame(gb, bg=C.faint, width=6)
        self.b_strip.pack(side="left", fill="y")
        mid = tk.Frame(gb, bg=C.card)
        mid.pack(side="left", fill="x", expand=True, padx=14, pady=(8, 6))
        self.b_title = tk.Label(mid, text="", bg=C.card, fg=C.text, font=FONT_BIG, anchor="w")
        self.b_title.pack(fill="x")
        self.b_detail = tk.Label(mid, text="", bg=C.card, fg=C.muted, font=FONT_UI, anchor="w")
        self.b_detail.pack(fill="x")
        self.b_prog = tk.Canvas(mid, height=6, bg=C.raised, highlightthickness=0, bd=0)
        self.b_prog.pack(fill="x", pady=(6, 0))
        btns = tk.Frame(gb, bg=C.card)
        btns.pack(side="right", padx=12)
        self.btn_stop = mk_button(btns, "Stop  (Esc)", self.stop_job, bg=C.raised)
        self.btn_stop.pack(side="left", padx=4)
        self.btn_snap = mk_button(btns, "Pointer to target  (F9)", self.snap_cursor, bg=C.raised)
        self.btn_snap.pack(side="left", padx=4)
        lf = tk.Frame(left, bg=C.panel, highlightbackground=C.line, highlightthickness=1)
        lf.grid(row=3, column=0, sticky="ew", padx=10, pady=(8, 10))
        self.logw = tk.Text(lf, height=7, bg=C.panel, fg=C.text, bd=0, highlightthickness=0, font=FONT_MONO, wrap="word",
                            state="disabled", padx=10, pady=6, cursor="arrow")
        self.logw.pack(fill="x")
        for tag, col in (("info", C.text), ("ok", C.green), ("warn", C.amber), ("err", C.red), ("dim", C.muted), ("act", C.yellow), ("ts", C.faint)):
            self.logw.tag_configure(tag, foreground=col)
        side_outer = tk.Frame(root, bg=C.panel, width=388)
        side_outer.grid(row=1, column=1, sticky="ns")
        side_outer.grid_propagate(False)
        side_outer.pack_propagate(False)
        self.side_cv = tk.Canvas(side_outer, bg=C.panel, highlightthickness=0, bd=0, width=388)
        self.side_cv.pack(fill="both", expand=True)
        side = tk.Frame(self.side_cv, bg=C.panel)
        self.side_win = self.side_cv.create_window(0, 0, window=side, anchor="nw", width=388)
        side.bind("<Configure>", lambda e: self.side_cv.configure(scrollregion=self.side_cv.bbox("all")))
        side_outer.bind("<Enter>", lambda e: self.root.bind_all("<MouseWheel>", self._side_wheel))
        side_outer.bind("<Leave>", lambda e: self.root.unbind_all("<MouseWheel>"))
        self.view.bind("<Enter>", lambda e: self.root.unbind_all("<MouseWheel>"))
        tk.Frame(side, bg=C.panel, height=10).pack()
        self._build_pose_card(side)
        self._build_joint_card(side)
        self._build_jog_card(side)
        self._build_goto_card(side)
        self._build_objects_card(side)
        self._build_digits_card(side)
        root.bind("<KeyPress>", self._key)
        self._set_controls()

    def _card(self, parent, title):
        f = tk.Frame(parent, bg=C.card, highlightbackground=C.line, highlightthickness=1)
        f.pack(fill="x", padx=12, pady=(0, 10))
        head = tk.Frame(f, bg=C.card)
        head.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(head, text=title, bg=C.card, fg=C.text, font=FONT_B).pack(side="left")
        body = tk.Frame(f, bg=C.card)
        body.pack(fill="x", padx=12, pady=(0, 12))
        return body, head

    def _build_pose_card(self, parent):
        body, head = self._card(parent, "Robot position")
        self.state_pill = tk.Label(head, text="", bg=C.card, fg=C.muted, font=FONT_SM)
        self.state_pill.pack(side="right")
        self.pose_lbl = []
        for r in range(2):
            for c in range(3):
                i = r * 3 + c
                cell = tk.Frame(body, bg=C.card)
                cell.grid(row=r, column=c, sticky="w", padx=(0, 16), pady=(0, 4))
                tk.Label(cell, text=f"{AXES[i]}  {UNITS[i]}", bg=C.card, fg=C.muted, font=FONT_SM).pack(anchor="w")
                v = tk.Label(cell, text="\u2013", bg=C.card, fg=C.text, font=FONT_NUM)
                v.pack(anchor="w")
                self.pose_lbl.append(v)
        self.read_lbl = tk.Label(body, text="", bg=C.card, fg=C.faint, font=FONT_SM, anchor="w")
        self.read_lbl.grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))

    def _build_joint_card(self, parent):
        body, head = self._card(parent, "Joints from the model")
        self.fit_lbl = tk.Label(head, text="", bg=C.card, fg=C.muted, font=FONT_SM)
        self.fit_lbl.pack(side="right")
        self.jbars, self.jlbls = [], []
        for i in range(6):
            tk.Label(body, text=f"J{i + 1}", bg=C.card, fg=C.muted, font=FONT_SM, width=3, anchor="w").grid(row=i, column=0, pady=2)
            jb = JointBar(body, K.JOINT_LIMITS_DEG[i, 0], K.JOINT_LIMITS_DEG[i, 1])
            jb.cv.grid(row=i, column=1, padx=6)
            lb = tk.Label(body, text="", bg=C.card, fg=C.text, font=FONT_MONO, width=8, anchor="e")
            lb.grid(row=i, column=2)
            self.jbars.append(jb)
            self.jlbls.append(lb)

    def _build_jog_card(self, parent):
        body, head = self._card(parent, "Aim the target")
        tk.Label(body, text="Step size", bg=C.card, fg=C.muted, font=FONT_SM).grid(row=0, column=0, columnspan=6, sticky="w")
        self.chips = []
        for i, (p, a) in enumerate(STEP_LEVELS):
            b = mk_button(body, f"{p:g}\n{a:g}\u00b0", lambda k=i: self.select_level(k), padx=0, pady=3, font=FONT_SM)
            b.config(highlightthickness=2, highlightbackground=C.card, width=5)
            b.grid(row=1, column=i, padx=1, pady=(2, 0), sticky="ew")
            body.grid_columnconfigure(i, weight=1, uniform="chip")
            self.chips.append(b)
        self.level_note = tk.Label(body, text="", bg=C.card, fg=C.muted, font=FONT_SM, anchor="w")
        self.level_note.grid(row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))
        mk_button(body, "Panel is here", self.mark_panel_level, padx=6, pady=1, font=FONT_SM).grid(row=2, column=4, columnspan=2, sticky="e", pady=(4, 0))
        self.jog_btns = []
        for blk, (title, names) in enumerate((("Position", ["X", "Y", "Z"]), ("Angle", ["Xa", "Ya", "Za"]))):
            tk.Label(body, text=title, bg=C.card, fg=C.muted, font=FONT_SM).grid(row=3 + blk * 3, column=0, columnspan=6, sticky="w", pady=(10, 2))
            for r, sg in enumerate("+-"):
                for c, ax in enumerate(names):
                    nm = f"{ax}{sg}"
                    plus = sg == "+"
                    b = mk_button(body, nm.replace("-", MINUS), lambda n=nm: self.jog_btn(n),
                                  bg="#33477f" if plus else C.raised, hover="#41599b" if plus else C.hover, pady=7)
                    b.grid(row=4 + blk * 3 + r, column=c * 2, columnspan=2, padx=2, pady=2, sticky="ew")
                    self.jog_btns.append(b)
        tk.Label(body, text="Jog buttons and keys move the TARGET (Go-to fields + ghost). Press Go to send it.", bg=C.card, fg=C.yellow,
                 font=FONT_SM, wraplength=330, justify="left").grid(row=10, column=0, columnspan=6, sticky="w", pady=(8, 0))
        self.btn_abort = mk_button(body, "Take me to Abort  (F10)", self.abort_panel, bg="#5a2a2a", hover="#7a3535", fg="#ffd9d6")
        self.btn_abort.grid(row=11, column=0, columnspan=6, sticky="ew", pady=(10, 0))

    def _build_goto_card(self, parent):
        body, head = self._card(parent, "Go to a pose")
        self.fields = [tk.StringVar() for _ in range(6)]
        for i in range(6):
            r, c = divmod(i, 3)
            cell = tk.Frame(body, bg=C.card)
            cell.grid(row=r, column=c, padx=(0, 8), pady=(0, 6), sticky="w")
            tk.Label(cell, text=f"{AXES[i]} ({UNITS[i]})", bg=C.card, fg=C.muted, font=FONT_SM).pack(anchor="w")
            e = tk.Entry(cell, textvariable=self.fields[i], width=8, bg=C.bg, fg=C.text, insertbackground=C.text, relief="flat",
                         font=("Consolas", 11), highlightthickness=1, highlightbackground=C.line, highlightcolor=C.yellow)
            e.pack(anchor="w", ipady=3)
            e.bind("<KeyRelease>", lambda ev: self._schedule_preview())
            e.bind("<Return>", lambda ev: self.go_pose())
        self.relative = tk.BooleanVar(value=False)
        tk.Checkbutton(body, text="Values are changes from the current pose", variable=self.relative, bg=C.card, fg=C.text,
                       selectcolor=C.raised, activebackground=C.card, activeforeground=C.text, font=FONT_SM, bd=0,
                       highlightthickness=0, command=self._schedule_preview).grid(row=2, column=0, columnspan=3, sticky="w")
        row = tk.Frame(body, bg=C.card)
        row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        mk_button(row, "Fill with current", self.fill_current, padx=10).pack(side="left")
        mk_button(row, "Clear", self.clear_fields, padx=10).pack(side="left", padx=6)
        self.btn_go = mk_button(row, "Go", self.go_pose, bg=C.yellow, hover="#ffd75a", fg="#15130a", padx=26)
        self.btn_go.pack(side="right")
        tk.Label(body, text="Saved poses", bg=C.card, fg=C.muted, font=FONT_SM).grid(row=4, column=0, columnspan=3, sticky="w", pady=(14, 2))
        self.wp_list = tk.Listbox(body, height=4, bg=C.bg, fg=C.text, selectbackground=C.yellow, selectforeground="#15130a",
                                  relief="flat", highlightthickness=1, highlightbackground=C.line, font=FONT_MONO, activestyle="none", exportselection=False)
        self.wp_list.grid(row=5, column=0, columnspan=3, sticky="ew")
        row = tk.Frame(body, bg=C.card)
        row.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        mk_button(row, "Save current", self.save_waypoint, padx=10).pack(side="left")
        mk_button(row, "Delete", self.delete_waypoint, padx=10).pack(side="left", padx=6)
        mk_button(row, "Go to it", self.go_waypoint, padx=10).pack(side="right")
        self._fill_waypoints()

    def _build_objects_card(self, parent):
        body, head = self._card(parent, "Map objects")
        r = tk.Frame(body, bg=C.card)
        r.pack(fill="x")
        mk_button(r, "New plate", lambda: self.start_teach("plate"), padx=10).pack(side="left")
        mk_button(r, "New hollow pipe", lambda: self.start_teach("pipe"), padx=10).pack(side="left", padx=6)
        mk_button(r, "Cancel", self.cancel_teach, padx=10).pack(side="left")
        r = tk.Frame(body, bg=C.card)
        r.pack(fill="x", pady=(6, 0))
        mk_button(r, "Pole from points…", self.import_pole_from_points, padx=10).pack(side="left")
        self.teach_lbl = tk.Label(body, text="", bg=C.card, fg=C.yellow, font=FONT_SM, wraplength=340, justify="left", anchor="w")
        self.teach_lbl.pack(fill="x", pady=(6, 0))
        r = tk.Frame(body, bg=C.card)
        r.pack(fill="x", pady=(6, 0))
        mk_button(r, "Set point (tool tip)", self.set_point, bg=C.yellow, hover="#ffd75a", fg="#15130a", padx=14).pack(side="left")
        mk_button(r, "Undo point", self.undo_point, padx=10).pack(side="left", padx=6)
        self.obj_list = tk.Listbox(body, height=5, bg=C.bg, fg=C.text, selectbackground=C.yellow, selectforeground="#15130a",
                                   relief="flat", highlightthickness=1, highlightbackground=C.line, font=FONT_MONO,
                                   activestyle="none", exportselection=False)
        self.obj_list.pack(fill="x", pady=(8, 0))
        self.obj_list.bind("<<ListboxSelect>>", self._obj_selected)
        r = tk.Frame(body, bg=C.card)
        r.pack(fill="x", pady=(6, 0))
        mk_button(r, "Delete", self.delete_obj, padx=10).pack(side="left")
        mk_button(r, "Edit size", self.edit_obj, padx=10).pack(side="left", padx=6)
        tk.Label(r, text="point to Go:", bg=C.card, fg=C.muted, font=FONT_SM).pack(side="left", padx=(8, 2))
        for k in range(3):
            mk_button(r, str(k + 1), lambda k=k: self.point_to_go(k), padx=6, pady=2, font=FONT_SM).pack(side="left", padx=1)
        self.show_objs, self.show_gizmo, self.show_axes = tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        for text, var in (("Show mapped objects", self.show_objs), ("Axis gizmo on the tool tip", self.show_gizmo),
                          ("Show coordinate axes", self.show_axes)):
            tk.Checkbutton(body, text=text, variable=var, bg=C.card, fg=C.text, selectcolor=C.raised, activebackground=C.card,
                           activeforeground=C.text, font=FONT_SM, bd=0, highlightthickness=0, command=self._mark_dirty).pack(anchor="w", pady=(4, 0))
        r = tk.Frame(body, bg=C.card)
        r.pack(fill="x", pady=(8, 0))
        tk.Label(r, text="Tool:", bg=C.card, fg=C.muted, font=FONT_SM).pack(side="left")
        mk_button(r, "Flip long-spike side", self.flip_spike, padx=8, pady=2, font=FONT_SM).pack(side="left", padx=4)
        mk_button(r, "Turn row 90\u00b0", self.turn_spike, padx=8, pady=2, font=FONT_SM).pack(side="left")
        self._fill_objs()

    def _build_digits_card(self, parent):
        body, head = self._card(parent, "Digit reader")
        self.digit_lbl = tk.Label(body, text="", bg=C.card, fg=C.muted, font=FONT_SM, wraplength=340, justify="left", anchor="w")
        self.digit_lbl.pack(fill="x")
        r = tk.Frame(body, bg=C.card)
        r.pack(fill="x", pady=(6, 0))
        mk_button(r, "Teach digits\u2026", self.teach_digits, bg=C.yellow, hover="#ffd75a", fg="#15130a", padx=10, pady=3, font=FONT_SM).pack(side="left")
        mk_button(r, "Save OCR snapshot", self.save_ocr_debug, padx=8, pady=3, font=FONT_SM).pack(side="left", padx=6)
        mk_button(r, "Reset learned", self.reset_digits, padx=8, pady=3, font=FONT_SM).pack(side="left")
        self._update_digit_label()

    # ------------------------------------------------------------------ small UI actions
    def _mark_dirty(self):
        self.dirty = True

    def _side_wheel(self, e):
        self.side_cv.yview_scroll(int(-e.delta / 120) * 2, "units")

    def _on_resize(self, e):
        self.vw, self.vh = max(e.width - 2, 50), max(e.height - 2, 50)
        self.dirty = True

    def _view_preset(self, name):
        if name == "reset":
            self.cam.reset()
        else:
            self.cam.preset(name)
        self.dirty = True

    def _drag_start(self, e):
        shift = bool(e.state & 0x0001)
        self.drag_mode = "orbit" if (e.num == 1 and not shift) else "pan"
        self.drag_last = (e.x, e.y)
        self.view.focus_set()

    def _drag_move(self, e):
        if not self.drag_mode:
            return
        dx, dy = e.x - self.drag_last[0], e.y - self.drag_last[1]
        self.drag_last = (e.x, e.y)
        if self.drag_mode == "orbit":
            self.cam.orbit(dx, dy)
        else:
            self.follow.set(False)
            self.cam.pan(dx, dy, self.vh)
        self.dirty = True

    def _wheel(self, e):
        self.cam.zoom(e.delta / 120.0)
        self.dirty = True

    def _setup_menu(self):
        m = tk.Menu(self.root, tearoff=0, bg=C.card, fg=C.text, activebackground=C.yellow, activeforeground="#15130a", bd=0, font=FONT_UI)
        m.add_command(label="Run full setup", command=lambda: self.run_setup(ALL_KEYS))
        m.add_separator()
        m.add_command(label="Number table only", command=lambda: self.run_setup(["region"]))
        m.add_command(label="Panel buttons only", command=lambda: self.run_setup(BUTTON_KEYS))
        m.add_command(label="IDLE / MOVING banner", command=lambda: self.run_setup(["status"]))
        m.add_command(label="Step size text", command=lambda: self.run_setup(["step_label"]))
        m.add_separator()
        m.add_command(label="Forget banner and step text", command=self.forget_optional)
        m.tk_popup(self.setup_btn.winfo_rootx(), self.setup_btn.winfo_rooty() + self.setup_btn.winfo_height())

    def forget_optional(self):
        self.cfg.set("status", None)
        self.cfg.set("step", None)
        self.cfg.save()
        self._apply_regions()
        with self.shared.lock:
            self.shared.step_level = None
        self.log("Banner and step-size text are no longer used.", "dim")

    def open_help(self):
        w = tk.Toplevel(self.root)
        w.title("How this works")
        w.configure(bg=C.panel)
        t = tk.Text(w, bg=C.panel, fg=C.text, font=("Segoe UI", 10), bd=0, highlightthickness=0, padx=22, pady=18, width=96, height=33, wrap="word", spacing3=3)
        t.insert("1.0", HELP_TEXT)
        t.config(state="disabled")
        t.pack()

    def open_settings(self):
        w = tk.Toplevel(self.root)
        w.title("Settings")
        w.configure(bg=C.panel)
        w.resizable(False, False)
        body = tk.Frame(w, bg=C.panel, padx=20, pady=16)
        body.pack()
        for r, (key, label, lo, hi, res) in enumerate(SETTING_DEFS):
            tk.Label(body, text=label, bg=C.panel, fg=C.text, font=FONT_UI, anchor="w").grid(row=r, column=0, sticky="w", pady=5)
            s = tk.Scale(body, from_=lo, to=hi, resolution=res, orient="horizontal", length=240, bg=C.panel, fg=C.text,
                         troughcolor=C.raised, highlightthickness=0, bd=0, activebackground=C.yellow, font=FONT_SM,
                         command=lambda v, k=key: self._set_setting(k, float(v)))
            s.set(self.settings[key])
            s.grid(row=r, column=1, padx=(14, 0))
        r = len(SETTING_DEFS)
        self._chk(body, r, "Check the pose after every click", "verify_each")
        self._chk(body, r + 1, "Step+ makes the step larger on my panel", "step_plus_larger")
        self._chk(body, r + 2, "Digit reader learns from confident reads", "self_learn")
        self._chk(body, r + 3, "Use the GPU 3D renderer (restart to apply)", "gpu")

    def _chk(self, parent, row, text, key):
        var = tk.BooleanVar(value=bool(self.settings.get(key, True)))

        def changed():
            self.settings[key] = bool(var.get())
            self._save_settings()
            self.dirty = True
        tk.Checkbutton(parent, text=text, variable=var, bg=C.panel, fg=C.text, selectcolor=C.raised, activebackground=C.panel,
                       activeforeground=C.text, font=FONT_UI, bd=0, highlightthickness=0, command=changed
                       ).grid(row=row, column=0, columnspan=2, sticky="w", pady=3)

    def _set_setting(self, key, value):
        self.settings[key] = int(value) if key == "click_tol" else value
        self._save_settings()

    # ------------------------------------------------------------------ setup
    def run_setup(self, keys):
        if self.setup_open:
            return
        if self.guide.active:
            self.guide.cancel("Setup opened")
        self.beacon.hide()
        self.setup_open = True
        self.root.withdraw()
        self.root.update()
        time.sleep(0.3)
        res = None
        try:
            res = Picker(self.root, keys).run()
        except Exception as exc:
            self.log(f"Setup failed: {exc}", "err")
        finally:
            self.root.deiconify()
            self.setup_open = False
        if not res:
            self.log("Setup cancelled \u2013 nothing changed.", "dim")
            return
        for src, dst in (("region", "region"), ("status", "status"), ("step", "step")):
            if src in res:
                self.cfg.set(dst, res[src])
        if "buttons" in res:
            self.buttons.update({k: (float(v[0]), float(v[1])) for k, v in res["buttons"].items()})
            self.cfg.set("buttons", {k: list(v) for k, v in self.buttons.items()})
        self.cfg.save()
        self._apply_regions()
        self.log("Setup saved." + ("" if self._controls_ready() else "  Panel buttons still incomplete."), "ok")
        if "region" in res:
            self.log("Tip: press \u2018Teach digits\u2026\u2019 once so the reader learns your screen\u2019s font.", "act")
        self._set_controls()

    # ------------------------------------------------------------------ step size chips
    def select_level(self, i):
        self.ui_level = int(np.clip(i, 0, N_LEVELS - 1))
        self._paint_chips()

    def mark_panel_level(self):
        self.set_panel_level(self.ui_level, "set by you")
        self.log(f"Okay \u2013 treating the panel as being at {fmt_step(self.ui_level)}.", "dim")

    def _paint_chips(self):
        for i, b in enumerate(self.chips):
            sel = i == self.ui_level
            restyle(b, C.yellow if sel else C.raised, "#ffd75a" if sel else C.hover, "#15130a" if sel else C.text)
            b.config(highlightbackground=C.blue if i == self.panel_level else C.card)
        snap = self.shared.snapshot()
        if snap.has_step:
            if snap.step_level is None:
                note = f"Panel step unknown \u00b7 {snap.step_status}"
            else:
                age = max(0.0, clock() - snap.step_t)
                note = f"Panel is at {fmt_step(snap.step_level)} (OCR, {age:.1f}s ago)"
        else:
            note = f"Panel is at {fmt_step(self.panel_level)} (set up step-size OCR to read it)"
        extra = "" if self.ui_level == self.panel_level else f"  \u2013  will press Step {abs(self.ui_level - self.panel_level)}\u00d7 first"
        self.level_note.config(text=note + extra)

    # ------------------------------------------------------------------ target editing + Go
    def _can_start(self):
        if self.guide.active:
            self.log("A job is running \u2013 press Stop first.", "warn")
            return False
        if self.setup_open:
            return False
        if not self._controls_ready():
            self.log("Set up the panel buttons first (Setup \u203a Panel buttons only).", "warn")
            return False
        if self.shared.snapshot().pose is None:
            self.log("Still waiting for a reading from the number table.", "warn")
            return False
        self.hot.stop_flag.clear()
        self.abort_pending = False
        return True

    def jog_btn(self, name):
        """Jog = edit the target fields by one step (the ghost previews it)."""
        if self.guide.active:
            self.log("A job is running \u2013 press Stop first.", "warn")
            return
        axis, sign = AXES.index(name[:-1]), (1 if name[-1] == "+" else -1)
        snap = self.shared.snapshot()
        if snap.pose is None:
            self.log("Still waiting for a reading from the number table.", "warn")
            return
        self.relative.set(False)
        try:
            cur = float(self.fields[axis].get().strip().replace(",", ".").replace(MINUS, "-"))
        except ValueError:
            cur = float(snap.pose[axis])
        nv = cur + sign * step_value(self.ui_level, axis)
        if axis >= 3:
            nv = wrap180(nv)
        self.fields[axis].set(_fmtv(nv))
        self._preview()

    def _read_fields(self):
        snap = self.shared.snapshot()
        tgt, any_val = [None] * 6, False
        for i, sv in enumerate(self.fields):
            s = sv.get().strip().replace(",", ".")
            if not s:
                continue
            try:
                v = float(s.replace(MINUS, "-"))
            except ValueError:
                return None, f"\u2018{s}\u2019 isn\u2019t a number."
            any_val = True
            base = snap.pose[i] if (snap.pose is not None and self.relative.get()) else 0.0
            tgt[i] = wrap180(base + v) if i >= 3 else base + v
        if not any_val:
            return None, ""
        if self.relative.get() and snap.pose is None:
            return None, "No reading yet."
        return tgt, ""

    def go_pose(self):
        tgt, err = self._read_fields()
        if tgt is None:
            self.log(err or "Type at least one value first.", "warn")
            return
        if not self._can_start():
            return
        snap = self.shared.snapshot()
        S = self.settings
        plan = plan_presses(snap.pose, tgt, self.panel_level, S["pos_tol"], S["ang_tol"], S["step_plus_larger"])
        self.guide.start(tgt, plan, "Go to pose", clock())
        self._set_controls()

    def fill_current(self):
        snap = self.shared.snapshot()
        if snap.pose is None:
            return
        self.relative.set(False)
        for i, sv in enumerate(self.fields):
            sv.set(fmt_pos(snap.pose[i]) if i < 3 else fmt_ang(snap.pose[i]))
        self._schedule_preview()

    def clear_fields(self):
        for sv in self.fields:
            sv.set("")
        if not self.guide.active:
            self.ghost = None
            self.dirty = True

    def _schedule_preview(self):
        if self.preview_job:
            self.root.after_cancel(self.preview_job)
        self.preview_job = self.root.after(60, self._preview)

    def _preview(self):
        self.preview_job = None
        if self.guide.active:
            return
        tgt, _ = self._read_fields()
        snap = self.shared.snapshot()
        if tgt is None or snap.pose is None:
            self.ghost = None
            self.dirty = True
            return
        full = np.array([snap.pose[i] if t is None else t for i, t in enumerate(tgt)], float)
        self.ghost_t, self.ghost_warned = None, True
        self.ik.request("ghost", full)

    def _fill_waypoints(self):
        self.wp_list.delete(0, "end")
        for w in self.waypoints:
            p = w["pose"]
            self.wp_list.insert("end", f"{w['name'][:14]:<14} {p[0]:+6.1f} {p[1]:+6.1f} {p[2]:+6.1f}")

    def save_waypoint(self):
        snap = self.shared.snapshot()
        if snap.pose is None:
            self.log("No reading to save yet.", "warn")
            return
        name = simpledialog.askstring("Save pose", "Name for this pose:", parent=self.root)
        if not name:
            return
        self.waypoints.append({"name": name.strip(), "pose": [float(v) for v in snap.pose]})
        self.cfg.set("waypoints", self.waypoints)
        self.cfg.save()
        self._fill_waypoints()

    def delete_waypoint(self):
        sel = self.wp_list.curselection()
        if sel:
            del self.waypoints[sel[0]]
            self.cfg.set("waypoints", self.waypoints)
            self.cfg.save()
            self._fill_waypoints()

    def go_waypoint(self):
        sel = self.wp_list.curselection()
        if not sel:
            self.log("Pick a saved pose in the list first.", "warn")
            return
        w = self.waypoints[sel[0]]
        self.relative.set(False)
        for i, sv in enumerate(self.fields):
            sv.set(fmt_pos(w["pose"][i]) if i < 3 else fmt_ang(w["pose"][i]))
        self.go_pose()

    # ------------------------------------------------------------------ safety actions
    def stop_job(self):
        if self.guide.active:
            self.guide.cancel("Stopped")
        self.hot.stop_flag.clear()

    def snap_cursor(self):
        if self.cursor.target:
            self.cursor.resnap()

    def abort_panel(self):
        if self.guide.active:
            self.guide.cancel("Stopped \u2013 going to Abort")
        pos = self.buttons.get("Abort")
        if not pos:
            self.log("Abort button isn\u2019t set up \u2013 Setup \u203a Panel buttons only.", "err")
            return
        self.cursor.glide(*pos)
        self.beacon.show(pos[0], pos[1], "Abort \u2013 click to stop the robot", "abort")
        self.abort_pending, self.abort_t = True, clock()
        self.log("Pointer is on Abort \u2013 click it to stop the robot.", "err")
        self.banner("fail", "Click Abort", "the pointer is on the panel\u2019s Abort button", None)

    def _key(self, e):
        try:
            f = self.root.focus_get()
        except Exception:
            f = None
        if isinstance(f, (tk.Entry, tk.Text, tk.Listbox)):
            return
        k = e.keysym.lower()
        if k in KEYMAP:
            self.jog_btn(KEYMAP[k])
        elif k in ("1", "2", "3", "4", "5", "6"):
            self.select_level(int(k) - 1)
        elif k == "home":
            self._view_preset("reset")
        elif k == "f":
            self.follow.set(not self.follow.get())
        elif k == "t":
            self.show_trail.set(not self.show_trail.get())
            self.dirty = True

    # ------------------------------------------------------------------ tool orientation
    def _tool_changed(self):
        self.cfg.set("tool", dict(K.TOOL))
        self.cfg.save()
        self.ik.last_live = None
        snap = self.shared.snapshot()
        if snap.pose is not None:
            self.ik.request("live", snap.pose)
        self.dirty = True

    def flip_spike(self):
        K.TOOL["sign"] = -K.TOOL["sign"]
        self._tool_changed()
        self.log("Long spike moved to the other side of the tool.", "dim")

    def turn_spike(self):
        K.TOOL["axis"] = 1 if K.TOOL["axis"] == 2 else 2
        self._tool_changed()
        self.log("Spike row turned 90\u00b0 on the tool plate.", "dim")

    # ------------------------------------------------------------------ digit reader tools
    def _update_digit_label(self):
        self.digit_lbl.config(text=self.bank.stats() + ("   (self-learned this session: %d)" % self.bank.n_self if self.bank.n_self else ""))

    def _grab_region_hidden(self, extra=()):
        """Grab the number table with our own windows hidden so they can't cover it."""
        reg = self.cfg.get("region")
        if not reg:
            self.log("Number table isn't set up yet.", "warn")
            return None
        wins = (self.root,) + tuple(extra)
        for w in wins:
            w.withdraw()
        self.root.update()
        time.sleep(0.3)
        try:
            return grab_gray(new_mss(), reg)
        finally:
            for w in wins:
                w.deiconify()

    def teach_digits(self):
        if not self.cfg.get("region"):
            self.log("Number table isn't set up yet.", "warn")
            return
        snap = self.shared.snapshot()
        w = tk.Toplevel(self.root)
        w.title("Teach digits")
        w.configure(bg=C.panel)
        w.transient(self.root)
        tk.Label(w, text="Type EXACTLY what the stream shows right now (including + or \u2212 and the decimal point).\n"
                         "Hold the robot still, then press Learn.  Do it at a few different poses so every digit 0\u20139 gets seen.",
                 bg=C.panel, fg=C.text, font=FONT_UI, justify="left").grid(row=0, column=0, columnspan=3, padx=16, pady=(14, 8))
        vars_ = []
        for i in range(6):
            r, c = divmod(i, 3)
            cell = tk.Frame(w, bg=C.panel)
            cell.grid(row=1 + r, column=c, padx=10, pady=4)
            tk.Label(cell, text=AXES[i], bg=C.panel, fg=C.muted, font=FONT_SM).pack(anchor="w")
            v = tk.StringVar()
            if snap.pose is not None:
                v.set(f"{snap.pose[i]:+.1f}" if i < 3 else f"{snap.pose[i]:+.0f}")
            tk.Entry(cell, textvariable=v, width=9, bg=C.bg, fg=C.text, insertbackground=C.text, relief="flat", font=("Consolas", 12),
                     highlightthickness=1, highlightbackground=C.line, highlightcolor=C.yellow).pack(ipady=3)
            vars_.append(v)
        msg = tk.Label(w, text="", bg=C.panel, fg=C.muted, font=FONT_SM, justify="left", wraplength=480)
        msg.grid(row=3, column=0, columnspan=3, padx=16, pady=(6, 0))

        def learn():
            got = self._grab_region_hidden(extra=(w,))
            if got is None:
                return
            _, gray = got
            lines, total = [], 0
            for i, cell in enumerate(split_cells(gray)):
                n, m = digits.teach(cell, i // 3, vars_[i].get(), self.bank)
                total += n
                lines.append(f"{AXES[i]}: " + (f"learned {n} digits" if m == "ok" else m))
            self.bank.save()
            msg.config(text="\n".join(lines) + f"\n\nTotal new samples: {total}", fg=C.green if total else C.amber)
            self._update_digit_label()
            self.log(f"Teach digits: {total} new samples.  {self.bank.stats()}", "ok" if total else "warn")

        row = tk.Frame(w, bg=C.panel)
        row.grid(row=4, column=0, columnspan=3, pady=(8, 14))
        mk_button(row, "Learn", learn, bg=C.yellow, hover="#ffd75a", fg="#15130a", padx=24).pack(side="left", padx=6)
        mk_button(row, "Close", w.destroy, padx=18).pack(side="left", padx=6)

    def reset_digits(self):
        self.bank.reset_learned()
        self._update_digit_label()
        self.log("Learned digit samples cleared.", "dim")

    def save_ocr_debug(self):
        got = self._grab_region_hidden()
        if got is None:
            return
        try:
            bgra, gray = got
            os.makedirs(DEBUG_DIR, exist_ok=True)
            cv2.imwrite(os.path.join(DEBUG_DIR, "region.png"), bgra)
            report = []
            for i, cell in enumerate(split_cells(gray)):
                cv2.imwrite(os.path.join(DEBUG_DIR, f"cell{i}_{AXES[i]}.png"), cell)
                r = digits.read_cell(cell, i // 3, self.bank)
                glyphs = digits.segment(cell)
                s = digits.upscale_factor(cell.shape[0])
                vis = cv2.cvtColor(cv2.resize(cell, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s > 1 else cell, cv2.COLOR_GRAY2BGR)
                for g in glyphs or []:
                    cv2.rectangle(vis, (g.x, g.y), (g.x + g.w, g.y + g.h), (0, 220, 255), 1)
                    cv2.putText(vis, g.kind if g.kind != 'digit' else str(g.label), (g.x, max(10, g.y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imwrite(os.path.join(DEBUG_DIR, f"cell{i}_{AXES[i]}_seg.png"), vis)
                line = f"{AXES[i]:>2}: " + (f"{r.text}  (score {r.score:.2f})" if r else f"NOT READ   glyphs: {digits.describe(glyphs or [])}")
                report.append(line)
                self.log(line, "dim")
            with open(os.path.join(DEBUG_DIR, "report.txt"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(report) + "\n")
            self.log(f"Saved cell images + segmentation to {DEBUG_DIR}", "ok")
        except Exception as exc:
            self.log(f"Snapshot failed: {exc}", "err")

    # ------------------------------------------------------------------ object mapping
    def _save_objs(self):
        self.cfg.set("objects", self.objects)
        self.cfg.save()
        self._fill_objs()
        self._objs_dirty = self.dirty = True

    def _fill_objs(self):
        self.obj_list.delete(0, "end")
        for o in self.objects:
            self.obj_list.insert("end", obj_summary(o))
        if 0 <= self.sel_i < len(self.objects):
            self.obj_list.selection_set(self.sel_i)

    def _obj_selected(self, _e=None):
        s = self.obj_list.curselection()
        self.sel_i = s[0] if s else -1
        self._objs_dirty = self.dirty = True

    def _teach_msg(self):
        t = self.teach
        if not t:
            self.teach_lbl.config(text="")
            self.banner("idle", "Ready", "Jog the target with the buttons, then press Go.", None)
            return
        steps = PLATE_STEPS if t["type"] == "plate" else PIPE_STEPS
        i, n = len(t["pts"]), len(steps)
        msg = f"Touch the tool tip to the {steps[i]}, then press Set point."
        self.teach_lbl.config(text=f"New {t['type']} \u2013 point {i + 1} of {n}: {msg}")
        self.banner("aim", f"New {t['type']}: point {i + 1} of {n}", msg, (i, n))

    def start_teach(self, kind):
        if self.guide.active:
            self.log("A job is running \u2013 press Stop first.", "warn")
            return
        self.teach = {"type": kind, "pts": []}
        self._teach_msg()
        self._objs_dirty = self.dirty = True

    def import_pole_from_points(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Pole from points")
        dialog.transient(self.root)
        dialog.geometry("560x330")
        tk.Label(dialog, text=("Paste exactly two pole centerline points, one per line, as X Y Z in inches.\n"
                               "Labeled panel read-outs or three bare numbers are accepted."),
                 justify="left", anchor="w").pack(fill="x", padx=10, pady=(10, 4))
        text_box = tk.Text(dialog, height=10, wrap="none")
        text_box.pack(fill="both", expand=True, padx=10, pady=4)
        result = {"text": None}
        actions = tk.Frame(dialog)
        actions.pack(fill="x", padx=10, pady=(4, 10))

        def accept():
            result["text"] = text_box.get("1.0", "end-1c")
            dialog.destroy()

        tk.Button(actions, text="Import", command=accept).pack(side="right", padx=(6, 0))
        tk.Button(actions, text="Cancel", command=dialog.destroy).pack(side="right")
        dialog.grab_set()
        self.root.wait_window(dialog)
        if result["text"] is None:
            return
        try:
            points = parse_xyz_points(result["text"])
        except ValueError as exc:
            messagebox.showerror("Pole from points", f"{exc}\n\nUse X Y Z values, one point per line.", parent=self.root)
            return
        if len(points) != 2:
            messagebox.showerror("Pole from points", f"Enter exactly two points; found {len(points)}.", parent=self.root)
            return
        if np.linalg.norm(np.asarray(points[1]) - np.asarray(points[0])) < 0.01:
            messagebox.showerror("Pole from points", "The two endpoints must be different.", parent=self.root)
            return
        number = 1 + sum(obj.get("type") == "pipe" for obj in self.objects)
        name = simpledialog.askstring("Pole name", "Name:", parent=self.root, initialvalue=f"Pole {number}")
        if not name or not name.strip():
            return
        outer = simpledialog.askfloat("Pole outside diameter", "Outside diameter (in):", parent=self.root,
                                      initialvalue=4.0, minvalue=0.1, maxvalue=60.0)
        if outer is None:
            return
        inner = simpledialog.askfloat("Pole inside diameter", "Inside diameter (in; enter 0 for a solid pole):",
                                      parent=self.root, initialvalue=min(3.0, outer - 0.01),
                                      minvalue=0.0, maxvalue=max(0.0, outer - 0.01))
        if inner is None:
            return
        item = {"type": "pipe", "name": name.strip(), "pts": points, "d": float(outer), "id": float(inner)}
        self.objects.append(item)
        self.sel_i = len(self.objects) - 1
        self._save_objs()
        self.log(f"Mapped {obj_summary(item)}", "ok")

    def cancel_teach(self):
        if self.teach:
            self.teach = None
            self.log("Mapping cancelled.", "dim")
        self._teach_msg()
        self._objs_dirty = self.dirty = True

    def undo_point(self):
        if self.teach and self.teach["pts"]:
            self.teach["pts"].pop()
            self._teach_msg()
            self._objs_dirty = self.dirty = True

    def set_point(self):
        t = self.teach
        if not t:
            self.log("Press New plate or New pipe first.", "warn")
            return
        snap = self.shared.snapshot()
        if snap.pose is None:
            self.log("No pose reading yet.", "warn")
            return
        if snap.has_status and (snap.busy or clock() - snap.last_moving_t < 0.5):
            self.log("The robot is still moving \u2013 wait for IDLE, then press Set point.", "warn")
            return
        t["pts"].append([float(x) for x in snap.pose[:3]])
        p = t["pts"][-1]
        self.log(f"Point {len(t['pts'])}: X {p[0]:+.1f}  Y {p[1]:+.1f}  Z {p[2]:+.1f} in", "ok")
        steps = PLATE_STEPS if t["type"] == "plate" else PIPE_STEPS
        if len(t["pts"]) < len(steps):
            self._teach_msg()
        else:
            self._finish_teach()
        self._objs_dirty = self.dirty = True

    def _finish_teach(self):
        t = self.teach
        kind, pts = t["type"], t["pts"]
        n = 1 + sum(o["type"] == kind for o in self.objects)
        if kind == "plate":
            val = simpledialog.askfloat("Plate thickness", "Thickness extending upward from the mapped face (in):", parent=self.root, initialvalue=0.5, minvalue=0.01, maxvalue=24)
            obj = {"type": "plate", "name": f"Plate {n}", "pts": pts, "t": float(val or 0.5)}
        else:
            val = simpledialog.askfloat("Pipe size", "Outside diameter (in):", parent=self.root, initialvalue=4.0, minvalue=0.1, maxvalue=60)
            if val is None:
                self.teach = None; self._teach_msg(); return
            inner = simpledialog.askfloat("Pipe size", "Inside diameter (in; enter 0 for a solid pipe):", parent=self.root, initialvalue=min(3.0, val - 0.01), minvalue=0.0, maxvalue=max(0.0, val - 0.01))
            if inner is None:
                self.teach = None; self._teach_msg(); return
            obj = {"type": "pipe", "name": f"Pipe {n}", "pts": pts, "d": float(val), "id": float(inner)}
        self.teach = None
        self.objects.append(obj)
        self.sel_i = len(self.objects) - 1
        self._save_objs()
        self.log(f"Mapped {obj_summary(obj)}", "ok")
        self._teach_msg()

    def delete_obj(self):
        if 0 <= self.sel_i < len(self.objects):
            del self.objects[self.sel_i]
            self.sel_i = -1
            self._save_objs()

    def edit_obj(self):
        if not (0 <= self.sel_i < len(self.objects)):
            return
        o = self.objects[self.sel_i]
        if o["type"] == "plate":
            v = simpledialog.askfloat("Plate thickness", "Thickness extending upward from the mapped face (in):", parent=self.root, initialvalue=o.get("t", 0.5), minvalue=0.01, maxvalue=24)
            if v:
                o["t"] = float(v)
        else:
            v = simpledialog.askfloat("Pipe size", "Outside diameter (in):", parent=self.root, initialvalue=o.get("d", 4.0), minvalue=0.1, maxvalue=60)
            if v is not None:
                inner = simpledialog.askfloat("Pipe size", "Inside diameter (in; enter 0 for a solid pipe):", parent=self.root, initialvalue=o.get("id", 0.0), minvalue=0.0, maxvalue=max(0.0, v - 0.01))
                if inner is not None:
                    o["d"], o["id"] = float(v), float(inner)
        self._save_objs()

    def point_to_go(self, k):
        if not (0 <= self.sel_i < len(self.objects)) or k >= len(self.objects[self.sel_i]["pts"]):
            self.log("Select a mapped object that has that point.", "warn")
            return
        p = self.objects[self.sel_i]["pts"][k]
        self.relative.set(False)
        for i in range(3):
            self.fields[i].set(_fmtv(p[i]))
        self._preview()

    def _rebuild_solids(self):
        soups, self._labels = [], []
        gold, green = hex_rgb(C.yellow), hex_rgb(C.green)
        for i, o in enumerate(self.objects):
            sel = i == self.sel_i
            res = plate_solid(o, sel) if o["type"] == "plate" else pipe_solid(o, sel)
            if res is None:
                continue
            soup, ctr = res
            soups.append(soup)
            self._labels.append((ctr + np.array([0, 0, 0.04]), o["name"], (200, 205, 215), False))
            if sel:
                for k, p in enumerate(o["pts"]):
                    self._labels.append((K.table_to_base(p), str(k + 1), green, True))
        if self.teach:
            for k, p in enumerate(self.teach["pts"]):
                self._labels.append((K.table_to_base(p), str(k + 1), gold, True))
        self._obj_soup = tuple(np.concatenate([s[j] for s in soups]) for j in range(3)) if soups else None
        self._objs_dirty = False

    # ------------------------------------------------------------------ enabling / chips / side refresh
    def _set_controls(self):
        busy = self.guide.active
        for b in self.jog_btns:
            b.config(state="disabled" if busy else "normal")
        self.btn_go.config(state="disabled" if busy else "normal")
        self.btn_stop.config(state="normal" if busy else "disabled")
        self.btn_snap.config(state="normal" if busy else "disabled")
        self._paint_chips()

    def _refresh_chips(self, snap):
        live = snap.state == "Live"
        txt = f"\u25cf Reading {snap.scan_rate:.0f}/s" if live else f"\u25cb {snap.state}"
        self._cfg_label(self.chip_read, txt, C.green if live else C.amber)
        ready = self._controls_ready() and HAVE_PYNPUT
        self._cfg_label(self.chip_ctl, "\u25cf Controls ready" if ready else "\u25cb Controls not set up", C.green if ready else C.muted)

    def _cfg_label(self, lbl, text, fg):
        if self._cache.get(id(lbl)) != (text, fg):
            self._cache[id(lbl)] = (text, fg)
            lbl.config(text=text, fg=fg)

    def _refresh_side(self, snap):
        self._refresh_chips(snap)
        if snap.pose is not None:
            for i in range(6):
                self._cfg_label(self.pose_lbl[i], fmt_pos(snap.pose[i]) if i < 3 else fmt_ang(snap.pose[i]), C.text)
        if snap.has_status:
            moving = bool(snap.busy)
            self._cfg_label(self.state_pill, "\u25cf Moving" if moving else "\u25cf Idle" if snap.busy is False else "\u25cb Unknown",
                            C.amber if moving else C.green if snap.busy is False else C.muted)
        else:
            self._cfg_label(self.state_pill, "no banner set", C.faint)
        self._cfg_label(self.read_lbl, f"{snap.scan_rate:.0f} scans/s \u00b7 read: {snap.raw}", C.faint)
        deg = np.degrees(self.q_vis)
        for i in range(6):
            self.jbars[i].set(deg[i])
            self._cfg_label(self.jlbls[i], f"{deg[i]:+7.1f}\u00b0", C.text)
        with self.shared.lock:
            live = self.shared.ik.get("live")
        if live is not None:
            self._cfg_label(self.fit_lbl, f"fit {live.pe:.1f} mm / {live.re:.1f}\u00b0", C.muted if live.pe < 15 else C.amber)

    # ------------------------------------------------------------------ main loop
    def tick(self):
        t0 = clock()
        try:
            self._tick(t0)
        except Exception:
            if t0 - self._cache.get("err_t", 0) > 5:
                self._cache["err_t"] = t0
                traceback.print_exc()
        self.root.after(max(1, int(16 - (clock() - t0) * 1000)), self.tick)

    def _tick(self, now):
        dt = min(0.1, now - self.last_tick)
        self.last_tick = now
        usage = self.gpu_monitor.latest
        if usage != self.gpu_usage:
            self.gpu_usage = usage
            self.dirty = True
        if self.setup_open:
            return
        was_active = self.guide.active
        self.guide.tick(now)
        if was_active != self.guide.active:
            self._set_controls()
        self.beacon.tick(now)
        if self.hot.snap_flag.is_set():
            self.hot.snap_flag.clear()
            self.snap_cursor()
        if self.hot.abort_flag.is_set():
            self.hot.abort_flag.clear()
            self.abort_panel()
        if self.abort_pending and now - self.abort_t > 60:
            self.abort_pending = False
            self.beacon.hide()
        snap = self.shared.snapshot()
        if not self.guide.active and snap.has_step and snap.step_level is not None and snap.step_level != self.panel_level:
            self.set_panel_level(snap.step_level, "read from the screen")
        with self.shared.lock:
            live, ghost = self.shared.ik.get("live"), self.shared.ik.get("ghost")
        if live is not None and live.t != self.live_t:
            self.live_t, self.q_target = live.t, live.q.copy()
        if snap.pose is not None and snap.seq != self.last_seq:
            self.last_seq = snap.seq
            self.trail_list.append(K.table_pose_to_target(snap.pose)[:3, 3].copy())
            del self.trail_list[:-260]
            self.trail_arr = np.array(self.trail_list)
            self.dirty = True
        diff = self.q_target - self.q_vis                          # smooth, speed-limited motion of the DRAWN arm
        if float(np.max(np.abs(diff))) > 2e-4:
            vmax = math.radians(float(self.settings.get("viz_deg_s", 25.0))) * dt
            self.q_vis = self.q_vis + np.clip(diff * (1.0 - math.exp(-dt * 14.0)), -vmax, vmax)
            self.dirty = True
        if ghost is not None and ghost.t != self.ghost_t and (self.guide.active or self._has_field_input()):
            self.ghost_t = ghost.t
            ok = ghost.pe < 15.0 and ghost.re < 6.0
            self.ghost = {"q": ghost.q.copy(), "ok": ok, "pose": ghost.pose}
            if not ok and not self.ghost_warned:
                self.ghost_warned = True
                self.log(f"Heads-up: my model says that pose may be out of reach ({ghost.pe:.0f} mm off). "
                         "The model is approximate \u2013 the real robot decides.", "warn")
            self.dirty = True
        sh = self.shared
        if sh.nrej != self._nrej_seen and now - self._rej_t > 2.0 and sh.last_rej is not None:
            new, old = sh.last_rej
            d = new - old
            d[3:] = wrap180(d[3:])
            i = int(np.argmax(np.abs(d)))
            self.log(f"Ignored a suspicious read ({sh.nrej - self._nrej_seen}x): {AXES[i]} {old[i]:+.1f} \u2192 {new[i]:+.1f}", "dim")
            self._nrej_seen, self._rej_t = sh.nrej, now
        if self.follow.get() and snap.pose is not None and self.trail_list:
            self.cam.target = self.cam.target + (self.trail_list[-1] - self.cam.target) * (1.0 - math.exp(-dt * 3.0))
            self.dirty = True
        if self.dirty:
            self._render(snap)
            self.dirty = False
        if now - self.last_side > 0.12:
            self.last_side = now
            self._refresh_side(snap)
            if self.bank.dirty and now - self._cache.get("bank_save", 0) > 20:
                self._cache["bank_save"] = now
                self.bank.save()
                self._update_digit_label()

    def _has_field_input(self):
        return any(sv.get().strip() for sv in self.fields)

    def _render(self, snap):
        if self._objs_dirty:
            self._rebuild_solids()
        w, h = self.vw, self.vh
        info = self.robot_mesh.update(self.q_vis)
        tris = self.robot_mesh.triangles()
        show_objs = bool(self.show_objs.get())
        if show_objs and self._obj_soup is not None:
            tris = tuple(np.concatenate([a, b]) for a, b in zip(tris, self._obj_soup))
        hud = {"fps": self.fps, "gpu": self.gpu_usage, "backend": self.renderer.name,
               "axes": bool(self.show_axes.get())}
        if snap.pose is not None:
            p = snap.pose
            hud["tip_text"] = f"{fmt_pos(p[0])}  {fmt_pos(p[1])}  {fmt_pos(p[2])} in"
        elif self.cfg.get("region"):
            hud["msg"] = "Waiting for the number table\u2026"
        else:
            hud["msg"] = "Open Setup to pick the number table"
        gt, gtip, gok = None, None, True
        if self.ghost is not None:
            ginfo = self.ghost_mesh.update(self.ghost["q"])
            gt, gtip, gok = self.ghost_mesh.triangles(), ginfo.tip, self.ghost["ok"]
            gp = self.ghost["pose"]
            hud["ghost_text"] = "target  " + "  ".join(f"{AXES[i]} {gp[i]:+.1f}" for i in range(6))
        fr = SimpleNamespace(tris=tris, ghost_tris=gt, ghost_col=hex_rgb(C.green if gok else C.amber), ghost_tip=gtip, ghost_ok=gok,
                             tip=info.tip, trail=self.trail_arr if self.show_trail.get() else None,
                             labels=self._labels if show_objs else [], gizmo=bool(self.show_gizmo.get()), hud=hud)
        t0 = clock()
        rgb = self.renderer.render(w, h, self.cam, fr)
        self.render_ms = 0.9 * self.render_ms + 0.1 * (clock() - t0) * 1000
        im = Image.fromarray(rgb)
        if self.photo is None or self.photo_size != (w, h):
            self.photo, self.photo_size = ImageTk.PhotoImage(im), (w, h)
            self.view.itemconfig(self.img_item, image=self.photo)
        else:
            self.photo.paste(im)
        self._fps_n += 1
        if clock() - self._fps_t >= 1.0:
            self.fps, self._fps_n, self._fps_t = self._fps_n / (clock() - self._fps_t), 0, clock()

    # ------------------------------------------------------------------ shutdown
    def shutdown(self):
        self.gpu_monitor.stop()
        self.gpu_monitor.join(timeout=2)
        try:
            self.cfg.set("window", self.root.geometry().split("+")[0])
            self.cfg.set("settings", self.settings)
            self.cfg.set("level", self.panel_level)
            self.cfg.save()
            self.bank.save()
        except Exception:
            pass
        self.cursor.stop()
        for part in (self.ocr, self.step_reader, self.ik, self.clicks, self.hot):
            try:
                part.stop()
            except Exception:
                pass
        self.beacon.destroy()
        self.root.destroy()


def _fmtv(v):
    return f"{v + 0.0:+.2f}".rstrip("0").rstrip(".")


def main():
    if WAYLAND_SESSION:
        try:
            from .qt_ui import main as wayland_main
            wayland_main()
        except Exception as exc:
            print("Native Wayland UI could not start:", exc)
            print("Install PySide6 and the Qt Wayland platform plugin for your Linux distribution.")
            traceback.print_exc()
        return
    try:
        print("R08 live twin (R-2000iA/200F) \u2013 starting.  Config:", CONFIG_FILE)
        App().run()
    except Exception:
        print("\nERROR")
        traceback.print_exc()
        try:
            input("\nPress Enter to close...")
        except Exception:
            pass
