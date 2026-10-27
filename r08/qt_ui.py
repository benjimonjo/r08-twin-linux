"""Native Qt UI used on Linux Wayland sessions.

The Tk UI remains the cross-platform desktop implementation. Tk on Unix is
Xlib based, so Wayland sessions use this Qt Widgets frontend instead.
"""
import os
os.environ["QT_QPA_PLATFORM"] = "wayland"

import math
import time
import numpy as np
import cv2
from PySide6 import QtCore, QtGui, QtWidgets

from .common import (C, AXES, UNITS, Config, CONFIG_FILE, DIGIT_BANK_FILE,
                     DEFAULT_SETTINGS, START_LEVEL, new_mss, hex_rgb, clock)
from . import kinematics as K
from .digits import DigitBank
from .model3d import Camera, RobotMesh
from .render import make_renderer
from .screen_reader import Shared, OcrWorker, IKWorker
from .screen_reader import split_cells
from . import digits
from .picker import ALL_KEYS, STAGE_DEFS, BUTTON_KEYS, compute_buttons
from .inputs import HAVE_PYNPUT
from .widgets import NEEDED_BUTTONS, fmt_pos, fmt_ang
from types import SimpleNamespace


STYLE = """
QMainWindow, QDialog { background: #13151a; color: #e8e9ed; }
QWidget { color: #e8e9ed; font: 10pt 'Noto Sans'; }
QFrame[card="true"] { background: #1f232b; border: 1px solid #303641; border-radius: 7px; }
QLabel[muted="true"] { color: #8d95a2; }
QPushButton { background: #2a2f39; border: 0; border-radius: 4px; padding: 7px 11px; }
QPushButton:hover { background: #353b47; }
QPushButton:disabled { color: #5d6573; }
QLineEdit, QComboBox, QListWidget, QTextEdit { background: #191c22; border: 1px solid #303641; border-radius: 3px; padding: 5px; }
QCheckBox { spacing: 7px; }
QScrollArea { border: none; }
"""


class SetupPicker(QtWidgets.QDialog):
    """Wayland-native screenshot setup dialog with local Qt keyboard handling."""
    def __init__(self, parent, keys, capture):
        super().__init__(parent)
        self.setWindowTitle("R08 screen setup")
        self.setWindowFlags(QtCore.Qt.WindowType.Window)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.keys, self.i, self.capture = list(keys), 0, capture
        self.rects, self.pts, self.markers = {}, {}, {}
        self.drag_start = self.drag_end = None
        self.mdrag = None
        self.result_data = None
        self.screen_rect = None
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self.setStyleSheet("background:#111318; color:#e8e9ed")
        monitor = capture.monitors[0]
        shot = capture.grab(monitor)
        self.image = QtGui.QImage(shot.bgra, shot.size[0], shot.size[1],
                                  shot.size[0] * 4, QtGui.QImage.Format.Format_ARGB32).copy()
        self.monitor = monitor
        self.resize(monitor["width"], monitor["height"])
        self.move(monitor["left"], monitor["top"])
        self.showFullScreen()
        self.setFocus()

    @property
    def stage(self):
        return self.keys[self.i]

    @property
    def kind(self):
        return STAGE_DEFS[self.stage][0]

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.drawImage(self.rect(), self.image)
        sx, sy = self.width() / self.image.width(), self.height() / self.image.height()
        p.save()
        p.scale(sx, sy)
        pen = QtGui.QPen(QtGui.QColor(C.yellow), 2)
        p.setPen(pen)
        for key, r in self.rects.items():
            x, y = r["left"] - self.monitor["left"], r["top"] - self.monitor["top"]
            p.drawRect(x, y, r["width"], r["height"])
        if self.drag_start and self.drag_end:
            p.drawRect(QtCore.QRect(self.drag_start, self.drag_end).normalized())
        for name, (x, y) in self.markers.items():
            x, y = x - self.monitor["left"], y - self.monitor["top"]
            p.drawEllipse(QtCore.QPointF(x, y), 12, 12)
            p.drawText(x + 14, y - 8, name)
        banner = QtCore.QRect(28, 24, min(self.image.width() - 56, 780), 126)
        p.fillRect(banner, QtGui.QColor(19, 21, 26, 238))
        p.setPen(QtGui.QPen(QtGui.QColor(C.yellow), 2))
        p.drawRect(banner)
        title, hint, optional = STAGE_DEFS[self.stage][1:4]
        p.setPen(QtGui.QColor(C.text))
        p.setFont(QtGui.QFont("Noto Sans", 15, QtGui.QFont.Weight.DemiBold))
        p.drawText(banner.adjusted(14, 12, -14, -76), QtCore.Qt.AlignmentFlag.AlignLeft, title)
        p.setFont(QtGui.QFont("Noto Sans", 10))
        p.setPen(QtGui.QColor(C.muted))
        p.drawText(banner.adjusted(14, 48, -14, -30), QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.TextFlag.TextWordWrap, hint)
        keys = "Drag, then Enter" if self.kind == "rect" else "Click the requested target"
        if self.stage in BUTTON_KEYS:
            keys += "  ·  S skips button setup"
        if len(self.capture.monitors) > 1:
            keys += "  ·  M changes screen"
        keys += "  ·  Backspace goes back  ·  Esc " + ("skips" if optional else "cancels")
        p.setPen(QtGui.QColor(C.faint))
        p.drawText(banner.adjusted(14, 92, -10, -8), QtCore.Qt.AlignmentFlag.AlignLeft, keys)
        p.restore(); p.end()

    def _image_point(self, e):
        return QtCore.QPoint(round(e.position().x() * self.image.width() / max(1, self.width())),
                             round(e.position().y() * self.image.height() / max(1, self.height())))

    def _next(self):
        self.drag_start = self.drag_end = None
        self.i += 1
        if self.i < len(self.keys) and self.keys[self.i] == "review":
            self.markers = compute_buttons(self.pts)
            if not self.markers:
                self._next()
                return
        if self.i >= len(self.keys):
            out = {}
            if "region" in self.rects: out["region"] = self.rects["region"]
            if "status" in self.rects: out["status"] = self.rects["status"]
            if "step_label" in self.rects: out["step"] = self.rects["step_label"]
            if self.markers:
                out["buttons"] = {k: [float(v[0]), float(v[1])] for k, v in self.markers.items()}
            elif self.pts:
                out["buttons"] = compute_buttons(self.pts)
            self.result_data = out
            self.accept()
        else:
            self.update()

    def keyPressEvent(self, e):
        if not (0 <= self.i < len(self.keys)):
            return
        if e.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            if self.kind == "rect" and self.drag_end:
                a, b = self.drag_start, self.drag_end
                x1, y1 = self._absolute(min(a.x(), b.x()), min(a.y(), b.y()))
                x2, y2 = self._absolute(max(a.x(), b.x()), max(a.y(), b.y()))
                if x2 - x1 >= 12 and y2 - y1 >= 8:
                    self.rects[self.stage] = {"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1}
                    self._next()
            elif self.kind == "review":
                self.mdrag = self._near_marker(point.x(), point.y())
                if self.mdrag is None: self._next()
        elif e.key() == QtCore.Qt.Key.Key_Escape:
            if STAGE_DEFS[self.stage][3]: self._next()
            else: self.reject()
        elif e.key() == QtCore.Qt.Key.Key_Backspace:
            if self.i:
                self.i -= 1
                self.rects.pop(self.stage, None); self.pts.pop(self.stage, None)
                self.update()
        elif e.key() == QtCore.Qt.Key.Key_M and len(self.capture.monitors) > 1:
            self.monitor = self.capture.monitors[(self.capture.monitors.index(self.monitor) + 1) % len(self.capture.monitors)]
            shot = self.capture.grab(self.monitor)
            self.image = QtGui.QImage(shot.bgra, shot.size[0], shot.size[1], shot.size[0] * 4,
                                      QtGui.QImage.Format.Format_ARGB32).copy()
            self.update()
        elif e.text().lower() == "s" and self.stage in BUTTON_KEYS:
            while self.i < len(self.keys) and self.stage in BUTTON_KEYS:
                self.pts.pop(self.stage, None); self.i += 1
            if self.i >= len(self.keys): self._next()
            else: self.update()
        else:
            super().keyPressEvent(e)

    def _absolute(self, x, y):
        return int(self.monitor["left"] + x * self.monitor["width"] / self.width()), int(self.monitor["top"] + y * self.monitor["height"] / self.height())

    def mousePressEvent(self, e):
        self.setFocus()
        point = self._image_point(e)
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.kind == "point":
                self.pts[self.stage] = self._absolute(point.x(), point.y())
                self._next()
            elif self.kind == "rect":
                self.drag_start = self.drag_end = point; self.update()
            elif self.kind == "review":
                self._next()

    def mouseMoveEvent(self, e):
        if self.drag_start:
            self.drag_end = self._image_point(e); self.update()
        elif self.mdrag:
            point = self._image_point(e)
            self.markers[self.mdrag] = list(self._absolute(point.x(), point.y()))
            self.update()

    def _near_marker(self, x, y):
        best = None
        for name, (ax, ay) in self.markers.items():
            px, py = ax - self.monitor["left"], ay - self.monitor["top"]
            distance = math.hypot(px - x, py - y)
            if distance < 22 and (best is None or distance < best[0]):
                best = (distance, name)
        return best[1] if best else None

    def mouseReleaseEvent(self, e):
        if self.kind == "rect" and self.drag_start:
            self.drag_end = self._image_point(e)
            self.update()
        self.mdrag = None


class QtApp:
    def __init__(self):
        self.cfg = Config(CONFIG_FILE)
        self.settings = dict(DEFAULT_SETTINGS); self.settings.update(self.cfg.get("settings", {}) or {})
        if self.cfg.get("tool"): K.TOOL.update(self.cfg.get("tool"))
        self.qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.qt.setStyle("Fusion"); self.qt.setStyleSheet(STYLE)
        self.window = QtWidgets.QMainWindow(); self.window.setWindowTitle("R08 live twin — FANUC R-2000iA/200F")
        self.window.resize(1450, 900); self.window.setMinimumSize(1000, 650)
        self.shared = Shared(); self.ik = IKWorker(self.shared); self.ik.start()
        self.shared.set_regions(region=self.cfg.get("region"), status=self.cfg.get("status"), step=self.cfg.get("step"))
        self.bank = DigitBank(DIGIT_BANK_FILE)
        self.ocr = OcrWorker(self.shared, self.settings, self.ik, self.bank); self.ocr.start()
        self.capture = None
        self.renderer, self.render_note = make_renderer(bool(self.settings.get("gpu", True)))
        self.robot_mesh, self.ghost_mesh, self.cam = RobotMesh(), RobotMesh(), Camera()
        self.q_vis = np.radians([0, 35, 25, 0, 40, 0]); self.q_target = self.q_vis.copy()
        self.live_t = self.ghost_t = None; self.trail = []; self.dirty = True
        self.image_label = None; self.last_tick = clock(); self.log_entries = []
        self._build()
        self._refresh()
        self.window.closeEvent = self._close_event
        self.window.show()
        self.timer = QtCore.QTimer(self.window); self.timer.timeout.connect(self._tick); self.timer.start(33)
        if not self.cfg.get("region"):
            QtCore.QTimer.singleShot(500, lambda: self.run_setup(ALL_KEYS))
        self.log(f"R08 live twin started. 3D: {self.renderer.name}.")
        if self.render_note: self.log(self.render_note)
        self.log("Native Wayland UI: global pointer automation is unavailable; viewer and screen setup are active.")

    def _build(self):
        central = QtWidgets.QWidget(); self.window.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central); root.setContentsMargins(10, 8, 10, 10)
        header = QtWidgets.QHBoxLayout(); root.addLayout(header)
        brand = QtWidgets.QLabel("R08  live twin  ·  FANUC R-2000iA/200F"); brand.setStyleSheet(f"color:{C.yellow};font-size:16px;font-weight:700")
        header.addWidget(brand); header.addStretch()
        self.read_state = QtWidgets.QLabel("Waiting for setup"); header.addWidget(self.read_state)
        setup = QtWidgets.QPushButton("Setup ▾"); setup.clicked.connect(self._setup_menu); header.addWidget(setup)
        helpb = QtWidgets.QPushButton("Help"); helpb.clicked.connect(self._help); header.addWidget(helpb)
        split = QtWidgets.QSplitter(); root.addWidget(split, 1)
        left = QtWidgets.QWidget(); ll = QtWidgets.QVBoxLayout(left); ll.setContentsMargins(0, 0, 4, 0)
        self.image_label = QtWidgets.QLabel("Open Setup to select the number table")
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(460, 360); self.image_label.setStyleSheet("background:#0f1115;border:1px solid #303641")
        ll.addWidget(self.image_label, 1)
        self.log_view = QtWidgets.QTextEdit(); self.log_view.setReadOnly(True); self.log_view.setMaximumHeight(110); ll.addWidget(self.log_view)
        split.addWidget(left)
        side = QtWidgets.QWidget(); side.setMinimumWidth(350); side.setMaximumWidth(440); sl = QtWidgets.QVBoxLayout(side)
        self.pose_labels = []
        box = self._group("Robot position"); grid = QtWidgets.QGridLayout(box)
        for i, axis in enumerate(AXES):
            lab = QtWidgets.QLabel(f"{axis} ({UNITS[i]})"); val = QtWidgets.QLabel("—")
            val.setStyleSheet("font-size:16px;font-weight:600")
            grid.addWidget(lab, i // 3 * 2, i % 3); grid.addWidget(val, i // 3 * 2 + 1, i % 3)
            self.pose_labels.append(val)
        sl.addWidget(box)
        self.capture_label = QtWidgets.QLabel("Not configured"); self.capture_label.setWordWrap(True); box.layout().addWidget(self.capture_label)
        status = self._group("Controls"); form = QtWidgets.QVBoxLayout(status)
        self.control_label = QtWidgets.QLabel("Guided controls require a desktop input portal on Wayland."); self.control_label.setWordWrap(True); form.addWidget(self.control_label)
        self.button_state = QtWidgets.QLabel("Panel buttons not set up"); form.addWidget(self.button_state)
        setupb = QtWidgets.QPushButton("Set up panel buttons"); setupb.clicked.connect(lambda: self.run_setup([k for k in ALL_KEYS if k != "region"]))
        form.addWidget(setupb); sl.addWidget(status)
        digit = self._group("Digit reader"); dl = QtWidgets.QVBoxLayout(digit)
        self.raw_label = QtWidgets.QLabel("Waiting for the number table…"); self.raw_label.setWordWrap(True); dl.addWidget(self.raw_label)
        teach = QtWidgets.QPushButton("Teach digits…"); teach.clicked.connect(self._teach_digits); dl.addWidget(teach); sl.addWidget(digit)
        sl.addStretch(); split.addWidget(side); split.setStretchFactor(0, 3); split.setStretchFactor(1, 1)
        self.window.menuBar().setNativeMenuBar(True)

    def _group(self, title):
        box = QtWidgets.QGroupBox(title); box.setStyleSheet("QGroupBox {border:1px solid #303641;border-radius:6px;margin-top:8px;padding:8px;font-weight:600} QGroupBox::title {subcontrol-origin:margin;left:8px;padding:0 4px}")
        return box

    def _setup_menu(self):
        menu = QtWidgets.QMenu(self.window)
        menu.addAction("Run full setup", lambda: self.run_setup(ALL_KEYS))
        menu.addAction("Number table", lambda: self.run_setup(["region"]))
        menu.addAction("Panel buttons", lambda: self.run_setup(BUTTON_KEYS))
        menu.addAction("Optional banner and step text", lambda: self.run_setup(["status", "step_label"]))
        menu.exec(QtGui.QCursor.pos())

    def run_setup(self, keys):
        try:
            if self.capture is None: self.capture = new_mss()
            dlg = SetupPicker(self.window, keys, self.capture)
            if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted or not dlg.result_data:
                return
            result = dlg.result_data
            for src, dst in (("region", "region"), ("status", "status"), ("step", "step")):
                if src in result: self.cfg.set(dst, result[src])
            if "buttons" in result:
                old = self.cfg.get("buttons", {}) or {}
                old.update(result["buttons"]); self.cfg.set("buttons", old)
            self.cfg.save(); self.shared.set_regions(region=self.cfg.get("region"), status=self.cfg.get("status"), step=self.cfg.get("step"))
            self.log("Setup saved.")
            if "region" in result: self.log("Tip: teach a few digits so the reader learns the stream font.")
            self._refresh()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self.window, "Setup failed", str(exc))
            self.log(f"Setup failed: {exc}")

    def _help(self):
        QtWidgets.QMessageBox.information(self.window, "R08 live twin", "Set up the number table to start screen reading. Use the Setup menu to calibrate panel buttons and optional status regions.\n\nThe Wayland frontend uses native Qt windows and the approved PipeWire screen stream. Guided global pointer automation is not enabled in this build.")

    def _teach_digits(self):
        if not self.cfg.get("region"):
            self.log("Set up the number table first."); return
        if self.capture is None:
            try: self.capture = new_mss()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self.window, "Teach digits", str(exc)); return
        text, ok = QtWidgets.QInputDialog.getText(self.window, "Teach digits", "Enter the six values currently shown, in X, Y, Z, Xa, Ya, Za order:")
        if ok:
            values = text.replace(",", " ").split()
            if len(values) != 6:
                QtWidgets.QMessageBox.warning(self.window, "Teach digits", "Enter exactly six values."); return
            try:
                bgra = np.asarray(self.capture.grab(self.cfg.get("region")))
                gray = cv2.cvtColor(bgra, cv2.COLOR_BGRA2GRAY)
                cells = split_cells(gray)
                added, messages = 0, []
                for i, (cell, truth) in enumerate(zip(cells, values)):
                    n, msg = digits.teach(cell, i, truth, self.bank)
                    added += n
                    if msg != "ok": messages.append(f"{AXES[i]}: {msg}")
                self.bank.save()
                detail = f"Added {added} digit samples."
                if messages: detail += "\n" + "\n".join(messages)
                QtWidgets.QMessageBox.information(self.window, "Teach digits", detail)
                self.log(detail.replace("\n", " · "))
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self.window, "Teach digits", str(exc))

    def log(self, text):
        stamp = time.strftime("%H:%M:%S")
        self.log_view.append(f"<span style='color:{C.faint}'>{stamp}</span>  {text}")

    def _refresh(self):
        self.button_state.setText("Panel buttons ready" if all(n in (self.cfg.get("buttons", {}) or {}) for n in NEEDED_BUTTONS) else "Panel buttons not set up")
        self.capture_label.setText("Number table configured" if self.cfg.get("region") else "Number table not configured")

    def _tick(self):
        now = clock(); dt = min(0.1, now - self.last_tick); self.last_tick = now
        snap = self.shared.snapshot()
        self.read_state.setText(f"● Reading {snap.scan_rate:.0f}/s" if snap.state == "Live" else f"○ {snap.state}")
        self.raw_label.setText(snap.raw or "Waiting for the number table…")
        if snap.pose is not None:
            for i, val in enumerate(snap.pose): self.pose_labels[i].setText(fmt_pos(val) if i < 3 else fmt_ang(val))
        with self.shared.lock:
            live = self.shared.ik.get("live")
        if live and live.t != self.live_t:
            self.live_t = live.t; self.q_target = live.q.copy()
        if snap.pose is not None and (not self.trail or snap.seq != getattr(self, "last_seq", -1)):
            self.last_seq = snap.seq; self.trail.append(K.table_pose_to_target(snap.pose)[:3, 3].copy()); self.trail = self.trail[-260:]
            self.dirty = True
        diff = self.q_target - self.q_vis
        if np.max(np.abs(diff)) > 2e-4:
            vmax = math.radians(float(self.settings.get("viz_deg_s", 25))) * dt
            self.q_vis += np.clip(diff * (1 - math.exp(-dt * 14)), -vmax, vmax); self.dirty = True
        if self.dirty: self._render(snap); self.dirty = False

    def _render(self, snap):
        w, h = max(400, self.image_label.width()), max(300, self.image_label.height())
        info = self.robot_mesh.update(self.q_vis)
        hud = {"backend": self.renderer.name}
        if snap.pose is not None: hud["tip_text"] = "  ".join(fmt_pos(snap.pose[i]) for i in range(3)) + " in"
        else: hud["msg"] = "Open Setup to select the number table"
        frame = SimpleNamespace(tris=self.robot_mesh.triangles(), ghost_tris=None, ghost_col=hex_rgb(C.green), ghost_tip=None,
                                ghost_ok=True, tip=info.tip, trail=np.asarray(self.trail) if self.trail else None,
                                labels=[], gizmo=True, hud=hud)
        rgb = self.renderer.render(w, h, self.cam, frame)
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def _close_event(self, event):
        try:
            self.cfg.set("settings", self.settings); self.cfg.save(); self.bank.save()
        except Exception: pass
        self.ocr.stop(); self.ik.stop()
        event.accept()

    def run(self):
        return self.qt.exec()


def main():
    app = QtApp()
    return app.run()
