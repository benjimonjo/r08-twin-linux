"""Native Qt UI used on Linux Wayland sessions.

The Tk UI remains the cross-platform desktop implementation. Tk on Unix is
Xlib based, so Wayland sessions use this Qt Widgets frontend instead.
"""
import os
os.environ["QT_QPA_PLATFORM"] = "wayland"

import math
import time
import re
import sys
import numpy as np
import cv2
from PySide6 import QtCore, QtGui, QtWidgets

from .common import (C, AXES, UNITS, Config, CONFIG_FILE, DIGIT_BANK_FILE,
                     DEFAULT_SETTINGS, START_LEVEL, STEP_LEVELS, DEBUG_DIR, wrap180, new_mss, hex_rgb, clock)
from . import kinematics as K
from .digits import DigitBank
from .model3d import Camera, RobotMesh, plate_solid, pipe_solid, obj_summary
from .render import make_renderer
from .screen_reader import Shared, OcrWorker, StepReader, IKWorker, HAVE_TESS, pytesseract
from .screen_reader import split_cells
from . import digits
from .picker import ALL_KEYS, STAGE_DEFS, BUTTON_KEYS, compute_buttons
from .inputs import HAVE_PYNPUT
from .widgets import NEEDED_BUTTONS, SETTING_DEFS, fmt_pos, fmt_ang
from .planner import plan_presses, step_value
from .wayland_input import WaylandPointer
from .github_updater import check_for_updates, install_update
from .gpu_usage import GpuUsageMonitor
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


class SceneLabel(QtWidgets.QLabel):
    dragged = QtCore.Signal(int, int, object)
    wheeled = QtCore.Signal(int)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self._last = None
    def mousePressEvent(self, event):
        self._last = event.position().toPoint(); self.setFocus()
    def mouseMoveEvent(self, event):
        if self._last is not None:
            p = event.position().toPoint(); delta = p - self._last
            self.dragged.emit(delta.x(), delta.y(), event.buttons())
            self._last = p
    def mouseReleaseEvent(self, event): self._last = None
    def wheelEvent(self, event):
        self.wheeled.emit(int(event.angleDelta().y() / 120)); event.accept()


class PortalSignals(QtCore.QObject):
    ready = QtCore.Signal(bool, str)


class UpdateWorker(QtCore.QThread):
    finished_result = QtCore.Signal(bool, object)
    def __init__(self, root, apply_info=None, parent=None):
        super().__init__(parent); self.root = root; self.apply_info = apply_info
    def run(self):
        try:
            result = install_update(self.apply_info) if self.apply_info else check_for_updates(self.root)
            self.finished_result.emit(True, result)
        except Exception as exc:
            self.finished_result.emit(False, str(exc))


class ButtonPreview(QtWidgets.QLabel):
    clicked = QtCore.Signal(float, float)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit(event.position().x(), event.position().y())


class ButtonReviewDialog(QtWidgets.QDialog):
    def __init__(self, parent, monitor, image, detected, current):
        super().__init__(parent)
        self.setWindowTitle("Review recognized panel buttons")
        self.monitor = dict(monitor); self.buttons = None
        self.detected = dict(detected)
        self.positions = {name: detected.get(name, current.get(name)) for name in NEEDED_BUTTONS}
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Select a button row, then click the center of that button in the screenshot to set its position. Source shows whether a value came from OCR or saved setup."))
        self.preview = ButtonPreview(); self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        h, w = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888).copy()
        self.source_pixmap = QtGui.QPixmap.fromImage(qimg).scaled(960, 500, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                                                  QtCore.Qt.TransformationMode.SmoothTransformation)
        self.preview.setFixedSize(self.source_pixmap.size()); layout.addWidget(self.preview, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.preview.clicked.connect(self._set_selected_position)
        self.table = QtWidgets.QTableWidget(len(NEEDED_BUTTONS), 4)
        self.table.setHorizontalHeaderLabels(["Button", "X screen px", "Y screen px", "Source"])
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        for col in (1, 2, 3): self.table.horizontalHeader().setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for row, name in enumerate(NEEDED_BUTTONS):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            point = self.positions[name]
            if point:
                self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{float(point[0]):.1f}"))
                self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{float(point[1]):.1f}"))
            source = "OCR" if name in detected else "Saved setup" if point else "Missing"
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(source))
        layout.addWidget(self.table)
        self.table.setCurrentCell(0, 0)
        self._render_preview()
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Save | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons)
        self.resize(1020, min(900, self.source_pixmap.height() + 390))

    def _render_preview(self):
        pix = self.source_pixmap.copy()
        qp = QtGui.QPainter(pix); qp.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        sx = pix.width() / float(self.monitor["width"])
        sy = pix.height() / float(self.monitor["height"])
        qp.setFont(QtGui.QFont("Noto Sans", 8, QtGui.QFont.Weight.Bold))
        for name in NEEDED_BUTTONS:
            point = self.positions[name]
            if point:
                x = (float(point[0]) - self.monitor["left"]) * sx
                y = (float(point[1]) - self.monitor["top"]) * sy
                color = QtGui.QColor(C.green if name in self.detected else C.yellow)
                qp.setPen(QtGui.QPen(QtGui.QColor("#15130a"), 1)); qp.setBrush(color)
                qp.drawEllipse(QtCore.QPointF(x, y), 7, 7); qp.drawText(QtCore.QPointF(x + 8, y - 5), name)
        qp.end()
        self.preview.setPixmap(pix)

    def _set_selected_position(self, px, py):
        row = self.table.currentRow()
        if row < 0: return
        name = NEEDED_BUTTONS[row]
        x = self.monitor["left"] + px * self.monitor["width"] / self.source_pixmap.width()
        y = self.monitor["top"] + py * self.monitor["height"] / self.source_pixmap.height()
        self.positions[name] = [x, y]
        self.table.item(row, 1).setText(f"{x:.1f}")
        self.table.item(row, 2).setText(f"{y:.1f}")
        self.table.item(row, 3).setText("Preview click")
        self.detected.pop(name, None)
        self._render_preview()

    def _accept(self):
        out = {}
        for row, name in enumerate(NEEDED_BUTTONS):
            try:
                x = float(self.table.item(row, 1).text()); y = float(self.table.item(row, 2).text())
            except (AttributeError, ValueError):
                QtWidgets.QMessageBox.warning(self, "Missing button", f"Provide valid coordinates for {name}."); return
            if not (self.monitor["left"] <= x < self.monitor["left"] + self.monitor["width"] and
                    self.monitor["top"] <= y < self.monitor["top"] + self.monitor["height"]):
                QtWidgets.QMessageBox.warning(self, "Invalid button position", f"{name} is outside the captured monitor."); return
            out[name] = [x, y]
        self.buttons = out; self.accept()


def detect_panel_buttons(capture):
    monitor = capture.monitors[0]
    image = np.asarray(capture.grab(monitor))
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    found = {}

    # The colored controls are much easier to locate than their small labels.
    # OCR each blue button separately at high scale, then save the button
    # rectangle center (the click target), rather than the text baseline.
    hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([88, 55, 65], np.uint8), np.array([140, 255, 255], np.uint8))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    button_config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+-"
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 12 or h < 8 or w * h < 120 or w / max(1, h) > 8:
            continue
        pad = max(2, round(min(w, h) * .12))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
        crop = gray[y0:y1, x0:x1]
        enlarged = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        binary = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        variants = [binary, cv2.bitwise_not(binary)]
        for variant in variants:
            label = pytesseract.image_to_string(variant, config=button_config, timeout=2)
            label = re.sub(r"[^A-Z+\-]", "", label.upper().replace("−", "-").replace("–", "-"))
            if label in NEEDED_BUTTONS:
                found[label] = ([monitor["left"] + x + w / 2, monitor["top"] + y + h / 2], 100)
                break

    # Find colored button bodies, including the green circular minus controls
    # and red Step-/Abort controls. Their centers are better click targets than
    # the nearby OCR text positions.
    saturated = cv2.inRange(hsv, np.array([0, 95, 65], np.uint8), np.array([179, 255, 255], np.uint8))
    saturated = cv2.morphologyEx(saturated, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    controls = []
    contours, _ = cv2.findContours(saturated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < 45 or w < 7 or h < 7 or w > 180 or h > 90:
            continue
        if w / max(h, 1) > 8 or h / max(w, 1) > 5:
            continue
        controls.append((x, y, w, h, x + w / 2, y + h / 2))

    # Run OCR on both contrast-equalized grayscale and adaptive-thresholded
    # images. The labels are tiny and white text can disappear in a plain
    # grayscale pass, especially on blue and green fills.
    scale = 3.0
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 7)
    text_candidates = []
    normalize = lambda s: re.sub(r"[^a-z+\-]", "", s.lower().replace("−", "-").replace("–", "-"))
    labels = {normalize(name): name for name in NEEDED_BUTTONS}
    for source in (clahe, adaptive):
        enlarged = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        data = pytesseract.image_to_data(enlarged, config="--oem 3 --psm 11",
                                         output_type=pytesseract.Output.DICT, timeout=8)
        lines = {}
        for i, word in enumerate(data["text"]):
            word = word.strip()
            if not word: continue
            try: confidence = float(data["conf"][i])
            except (TypeError, ValueError): confidence = 0
            if confidence < 4: continue
            key = (data["page_num"][i], data["block_num"][i], data["par_num"][i], data["line_num"][i])
            left, top, width, height = (int(data[k][i]) for k in ("left", "top", "width", "height"))
            lines.setdefault(key, []).append((left, top, width, height, word, confidence))
        for words in lines.values():
            words.sort(key=lambda item: item[0])
            for i in range(len(words)):
                for count in (1, 2, 3):
                    span = words[i:i + count]
                    if len(span) != count: continue
                    name = labels.get(normalize("".join(x[4] for x in span)))
                    if not name: continue
                    confidence = sum(x[5] for x in span) / count
                    left = min(x[0] for x in span); top = min(x[1] for x in span)
                    right = max(x[0] + x[2] for x in span); bottom = max(x[1] + x[3] for x in span)
                    cx, cy = (left + right) / (2 * scale), (top + bottom) / (2 * scale)
                    text_candidates.append((name, confidence, cx, cy))

    for name, confidence, cx, cy in text_candidates:
        nearby = []
        for x, y, w, h, bx, by in controls:
            # Distance to the colored body, zero when the label overlaps it.
            dx = max(x - cx, 0, cx - (x + w))
            dy = max(y - cy, 0, cy - (y + h))
            distance = math.hypot(dx, dy)
            if distance <= 55:
                nearby.append((distance, bx, by))
        if nearby:
            _, bx, by = min(nearby)
            point = [monitor["left"] + bx, monitor["top"] + by]
        else:
            point = [monitor["left"] + cx, monitor["top"] + cy]
        # Prefer the color-crop result (confidence 100) when available.
        if name not in found or confidence > found[name][1]: found[name] = (point, confidence)
    return monitor, image, {name: value[0] for name, value in found.items()}


class ButtonRecognitionWorker(QtCore.QThread):
    finished_result = QtCore.Signal(bool, object)
    def __init__(self, capture, parent=None):
        super().__init__(parent); self.capture = capture
    def run(self):
        try: self.finished_result.emit(True, detect_panel_buttons(self.capture))
        except Exception as exc: self.finished_result.emit(False, str(exc))


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
        self.step_reader = StepReader(self.shared); self.step_reader.start()
        self.capture = None
        self.renderer, self.render_note = make_renderer(bool(self.settings.get("gpu", True)))
        self.robot_mesh, self.ghost_mesh, self.cam = RobotMesh(), RobotMesh(), Camera()
        self.q_vis = np.radians([0, 35, 25, 0, 40, 0]); self.q_target = self.q_vis.copy()
        self.live_t = self.ghost_t = None; self.ghost = None; self.trail = []; self.dirty = True
        self.follow_tool = False; self.show_trail = True
        self.guided_plan = []; self.guided_index = 0; self.guided_before = None
        self.portal_signals = PortalSignals(); self.portal_signals.ready.connect(self._on_input_ready)
        self.wayland_pointer = None; self.automated_run = False
        self.auto_next_allowed = 0.0; self.auto_verify_deadline = 0.0
        self.update_worker = None
        self.image_label = None; self.last_tick = clock(); self.log_entries = []
        self.fps = 0.0; self._fps_count = 0; self._fps_time = clock()
        self.gpu_monitor = GpuUsageMonitor(); self.gpu_monitor.start()
        self.gpu_usage = None
        self._build()
        self._refresh()
        self.window.closeEvent = self._close_event
        self.window.show()
        QtCore.QTimer.singleShot(1200, lambda: self._check_updates(manual=False))
        self.stop_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape), self.window)
        self.stop_shortcut.activated.connect(self._stop_guided_move)
        self.timer = QtCore.QTimer(self.window); self.timer.timeout.connect(self._tick); self.timer.start(33)
        if not self.cfg.get("region"):
            QtCore.QTimer.singleShot(500, lambda: self.run_setup(ALL_KEYS))
        self.log(f"R08 live twin started. 3D: {self.renderer.name}.")
        if self.render_note: self.log(self.render_note)
        self.log("Native Wayland UI: manual guidance is available; optional pointer automation uses the KDE RemoteDesktop portal.")

    def _build(self):
        central = QtWidgets.QWidget(); self.window.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central); root.setContentsMargins(10, 8, 10, 10)
        header = QtWidgets.QHBoxLayout(); root.addLayout(header)
        brand = QtWidgets.QLabel("R08  live twin  ·  FANUC R-2000iA/200F"); brand.setStyleSheet(f"color:{C.yellow};font-size:16px;font-weight:700")
        header.addWidget(brand); header.addStretch()
        self.read_state = QtWidgets.QLabel("Waiting for setup"); header.addWidget(self.read_state)
        setup = QtWidgets.QPushButton("Setup ▾"); setup.clicked.connect(self._setup_menu); header.addWidget(setup)
        helpb = QtWidgets.QPushButton("Help"); helpb.clicked.connect(self._help); header.addWidget(helpb)
        settingsb = QtWidgets.QPushButton("Settings"); settingsb.clicked.connect(self._settings); header.addWidget(settingsb)
        self.update_button = QtWidgets.QPushButton("Check GitHub updates"); self.update_button.clicked.connect(self._update_button_clicked); header.addWidget(self.update_button)
        reloadb = QtWidgets.QPushButton("Reload scripts (dev)")
        reloadb.setToolTip("Save settings and restart R08 to load the current Python files")
        reloadb.clicked.connect(lambda: self._reload_application()); header.addWidget(reloadb)
        split = QtWidgets.QSplitter(); root.addWidget(split, 1)
        left = QtWidgets.QWidget(); ll = QtWidgets.QVBoxLayout(left); ll.setContentsMargins(0, 0, 4, 0)
        self.image_label = SceneLabel("Open Setup to select the number table")
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(460, 360); self.image_label.setStyleSheet("background:#0f1115;border:1px solid #303641")
        self.image_label.dragged.connect(self._camera_drag); self.image_label.wheeled.connect(self._camera_zoom)
        ll.addWidget(self.image_label, 1)
        views = QtWidgets.QHBoxLayout(); ll.insertLayout(0, views)
        for name in ("iso", "front", "side", "top"):
            b = QtWidgets.QPushButton(name.title()); b.clicked.connect(lambda checked=False, n=name: self._camera_preset(n)); views.addWidget(b)
        reset_view = QtWidgets.QPushButton("Reset view"); reset_view.clicked.connect(lambda: self._camera_preset("reset")); views.addWidget(reset_view)
        follow = QtWidgets.QCheckBox("Follow tool"); follow.toggled.connect(self._set_follow); views.addWidget(follow)
        trail = QtWidgets.QCheckBox("Trail"); trail.setChecked(True); trail.toggled.connect(self._set_trail); views.addWidget(trail)
        self.log_view = QtWidgets.QTextEdit(); self.log_view.setReadOnly(True); self.log_view.setMaximumHeight(110); ll.addWidget(self.log_view)
        split.addWidget(left)
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setMinimumWidth(370); scroll.setMaximumWidth(470)
        side = QtWidgets.QWidget(); sl = QtWidgets.QVBoxLayout(side); sl.setContentsMargins(4, 2, 4, 2)
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
        self.control_label = QtWidgets.QLabel("Guided panel moves are available in manual-confirm mode. The app cannot click the robot panel through Wayland."); self.control_label.setWordWrap(True); form.addWidget(self.control_label)
        self.button_state = QtWidgets.QLabel("Panel buttons not set up"); form.addWidget(self.button_state)
        setupb = QtWidgets.QPushButton("Set up panel buttons"); setupb.clicked.connect(lambda: self.run_setup([k for k in ALL_KEYS if k != "region"]))
        form.addWidget(setupb); sl.addWidget(status)
        self.recognize_button = QtWidgets.QPushButton("Recognize panel buttons automatically…"); self.recognize_button.clicked.connect(self._recognize_buttons); form.addWidget(self.recognize_button)
        movement = self._group("Movement options"); ml = QtWidgets.QVBoxLayout(movement)
        self.step_combo = QtWidgets.QComboBox()
        for pos, ang in STEP_LEVELS: self.step_combo.addItem(f"{pos:g} in / {ang:g}°")
        self.step_combo.setCurrentIndex(int(np.clip(self.cfg.get("level", START_LEVEL), 0, len(STEP_LEVELS)-1)))
        ml.addWidget(QtWidgets.QLabel("Jog step size")); ml.addWidget(self.step_combo)
        self.step_read_label = QtWidgets.QLabel("Continuous step OCR starts when the step-size text region is configured. Install Tesseract for OCR.")
        self.step_read_label.setWordWrap(True); ml.addWidget(self.step_read_label)
        step_setup = QtWidgets.QPushButton("Set up step-size reading region"); step_setup.clicked.connect(lambda: self.run_setup(["step_label"])); ml.addWidget(step_setup)
        jog = QtWidgets.QGridLayout(); ml.addLayout(jog)
        for axis in range(6):
            label = f"{AXES[axis]} ({UNITS[axis]})"
            jog.addWidget(QtWidgets.QLabel(label), axis // 3 * 3, axis % 3)
            for offset, sign in enumerate((1, -1)):
                b = QtWidgets.QPushButton("+" if sign > 0 else "−")
                b.clicked.connect(lambda checked=False, a=axis, s=sign: self._jog_target(a, s))
                jog.addWidget(b, axis // 3 * 3 + offset + 1, axis % 3)
        self.target_fields = []
        grid = QtWidgets.QGridLayout(); ml.addLayout(grid)
        for i, axis in enumerate(AXES):
            grid.addWidget(QtWidgets.QLabel(f"{axis}"), i // 3 * 2, i % 3)
            field = QtWidgets.QLineEdit(); field.setPlaceholderText("current" if i < 3 else "current")
            field.textChanged.connect(self._preview_target); grid.addWidget(field, i // 3 * 2 + 1, i % 3)
            self.target_fields.append(field)
        self.relative_box = QtWidgets.QCheckBox("Values are changes from current pose")
        ml.addWidget(self.relative_box)
        row = QtWidgets.QHBoxLayout(); ml.addLayout(row)
        fill = QtWidgets.QPushButton("Fill current"); fill.clicked.connect(self._fill_target); row.addWidget(fill)
        clear = QtWidgets.QPushButton("Clear"); clear.clicked.connect(lambda: [f.clear() for f in self.target_fields]); row.addWidget(clear)
        plan = QtWidgets.QPushButton("Plan movement"); plan.clicked.connect(self._plan_movement); ml.addWidget(plan)
        guide_row = QtWidgets.QHBoxLayout(); ml.addLayout(guide_row)
        self.guide_start = QtWidgets.QPushButton("Start guided move"); self.guide_start.clicked.connect(self._start_guided_move); guide_row.addWidget(self.guide_start)
        self.guide_next = QtWidgets.QPushButton("Confirm panel action"); self.guide_next.clicked.connect(self._confirm_guided_action); self.guide_next.setEnabled(False); guide_row.addWidget(self.guide_next)
        self.guide_stop = QtWidgets.QPushButton("Stop guide"); self.guide_stop.clicked.connect(self._stop_guided_move); self.guide_stop.setEnabled(False); guide_row.addWidget(self.guide_stop)
        auto_row = QtWidgets.QHBoxLayout(); ml.addLayout(auto_row)
        self.auto_input_check = QtWidgets.QCheckBox("Automated input (Wayland portal)")
        self.auto_input_check.setChecked(bool(self.cfg.get("auto_input", False)))
        self.auto_input_check.toggled.connect(self._auto_input_changed); auto_row.addWidget(self.auto_input_check)
        self.click_interval = QtWidgets.QDoubleSpinBox(); self.click_interval.setRange(1.0, 10.0); self.click_interval.setSingleStep(0.5); self.click_interval.setSuffix(" s between clicks")
        self.click_interval.setValue(float(self.cfg.get("click_interval", 2.0))); self.click_interval.valueChanged.connect(self._save_click_interval); auto_row.addWidget(self.click_interval)
        self.plan_label = QtWidgets.QLabel("Jog buttons adjust the target pose. Movement plans list the panel steps for you to perform.")
        self.plan_label.setWordWrap(True); ml.addWidget(self.plan_label)
        self.waypoints = list(self.cfg.get("waypoints", []) or [])
        self.objects = list(self.cfg.get("objects", []) or []); self.teach = None; self.selected_object = -1
        wp_row = QtWidgets.QHBoxLayout(); ml.addLayout(wp_row)
        self.waypoint_combo = QtWidgets.QComboBox(); self._refresh_waypoints(); wp_row.addWidget(self.waypoint_combo, 1)
        save = QtWidgets.QPushButton("Save current"); save.clicked.connect(self._save_waypoint); wp_row.addWidget(save)
        wp_actions = QtWidgets.QHBoxLayout(); ml.addLayout(wp_actions)
        to_wp = QtWidgets.QPushButton("Load pose"); to_wp.clicked.connect(self._load_waypoint); wp_actions.addWidget(to_wp)
        del_wp = QtWidgets.QPushButton("Delete pose"); del_wp.clicked.connect(self._delete_waypoint); wp_actions.addWidget(del_wp)
        sl.addWidget(movement)
        objects = self._group("Map objects"); ol = QtWidgets.QVBoxLayout(objects)
        objrow = QtWidgets.QHBoxLayout(); ol.addLayout(objrow)
        for title, kind in (("New plate", "plate"), ("New hollow pipe", "pipe")):
            b = QtWidgets.QPushButton(title); b.clicked.connect(lambda checked=False, k=kind: self._start_teach(k)); objrow.addWidget(b)
        points_plate = QtWidgets.QPushButton("Plate from points…"); points_plate.clicked.connect(self._add_plate_from_points); ol.addWidget(points_plate)
        self.teach_label = QtWidgets.QLabel("Map surfaces by positioning the tool tip and recording pose points."); self.teach_label.setWordWrap(True); ol.addWidget(self.teach_label)
        pointrow = QtWidgets.QHBoxLayout(); ol.addLayout(pointrow)
        add_point = QtWidgets.QPushButton("Set point"); add_point.clicked.connect(self._set_point); pointrow.addWidget(add_point)
        undo = QtWidgets.QPushButton("Undo point"); undo.clicked.connect(self._undo_point); pointrow.addWidget(undo)
        cancel = QtWidgets.QPushButton("Cancel map"); cancel.clicked.connect(self._cancel_teach); pointrow.addWidget(cancel)
        self.object_list = QtWidgets.QListWidget(); self.object_list.currentRowChanged.connect(self._select_object); ol.addWidget(self.object_list)
        objactions = QtWidgets.QHBoxLayout(); ol.addLayout(objactions)
        delete_obj = QtWidgets.QPushButton("Delete"); delete_obj.clicked.connect(self._delete_object); objactions.addWidget(delete_obj)
        edit_obj = QtWidgets.QPushButton("Edit size"); edit_obj.clicked.connect(self._edit_object); objactions.addWidget(edit_obj)
        for idx in range(3):
            b = QtWidgets.QPushButton(f"Point {idx+1} to target"); b.clicked.connect(lambda checked=False, n=idx: self._object_point_to_target(n)); ol.addWidget(b)
        self.show_objects = QtWidgets.QCheckBox("Show mapped objects"); self.show_objects.setChecked(True); self.show_objects.toggled.connect(self._mark_dirty); ol.addWidget(self.show_objects)
        self.show_gizmo = QtWidgets.QCheckBox("Show tool axis gizmo"); self.show_gizmo.setChecked(True); self.show_gizmo.toggled.connect(self._mark_dirty); ol.addWidget(self.show_gizmo)
        self._refresh_objects(); sl.addWidget(objects)
        digit = self._group("Digit reader"); dl = QtWidgets.QVBoxLayout(digit)
        self.raw_label = QtWidgets.QLabel("Waiting for the number table…"); self.raw_label.setWordWrap(True); dl.addWidget(self.raw_label)
        teach = QtWidgets.QPushButton("Teach digits…"); teach.clicked.connect(self._teach_digits); dl.addWidget(teach); sl.addWidget(digit)
        debug = QtWidgets.QPushButton("Save OCR snapshot"); debug.clicked.connect(self._save_ocr_snapshot); dl.addWidget(debug)
        sl.addStretch(); scroll.setWidget(side); split.addWidget(scroll); split.setStretchFactor(0, 3); split.setStretchFactor(1, 1)
        self.window.menuBar().setNativeMenuBar(True)

    def _mark_dirty(self, *_): self.dirty = True

    def _camera_preset(self, name):
        if name == "reset": self.cam.reset()
        else: self.cam.preset(name)
        self.dirty = True

    def _camera_drag(self, dx, dy, buttons):
        if buttons & QtCore.Qt.MouseButton.RightButton:
            self.cam.orbit(dx, dy)
        else:
            self.cam.pan(dx, dy, max(1, self.image_label.height()))
        self.dirty = True

    def _camera_zoom(self, notches):
        self.cam.zoom(notches); self.dirty = True

    def _set_follow(self, value): self.follow_tool = bool(value)
    def _set_trail(self, value): self.show_trail = bool(value); self.dirty = True

    def _auto_input_changed(self, enabled):
        self.cfg.set("auto_input", bool(enabled)); self.cfg.save()
        self.control_label.setText("Automated input is enabled as an option; each run asks you to confirm and KDE must grant pointer access." if enabled else "Guided panel moves are available in manual-confirm mode. The app cannot click the robot panel through Wayland unless automated input is enabled and granted.")

    def _save_click_interval(self, value):
        self.cfg.set("click_interval", float(value)); self.cfg.save()

    def _settings(self):
        dlg = QtWidgets.QDialog(self.window); dlg.setWindowTitle("Settings"); form = QtWidgets.QFormLayout(dlg)
        controls = {}
        for key, label, low, high, step in SETTING_DEFS:
            spin = QtWidgets.QDoubleSpinBox(); spin.setRange(low, high); spin.setSingleStep(step); spin.setDecimals(3)
            spin.setValue(float(self.settings.get(key, low))); controls[key] = spin; form.addRow(label, spin)
        toggles = {}
        for key, label in (("verify_each", "Verify pose after each action"), ("step_plus_larger", "Step+ increases the step size"),
                           ("self_learn", "Learn from confident digit reads"), ("gpu", "Use GPU renderer (restart to apply)")):
            check = QtWidgets.QCheckBox(label); check.setChecked(bool(self.settings.get(key, True))); toggles[key] = check; form.addRow(check)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept); buttons.rejected.connect(dlg.reject); form.addRow(buttons)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            for key, spin in controls.items(): self.settings[key] = spin.value()
            for key, check in toggles.items(): self.settings[key] = check.isChecked()
            self.cfg.set("settings", self.settings); self.cfg.save()

    def _save_ocr_snapshot(self):
        if not self.cfg.get("region"):
            self.log("Set up the number table first."); return
        try:
            if self.capture is None: self.capture = new_mss()
            os.makedirs(DEBUG_DIR, exist_ok=True)
            bgra = np.asarray(self.capture.grab(self.cfg.get("region")))
            gray = cv2.cvtColor(bgra, cv2.COLOR_BGRA2GRAY)
            cv2.imwrite(os.path.join(DEBUG_DIR, "region.png"), bgra)
            report = []
            for i, cell in enumerate(split_cells(gray)):
                cv2.imwrite(os.path.join(DEBUG_DIR, f"cell{i}_{AXES[i]}.png"), cell)
                result = digits.read_cell(cell, i // 3, self.bank)
                report.append(f"{AXES[i]:>2}: {result.text} (score {result.score:.2f})" if result else f"{AXES[i]:>2}: NOT READ")
            with open(os.path.join(DEBUG_DIR, "report.txt"), "w", encoding="utf-8") as out: out.write("\n".join(report) + "\n")
            self.log(f"OCR snapshot saved in {DEBUG_DIR}")
        except Exception as exc: self.log(f"OCR snapshot failed: {exc}")

    def _refresh_objects(self):
        self.object_list.clear()
        for obj in self.objects: self.object_list.addItem(obj_summary(obj))

    def _add_plate_from_points(self):
        prompt = ("Enter exactly three points, one per line, as X Y Z in inches.\n"
                  "Optional numbered format is accepted. Example:\n"
                  "1. Point 1: X +59.0  Y +51.1  Z -28.2 in\n"
                  "2. Point 2: X +57.2  Y -52.1  Z -27.3 in\n"
                  "3. Point 3: X +38.9  Y -46.3  Z -27.1 in")
        text, ok = QtWidgets.QInputDialog.getMultiLineText(self.window, "Plate from points", prompt)
        if not ok: return
        import re
        pts = []
        for line in text.splitlines():
            if not line.strip(): continue
            if line.strip().lower().startswith("mapped "): continue
            # Accept either three bare numbers or the labeled panel read-out format.
            labeled = re.search(r"X\s*([+-]?\d+(?:\.\d+)?)\s+Y\s*([+-]?\d+(?:\.\d+)?)\s+Z\s*([+-]?\d+(?:\.\d+)?)", line, re.I)
            if labeled:
                pts.append([float(v) for v in labeled.groups()])
                continue
            nums = re.findall(r"[+-]?\d+(?:\.\d+)?", line)
            # Strip a leading point index such as '1.' or 'Point 1'.
            if len(nums) >= 4 and re.match(r"\s*\d+\s*[.)]", line): nums = nums[1:]
            if len(nums) != 3:
                QtWidgets.QMessageBox.warning(self.window, "Plate from points", f"Could not read this line:\n{line}\n\nUse X Y Z values, one point per line.")
                return
            pts.append([float(v) for v in nums])
        if len(pts) != 3:
            QtWidgets.QMessageBox.warning(self.window, "Plate from points", f"Enter exactly three points; found {len(pts)}.")
            return
        trial = {"type": "plate", "name": "Plate", "pts": pts, "t": 0.5}
        from .model3d import plate_dims
        dims = plate_dims(trial)
        if dims is None:
            QtWidgets.QMessageBox.warning(self.window, "Plate from points", "The three points must be distinct and non-collinear.")
            return
        name, ok = QtWidgets.QInputDialog.getText(self.window, "Plate name", "Name:", text=f"Plate {1 + sum(o.get('type') == 'plate' for o in self.objects)}")
        if not ok or not name.strip(): return
        thickness, ok = QtWidgets.QInputDialog.getDouble(self.window, "Plate thickness", "Thickness extending upward from the mapped face (in):", 0.5, 0.01, 24.0, 2)
        if not ok: return
        trial.update(name=name.strip(), t=thickness)
        self.objects.append(trial); self.cfg.set("objects", self.objects); self.cfg.save()
        self._refresh_objects(); self.object_list.setCurrentRow(len(self.objects) - 1); self.dirty = True
        self.teach_label.setText(f"Mapped {obj_summary(trial)}")
        self.log(f"Mapped {obj_summary(trial)}")

    def _start_teach(self, kind):
        self.teach = {"type": kind, "pts": []}
        self.teach_label.setText(f"New {kind}: record {3 if kind == 'plate' else 2} tool tip points, then size the object.")

    def _cancel_teach(self):
        self.teach = None; self.teach_label.setText("Mapping cancelled.")

    def _undo_point(self):
        if self.teach and self.teach["pts"]: self.teach["pts"].pop()
        if self.teach: self.teach_label.setText(f"{len(self.teach['pts'])} point(s) recorded.")

    def _set_point(self):
        if not self.teach:
            self.log("Choose New plate or New pipe first."); return
        pose = self._current_pose()
        if pose is None:
            self.log("No readable pose yet."); return
        self.teach["pts"].append(pose[:3].tolist())
        required = 3 if self.teach["type"] == "plate" else 2
        if len(self.teach["pts"]) < required:
            self.teach_label.setText(f"Point {len(self.teach['pts'])} recorded. Position the tip at the next point."); return
        kind = self.teach["type"]
        if kind == "plate":
            value, ok = QtWidgets.QInputDialog.getDouble(self.window, "Plate thickness", "Thickness extending upward from the mapped face (in):", 0.5, 0.01, 24.0, 2)
            if not ok: self.teach["pts"].pop(); return
            inner = None
        else:
            value, ok = QtWidgets.QInputDialog.getDouble(self.window, "Pipe outside diameter", "Outside diameter (in):", 4.0, 0.1, 60.0, 2)
            if not ok: self.teach["pts"].pop(); return
            inner, ok = QtWidgets.QInputDialog.getDouble(self.window, "Pipe inside diameter", "Inside diameter (in; enter 0 for a solid pipe):", min(3.0, value - 0.01), 0.0, max(0.0, value - 0.01), 2)
            if not ok: self.teach["pts"].pop(); return
        n = 1 + sum(o["type"] == kind for o in self.objects)
        item = {"type": kind, "name": f"{kind.title()} {n}", "pts": self.teach["pts"], "t" if kind == "plate" else "d": value}
        if kind == "pipe": item["id"] = inner
        self.objects.append(item); self.cfg.set("objects", self.objects); self.cfg.save()
        self.teach = None; self._refresh_objects(); self.object_list.setCurrentRow(len(self.objects)-1); self.dirty = True
        self.teach_label.setText(f"Mapped {obj_summary(item)}")

    def _select_object(self, row):
        self.selected_object = row; self.dirty = True

    def _delete_object(self):
        i = self.object_list.currentRow()
        if i < 0: return
        del self.objects[i]; self.cfg.set("objects", self.objects); self.cfg.save(); self._refresh_objects(); self.dirty = True

    def _edit_object(self):
        i = self.object_list.currentRow()
        if i < 0: return
        obj = self.objects[i]
        if obj["type"] == "plate":
            val, ok = QtWidgets.QInputDialog.getDouble(self.window, "Plate thickness", "Thickness extending upward from the mapped face (in):", float(obj.get("t", .5)), .01, 24, 2)
            if ok: obj["t"] = val
        else:
            val, ok = QtWidgets.QInputDialog.getDouble(self.window, "Pipe outside diameter", "Outside diameter (in):", float(obj.get("d", 4)), .1, 60, 2)
            if not ok: return
            inner, ok = QtWidgets.QInputDialog.getDouble(self.window, "Pipe inside diameter", "Inside diameter (in; enter 0 for a solid pipe):", float(obj.get("id", 0)), 0, max(0.0, val - .01), 2)
            if not ok: return
            obj.update(d=val, id=inner)
            self.cfg.set("objects", self.objects); self.cfg.save(); self._refresh_objects(); self.object_list.setCurrentRow(i); self.dirty = True
            return
        if ok:
            self.cfg.set("objects", self.objects); self.cfg.save(); self._refresh_objects(); self.object_list.setCurrentRow(i); self.dirty = True

    def _object_point_to_target(self, n):
        i = self.object_list.currentRow()
        if i < 0 or n >= len(self.objects[i]["pts"]):
            self.log("Select an object with that point recorded."); return
        self.relative_box.setChecked(False)
        for field, value in zip(self.target_fields[:3], self.objects[i]["pts"][n]): field.setText(f"{float(value):.1f}")


    def _group(self, title):
        box = QtWidgets.QGroupBox(title); box.setStyleSheet("QGroupBox {border:1px solid #303641;border-radius:6px;margin-top:8px;padding:8px;font-weight:600} QGroupBox::title {subcontrol-origin:margin;left:8px;padding:0 4px}")
        return box

    def _current_pose(self):
        pose = self.shared.snapshot().pose
        return None if pose is None else np.asarray(pose, dtype=float)

    def _target_values(self):
        pose = self._current_pose()
        if pose is None:
            return None, "Waiting for a readable robot pose."
        target = [None] * 6
        any_value = False
        for i, field in enumerate(self.target_fields):
            raw = field.text().strip().replace(",", ".")
            if not raw: continue
            try: value = float(raw.replace("−", "-"))
            except ValueError: return None, f"{raw!r} is not a number."
            if self.relative_box.isChecked(): value += pose[i]
            target[i] = wrap180(value) if i >= 3 else value
            any_value = True
        if not any_value: return None, "Enter at least one target value."
        return target, ""

    def _fill_target(self):
        pose = self._current_pose()
        if pose is None:
            self.log("Waiting for a readable robot pose."); return
        self.relative_box.setChecked(False)
        for i, field in enumerate(self.target_fields):
            field.setText(f"{pose[i]:.1f}")

    def _jog_target(self, axis, sign):
        pose = self._current_pose()
        if pose is None:
            self.log("Waiting for a readable robot pose."); return
        field = self.target_fields[axis]
        try: value = float(field.text().strip().replace(",", ".").replace("−", "-"))
        except ValueError: value = float(pose[axis])
        if self.relative_box.isChecked(): value += pose[axis]
        value += sign * step_value(self.step_combo.currentIndex(), axis)
        if axis >= 3: value = wrap180(value)
        self.relative_box.setChecked(False)
        field.setText(f"{value:.1f}")

    def _preview_target(self, *_):
        target, _ = self._target_values()
        pose = self._current_pose()
        if target is None or pose is None: return
        full = np.array([pose[i] if value is None else value for i, value in enumerate(target)], float)
        self.ik.request("ghost", full)
        self.dirty = True

    def _plan_movement(self):
        target, error = self._target_values()
        pose = self._current_pose()
        if target is None:
            self.log(error); return
        plan = plan_presses(pose, target, int(self.step_combo.currentIndex()), self.settings["pos_tol"],
                            self.settings["ang_tol"], self.settings["step_plus_larger"])
        self.guided_plan = plan
        if not plan:
            self.plan_label.setText("Target is already within the configured tolerance.")
        else:
            self.plan_label.setText(f"Movement plan ready: {len(plan)} panel actions. Start the guide to follow them one at a time.")
        self.guide_start.setEnabled(bool(plan) and not self.guided_next.isEnabled())
        self.log(f"Movement plan prepared: {len(plan)} panel actions.")

    def _start_guided_move(self):
        if not self.guided_plan:
            self._plan_movement()
        if not self.guided_plan: return
        if self._current_pose() is None:
            self.log("Cannot start the guide without a readable pose."); return
        if self.auto_input_check.isChecked():
            self._start_automated_move(); return
        self.guided_index = 0; self.guided_before = self._current_pose().copy()
        self.guide_start.setEnabled(False); self.guide_next.setEnabled(True); self.guide_stop.setEnabled(True)
        self._show_guided_action()

    def _show_guided_action(self):
        action = self.guided_plan[self.guided_index]
        name = action.btn.replace("-", "−")
        suffix = f"  ·  {action.label}"
        pos = (self.cfg.get("buttons", {}) or {}).get(action.btn)
        location = f" (setup screen position {int(pos[0])}, {int(pos[1])})" if pos and len(pos) >= 2 else ""
        self.plan_label.setText(f"Action {self.guided_index + 1} of {len(self.guided_plan)}: press {name}{location} on the robot panel{suffix}. Then click ‘Confirm panel action’. The app does not press it for you.")

    def _confirm_guided_action(self):
        if not self.guided_plan or self.guided_index >= len(self.guided_plan): return
        action = self.guided_plan[self.guided_index]
        pose = self._current_pose()
        verified = self._guided_action_verified(action, pose)
        if not verified:
            answer = QtWidgets.QMessageBox.question(
                self.window, "Confirm panel action",
                "The screen reader cannot verify this action yet. Confirm that you pressed the requested panel button and the robot/panel responded.",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No)
            if answer != QtWidgets.QMessageBox.StandardButton.Yes: return
        self.guided_index += 1
        if self.guided_index >= len(self.guided_plan):
            self.guide_next.setEnabled(False); self.guide_stop.setEnabled(False); self.guide_start.setEnabled(False)
            self.plan_label.setText("Guided sequence complete. Compare the displayed pose with your intended target.")
            self.log("Guided panel sequence completed."); return
        self.guided_before = None if pose is None else pose.copy()
        self._show_guided_action()

    def _guided_action_verified(self, action, pose):
        if action.kind == "jog" and pose is not None and self.guided_before is not None:
            delta = float(pose[action.axis] - self.guided_before[action.axis])
            if action.axis >= 3: delta = wrap180(delta)
            resolution = 0.1 if action.axis < 3 else 1.0
            return delta * action.sign > 0 and abs(delta) >= resolution * 0.75
        if action.kind == "step":
            snap = self.shared.snapshot()
            return snap.step_level is not None and snap.step_level == action.level
        return False

    def _start_automated_move(self):
        buttons = self.cfg.get("buttons", {}) or {}
        missing = sorted({p.btn for p in self.guided_plan if p.btn not in buttons})
        if missing:
            QtWidgets.QMessageBox.warning(self.window, "Automated input", "Set up these panel button positions first: " + ", ".join(missing))
            return
        window_rect = self.window.frameGeometry()
        covered = [name for name in {p.btn for p in self.guided_plan}
                   if window_rect.contains(QtCore.QPoint(round(buttons[name][0]), round(buttons[name][1])))]
        if covered:
            QtWidgets.QMessageBox.warning(self.window, "Automated input", "Move the R08 window away from these panel buttons before starting: " + ", ".join(sorted(covered)))
            return
        answer = QtWidgets.QMessageBox.warning(
            self.window, "Automated robot panel input",
            f"R08 will move the pointer and click the calibrated robot panel buttons, every {self.click_interval.value():g} seconds. "
            "This can move the robot. Make sure the work area is clear and the robot is safe to move. Keep this window visible so you can stop the sequence. KDE will ask you to grant pointer access.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel)
        if answer != QtWidgets.QMessageBox.StandardButton.Yes: return
        try:
            if self.capture is None: self.capture = new_mss()
            monitor = self.capture.monitors[0]
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self.window, "Automated input", str(exc)); return
        self.guided_index = 0; self.guided_before = self._current_pose().copy()
        self.automated_run = True
        self.guide_start.setEnabled(False); self.guide_next.setEnabled(False); self.guide_stop.setEnabled(True)
        self.plan_label.setText("Requesting KDE RemoteDesktop pointer permission… Approve the portal prompt to continue.")
        self.log("Requesting KDE RemoteDesktop pointer permission for automated panel clicks.")
        self.wayland_pointer = WaylandPointer(monitor, lambda ok, message: self.portal_signals.ready.emit(ok, message))

    def _on_input_ready(self, ok, message):
        if not self.automated_run:
            if self.wayland_pointer: self.wayland_pointer.close()
            self.wayland_pointer = None; return
        if not ok:
            self.automated_run = False; self.wayland_pointer = None
            self.guide_start.setEnabled(True); self.guide_next.setEnabled(False); self.guide_stop.setEnabled(False)
            self.plan_label.setText("Automated input could not start. Use manual guidance or check KDE portal support.")
            QtWidgets.QMessageBox.warning(self.window, "Automated input unavailable", message)
            self.log(f"Automated input unavailable: {message}"); return
        self.log("KDE granted pointer access. Starting the calibrated panel sequence.")
        self._auto_click_current()

    def _auto_click_current(self):
        if not self.automated_run or not self.wayland_pointer: return
        if self.guided_index >= len(self.guided_plan):
            self._finish_automated_move(); return
        action = self.guided_plan[self.guided_index]
        pos = (self.cfg.get("buttons", {}) or {}).get(action.btn)
        if not pos or len(pos) < 2:
            self._stop_automated_on_error(f"No calibrated position for {action.btn}."); return
        try:
            self.plan_label.setText(f"Automated action {self.guided_index + 1}/{len(self.guided_plan)}: clicking {action.btn.replace('-', '−')} — {action.label}")
            self.wayland_pointer.click(float(pos[0]), float(pos[1]))
            now = clock()
            self.auto_next_allowed = now + self.click_interval.value()
            self.auto_verify_deadline = now + max(self.click_interval.value(), self.settings["motion_wait"] + self.settings["settle"])
            QtCore.QTimer.singleShot(150, self._poll_auto_action)
        except Exception as exc:
            self._stop_automated_on_error(str(exc))

    def _poll_auto_action(self):
        if not self.automated_run or self.guided_index >= len(self.guided_plan): return
        action = self.guided_plan[self.guided_index]
        pose = self._current_pose()
        if not self._guided_action_verified(action, pose):
            if clock() < self.auto_verify_deadline:
                QtCore.QTimer.singleShot(150, self._poll_auto_action)
                return
            self.automated_run = False
            if self.wayland_pointer: self.wayland_pointer.close()
            self.wayland_pointer = None
            self.guide_next.setEnabled(True); self.guide_stop.setEnabled(True)
            self.plan_label.setText("Automatic sequence paused: the screen reader did not verify this action. Check the robot and reading, then use manual confirm or stop.")
            self.log(f"Automatic sequence paused after {action.btn}: no expected read-back change was detected.")
            return
        self.guided_index += 1
        self.guided_before = None if pose is None else pose.copy()
        if self.guided_index >= len(self.guided_plan): self._finish_automated_move()
        else:
            wait_ms = max(0, round((self.auto_next_allowed - clock()) * 1000))
            QtCore.QTimer.singleShot(wait_ms, self._auto_click_current)

    def _finish_automated_move(self):
        self.automated_run = False
        if self.wayland_pointer: self.wayland_pointer.close()
        self.wayland_pointer = None
        self.guide_next.setEnabled(False); self.guide_stop.setEnabled(False); self.guide_start.setEnabled(False)
        self.plan_label.setText("Automated sequence complete. Compare the displayed pose with your target.")
        self.log("Automated panel sequence completed and verified.")

    def _stop_automated_on_error(self, message):
        self.automated_run = False
        if self.wayland_pointer: self.wayland_pointer.close()
        self.wayland_pointer = None
        self.guide_next.setEnabled(True); self.guide_stop.setEnabled(True)
        self.plan_label.setText("Automated sequence stopped: " + message)
        self.log("Automated sequence stopped: " + message)

    def _stop_guided_move(self):
        self.automated_run = False
        if self.wayland_pointer: self.wayland_pointer.close()
        self.wayland_pointer = None
        self.guided_plan = []; self.guided_index = 0; self.guided_before = None
        self.guide_next.setEnabled(False); self.guide_stop.setEnabled(False); self.guide_start.setEnabled(False)
        self.plan_label.setText("Guided movement stopped."); self.log("Guided movement stopped.")

    def _refresh_waypoints(self):
        self.waypoint_combo.clear()
        for item in self.waypoints:
            self.waypoint_combo.addItem(item.get("name", "Saved pose"))

    def _save_waypoint(self):
        pose = self._current_pose()
        if pose is None:
            self.log("No readable pose to save yet."); return
        name, ok = QtWidgets.QInputDialog.getText(self.window, "Save pose", "Name for this pose:")
        if not ok or not name.strip(): return
        self.waypoints.append({"name": name.strip(), "pose": pose.tolist()})
        self.cfg.set("waypoints", self.waypoints); self.cfg.save(); self._refresh_waypoints()
        self.waypoint_combo.setCurrentIndex(len(self.waypoints) - 1)

    def _load_waypoint(self):
        index = self.waypoint_combo.currentIndex()
        if index < 0: return
        self.relative_box.setChecked(False)
        for field, value in zip(self.target_fields, self.waypoints[index]["pose"]): field.setText(f"{float(value):.1f}")

    def _delete_waypoint(self):
        index = self.waypoint_combo.currentIndex()
        if index < 0: return
        del self.waypoints[index]
        self.cfg.set("waypoints", self.waypoints); self.cfg.save(); self._refresh_waypoints()

    def _setup_menu(self):
        menu = QtWidgets.QMenu(self.window)
        menu.addAction("Run full setup", lambda: self.run_setup(ALL_KEYS))
        menu.addAction("Number table", lambda: self.run_setup(["region"]))
        menu.addAction("Panel buttons", lambda: self.run_setup(BUTTON_KEYS))
        menu.addAction("Recognize panel buttons automatically…", self._recognize_buttons)
        menu.addAction("Step-size text region", lambda: self.run_setup(["step_label"]))
        menu.addAction("Optional banner and step text", lambda: self.run_setup(["status", "step_label"]))
        menu.exec(QtGui.QCursor.pos())

    def _recognize_buttons(self):
        if not HAVE_TESS or pytesseract is None:
            QtWidgets.QMessageBox.warning(self.window, "Button recognition", "Button recognition needs pytesseract and the Tesseract executable. Install them, then retry.")
            return
        try:
            if self.capture is None: self.capture = new_mss()
            if getattr(self, "button_recognition_worker", None) and self.button_recognition_worker.isRunning(): return
            self.recognize_button.setEnabled(False); self.recognize_button.setText("Recognizing labels…")
            self.button_recognition_worker = ButtonRecognitionWorker(self.capture, self.window)
            self.button_recognition_worker.finished_result.connect(self._button_recognition_finished)
            self.button_recognition_worker.start()
        except Exception as exc:
            self.recognize_button.setEnabled(True); self.recognize_button.setText("Recognize panel buttons automatically…")
            QtWidgets.QMessageBox.warning(self.window, "Button recognition failed", str(exc))

    def _button_recognition_finished(self, ok, result):
        self.recognize_button.setEnabled(True); self.recognize_button.setText("Recognize panel buttons automatically…")
        if not ok:
            QtWidgets.QMessageBox.warning(self.window, "Button recognition failed", str(result)); return
        monitor, image, detected = result
        if not detected:
            QtWidgets.QMessageBox.information(self.window, "Button recognition", "No panel button labels were detected. Move the R08 window away, ensure the panel is visible, or use manual panel setup.")
            return
        dlg = ButtonReviewDialog(self.window, monitor, image, detected, self.cfg.get("buttons", {}) or {})
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted or not dlg.buttons: return
        self.cfg.set("buttons", dlg.buttons); self.cfg.save(); self._refresh()
        self.log(f"Saved recognized panel button positions ({len(detected)} labels detected; review confirmed).")

    def _update_button_clicked(self):
        if getattr(self, "available_update", None):
            self._apply_available_update()
        else:
            self._check_updates(manual=True)

    def _check_updates(self, manual=True):
        if self.update_worker and self.update_worker.isRunning(): return
        self.update_button.setEnabled(False); self.update_button.setText("Checking GitHub…")
        self.update_check_manual = bool(manual)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.update_worker = UpdateWorker(root, parent=self.window)
        self.update_worker.finished_result.connect(self._update_check_finished)
        self.update_worker.start()

    def _update_check_finished(self, ok, result):
        self.update_button.setEnabled(True)
        if not ok:
            self.available_update = None
            self.update_button.setText("Retry update check")
            self.update_button.setToolTip(str(result))
            if self.update_check_manual: QtWidgets.QMessageBox.warning(self.window, "GitHub updater", str(result))
            return
        if result["behind"] <= 0:
            self.available_update = None
            self.update_button.setText("Check GitHub updates")
            self.update_button.setToolTip("Check the GitHub main branch for updates")
            if result["ahead"]:
                text = "This checkout has local commits not on GitHub; automatic fast-forward is unavailable."
            else: text = "This checkout is up to date with GitHub."
            if self.update_check_manual: QtWidgets.QMessageBox.information(self.window, "GitHub updater", text)
            return
        self.available_update = result
        self.update_button.setText("Update available")
        self.update_button.setToolTip(f"{result['behind']} commit(s) available from GitHub. Click to review and install.")
        if self.update_check_manual: self._apply_available_update()

    def _apply_available_update(self):
        result = self.available_update
        if not result: return
        if result["ahead"]:
            QtWidgets.QMessageBox.warning(self.window, "GitHub updater", "This branch has local commits as well as GitHub commits. Automatic fast-forward is unavailable.")
            return
        if result["staged"]:
            QtWidgets.QMessageBox.warning(self.window, "GitHub updater", "Unstage or commit your staged changes before updating.")
            return
        if result["blocked"]:
            text = "A GitHub update is available, but local source edits must be committed or saved before installing:\n\n" + "\n".join(result["blocked"][:12])
            QtWidgets.QMessageBox.warning(self.window, "GitHub updater", text); return
        text = f"GitHub has {result['behind']} new commit(s) on {result['target']}. Your local config and digit bank will be preserved. Install the fast-forward update now?"
        answer = QtWidgets.QMessageBox.question(self.window, "GitHub updater", text,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No)
        if answer != QtWidgets.QMessageBox.StandardButton.Yes: return
        self.update_button.setEnabled(False); self.update_button.setText("Updating…")
        self.available_update = None
        self.update_worker = UpdateWorker(result["root"], result, self.window)
        self.update_worker.finished_result.connect(self._update_install_finished)
        self.update_worker.start()

    def _update_install_finished(self, ok, result):
        self.update_button.setEnabled(True); self.update_button.setText("Check GitHub updates")
        if not ok:
            QtWidgets.QMessageBox.warning(self.window, "GitHub updater", str(result)); return
        self.log(f"Updated source to {result[:12]}.")
        answer = QtWidgets.QMessageBox.question(self.window, "Update installed", "Restart now to load the updated code?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes)
        if answer == QtWidgets.QMessageBox.StandardButton.Yes: self._reload_application(confirm=False)

    def _reload_application(self, confirm=True):
        if self.automated_run or self.guide_next.isEnabled():
            QtWidgets.QMessageBox.warning(self.window, "Reload unavailable", "Stop the guided movement and make sure the robot is settled before reloading.")
            return
        if self.update_worker and self.update_worker.isRunning():
            QtWidgets.QMessageBox.warning(self.window, "Reload unavailable", "Wait for the GitHub update check or install to finish before reloading.")
            return
        if confirm:
            answer = QtWidgets.QMessageBox.question(self.window, "Reload development code", "Save settings and restart R08 from the current Python files?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No)
            if answer != QtWidgets.QMessageBox.StandardButton.Yes: return
        try:
            self.cfg.set("settings", self.settings); self.cfg.save(); self.bank.save()
        except Exception: pass
        if hasattr(self, "timer"): self.timer.stop()
        if self.wayland_pointer:
            try: self.wayland_pointer.close()
            except Exception: pass
            self.wayland_pointer = None
        for worker in (self.ocr, self.step_reader, self.ik):
            worker.stop()
        for worker in (self.ocr, self.step_reader, self.ik):
            worker.join(timeout=2)
        os.execv(sys.executable, [sys.executable, *sys.argv])

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
        QtWidgets.QMessageBox.information(self.window, "R08 live twin", "Set up the number table to start screen reading. Use the Setup menu to calibrate panel buttons and optional status regions.\n\nMovement planning and step-by-step manual guidance are available. Press each requested panel control yourself, then confirm it in the app; the Wayland portal does not provide global panel clicks in this build.")

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
        usage = self.gpu_monitor.latest
        if usage != self.gpu_usage:
            self.gpu_usage = usage; self.dirty = True
        snap = self.shared.snapshot()
        self.read_state.setText(f"● Reading {snap.scan_rate:.0f}/s" if snap.state == "Live" else f"○ {snap.state}")
        self.raw_label.setText(snap.raw or "Waiting for the number table…")
        if not snap.has_step:
            self.step_read_label.setText("Continuous step OCR is waiting for its region; run full setup or set the step-size text region.")
        elif snap.step_level is None:
            self.step_read_label.setText("Step size (continuous OCR): " + snap.step_status)
        else:
            age = max(0.0, now - snap.step_t)
            pos, ang = STEP_LEVELS[snap.step_level]
            freshness = "current" if age <= 2.0 else f"stale · last read {age:.1f}s ago · {snap.step_status}"
            self.step_read_label.setText(f"Step size (continuous OCR): {pos:g} in / {ang:g}°  ·  {freshness}")
            if age <= 2.0 and self.step_combo.currentIndex() != snap.step_level:
                self.step_combo.setCurrentIndex(snap.step_level)
        if snap.pose is not None:
            for i, val in enumerate(snap.pose): self.pose_labels[i].setText(fmt_pos(val) if i < 3 else fmt_ang(val))
        with self.shared.lock:
            live, ghost = self.shared.ik.get("live"), self.shared.ik.get("ghost")
        if live and live.t != self.live_t:
            self.live_t = live.t; self.q_target = live.q.copy()
        if ghost and ghost.t != self.ghost_t:
            self.ghost_t = ghost.t
            self.ghost = {"q": ghost.q.copy(), "ok": ghost.pe < 15.0 and ghost.re < 6.0, "pose": ghost.pose}
        if snap.pose is not None and (not self.trail or snap.seq != getattr(self, "last_seq", -1)):
            self.last_seq = snap.seq; self.trail.append(K.table_pose_to_target(snap.pose)[:3, 3].copy()); self.trail = self.trail[-260:]
            self.dirty = True
        if self.follow_tool and snap.pose is not None:
            tip = K.table_pose_to_target(snap.pose)[:3, 3]
            if np.linalg.norm(tip - self.cam.target) > .005:
                self.cam.target = self.cam.target + (tip - self.cam.target) * (1 - math.exp(-dt * 3)); self.dirty = True
        diff = self.q_target - self.q_vis
        if np.max(np.abs(diff)) > 2e-4:
            vmax = math.radians(float(self.settings.get("viz_deg_s", 25))) * dt
            self.q_vis += np.clip(diff * (1 - math.exp(-dt * 14)), -vmax, vmax); self.dirty = True
        if self.dirty: self._render(snap); self.dirty = False

    def _render(self, snap):
        w, h = max(400, self.image_label.width()), max(300, self.image_label.height())
        info = self.robot_mesh.update(self.q_vis)
        hud = {"backend": self.renderer.name, "fps": self.fps, "gpu": self.gpu_usage}
        if snap.pose is not None: hud["tip_text"] = "  ".join(fmt_pos(snap.pose[i]) for i in range(3)) + " in"
        else: hud["msg"] = "Open Setup to select the number table"
        ghost_tris = ghost_tip = None; ghost_ok = True; ghost_col = hex_rgb(C.green)
        hud["ghost_text"] = None
        if self.ghost:
            gi = self.ghost_mesh.update(self.ghost["q"])
            ghost_tris, ghost_tip, ghost_ok = self.ghost_mesh.triangles(), gi.tip, self.ghost["ok"]
            ghost_col = hex_rgb(C.green if ghost_ok else C.amber)
            hud["ghost_text"] = "target  " + "  ".join(f"{AXES[i]} {self.ghost['pose'][i]:+.1f}" for i in range(6))
        tris = self.robot_mesh.triangles(); labels = []
        if self.show_objects.isChecked():
            soups = []
            for i, obj in enumerate(self.objects):
                res = plate_solid(obj, i == self.selected_object) if obj["type"] == "plate" else pipe_solid(obj, i == self.selected_object)
                if res is None: continue
                soup, center = res; soups.append(soup)
                labels.append((center + np.array([0, 0, .04]), obj["name"], (200, 205, 215), False))
                if i == self.selected_object:
                    for k, point in enumerate(obj["pts"]): labels.append((K.table_to_base(point), str(k + 1), hex_rgb(C.green), True))
            if self.teach:
                for k, point in enumerate(self.teach["pts"]): labels.append((K.table_to_base(point), str(k + 1), hex_rgb(C.yellow), True))
            if soups: tris = tuple(np.concatenate([a, b]) for a, b in zip(tris, tuple(np.concatenate([s[j] for s in soups]) for j in range(3))))
        frame = SimpleNamespace(tris=tris, ghost_tris=ghost_tris, ghost_col=ghost_col, ghost_tip=ghost_tip,
                                ghost_ok=ghost_ok, tip=info.tip, trail=np.asarray(self.trail) if self.show_trail and self.trail else None,
                                labels=labels, gizmo=self.show_gizmo.isChecked(), hud=hud)
        rgb = self.renderer.render(w, h, self.cam, frame)
        self._fps_count += 1
        elapsed = clock() - self._fps_time
        if elapsed >= 1.0:
            self.fps = self._fps_count / elapsed
            self._fps_count = 0; self._fps_time = clock()
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def _close_event(self, event):
        self.automated_run = False
        if self.wayland_pointer: self.wayland_pointer.close()
        self.wayland_pointer = None
        try:
            self.cfg.set("settings", self.settings); self.cfg.save(); self.bank.save()
        except Exception: pass
        self.ocr.stop(); self.step_reader.stop(); self.ik.stop()
        self.gpu_monitor.stop(); self.gpu_monitor.join(timeout=2)
        event.accept()

    def run(self):
        return self.qt.exec()


def main():
    app = QtApp()
    return app.run()
