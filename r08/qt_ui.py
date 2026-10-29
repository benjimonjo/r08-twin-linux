"""Native Qt UI used on Linux Wayland sessions.

The Tk UI remains the cross-platform desktop implementation. Tk on Unix is
Xlib based, so Wayland sessions use this Qt Widgets frontend instead.
"""
import os
import base64
os.environ["QT_QPA_PLATFORM"] = "wayland"

import math
import time
import re
import sys
import signal
import numpy as np
import cv2
from PySide6 import QtCore, QtGui, QtWidgets

from .common import (C, AXES, UNITS, Config, CONFIG_FILE, DIGIT_BANK_FILE,
                     DEFAULT_SETTINGS, START_LEVEL, STEP_LEVELS, DEBUG_DIR, wrap180, new_mss, set_capture_source,
                     hex_rgb, clock)
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
from .planner import Press, plan_presses, step_value
from .wayland_input import WaylandPointer
from .github_updater import check_for_updates, install_update
from .gpu_usage import GpuUsageMonitor
from .object_points import parse_xyz_points
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


class WorkspaceScanWorker(QtCore.QThread):
    finished_result = QtCore.Signal(bool, object)

    def __init__(self, sample_count=20000, parent=None):
        super().__init__(parent)
        self.sample_count = sample_count

    def run(self):
        try:
            self.finished_result.emit(True, K.workspace_bounds(self.sample_count))
        except Exception as exc:
            self.finished_result.emit(False, str(exc))


class ButtonPreview(QtWidgets.QLabel):
    clicked = QtCore.Signal(float, float)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit(event.position().x(), event.position().y())


class CameraCalibrationDialog(QtWidgets.QDialog):
    def __init__(self, parent, image, points):
        super().__init__(parent)
        self.setWindowTitle("Calibrate camera projection")
        self.points = points
        self.image_points = []
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(
            "Click the visible robot landmarks in the listed order. Use at least six landmarks; "
            "the first landmark is the base, followed by J1 through J6, flange, and tool tip."))
        self.preview = ButtonPreview()
        rgb = np.ascontiguousarray(image[:, :, :3])
        h, w = rgb.shape[:2]
        self.source_width, self.source_height = w, h
        qimg = QtGui.QImage(rgb.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888).copy()
        self.pixmap = QtGui.QPixmap.fromImage(qimg).scaled(
            1000, 650, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation)
        self.preview.setPixmap(self.pixmap)
        self.preview.setFixedSize(self.pixmap.size())
        self.preview.clicked.connect(self._clicked)
        layout.addWidget(self.preview, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._refresh()

    def _clicked(self, x, y):
        if len(self.image_points) >= len(self.points):
            return
        self.image_points.append([
            x * self.source_width / self.pixmap.width(),
            y * self.source_height / self.pixmap.height(),
        ])
        self._refresh()

    def _refresh(self):
        next_name = self.points[len(self.image_points)][0] if len(self.image_points) < len(self.points) else "complete"
        self.status.setText(f"Landmark {len(self.image_points) + 1}/{len(self.points)}: {next_name}")


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


class DebugDialog(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("R08 Debug")
        self.resize(900, 600)
        layout = QtWidgets.QVBoxLayout(self)
        self.output = QtWidgets.QTextEdit()
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.output)
        clear = QtWidgets.QPushButton("Clear")
        clear.clicked.connect(self.output.clear)
        layout.addWidget(clear)


def detect_panel_buttons(capture, button_region=None):
    monitor = dict(button_region or capture.monitors[0])
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
    button_config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-"
    color_ocr_labels = {f"{axis}{sign}" for axis in AXES for sign in "+-"} | {
        "Step+", "Step-", "Camera Step+", "Camera Step-",
        "Abort", "Backup", "Safe J23", "Safe J456"}
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
            if label in color_ocr_labels:
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

    def row_groups(items):
        groups = []
        for item in sorted(items, key=lambda value: value[5]):
            for group in groups:
                if abs(item[5] - np.mean([x[5] for x in group])) <= max(item[3], 12) * 0.65:
                    group.append(item)
                    break
            else:
                groups.append([item])
        return [sorted(group, key=lambda value: value[4]) for group in groups]

    # The panel layout is regular even when OCR is poor: position and angle
    # each contain two rows of three colored buttons. Use those rows as a
    # structural fallback and reserve OCR for the irregular jog controls.
    grid_rows = [group for group in row_groups(controls) if len(group) >= 3]
    def remember(name, item):
        point = [monitor["left"] + item[4], monitor["top"] + item[5]]
        found[name] = (point, 110)

    if len(grid_rows) >= 2:
        grid_rows.sort(key=lambda group: np.mean([x[5] for x in group]))
        grid_floor = 0
        for plus_names, minus_names in (
            (("X+", "Y+", "Z+"), ("X-", "Y-", "Z-")),
            (("Xa+", "Ya+", "Za+"), ("Xa-", "Ya-", "Za-")),
        ):
            if not grid_rows:
                break
            plus = grid_rows.pop(0)
            if len(plus) < 3:
                continue
            grid_floor = max(grid_floor, max(x[5] for x in plus))
            for name, item in zip(plus_names, plus[:3]):
                remember(name, item)
                # The negative controls use the same x positions and size,
                # but are dark gray and therefore absent from the color mask.
                minus = (item[0], item[1] + item[3] + 8, item[2], item[3],
                         item[4], item[5] + item[3] + 8)
                remember(minus_names[plus_names.index(name)], minus)

    # Camera controls use the same three-column layout as the robot blocks.
    # Their lower row is usually dark gray, so infer it from the visible
    # Up/Right/Zoom+ row.
    camera_top = None
    if grid_rows:
        camera_top = grid_rows.pop(0)
        if len(camera_top) >= 3:
            for name, item in zip(("Up", "Right", "Zoom+"), camera_top[:3]):
                remember(name, item)
                minus = (item[0], item[1] + item[3] + 8, item[2], item[3],
                         item[4], item[5] + item[3] + 8)
                lower_name = {"Up": "Down", "Right": "Left", "Zoom+": "Zoom-"}[name]
                remember(lower_name, minus)
            camera_bottom_y = camera_top[0][5] + camera_top[0][3] + 8
            switch_candidates = [
                item for item in controls
                if camera_bottom_y + 20 < item[5] < camera_bottom_y + 80
            ]
            if switch_candidates:
                remember("Switch Camera", min(switch_candidates, key=lambda item: item[5]))
            # Reset Camera is a dark button aligned with the lower camera row.
            last = camera_top[2]
            reset = (last[0], last[1] + last[3] + 8, 110, last[3],
                     last[4] + last[2] / 2 + 48 + 55, camera_bottom_y)
            remember("Reset Camera", reset)

    camera_section_top = camera_top[0][1] if camera_top else float("inf")
    special_rows = [
        group for group in row_groups(controls)
        if len(group) in (1, 2)
        and np.mean([x[5] for x in group]) > grid_floor
        and np.mean([x[5] for x in group]) < camera_section_top - 12
    ]
    special_rows.sort(key=lambda group: np.mean([x[5] for x in group]))
    if special_rows:
        step_row = next((group for group in special_rows if len(group) == 2), None)
        if step_row:
            remember("Step+", step_row[0]); remember("Step-", step_row[1])
        abort_row = next((group for group in special_rows if len(group) == 1), None)
        if abort_row:
            item = abort_row[0]
            remember("Abort", item)

    if "Step+" not in found and "Step-" not in found:
        arm_step_candidates = [
            item for item in controls
            if grid_floor < item[5] < camera_section_top - 12
            and item[2] >= 50 and item[3] >= 24
        ]
        if arm_step_candidates:
            item = min(arm_step_candidates, key=lambda value: value[5])
            half = item[2] / 2
            row_height = min(item[3], 22)
            plus = (item[0], item[1], half - 2, row_height,
                    item[0] + (half - 2) / 2, item[1] + row_height / 2)
            minus_x = item[0] + half + 2
            minus = (minus_x, item[1], half - 2, row_height,
                     minus_x + (half - 2) / 2, item[1] + row_height / 2)
            remember("Step+", plus)
            remember("Step-", minus)

    # The two camera step buttons can touch and become one saturated contour.
    # Split that wide row rather than allowing OCR to merge both labels into
    # the robot Step+ / Step- entries.
    camera_y = max((point[1] for name, point in found.items()
                    if name in ("Down", "Left", "Zoom-")), default=0)
    camera_step_floor = (
        camera_top[0][5] + camera_top[0][3] + 35
        if camera_top and len(camera_top) >= 3
        else camera_y + 55
    )
    camera_step_rows = [
        item for item in controls
        if item[5] > camera_step_floor and item[2] >= 45 and item[3] <= 35
    ]
    # Prefer the row immediately below the "Camera Jog controls" heading.
    # This avoids selecting similarly sized buttons in unrelated windows when
    # the panel is captured as part of a larger desktop.
    heading_y = None
    heading_data = pytesseract.image_to_data(
        cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
        config="--oem 3 --psm 11",
        output_type=pytesseract.Output.DICT,
        timeout=8,
    )
    camera_words = []
    for i, word in enumerate(heading_data["text"]):
        token = re.sub(r"[^a-z]", "", word.lower())
        if token in {"camera", "jog", "controls"}:
            camera_words.append((int(heading_data["top"][i]) / 3,
                                 int(heading_data["height"][i]) / 3))
    if camera_words:
        heading_y = max(y for y, _ in camera_words)
        headed_rows = [item for item in camera_step_rows if item[5] > heading_y + 18]
        camera_step_rows = headed_rows
    if camera_step_rows:
        # A single wide contour is not enough evidence: splitting it can put
        # Camera Step+ on the Camera Step- button. Require two distinct
        # controls, preferably identified by their green/red button colors.
        row = min(camera_step_rows, key=lambda value: value[5])
        row_y = row[5]
        same_row = sorted(
            [item for item in camera_step_rows if abs(item[5] - row_y) <= 12],
            key=lambda value: value[4])
        if len(same_row) >= 2:
            remember("Camera Step+", same_row[0])
            remember("Camera Step-", same_row[-1])

    def find_camera_step_colors():
        candidates = []
        for item in controls:
            if item[5] <= camera_step_floor or item[2] < 20 or item[3] < 8:
                continue
            px, py = round(item[4]), round(item[5])
            if not (0 <= px < hsv.shape[1] and 0 <= py < hsv.shape[0]):
                continue
            hue, saturation, value = hsv[py, px]
            kind = "plus" if 35 <= int(hue) <= 95 else "minus" if int(hue) <= 15 or int(hue) >= 165 else None
            if kind:
                candidates.append((kind, item))
        greens = [item for kind, item in candidates if kind == "plus"]
        reds = [item for kind, item in candidates if kind == "minus"]
        if not greens or not reds:
            return None
        pair = min(
            ((green, red) for green in greens for red in reds
             if abs(green[5] - red[5]) <= 18 and green[4] < red[4]),
            key=lambda pair: abs(pair[0][5] - pair[1][5]) + abs(pair[0][4] - pair[1][4]),
            default=None)
        return pair

    color_pair = find_camera_step_colors()
    if color_pair is not None:
        remember("Camera Step+", color_pair[0])
        remember("Camera Step-", color_pair[1])

    # OCR identifies the two irregular camera actions; a color-only fallback
    # uses their stable placement below the camera grid.
    # Run OCR on both contrast-equalized grayscale and adaptive-thresholded
    # images. The labels are tiny and white text can disappear in a plain
    # grayscale pass, especially on blue and green fills.
    scale = 3.0
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 7)
    text_candidates = []
    normalize = lambda s: re.sub(r"[^a-z0-9+\-]", "", s.lower().replace("−", "-").replace("–", "-"))
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
        if name == "Reset Camera":
            # This control is dark gray, so its label is the most reliable
            # location when no colored body is available for association.
            point = [monitor["left"] + cx, monitor["top"] + cy]
        if name in {"Abort", "Backup", "Safe J23", "Safe J456"}:
            # These labels are also present in the robot-mode card and its
            # buttons are dark, so the OCR text center is safer than associating
            # the label with an unrelated saturated control.
            point = [monitor["left"] + cx, monitor["top"] + cy]
        if name in {"Step+", "Step-"} and cy >= camera_section_top:
            # Generic Step labels in the camera section must not replace the
            # arm step buttons. Camera step controls have their own names.
            continue
        # Prefer the color-crop result (confidence 100) when available.
        if name not in found or confidence > found[name][1]: found[name] = (point, confidence)

    # Re-apply the color-identified camera pair after OCR so a misread
    # "Camera Step+" token cannot move the plus marker onto the red minus
    # button.
    if color_pair is not None:
        remember("Camera Step+", color_pair[0])
        remember("Camera Step-", color_pair[1])

    # The compact safe-position card often renders both labels as one OCR
    # token, and the J in J456 may be read as a 1. Re-read that card at a
    # larger scale so the two distinct safe buttons cannot collapse together.
    if "Backup" in found:
        backup_y = int(found["Backup"][1] - monitor["top"])
        y0 = max(0, backup_y - 28)
        lower = gray[y0:]
        enlarged = cv2.resize(lower, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
        data = pytesseract.image_to_data(
            enlarged,
            config="--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            output_type=pytesseract.Output.DICT,
            timeout=8,
        )
        for i, word in enumerate(data["text"]):
            token = normalize(word)
            if token not in {"safej23", "safej456", "safej1456"}:
                continue
            left, top = int(data["left"][i]), int(data["top"][i])
            width, height = int(data["width"][i]), int(data["height"][i])
            name = "Safe J23" if token == "safej23" else "Safe J456"
            found[name] = (
                [monitor["left"] + (left + width / 2) / 6,
                 monitor["top"] + y0 + (top + height / 2) / 6],
                120,
            )
    return monitor, image, {name: value[0] for name, value in found.items()}


def detect_step_label_region(image_bgra, monitor):
    """Find the first arm-panel 'Step size:' line for continuous OCR."""
    gray = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2GRAY)
    enlarged = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    data = pytesseract.image_to_data(
        enlarged, config="--oem 3 --psm 11",
        output_type=pytesseract.Output.DICT, timeout=8)
    words = []
    for i, text in enumerate(data["text"]):
        token = re.sub(r"[^a-z]", "", text.lower())
        if token not in {"step", "size"}:
            continue
        words.append((
            int(data["left"][i]) / 3, int(data["top"][i]) / 3,
            int(data["width"][i]) / 3, int(data["height"][i]) / 3))
    if not words:
        return None
    top = min(item[1] for item in words)
    line = [item for item in words if abs(item[1] - top) <= 24]
    left = max(0, min(item[0] for item in line) - 12)
    right = min(image_bgra.shape[1], max(item[0] + item[2] for item in line) + 180)
    bottom = min(image_bgra.shape[0], max(item[1] + item[3] for item in line) + 12)
    return {
        "left": int(monitor["left"] + left),
        "top": int(monitor["top"] + top - 8),
        "width": max(1, int(right - left)),
        "height": max(1, int(bottom - top + 8)),
    }


def make_button_panel_calibration(image_bgra, monitor, buttons):
    """Store button centers relative to a compact image template of their panel."""
    points = {name: buttons[name] for name in NEEDED_BUTTONS
              if name in buttons and buttons[name] and len(buttons[name]) >= 2 and
              monitor["left"] <= float(buttons[name][0]) < monitor["left"] + monitor["width"] and
              monitor["top"] <= float(buttons[name][1]) < monitor["top"] + monitor["height"]}
    if len(points) < 4:
        return None
    xs = [float(p[0]) for p in points.values()]; ys = [float(p[1]) for p in points.values()]
    pad = 24
    left = max(float(monitor["left"]), min(xs) - pad)
    top = max(float(monitor["top"]), min(ys) - pad)
    right = min(float(monitor["left"] + monitor["width"]), max(xs) + pad + 1)
    bottom = min(float(monitor["top"] + monitor["height"]), max(ys) + pad + 1)
    x0, y0 = int(left - monitor["left"]), int(top - monitor["top"])
    x1, y1 = int(right - monitor["left"]), int(bottom - monitor["top"])
    crop = np.asarray(image_bgra)[y0:y1, x0:x1]
    if crop.size == 0 or crop.shape[0] < 24 or crop.shape[1] < 24:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY) if crop.shape[2] == 4 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 130)
    ok, encoded = cv2.imencode(".png", edges)
    if not ok:
        return None
    width, height = x1 - x0, y1 - y0
    relative = {name: [(float(p[0]) - left) / width, (float(p[1]) - top) / height]
                for name, p in points.items()}
    return {"rect": {"left": left, "top": top, "width": width, "height": height},
            "relative": relative, "template": base64.b64encode(encoded.tobytes()).decode("ascii")}


def resolve_panel_button(capture, calibration, name, fallback, button_region=None, search_full=False):
    """Find the calibrated button panel in the current monitor image and map a relative point."""
    if not calibration or name not in calibration.get("relative", {}):
        return fallback
    monitor = dict(button_region or capture.monitors[0])
    shot = np.asarray(capture.grab(monitor))
    encoded = np.frombuffer(base64.b64decode(calibration["template"]), np.uint8)
    template = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if template is None or not template.size:
        raise RuntimeError("The saved movement-panel template is invalid. Re-run panel button setup.")
    gray = cv2.cvtColor(shot, cv2.COLOR_BGRA2GRAY) if shot.shape[2] == 4 else cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 130)
    rect = calibration["rect"]
    ox, oy = int(rect["left"] - monitor["left"]), int(rect["top"] - monitor["top"])
    tw, th = template.shape[1], template.shape[0]
    if search_full:
        # A window capture has local coordinates only. Find its calibrated
        # controls inside a desktop monitor capture to recover global coords.
        factor = 4
        small_template = cv2.resize(template, (max(1, tw // factor), max(1, th // factor)), interpolation=cv2.INTER_AREA)
        small_edges = cv2.resize(edges, (max(1, edges.shape[1] // factor), max(1, edges.shape[0] // factor)), interpolation=cv2.INTER_AREA)
        if small_edges.shape[0] < small_template.shape[0] or small_edges.shape[1] < small_template.shape[1]:
            raise RuntimeError("The movement panel is larger than the selected monitor capture.")
        coarse = cv2.matchTemplate(small_edges, small_template, cv2.TM_CCOEFF_NORMED)
        _, coarse_score, _, coarse_loc = cv2.minMaxLoc(coarse)
        if coarse_score < .45:
            raise RuntimeError(f"Could not find the window's movement controls on the selected monitor (match {coarse_score:.2f}).")
        guess_x, guess_y = coarse_loc[0] * factor, coarse_loc[1] * factor
        radius = 12
        sx, sy = max(0, guess_x - radius), max(0, guess_y - radius)
        ex = min(edges.shape[1], guess_x + tw + radius)
        ey = min(edges.shape[0], guess_y + th + radius)
        search = edges[sy:ey, sx:ex]
        if search.shape[0] < th or search.shape[1] < tw:
            raise RuntimeError("Could not refine the movement-panel location on the selected monitor.")
        scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(scores)
        if score < .48:
            raise RuntimeError(f"Could not reliably align the window controls to the monitor (match {score:.2f}).")
        rel_x, rel_y = calibration["relative"][name]
        return [monitor["left"] + sx + location[0] + rel_x * tw,
                monitor["top"] + sy + location[1] + rel_y * th]

    margin_x, margin_y = max(tw, int(tw * .7)), max(th, int(th * .7))
    sx, sy = max(0, ox - margin_x), max(0, oy - margin_y)
    ex, ey = min(edges.shape[1], ox + tw + margin_x), min(edges.shape[0], oy + th + margin_y)
    search = edges[sy:ey, sx:ex]
    if search.shape[0] < th or search.shape[1] < tw:
        raise RuntimeError("Could not find the movement-button panel on the captured screen.")
    scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(scores)
    if score < .42:
        raise RuntimeError(f"Could not reliably locate the movement-button panel (match {score:.2f}). Re-run panel button setup.")
    rel_x, rel_y = calibration["relative"][name]
    return [monitor["left"] + sx + location[0] + rel_x * tw,
            monitor["top"] + sy + location[1] + rel_y * th]


def _point_is_blue_button(capture, point, radius=18):
    """Require an automated target to still lie on a visible panel button."""
    monitor = dict(capture.monitors[0])
    x = float(point[0]) - float(monitor["left"])
    y = float(point[1]) - float(monitor["top"])
    if not (0 <= x < float(monitor["width"]) and 0 <= y < float(monitor["height"])):
        return False
    shot = np.asarray(capture.grab(monitor))
    hsv = cv2.cvtColor(shot[:, :, :3], cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([88, 55, 65], np.uint8), np.array([140, 255, 255], np.uint8))
    ix, iy = round(x), round(y)
    y0, y1 = max(0, iy - radius), min(blue.shape[0], iy + radius + 1)
    x0, x1 = max(0, ix - radius), min(blue.shape[1], ix + radius + 1)
    return bool(np.any(blue[y0:y1, x0:x1]))


class ButtonRecognitionWorker(QtCore.QThread):
    finished_result = QtCore.Signal(bool, object)
    def __init__(self, capture=None, button_region=None, source="monitor", parent=None):
        super().__init__(parent); self.capture, self.button_region, self.source = capture, button_region, source
    def run(self):
        try:
            capture = self.capture or new_mss(separate=True, source=self.source)
            self.finished_result.emit(True, (capture, detect_panel_buttons(capture, self.button_region)))
        except Exception as exc: self.finished_result.emit(False, str(exc))


class CaptureInitWorker(QtCore.QThread):
    finished_result = QtCore.Signal(bool, object)
    def __init__(self, separate=False, source=None, restore_token=None, parent=None):
        super().__init__(parent)
        self.separate, self.source, self.restore_token = separate, source, restore_token
    def run(self):
        try:
            capture = new_mss(
                separate=self.separate,
                source=self.source or ("monitor" if self.separate else None),
                restore_token=self.restore_token)
            self.finished_result.emit(True, capture)
        except Exception as exc:
            self.finished_result.emit(False, str(exc))


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
        self.screenshot_bgra = np.asarray(shot).copy()
        self.image = QtGui.QImage(shot.bgra, shot.size[0], shot.size[1],
                                  shot.size[0] * 4, QtGui.QImage.Format.Format_ARGB32).copy()
        self.monitor = monitor
        if monitor.get("source") == "window":
            # A captured app window is not a desktop overlay. Showing this
            # calibration dialog fullscreen made Plasma hide/minimize R08 and
            # looked like the requested app picker had vanished.
            screen = (parent.screen() if parent is not None else None) or QtWidgets.QApplication.primaryScreen()
            area = screen.availableGeometry() if screen is not None else QtCore.QRect(0, 0, 1280, 800)
            scale = min(1.0, area.width() * 0.92 / self.image.width(), area.height() * 0.84 / self.image.height())
            self.resize(max(480, round(self.image.width() * scale)), max(320, round(self.image.height() * scale)))
            self.setWindowTitle("Set up movement controls in selected window")
            self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
            self.show()
            self.frameGeometry().moveCenter(area.center())
            self.move(self.frameGeometry().topLeft())
            self.raise_()
            self.activateWindow()
        else:
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
            if "camera_region" in self.rects: out["camera_region"] = self.rects["camera_region"]
            if "button_region" in self.rects: out["button_region"] = self.rects["button_region"]
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
                self._next()
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
        return int(self.monitor["left"] + x), int(self.monitor["top"] + y)

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
        self.capture_source = self.cfg.get("capture_source", "monitor")
        set_capture_source(self.capture_source)
        self.settings = dict(DEFAULT_SETTINGS); self.settings.update(self.cfg.get("settings", {}) or {})
        if self.cfg.get("tool"): K.TOOL.update(self.cfg.get("tool"))
        self.qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.qt.setStyle("Fusion"); self.qt.setStyleSheet(STYLE)
        self.window = QtWidgets.QMainWindow(); self.window.setWindowTitle("R08 live twin — FANUC R-2000iA/200F")
        self.window.resize(1450, 900); self.window.setMinimumSize(1000, 650)
        self.capture = None
        self.button_capture = None
        self.pointer_capture = None
        self.shared = Shared(); self.ik = IKWorker(self.shared); self.ik.start()
        self.shared.set_regions(region=self.cfg.get("region"), status=self.cfg.get("status"), step=self.cfg.get("step"))
        self.bank = DigitBank(DIGIT_BANK_FILE)
        self.ocr = OcrWorker(self.shared, self.settings, self.ik, self.bank); self.ocr.start()
        self.step_reader = StepReader(self.shared, self.button_capture); self.step_reader.start()
        self.capture_init_worker = None
        self.capture_init_purpose = None
        self.capture_init_restore_token = None
        self.portal_selection_hint = None
        self.auto_recognition_after_region = False
        self.startup_capture_started = False
        self.pending_setup_keys = None
        self.setup_queue = []
        self._interrupt_requested = False
        self.renderer, self.render_note = make_renderer(bool(self.settings.get("gpu", True)))
        self.robot_mesh, self.ghost_mesh, self.cam = RobotMesh(), RobotMesh(), Camera()
        self.q_vis = np.radians([0, 35, 25, 0, 40, 0]); self.q_target = self.q_vis.copy()
        self.q_vis_velocity = np.zeros(6)
        self.live_t = self.ghost_t = None; self.ghost = None; self.trail = []; self.dirty = True
        self.follow_tool = False; self.show_trail = True
        self.guided_plan = []; self.guided_index = 0; self.guided_before = None
        self.portal_signals = PortalSignals(); self.portal_signals.ready.connect(self._on_input_ready)
        self.wayland_pointer = None; self.automated_run = False
        self.auto_next_allowed = 0.0; self.auto_verify_deadline = 0.0
        self.update_worker = None
        self.workspace_worker = None
        self.image_label = None; self.last_tick = clock(); self.log_entries = []
        self.debug_dialog = None
        self._last_debug_raw = None
        self._last_debug_step = None
        self._last_debug_step_level = None
        self.fps = 0.0; self._fps_count = 0; self._fps_time = clock()
        self.gpu_monitor = GpuUsageMonitor(); self.gpu_monitor.start()
        self.gpu_usage = None
        self._build()
        self.debug_dialog = DebugDialog(self.window)
        for entry in self.log_entries:
            self.debug_dialog.output.append(entry)
        self.log_entries.clear()
        self._refresh()
        self.window.closeEvent = self._close_event
        self.window.show()
        QtCore.QTimer.singleShot(1200, lambda: self._check_updates(manual=False))
        self.stop_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape), self.window)
        self.stop_shortcut.activated.connect(self._stop_guided_move)
        self.timer = QtCore.QTimer(self.window); self.timer.timeout.connect(self._tick); self.timer.start(33)
        QtCore.QTimer.singleShot(500, self._start_startup_captures)
        self.log(f"R08 live twin started. 3D: {self.renderer.name}.")
        if self.render_note: self.log(self.render_note)
        self.log("Native Wayland UI: manual guidance is available; optional pointer automation uses the KDE RemoteDesktop portal.")

    def _start_startup_captures(self):
        if self.startup_capture_started or self.capture_init_worker is not None:
            return
        self.startup_capture_started = True
        self.log("Preparing startup capture: main R08 stream first, movement-controls stream second.")
        if self.capture is None:
            source = self.capture_source
            detail = ("Choose the window containing the number table and camera stream."
                      if source == "window" else
                      "Choose the monitor containing the number table and camera stream.")
            self.portal_selection_hint = self._show_portal_selection_hint(
                "Select main R08 capture",
                "Choose the primary R08 screen share.",
                detail)
            self.capture_init_purpose = "startup_stream"
            self.capture_init_worker = CaptureInitWorker(
                separate=False, source=source,
                restore_token=self._capture_restore_token(False, source),
                parent=self.window)
            self.capture_init_worker.finished_result.connect(self._capture_init_finished)
            self.capture_init_worker.start()
            self.log("Opening the main screen-share chooser. Select the number table and camera stream…")
            return
        self.log("Main R08 capture already active; continuing to movement-controls capture.")
        self._start_startup_button_capture()

    def _start_startup_button_capture(self):
        if self.button_capture is not None:
            if not self.cfg.get("region"):
                self.run_setup(ALL_KEYS)
            return
        source = self.cfg.get("button_capture_source", "monitor")
        detail = ("Choose the Discord window containing the movement controls, not the camera-only window."
                  if source == "window" else
                  "Choose the monitor containing the movement controls.")
        self.portal_selection_hint = self._show_portal_selection_hint(
            "Select movement controls",
            "Choose the separate movement-controls screen share.",
            detail)
        self.capture_init_purpose = "startup_button"
        self.capture_init_worker = CaptureInitWorker(
            separate=True, source=source,
            restore_token=self._capture_restore_token(True, source),
            parent=self.window)
        self.capture_init_worker.finished_result.connect(self._capture_init_finished)
        self.capture_init_worker.start()
        self.log("Opening the movement-controls screen-share chooser…")

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
        debugb = QtWidgets.QPushButton("Debug"); debugb.clicked.connect(self._toggle_debug); header.addWidget(debugb)
        self.update_button = QtWidgets.QPushButton("Check GitHub updates"); self.update_button.clicked.connect(self._update_button_clicked); header.addWidget(self.update_button)
        reloadb = QtWidgets.QPushButton("Reload scripts (dev)")
        reloadb.setToolTip("Save settings and restart R08 to load the current Python files")
        reloadb.clicked.connect(lambda: self._reload_application()); header.addWidget(reloadb)
        tabs = QtWidgets.QTabWidget(); root.addWidget(tabs, 1)
        split = QtWidgets.QSplitter(); tabs.addTab(split, "Live twin")
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
        self.gaussian_splats = QtWidgets.QCheckBox("Gaussian splat view")
        self.gaussian_splats.setToolTip("Render mapped objects as soft splats and show the active Discord camera as Gaussian image points")
        self.gaussian_splats.toggled.connect(self._mark_dirty); views.addWidget(self.gaussian_splats)
        self.camera_texture = QtWidgets.QCheckBox("Map camera onto robot")
        self.camera_texture.setToolTip("Show the live camera texture projected onto the reconstructed robot after calibration.")
        self.camera_texture.setChecked(bool(self.cfg.get("camera_texture_enabled", False)))
        self.camera_texture.toggled.connect(self._camera_texture_toggled); views.addWidget(self.camera_texture)
        self.camera_source = QtWidgets.QComboBox(); self.camera_source.addItems(["Stream camera 1", "Stream camera 2"])
        self.camera_source.setToolTip("Select the label for the camera currently selected in the Discord stream")
        self.camera_source.currentIndexChanged.connect(self._mark_dirty); views.addWidget(self.camera_source)
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
        sl.addWidget(status)
        camera_controls = self._group("Panel camera controls")
        camera_grid = QtWidgets.QGridLayout(camera_controls)
        for index, name in enumerate(("Up", "Down", "Left", "Right", "Zoom+", "Zoom-")):
            button = QtWidgets.QPushButton(name)
            button.clicked.connect(lambda checked=False, n=name: self._panel_camera_action(n))
            camera_grid.addWidget(button, index // 3, index % 3)
        for column, name in enumerate(("Camera Step+", "Camera Step-")):
            button = QtWidgets.QPushButton(name)
            button.clicked.connect(lambda checked=False, n=name: self._panel_camera_action(n))
            camera_grid.addWidget(button, 3, column)
        reset_camera = QtWidgets.QPushButton("Reset Camera")
        reset_camera.clicked.connect(lambda: self._panel_camera_action("Reset Camera"))
        camera_grid.addWidget(reset_camera, 2, 0, 1, 2)
        switch_camera = QtWidgets.QPushButton("Switch Camera")
        switch_camera.clicked.connect(lambda: self._panel_camera_action("Switch Camera"))
        camera_grid.addWidget(switch_camera, 2, 2)
        sl.addWidget(camera_controls)
        safety = self._group("Panel safety controls")
        safety_grid = QtWidgets.QHBoxLayout(safety)
        for name in ("Abort", "Backup", "Safe J23", "Safe J456"):
            button = QtWidgets.QPushButton(name)
            button.clicked.connect(lambda checked=False, n=name: self._panel_camera_action(n))
            safety_grid.addWidget(button)
        sl.addWidget(safety)
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
        import_row = QtWidgets.QHBoxLayout(); ol.addLayout(import_row)
        points_plate = QtWidgets.QPushButton("Plate from points…"); points_plate.clicked.connect(self._add_plate_from_points); import_row.addWidget(points_plate)
        points_pole = QtWidgets.QPushButton("Pole from points…"); points_pole.clicked.connect(self._add_pole_from_points); import_row.addWidget(points_pole)
        self.teach_label = QtWidgets.QLabel("Map surfaces by positioning the tool tip and recording pose points."); self.teach_label.setWordWrap(True); ol.addWidget(self.teach_label)
        pointrow = QtWidgets.QHBoxLayout(); ol.addLayout(pointrow)
        add_point = QtWidgets.QPushButton("Set point"); add_point.clicked.connect(self._set_point); pointrow.addWidget(add_point)
        undo = QtWidgets.QPushButton("Undo point"); undo.clicked.connect(self._undo_point); pointrow.addWidget(undo)
        cancel = QtWidgets.QPushButton("Cancel map"); cancel.clicked.connect(self._cancel_teach); pointrow.addWidget(cancel)
        self.object_list = QtWidgets.QListWidget(); self.object_list.currentRowChanged.connect(self._select_object); ol.addWidget(self.object_list)
        objactions = QtWidgets.QHBoxLayout(); ol.addLayout(objactions)
        delete_obj = QtWidgets.QPushButton("Delete"); delete_obj.clicked.connect(self._delete_object); objactions.addWidget(delete_obj)
        edit_obj = QtWidgets.QPushButton("Edit size"); edit_obj.clicked.connect(self._edit_object); objactions.addWidget(edit_obj)
        edit_points = QtWidgets.QPushButton("Edit points"); edit_points.clicked.connect(self._edit_object_points); objactions.addWidget(edit_points)
        for idx in range(3):
            b = QtWidgets.QPushButton(f"Point {idx+1} to target"); b.clicked.connect(lambda checked=False, n=idx: self._object_point_to_target(n)); ol.addWidget(b)
        self.show_objects = QtWidgets.QCheckBox("Show mapped objects"); self.show_objects.setChecked(True); self.show_objects.toggled.connect(self._mark_dirty); ol.addWidget(self.show_objects)
        self._refresh_objects(); sl.addWidget(objects)
        digit = self._group("Digit reader"); dl = QtWidgets.QVBoxLayout(digit)
        self.raw_label = QtWidgets.QLabel("Waiting for the number table…"); self.raw_label.setWordWrap(True); dl.addWidget(self.raw_label)
        teach = QtWidgets.QPushButton("Teach digits…"); teach.clicked.connect(self._teach_digits); dl.addWidget(teach); sl.addWidget(digit)
        debug = QtWidgets.QPushButton("Save OCR snapshot"); debug.clicked.connect(self._save_ocr_snapshot); dl.addWidget(debug)
        sl.addStretch(); scroll.setWidget(side); split.addWidget(scroll); split.setStretchFactor(0, 3); split.setStretchFactor(1, 1)
        self._build_config_tab(tabs)
        self.window.menuBar().setNativeMenuBar(True)

    def _mark_dirty(self, *_): self.dirty = True

    def _capture_token_role(self, separate, source=None, purpose=None):
        if purpose == "automated_pointer":
            return "pointer_" + (source or "monitor")
        return ("panel_" + (source or self.cfg.get("button_capture_source", "monitor"))
                if separate else "primary")

    def _capture_restore_token(self, separate, source=None, purpose=None):
        tokens = self.cfg.get("pipewire_restore_tokens", {}) or {}
        role = self._capture_token_role(separate, source, purpose)
        return tokens.get(role)

    def _save_capture_restore_token(self, capture, separate, source=None, purpose=None):
        token = getattr(capture, "new_restore_token", None)
        if not token:
            return
        tokens = self.cfg.get("pipewire_restore_tokens", {}) or {}
        role = self._capture_token_role(separate, source, purpose)
        tokens[role] = str(token)
        self.cfg.set("pipewire_restore_tokens", tokens)
        self.cfg.save()

    def _toggle_debug(self):
        if self.debug_dialog is None:
            return
        self.debug_dialog.setVisible(not self.debug_dialog.isVisible())
        if self.debug_dialog.isVisible():
            self.debug_dialog.raise_()
            self.debug_dialog.activateWindow()

    def _camera_texture_toggled(self, enabled):
        self.cfg.set("camera_texture_enabled", bool(enabled))
        self.cfg.save()
        self.dirty = True

    def _clear_pipewire_tokens(self):
        answer = QtWidgets.QMessageBox.question(
            self.window, "Clear PipeWire tokens",
            "Forget all saved PipeWire screen-share permissions? The desktop chooser will appear again the next time a capture is opened.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No)
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.cfg.set("pipewire_restore_tokens", {})
        self.cfg.save()
        for attr in ("pointer_capture", "button_capture", "capture"):
            capture = getattr(self, attr, None)
            if capture is not None:
                try:
                    capture.close()
                except Exception:
                    pass
                setattr(self, attr, None)
        if self.wayland_pointer is not None:
            self.wayland_pointer.close()
            self.wayland_pointer = None
        self.log("Saved PipeWire screen-share tokens cleared.")
        self.debug_camera_status.setText("Saved PipeWire tokens cleared; the next capture will ask again.")

    def _refresh_debug_calibration_status(self):
        calibration = self.cfg.get("camera_texture")
        if not calibration:
            self.debug_calibration_status.setText("Camera projection: not calibrated.")
            return
        names = calibration.get("landmarks", [])
        self.debug_calibration_status.setText(
            f"Camera projection: calibrated with {len(names)} landmarks "
            f"({', '.join(names) if names else 'no landmark names saved'}).")

    def _camera_landmarks(self, q):
        info = self.robot_mesh.update(np.asarray(q, float))
        joints = [point.copy() for point in info.joints]
        # FANUC's J4 wrist-roll axis is in the wrist cluster, next to J5.
        # The raw DH origin at index 3 is the wrist-link offset, not the
        # visible J4 center used for camera calibration.
        joints[3] = joints[4].copy()
        return [np.zeros(3)] + joints + [
            info.flange.copy(), info.tip.copy()]

    def _update_debug_camera_view(self):
        if not hasattr(self, "debug_camera_image"):
            return
        frame = getattr(self, "camera_frame", None)
        if frame is None or not frame.size:
            return
        image = frame.copy()
        calibration = self.cfg.get("camera_texture") or {}
        if calibration.get("camera_matrix") and calibration.get("rvec") and calibration.get("tvec"):
            points = np.asarray(self._camera_landmarks(self.q_vis), np.float32)
            count = len(calibration.get("landmarks", []))
            projected, _ = cv2.projectPoints(
                points[:count], np.asarray(calibration["rvec"], float),
                np.asarray(calibration["tvec"], float),
                np.asarray(calibration["camera_matrix"], float), np.zeros(5))
            marker_points = projected.reshape(-1, 2)
        else:
            marker_points = calibration.get("image_points", [])
        for point in marker_points:
            x, y = np.rint(point).astype(int)
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.drawMarker(image, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
        h, w = image.shape[:2]
        qimg = QtGui.QImage(np.ascontiguousarray(image).data, w, h, w * 3,
                            QtGui.QImage.Format.Format_RGB888).copy()
        self.debug_camera_image.setPixmap(
            QtGui.QPixmap.fromImage(qimg).scaled(
                900, 340, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation))
        self.debug_camera_status.setText(
            f"Live camera frame: {w} × {h} px  ·  updated {time.strftime('%H:%M:%S')}")

    def _calibrate_camera_texture(self):
        camera_region = self.cfg.get("camera_region")
        if not camera_region:
            QtWidgets.QMessageBox.warning(self.window, "Camera calibration",
                                          "Select the Discord camera video region first.")
            return
        try:
            if self.capture is None:
                self.capture = new_mss()
            shot = np.asarray(self.capture.grab(camera_region))
            landmark_points = self._camera_landmarks(self.q_vis)
            landmarks = list(zip(
                ["Base", "J1", "J2", "J3", "J4", "J5", "J6", "Flange", "Tool tip"],
                landmark_points))
            dialog = CameraCalibrationDialog(self.window, shot, landmarks)
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return
            if len(dialog.image_points) < 6:
                QtWidgets.QMessageBox.warning(self.window, "Camera calibration",
                                              "Select at least six robot landmarks.")
                return
            object_points = np.asarray([point for _, point in landmarks[:len(dialog.image_points)]], np.float32)
            image_points = np.asarray(dialog.image_points, np.float32)
            height, width = shot.shape[:2]
            focal = width / (2.0 * math.tan(math.radians(60.0) / 2.0))
            camera_matrix = np.array([[focal, 0, width / 2.0],
                                      [0, focal, height / 2.0],
                                      [0, 0, 1]], np.float64)
            ok, rvec, tvec = cv2.solvePnP(
                object_points, image_points, camera_matrix, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                raise RuntimeError("OpenCV could not solve the camera pose from these points.")
            self.cfg.set("camera_texture", {
                "camera_matrix": camera_matrix.tolist(),
                "rvec": rvec.reshape(-1).tolist(),
                "tvec": tvec.reshape(-1).tolist(),
                "image_size": [width, height],
                "landmarks": [name for name, _ in landmarks[:len(dialog.image_points)]],
                "image_points": np.asarray(dialog.image_points, float).tolist(),
                "calibration_q": self.q_vis.tolist(),
            })
            self.cfg.set("camera_texture_enabled", True)
            self.cfg.save()
            self.camera_texture.setChecked(True)
            self._refresh_debug_calibration_status()
            self.log("Camera texture calibrated from robot landmarks.")
            self.dirty = True
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self.window, "Camera calibration", str(exc))

    def _build_config_tab(self, tabs):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)
        title = QtWidgets.QLabel("Configuration")
        title.setStyleSheet(f"color:{C.yellow};font-size:18px;font-weight:700")
        layout.addWidget(title)
        layout.addWidget(QtWidgets.QLabel(
            "Run any setup step again here. Screen regions and calibrated panel buttons are saved in the current configuration."))

        capture = self._group("Capture sources")
        capture_layout = QtWidgets.QGridLayout(capture)
        capture_layout.addWidget(QtWidgets.QLabel("Primary pose/step capture"), 0, 0)
        primary = QtWidgets.QPushButton("Choose monitor or window")
        primary.clicked.connect(lambda: self._setup_menu())
        capture_layout.addWidget(primary, 0, 1)
        pose = QtWidgets.QPushButton("Select number table")
        pose.clicked.connect(lambda: self.run_setup(["region"]))
        capture_layout.addWidget(pose, 1, 0)
        camera = QtWidgets.QPushButton("Select Discord camera video")
        camera.clicked.connect(lambda: self.run_setup(["camera_region"]))
        capture_layout.addWidget(camera, 1, 1)
        shared_note = QtWidgets.QLabel(
            "The number table and Discord camera region share the primary capture stream when they are on the same screen share.")
        shared_note.setWordWrap(True)
        shared_note.setProperty("muted", True)
        capture_layout.addWidget(shared_note, 3, 0, 1, 2)
        clear_tokens = QtWidgets.QPushButton("Clear saved PipeWire tokens")
        clear_tokens.setToolTip("Forget saved screen-share permissions so the desktop chooser appears next time.")
        clear_tokens.clicked.connect(self._clear_pipewire_tokens)
        capture_layout.addWidget(clear_tokens, 4, 0, 1, 2)
        calibrate = QtWidgets.QPushButton("Calibrate camera using robot points")
        calibrate.clicked.connect(self._calibrate_camera_texture)
        capture_layout.addWidget(calibrate, 5, 0, 1, 2)
        layout.addWidget(capture)

        panel = self._group("Movement panel")
        panel_layout = QtWidgets.QGridLayout(panel)
        actions = [
            ("Select panel screen share", lambda: self._reselect_button_screen_share("monitor")),
            ("Select panel window", lambda: self._reselect_button_screen_share("window")),
            ("Run full setup", lambda: self.run_setup(ALL_KEYS)),
            ("Set up all panel buttons", lambda: self.run_setup(BUTTON_KEYS)),
            ("Recognize panel buttons", self._recognize_buttons),
            ("Select step-size text", lambda: self.run_setup(["step_label"])),
            ("Select IDLE/MOVING banner", lambda: self.run_setup(["status"])),
            ("Set status and step text", lambda: self.run_setup(["status", "step_label"])),
        ]
        for index, (label, callback) in enumerate(actions):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(callback)
            panel_layout.addWidget(button, index // 2, index % 2)
        layout.addWidget(panel)

        workspace = self._group("Reachable workspace")
        workspace_layout = QtWidgets.QVBoxLayout(workspace)
        self.workspace_label = QtWidgets.QLabel("No workspace scan has been run.")
        self.workspace_label.setWordWrap(True)
        workspace_layout.addWidget(self.workspace_label)
        scan = QtWidgets.QPushButton("Find arm reach limits")
        scan.setToolTip("Samples the configured joint limits and estimates the TCP bounding box.")
        scan.clicked.connect(self._scan_workspace)
        workspace_layout.addWidget(scan)
        layout.addWidget(workspace)

        ocr = self._group("OCR snapshots")
        ocr_layout = QtWidgets.QVBoxLayout(ocr)
        ocr_layout.addWidget(QtWidgets.QLabel(
            "Capture the current number table to inspect the region and each cell used by the OCR reader."))
        snapshot_row = QtWidgets.QHBoxLayout()
        snapshot = QtWidgets.QPushButton("Capture OCR snapshot")
        snapshot.clicked.connect(self._save_ocr_snapshot)
        snapshot_row.addWidget(snapshot)
        self.ocr_snapshot_status = QtWidgets.QLabel("No snapshot captured.")
        self.ocr_snapshot_status.setProperty("muted", True)
        ocr_layout.addLayout(snapshot_row)
        self.ocr_snapshot_status.setWordWrap(True)
        ocr_layout.addWidget(self.ocr_snapshot_status)
        self.ocr_region_preview = QtWidgets.QLabel("Region preview")
        self.ocr_region_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.ocr_region_preview.setMinimumHeight(70)
        self.ocr_region_preview.setMaximumHeight(120)
        self.ocr_region_preview.setStyleSheet("background:#0f1115;border:1px solid #303641")
        ocr_layout.addWidget(self.ocr_region_preview)
        self.ocr_cell_previews = []
        cell_grid = QtWidgets.QGridLayout()
        for i, axis in enumerate(AXES):
            cell = QtWidgets.QLabel(f"{axis}\nNo image")
            cell.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            cell.setMinimumSize(125, 54)
            cell.setMaximumHeight(78)
            cell.setStyleSheet("background:#0f1115;border:1px solid #303641")
            self.ocr_cell_previews.append(cell)
            cell_grid.addWidget(cell, i // 3, i % 3)
        ocr_layout.addLayout(cell_grid)
        layout.addWidget(ocr)
        self._update_ocr_snapshot_preview()

        debug = self._group("Debug view")
        debug_layout = QtWidgets.QVBoxLayout(debug)
        self.debug_camera_preview = QtWidgets.QCheckBox("Show live camera preview")
        self.debug_camera_preview.setToolTip("Capture the configured Discord camera region and show it here for calibration diagnostics.")
        self.debug_camera_preview.toggled.connect(self._mark_dirty)
        debug_layout.addWidget(self.debug_camera_preview)
        self.debug_camera_status = QtWidgets.QLabel("Camera preview is disabled.")
        self.debug_camera_status.setWordWrap(True)
        debug_layout.addWidget(self.debug_camera_status)
        self.debug_camera_image = QtWidgets.QLabel("No camera frame")
        self.debug_camera_image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.debug_camera_image.setMinimumHeight(120)
        self.debug_camera_image.setMaximumHeight(360)
        self.debug_camera_image.setStyleSheet("background:#0f1115;border:1px solid #303641")
        debug_layout.addWidget(self.debug_camera_image)
        self.debug_calibration_status = QtWidgets.QLabel()
        self.debug_calibration_status.setWordWrap(True)
        debug_layout.addWidget(self.debug_calibration_status)
        layout.addWidget(debug)
        self._refresh_debug_calibration_status()
        layout.addStretch()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(page)
        tabs.addTab(scroll, "Configuration")

    def _scan_workspace(self):
        if self.workspace_worker and self.workspace_worker.isRunning():
            return
        self.workspace_label.setText("Scanning joint limits… this may take a few seconds.")
        self.workspace_worker = WorkspaceScanWorker(parent=self.window)
        self.workspace_worker.finished_result.connect(self._workspace_scan_finished)
        self.workspace_worker.start()

    def _workspace_scan_finished(self, ok, result):
        if not ok:
            self.workspace_label.setText(f"Workspace scan failed: {result}")
            self.log(f"Workspace scan failed: {result}")
            return
        lo = np.asarray(result["min_m"]) * 1000.0
        hi = np.asarray(result["max_m"]) * 1000.0
        radius = float(result["max_radius_m"]) * 1000.0
        text = (f"Estimated TCP limits from {result['samples']:,} joint samples:\n"
                f"X {lo[0]:+.0f} to {hi[0]:+.0f} mm, "
                f"Y {lo[1]:+.0f} to {hi[1]:+.0f} mm, "
                f"Z {lo[2]:+.0f} to {hi[2]:+.0f} mm\n"
                f"Maximum sampled distance from base: {radius:.0f} mm.")
        self.workspace_label.setText(text)
        self.cfg.set("workspace_bounds", {
            "min_m": np.asarray(result["min_m"]).tolist(),
            "max_m": np.asarray(result["max_m"]).tolist(),
            "max_radius_m": result["max_radius_m"],
            "samples": result["samples"],
        })
        self.cfg.save()
        self.log("Workspace reach limits estimated from joint-limit samples.")

    def _panel_camera_action(self, name):
        if self.guide_next.isEnabled() or self.automated_run:
            self.log("A guided panel action is already running. Stop it first.")
            return
        self.guided_plan = [Press(name, kind="camera", verifiable=False, label=f"panel camera: {name}")]
        self.guided_index = 0
        self._start_guided_move()

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
                           ("self_learn", "Learn from confident digit reads"), ("gpu", "Use GPU renderer (restart to apply)"),
                           ("show_axes", "Show coordinate axes and tool axis gizmo"),
                           ("show_perf", "Show FPS/GPU frequency counter")):
            check = QtWidgets.QCheckBox(label); check.setChecked(bool(self.settings.get(key, True))); toggles[key] = check; form.addRow(check)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept); buttons.rejected.connect(dlg.reject); form.addRow(buttons)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            for key, spin in controls.items(): self.settings[key] = spin.value()
            for key, check in toggles.items(): self.settings[key] = check.isChecked()
            self.cfg.set("settings", self.settings); self.cfg.save()
            self.dirty = True

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
            self._update_ocr_snapshot_preview(report)
            self.log(f"OCR snapshot saved in {DEBUG_DIR}")
        except Exception as exc: self.log(f"OCR snapshot failed: {exc}")

    def _update_ocr_snapshot_preview(self, report=None):
        if not hasattr(self, "ocr_region_preview"):
            return
        region_path = os.path.join(DEBUG_DIR, "region.png")
        if not os.path.isfile(region_path):
            return
        region_pixmap = QtGui.QPixmap(region_path)
        self.ocr_region_preview.setPixmap(
            region_pixmap.scaled(720, 150, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                 QtCore.Qt.TransformationMode.SmoothTransformation))
        for i, axis in enumerate(AXES):
            path = os.path.join(DEBUG_DIR, f"cell{i}_{axis}.png")
            if not os.path.isfile(path):
                continue
            pixmap = QtGui.QPixmap(path)
            self.ocr_cell_previews[i].setPixmap(
                pixmap.scaled(220, 80, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                              QtCore.Qt.TransformationMode.SmoothTransformation))
            self.ocr_cell_previews[i].setToolTip(path)
        if report is None:
            report_path = os.path.join(DEBUG_DIR, "report.txt")
            if os.path.isfile(report_path):
                with open(report_path, encoding="utf-8") as source:
                    report = [line.strip() for line in source if line.strip()]
        if report:
            self.ocr_snapshot_status.setText("  ·  ".join(report))
        else:
            self.ocr_snapshot_status.setText(f"Snapshot files found in {DEBUG_DIR}")

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
        try: pts = parse_xyz_points(text)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self.window, "Plate from points", f"{exc}\n\nUse X Y Z values, one point per line.")
            return
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
        thickness, ok = QtWidgets.QInputDialog.getDouble(self.window, "Plate thickness", "Thickness extending downward from the mapped top face (in):", 0.5, 0.01, 24.0, 2)
        if not ok: return
        trial.update(name=name.strip(), t=thickness)
        self.objects.append(trial); self.cfg.set("objects", self.objects); self.cfg.save()
        self._refresh_objects(); self.object_list.setCurrentRow(len(self.objects) - 1); self.dirty = True
        self.teach_label.setText(f"Mapped {obj_summary(trial)}")
        self.log(f"Mapped {obj_summary(trial)}")

    def _add_pole_from_points(self):
        prompt = ("Enter exactly two pole centerline points, one per line, as X Y Z in inches.\n"
                  "You can paste the labeled panel read-out format or three bare numbers per line.\n"
                  "Example:\n1. X +59.0  Y +51.1  Z -28.2 in\n2. X +57.2  Y -52.1  Z -27.3 in")
        text, ok = QtWidgets.QInputDialog.getMultiLineText(self.window, "Pole from points", prompt)
        if not ok: return
        try: pts = parse_xyz_points(text)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self.window, "Pole from points", f"{exc}\n\nUse X Y Z values, one point per line.")
            return
        if len(pts) != 2:
            QtWidgets.QMessageBox.warning(self.window, "Pole from points", f"Enter exactly two points; found {len(pts)}.")
            return
        if np.linalg.norm(np.asarray(pts[1]) - np.asarray(pts[0])) < 0.01:
            QtWidgets.QMessageBox.warning(self.window, "Pole from points", "The two endpoints must be different.")
            return
        number = 1 + sum(obj.get("type") == "pipe" for obj in self.objects)
        name, ok = QtWidgets.QInputDialog.getText(self.window, "Pole name", "Name:", text=f"Pole {number}")
        if not ok or not name.strip(): return
        outer, ok = QtWidgets.QInputDialog.getDouble(self.window, "Pole outside diameter", "Outside diameter (in):", 4.0, 0.1, 60.0, 2)
        if not ok: return
        inner, ok = QtWidgets.QInputDialog.getDouble(self.window, "Pole inside diameter", "Inside diameter (in; enter 0 for a solid pole):", min(3.0, outer - 0.01), 0.0, max(0.0, outer - 0.01), 2)
        if not ok: return
        item = {"type": "pipe", "name": name.strip(), "pts": pts, "d": outer, "id": inner}
        self.objects.append(item); self.cfg.set("objects", self.objects); self.cfg.save()
        self._refresh_objects(); self.object_list.setCurrentRow(len(self.objects) - 1); self.dirty = True
        self.teach_label.setText(f"Mapped {obj_summary(item)}")
        self.log(f"Mapped {obj_summary(item)}")

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
            value, ok = QtWidgets.QInputDialog.getDouble(self.window, "Plate thickness", "Thickness extending downward from the mapped top face (in):", 0.5, 0.01, 24.0, 2)
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
            val, ok = QtWidgets.QInputDialog.getDouble(self.window, "Plate thickness", "Thickness extending downward from the mapped top face (in):", float(obj.get("t", .5)), .01, 24, 2)
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

    def _edit_object_points(self):
        i = self.object_list.currentRow()
        if i < 0:
            return
        obj = self.objects[i]
        if obj.get("type") != "plate":
            QtWidgets.QMessageBox.information(
                self.window, "Edit points",
                "Only plate points can be edited here. Pipe endpoints are still edited by remapping the pipe.")
            return
        points = obj.get("pts", [])
        if len(points) != 3:
            QtWidgets.QMessageBox.warning(self.window, "Edit points", "This plate does not have exactly three points.")
            return

        dialog = QtWidgets.QDialog(self.window)
        dialog.setWindowTitle(f"Edit points — {obj.get('name', 'Plate')}")
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel(
            "Enter the three top-face points in FANUC WORLD coordinates (inches). "
            "The points must be distinct and non-collinear."))
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Point"), 0, 0)
        for col, axis in enumerate(("X", "Y", "Z"), 1):
            grid.addWidget(QtWidgets.QLabel(axis), 0, col)
        fields = []
        for row, point in enumerate(points, 1):
            grid.addWidget(QtWidgets.QLabel(str(row)), row, 0)
            row_fields = []
            for col, value in enumerate(point[:3], 1):
                field = QtWidgets.QDoubleSpinBox()
                field.setRange(-100000.0, 100000.0)
                field.setDecimals(3)
                field.setSingleStep(0.1)
                field.setValue(float(value))
                row_fields.append(field)
                grid.addWidget(field, row, col)
            fields.append(row_fields)
        layout.addLayout(grid)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        updated = [[field.value() for field in row_fields] for row_fields in fields]
        from .model3d import plate_dims
        if plate_dims({"type": "plate", "pts": updated}) is None:
            QtWidgets.QMessageBox.warning(
                self.window, "Edit points",
                "The three points must be distinct and non-collinear.")
            return
        obj["pts"] = updated
        self.cfg.set("objects", self.objects)
        self.cfg.save()
        self._refresh_objects()
        self.object_list.setCurrentRow(i)
        self.dirty = True
        self.teach_label.setText(f"Updated {obj_summary(obj)}")
        self.log(f"Updated points for {obj_summary(obj)}")

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
        self.guide_start.setEnabled(bool(plan) and not self.guide_next.isEnabled())
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
        self.plan_label.setText(f"Action {self.guided_index + 1} of {len(self.guided_plan)}: press {name} on the robot panel{suffix}. Then click ‘Confirm panel action’. The app does not press it for you.")

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
        if not action.verifiable:
            return True
        if action.kind == "jog" and pose is not None and self.guided_before is not None:
            delta = float(pose[action.axis] - self.guided_before[action.axis])
            if action.axis >= 3: delta = wrap180(delta)
            resolution = 0.1 if action.axis < 3 else 1.0
            return delta * action.sign > 0 and abs(delta) >= resolution * 0.75
        if action.kind == "step":
            snap = self.shared.snapshot()
            return snap.step_level is not None and snap.step_level == action.level
        return False

    def _show_portal_selection_hint(self, title, instruction, detail):
        dialog = QtWidgets.QDialog(self.window)
        dialog.setWindowTitle(title)
        dialog.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        dialog.setWindowFlags(
            QtCore.Qt.WindowType.Dialog |
            QtCore.Qt.WindowType.WindowStaysOnTopHint)
        layout = QtWidgets.QVBoxLayout(dialog)
        label = QtWidgets.QLabel(f"<b>{instruction}</b><br><br>{detail}<br><br>"
                                 "Press Continue, then make the selection in KDE.")
        label.setWordWrap(True)
        layout.addWidget(label)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Continue")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.adjustSize()
        window_rect = self.window.frameGeometry()
        screen = QtGui.QGuiApplication.screenAt(window_rect.center())
        available = screen.availableGeometry() if screen else QtGui.QGuiApplication.primaryScreen().availableGeometry()
        x = window_rect.right() + 12
        if x + dialog.width() > available.right():
            x = max(available.left(), window_rect.left() - dialog.width() - 12)
        y = max(available.top(), min(window_rect.top(), available.bottom() - dialog.height()))
        dialog.move(x, y)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return dialog

    def _start_automated_move(self):
        buttons = self.cfg.get("buttons", {}) or {}
        window_source = self.cfg.get("button_capture_source", "monitor") == "window"
        if not self.cfg.get("button_panel"):
            QtWidgets.QMessageBox.warning(self.window, "Panel box not calibrated",
                "Run panel button setup or button recognition once to save the movement-panel box and relative button positions.")
            return
        missing = sorted({p.btn for p in self.guided_plan if p.btn not in buttons})
        if missing:
            QtWidgets.QMessageBox.warning(self.window, "Automated input", "Set up these panel button positions first: " + ", ".join(missing))
            return
        if self.button_capture is None:
            if self.capture_init_worker and self.capture_init_worker.isRunning():
                self.log("Waiting for the current screen-share selection to finish before automated input.")
                return
            self.portal_selection_hint = self._show_portal_selection_hint(
                    "Select movement controls",
                    "Choose the Discord window containing the robot movement controls.",
                    "Do not choose the camera-only window. This selection is used to locate and calibrate the buttons.")
            self.capture_init_purpose = "automated_input"
            self.capture_init_worker = CaptureInitWorker(
                separate=True,
                source=self.cfg.get("button_capture_source", "monitor"),
                restore_token=self._capture_restore_token(
                    True, self.cfg.get("button_capture_source", "monitor")),
                parent=self.window)
            self.capture_init_worker.finished_result.connect(self._capture_init_finished)
            self.capture_init_worker.start()
            self.log("Select the movement-button window in the screen-share prompt…" if window_source else
                     "Select the movement-button monitor in the screen-share prompt…")
            return
        if window_source and self.pointer_capture is None:
            if self.capture_init_worker and self.capture_init_worker.isRunning():
                self.log("Waiting for the current screen-share selection to finish before automated input.")
                return
            self.portal_selection_hint = self._show_portal_selection_hint(
                    "Select pointer workspace",
                    "Choose the full workspace monitor containing the movement-control window.",
                    "R08 needs the monitor/workspace stream for pointer coordinates. Do not select a window here.")
            self.capture_init_purpose = "automated_pointer"
            self.capture_init_worker = CaptureInitWorker(
                # KDE currently returns the full virtual desktop for a
                # RemoteDesktop WINDOW request. Use a monitor stream and
                # locate the calibrated panel with its saved template.
                separate=True, source="monitor",
                # Do not restore the pointer window token. KDE can associate
                # a restored window token with a different video window after
                # Discord recreates its panels, so pointer input must always
                # let the user select the intended control window.
                restore_token=None if window_source else self._capture_restore_token(
                    True, "monitor", "automated_pointer"),
                parent=self.window)
            self.capture_init_worker.finished_result.connect(self._capture_init_finished)
            self.capture_init_worker.start()
            self.log("Select the monitor containing the movement-control window so R08 can locate it for pointer input…" if window_source else
                     "Select the monitor containing that window so R08 can map its controls to desktop clicks…")
            return
        answer = QtWidgets.QMessageBox.warning(
            self.window, "Automated robot panel input",
            f"R08 will move the pointer and click the calibrated robot panel buttons, every {self.click_interval.value():g} seconds. "
            "This can move the robot. Make sure the work area is clear and the robot is safe to move. Keep this window visible so you can stop the sequence. KDE will ask you to grant pointer access.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel)
        if answer != QtWidgets.QMessageBox.StandardButton.Yes: return
        try:
            capture = self.pointer_capture if window_source else self.button_capture
            monitors = list(capture.monitors[1:] or capture.monitors)
            panel = self.cfg.get("button_panel", {}) or {}
            rect = panel.get("rect", {}) if isinstance(panel, dict) else {}
            buttons = self.cfg.get("buttons", {}) or {}
            anchor = (
                float(rect.get("left", 0)) + float(rect.get("width", 0)) / 2,
                float(rect.get("top", 0)) + float(rect.get("height", 0)) / 2,
            )
            if not rect.get("width") or not rect.get("height"):
                points = [p for p in buttons.values() if isinstance(p, (list, tuple)) and len(p) >= 2]
                if not points:
                    raise RuntimeError("The calibrated panel has no usable button coordinates.")
                anchor = (float(points[0][0]), float(points[0][1]))
            if window_source:
                monitor = monitors[0]
            else:
                containing = [
                    m for m in monitors
                    if float(m["left"]) <= anchor[0] < float(m["left"]) + float(m["width"])
                    and float(m["top"]) <= anchor[1] < float(m["top"]) + float(m["height"])
                ]
                if not containing:
                    raise RuntimeError(
                        f"The calibrated movement panel at {anchor[0]:.0f},{anchor[1]:.0f} "
                        "is not on any currently captured monitor. Re-run panel setup."
                    )
                monitor = containing[0]
            if window_source:
                # The pointer portal may return a different window even when
                # the chooser was opened on the correct screen. Verify the
                # selected window contains the saved movement-panel template
                # before granting it permission to receive clicks.
                panel = self.cfg.get("button_panel", {}) or {}
                template_data = panel.get("template") if isinstance(panel, dict) else None
                if not template_data:
                    raise RuntimeError("The calibrated panel template is missing. Re-run panel window setup.")
                probe_name = next((name for name in self.cfg.get("buttons", {}) if name in NEEDED_BUTTONS), None)
                if probe_name is None:
                    raise RuntimeError("The calibrated panel has no recognized controls. Re-run panel recognition.")
                resolve_panel_button(
                    capture, panel, probe_name,
                    (self.cfg.get("buttons", {}) or {})[probe_name],
                    None,
                    search_full=True,
                )
        except Exception as exc:
            message = str(exc)
            if window_source:
                message += "\n\nR08 could not locate the calibrated movement panel on the selected monitor. Restart automated input and select the monitor containing the movement-control window."
                try:
                    capture.close()
                except Exception:
                    pass
                self.pointer_capture = None
            QtWidgets.QMessageBox.critical(self.window, "Automated input", message); return
        self.guided_index = 0; self.guided_before = self._current_pose().copy()
        self.automated_run = True
        self.guide_start.setEnabled(False); self.guide_next.setEnabled(False); self.guide_stop.setEnabled(True)
        self.plan_label.setText("Requesting KDE RemoteDesktop pointer permission… Approve the portal prompt to continue.")
        self.log("Requesting KDE RemoteDesktop pointer permission for automated panel clicks.")
        self.wayland_pointer = WaylandPointer(
            monitor,
            lambda ok, message: self.portal_signals.ready.emit(ok, message),
            output_type="monitor")

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
        self.log(f"KDE granted pointer access. {message} Starting the calibrated panel sequence.")
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
            window_source = self.cfg.get("button_capture_source", "monitor") == "window"
            capture = self.pointer_capture if window_source else self.button_capture
            pos = resolve_panel_button(capture, self.cfg.get("button_panel"), action.btn, pos,
                                       None if window_source else self.cfg.get("button_region"),
                                       search_full=window_source)
            if window_source and not _point_is_blue_button(capture, pos):
                monitor = capture.monitors[0]
                raise RuntimeError(
                    f"The calibrated target for {action.btn} is not on a visible blue "
                    f"panel button at ({pos[0]:.0f},{pos[1]:.0f}); the panel layout "
                    f"or scale has changed. Re-run movement-panel button setup "
                    f"with the complete control window visible "
                    f"({monitor['width']}×{monitor['height']} at "
                    f"{monitor['left']},{monitor['top']}).")
            panel_monitor = capture.monitors[0]
            self.log(
                f"Panel target {action.btn}: ({pos[0]:.0f},{pos[1]:.0f}); "
                f"capture {panel_monitor['left']},{panel_monitor['top']} "
                f"{panel_monitor['width']}×{panel_monitor['height']}")
            if self.window.frameGeometry().contains(QtCore.QPoint(round(pos[0]), round(pos[1]))):
                raise RuntimeError("Move the R08 window away from the detected movement-button panel, then resume.")
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
        capture_menu = menu.addMenu("Primary capture source")
        capture_menu.addAction("Capture a monitor", lambda: self._set_capture_source_mode("monitor"))
        capture_menu.addAction("Capture a window", lambda: self._set_capture_source_mode("window"))
        menu.addSeparator()
        menu.addAction("Run full setup", lambda: self.run_setup(ALL_KEYS))
        menu.addAction("Number table", lambda: self.run_setup(["region"]))
        menu.addAction("Panel buttons", lambda: self.run_setup(BUTTON_KEYS))
        movement_capture_menu = menu.addMenu("Movement-button capture source")
        movement_capture_menu.addAction("Choose a screen", lambda: self._reselect_button_screen_share("monitor"))
        movement_capture_menu.addAction("Choose a window", lambda: self._reselect_button_screen_share("window"))
        menu.addAction("Discord camera video region", lambda: self.run_setup(["camera_region"]))
        menu.addAction("Recognize panel buttons automatically…", self._recognize_buttons)
        menu.addAction("Step-size text region", lambda: self.run_setup(["step_label"]))
        menu.addAction("Optional banner and step text", lambda: self.run_setup(["status", "step_label"]))
        menu.exec(QtGui.QCursor.pos())

    def _set_capture_source_mode(self, source):
        current = self.cfg.get("capture_source", "monitor")
        if source == current:
            return
        if not os.environ.get("WAYLAND_DISPLAY"):
            QtWidgets.QMessageBox.information(self.window, "Capture source",
                "Window capture is provided by the Wayland desktop portal. This session is not running on Wayland.")
            return
        if self.automated_run or self.guide_next.isEnabled() or (self.update_worker and self.update_worker.isRunning()):
            QtWidgets.QMessageBox.warning(self.window, "Capture source", "Stop the current movement and wait for updater activity to finish before changing the capture source.")
            return
        answer = QtWidgets.QMessageBox.question(self.window, "Change capture source",
            f"R08 will restart and the next screen-share prompt will let you choose a {source}. "
            "The number-table, banner, step-text, and Discord-video regions must be selected again for the new capture.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No)
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.cfg.set("capture_source", source)
        for key in ("region", "status", "step", "camera_region"):
            self.cfg.set(key, None)
        self.cfg.save()
        self._reload_application(confirm=False)

    def _reselect_button_screen_share(self, source="monitor"):
        if self.capture_init_worker and self.capture_init_worker.isRunning():
            QtWidgets.QMessageBox.information(self.window, "Movement-button screen share",
                "Wait for the current screen-share portal selection to finish first.")
            return
        if self.button_capture is not None:
            try: self.button_capture.close()
            except Exception: pass
            self.button_capture = None
        if self.pointer_capture is not None:
            try: self.pointer_capture.close()
            except Exception: pass
            self.pointer_capture = None
        # These local pixel regions and templates belong to the old portal source.
        self.cfg.set("button_region", None)
        self.cfg.set("button_panel", None)
        self.cfg.set("button_capture_source", source)
        if source == "window":
            tokens = self.cfg.get("pipewire_restore_tokens", {}) or {}
            tokens.pop("panel_window", None)
            self.cfg.set("pipewire_restore_tokens", tokens)
        self.cfg.save()
        self.run_setup(["button_region"])

    def _recognize_buttons(self):
        if not HAVE_TESS or pytesseract is None:
            QtWidgets.QMessageBox.warning(self.window, "Button recognition", "Button recognition needs pytesseract and the Tesseract executable. Install them, then retry.")
            return
        try:
            if getattr(self, "button_recognition_worker", None) and self.button_recognition_worker.isRunning(): return
            self.log("Starting automatic movement-panel button recognition.")
            self.button_state.setText("Recognizing panel buttons…")
            self.button_recognition_worker = ButtonRecognitionWorker(
                self.button_capture, self.cfg.get("button_region"),
                self.cfg.get("button_capture_source", "monitor"), self.window)
            self.button_recognition_worker.finished_result.connect(self._button_recognition_finished)
            self.button_recognition_worker.start()
        except Exception as exc:
            self.button_state.setText("Panel button recognition failed")
            QtWidgets.QMessageBox.warning(self.window, "Button recognition failed", str(exc))

    def _button_recognition_finished(self, ok, result):
        self.button_state.setText("Panel buttons not set up" if not ok else "Panel buttons recognized")
        if not ok:
            self.log(f"Automatic button recognition failed: {result}")
            QtWidgets.QMessageBox.warning(self.window, "Button recognition failed", str(result)); return
        self.button_capture, (monitor, image, detected) = result
        self._save_capture_restore_token(
            self.button_capture, True, self.cfg.get("button_capture_source", "monitor"))
        self.log(f"Automatic button recognition found {len(detected)} label(s).")
        for name, point in sorted(detected.items()):
            self.log(f"  {name}: ({point[0]:.1f}, {point[1]:.1f})")
        step_region = detect_step_label_region(image, monitor)
        if step_region is not None:
            self.cfg.set("step", step_region)
            self.cfg.save()
            self.shared.set_regions(step=step_region, keep=True)
            self.step_reader.set_capture(self.button_capture)
            self.log(f"Automatic step-size recognition selected {step_region}.")
        else:
            self.log("Automatic step-size recognition found no matching text.")
        if not detected:
            QtWidgets.QMessageBox.information(self.window, "Button recognition", "No panel button labels were detected. Move the R08 window away, ensure the panel is visible, or use manual panel setup.")
            return
        dlg = ButtonReviewDialog(self.window, monitor, image, detected, self.cfg.get("buttons", {}) or {})
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted or not dlg.buttons: return
        self.cfg.set("buttons", dlg.buttons)
        panel = make_button_panel_calibration(image, monitor, dlg.buttons)
        if panel: self.cfg.set("button_panel", panel)
        self.cfg.save(); self._refresh()
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
        if self.capture_init_worker and self.capture_init_worker.isRunning():
            QtWidgets.QMessageBox.warning(self.window, "Reload unavailable", "Wait for the screen-share portal selection to finish before reloading.")
            return
        if getattr(self, "button_recognition_worker", None) and self.button_recognition_worker.isRunning():
            QtWidgets.QMessageBox.warning(self.window, "Reload unavailable", "Wait for button recognition to finish before reloading.")
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
        self._close_capture_sources()
        os.execv(sys.executable, [sys.executable, *sys.argv])

    def run_setup(self, keys):
        try:
            panel_keys = set(BUTTON_KEYS) | {"button_region", "step_label"}
            if any(key in panel_keys for key in keys) and any(key not in panel_keys for key in keys):
                stream_keys = [key for key in keys if key not in panel_keys]
                button_keys = [key for key in keys if key in panel_keys]
                self.setup_queue = [part for part in (stream_keys, button_keys) if part]
                self._run_next_setup()
                return
            separate = all(key in panel_keys for key in keys)
            capture = self.button_capture if separate else self.capture
            if capture is None:
                if self.capture_init_worker and self.capture_init_worker.isRunning():
                    self.log("Waiting for the current screen-share selection to finish.")
                    return
                if separate:
                    source = self.cfg.get("button_capture_source", "monitor")
                    if source == "window":
                        self.portal_selection_hint = self._show_portal_selection_hint(
                            "Select movement controls",
                            "Choose the Discord window containing the robot movement controls.",
                            "Do not choose the camera-only window.")
                    else:
                        self.portal_selection_hint = self._show_portal_selection_hint(
                            "Select movement-control monitor",
                            "Choose the monitor containing the robot movement controls.",
                            "The controls must be visible in the selected monitor.")
                self.pending_setup_keys = list(keys)
                self.capture_init_purpose = "setup_button" if separate else "setup_stream"
                self.capture_init_worker = CaptureInitWorker(
                    separate=separate,
                    source=self.cfg.get("button_capture_source", "monitor") if separate else None,
                    restore_token=self._capture_restore_token(
                        separate,
                        self.cfg.get("button_capture_source", "monitor") if separate else None),
                    parent=self.window)
                self.capture_init_worker.finished_result.connect(self._capture_init_finished)
                if separate and self.cfg.get("button_capture_source", "monitor") == "window":
                    # Make sure the portal request starts from a visible,
                    # activated Qt window. On Plasma, launching it while R08
                    # is minimized/fullscreen can leave the chooser behind it.
                    self.window.showNormal(); self.window.raise_(); self.window.activateWindow()
                    QtCore.QTimer.singleShot(180, self.capture_init_worker.start)
                    self.log("Opening KDE's screen-share chooser. Select the Windows tab, choose the app with the movement controls, then press Share.")
                else:
                    self.capture_init_worker.start()
                    self.log("Opening the separate movement-button screen share…" if separate else "Opening the screen/window share…")
                return
            self._open_setup_dialog(keys, capture)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self.window, "Setup failed", str(exc))
            self.log(f"Setup failed: {exc}")

    def _open_setup_dialog(self, keys, capture):
        dlg = SetupPicker(self.window, keys, capture)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted or not dlg.result_data:
            if self.auto_recognition_after_region and keys == ["button_region"]:
                self.auto_recognition_after_region = False
            self.setup_queue.clear()
            return
        result = dlg.result_data
        for src, dst in (("region", "region"), ("status", "status"), ("step", "step"),
                         ("camera_region", "camera_region"), ("button_region", "button_region")):
            if src in result: self.cfg.set(dst, result[src])
        if "buttons" in result:
            old = self.cfg.get("buttons", {}) or {}
            old.update(result["buttons"]); self.cfg.set("buttons", old)
            panel = make_button_panel_calibration(dlg.screenshot_bgra, dlg.monitor, old)
            if panel: self.cfg.set("button_panel", panel)
        self.cfg.save(); self.shared.set_regions(region=self.cfg.get("region"), status=self.cfg.get("status"), step=self.cfg.get("step"))
        self.log("Setup saved.")
        if "region" in result: self.log("Tip: teach a few digits so the reader learns the stream font.")
        self._refresh()
        if self.auto_recognition_after_region and "button_region" in result:
            self.auto_recognition_after_region = False
            self.log("Controls region saved. Starting automatic button recognition…")
            self._recognize_buttons()
            return
        if self.setup_queue:
            QtCore.QTimer.singleShot(0, self._run_next_setup)

    def _run_next_setup(self):
        if self.setup_queue:
            self.run_setup(self.setup_queue.pop(0))

    def _retry_capture_without_token(self, purpose, setup_keys):
        source = self.cfg.get("button_capture_source", "monitor") if purpose in (
            "setup_button", "startup_button", "automated_input") else self.capture_source
        separate = purpose in ("setup_button", "startup_button", "automated_input", "automated_pointer")
        role = self._capture_token_role(separate, "monitor" if purpose == "automated_pointer" else source, purpose)
        tokens = self.cfg.get("pipewire_restore_tokens", {}) or {}
        if not tokens.get(role):
            return False
        answer = QtWidgets.QMessageBox.question(
            self.window, "Screen-share permission expired",
            "The saved permission for this capture is no longer valid. Pick the screen or window again?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes)
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return False
        tokens.pop(role, None)
        self.cfg.set("pipewire_restore_tokens", tokens)
        self.cfg.save()
        self.log(f"Saved screen-share token '{role}' was invalid; requesting a new selection.")
        if purpose == "startup_stream":
            self.startup_capture_started = False
            self._start_startup_captures()
        elif purpose == "startup_button":
            self._start_startup_button_capture()
        elif purpose in ("setup_button", "setup_stream"):
            self.run_setup(setup_keys or (["button_region"] if purpose == "setup_button" else ALL_KEYS))
        elif purpose in ("automated_input", "automated_pointer"):
            self._start_automated_move()
        elif purpose == "camera_capture":
            self.camera_capture_error_time = 0.0
        return True

    def _capture_init_finished(self, ok, result):
        purpose = self.capture_init_purpose
        setup_keys = self.pending_setup_keys
        self.capture_init_worker = None; self.capture_init_purpose = None
        if not ok:
            self.pending_setup_keys = None; self.setup_queue.clear()
            if self.portal_selection_hint is not None:
                self.portal_selection_hint.close()
                self.portal_selection_hint.deleteLater()
                self.portal_selection_hint = None
            if purpose == "camera_capture":
                self.camera_capture_error_time = clock()
                self.log(f"Discord camera capture failed: {result}")
            else:
                if not self._retry_capture_without_token(purpose, setup_keys):
                    QtWidgets.QMessageBox.warning(self.window, "Screen share unavailable", str(result))
            return
        if self.portal_selection_hint is not None:
            self.portal_selection_hint.close()
            self.portal_selection_hint.deleteLater()
            self.portal_selection_hint = None
        if purpose in ("setup_button", "automated_input", "startup_button"):
            self.button_capture = result
            self._save_capture_restore_token(
                result, True, self.cfg.get("button_capture_source", "monitor"))
            self.step_reader.set_capture(result)
        elif purpose == "automated_pointer":
            self.pointer_capture = result
            self._save_capture_restore_token(result, True, "monitor", purpose)
        else:
            self.capture = result
            self._save_capture_restore_token(result, False)
        if purpose == "startup_stream":
            self.log("Main R08 capture ready.")
            self._start_startup_button_capture()
            return
        if purpose == "startup_button":
            self.log("Movement-controls capture ready.")
            if self.cfg.get("button_capture_source", "monitor") == "window":
                self.auto_recognition_after_region = True
                self.log("Select the movement-controls region for automatic recognition…")
                self.run_setup(["button_region"])
            elif not self.cfg.get("region"):
                self.run_setup(ALL_KEYS)
            return
        if (purpose == "setup_button" and
                self.cfg.get("button_capture_source", "monitor") == "window"):
            self.auto_recognition_after_region = True
            self.log("Movement-control window shared. Select the controls region for recognition…")
            self.run_setup(["button_region"])
            return
        if purpose in ("setup_button", "setup_stream"):
            keys = self.pending_setup_keys; self.pending_setup_keys = None
            if keys: self.run_setup(keys)
        elif purpose in ("automated_input", "automated_pointer"):
            self._start_automated_move()

    def _close_capture_sources(self):
        seen = set()
        for capture in (self.pointer_capture, self.button_capture, self.capture):
            if capture is None or id(capture) in seen: continue
            seen.add(id(capture))
            try:
                capture.close()
            except Exception:
                pass

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
        line = f"<span style='color:{C.faint}'>{stamp}</span>  {text}"
        if self.debug_dialog is not None:
            self.debug_dialog.output.append(line)
        else:
            self.log_entries.append(text)

    def _refresh(self):
        self.button_state.setText("Panel buttons ready" if all(n in (self.cfg.get("buttons", {}) or {}) for n in NEEDED_BUTTONS) else "Panel buttons not set up")
        if self.cfg.get("button_panel"):
            self.button_state.setText(self.button_state.text() + " · relative panel tracking on")
        self.capture_label.setText("Number table configured" if self.cfg.get("region") else "Number table not configured")

    def _tick(self):
        now = clock(); dt = min(0.1, now - self.last_tick); self.last_tick = now
        usage = self.gpu_monitor.latest
        if usage != self.gpu_usage:
            self.gpu_usage = usage; self.dirty = True
        snap = self.shared.snapshot()
        self.read_state.setText(f"● Reading {snap.scan_rate:.0f}/s" if snap.state == "Live" else f"○ {snap.state}")
        self.raw_label.setText(snap.raw or "Waiting for the number table…")
        if snap.raw and snap.raw != self._last_debug_raw:
            self._last_debug_raw = snap.raw
            self.log(f"OCR pose readout: {snap.raw}")
        if snap.step_status != self._last_debug_step:
            self._last_debug_step = snap.step_status
            self.log(f"Step-size OCR: {snap.step_status}")
        if snap.step_level != self._last_debug_step_level:
            self._last_debug_step_level = snap.step_level
            if snap.step_level is not None:
                pos, ang = STEP_LEVELS[snap.step_level]
                self.log(f"Step-size OCR recognized: {pos:g} in / {ang:g}°")
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
            self.live_t = live.t; self.q_target = K.unwrap_q_to_reference(live.q, self.q_vis)
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
        for i in K.INFINITE_JOINTS:
            diff[i] = math.radians(wrap180(math.degrees(diff[i])))
        if np.max(np.abs(diff)) > 2e-4:
            vmax = math.radians(float(self.settings.get("viz_deg_s", 25)))
            target_velocity = np.clip(diff * 8.0, -vmax, vmax)
            accel = math.radians(90.0)
            self.q_vis_velocity += np.clip(target_velocity - self.q_vis_velocity, -accel * dt, accel * dt)
            self.q_vis += self.q_vis_velocity * dt; self.dirty = True
        else:
            self.q_vis_velocity *= math.exp(-dt * 12.0)
        camera_region = self.cfg.get("camera_region")
        if ((self.gaussian_splats.isChecked() or self.camera_texture.isChecked() or
             self.debug_camera_preview.isChecked()) and camera_region and
                now - getattr(self, "camera_frame_time", 0.0) >= 0.20 and
                now - getattr(self, "camera_capture_error_time", 0.0) > 5.0):
            try:
                if self.capture is None:
                    if not (self.capture_init_worker and self.capture_init_worker.isRunning()):
                        self.capture_init_purpose = "camera_capture"
                        self.capture_init_worker = CaptureInitWorker(
                            restore_token=self._capture_restore_token(False),
                            parent=self.window)
                        self.capture_init_worker.finished_result.connect(self._capture_init_finished)
                        self.capture_init_worker.start()
                else:
                    shot = np.asarray(self.capture.grab(camera_region))
                    self.camera_frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2RGB)
                    self.camera_frame_time = now; self.dirty = True
                    self._update_debug_camera_view()
            except Exception as exc:
                if now - getattr(self, "camera_capture_error_time", 0.0) > 5:
                    self.camera_capture_error_time = now
                    self.log(f"Discord camera capture failed: {exc}")
        if self.dirty: self._render(snap); self.dirty = False

    def _render(self, snap):
        w, h = max(400, self.image_label.width()), max(300, self.image_label.height())
        info = self.robot_mesh.update(self.q_vis)
        hud = {"backend": self.renderer.name,
               "axes": bool(self.settings.get("show_axes", True)), "gaussian_splats": self.gaussian_splats.isChecked()}
        if self.settings.get("show_perf", True):
            hud.update(fps=self.fps, gpu=self.gpu_usage)
        if self.gaussian_splats.isChecked() and getattr(self, "camera_frame", None) is not None:
            hud["camera_view"] = self.camera_frame
            hud["camera_name"] = f"Discord camera {self.camera_source.currentIndex() + 1}"
        calibration = self.cfg.get("camera_texture")
        if (self.camera_texture.isChecked() and calibration and
                getattr(self, "camera_frame", None) is not None):
            hud["camera_view"] = self.camera_frame
            hud["camera_texture"] = calibration
            hud["camera_landmarks"] = self._camera_landmarks(self.q_vis)
        if snap.pose is not None: hud["tip_text"] = "  ".join(fmt_pos(snap.pose[i]) for i in range(3)) + " in"
        else: hud["msg"] = "Open Setup to select the number table"
        ghost_tris = ghost_tip = None; ghost_ok = True; ghost_col = hex_rgb(C.green)
        hud["ghost_text"] = None
        if self.ghost:
            gi = self.ghost_mesh.update(self.ghost["q"])
            ghost_tris, ghost_tip, ghost_ok = self.ghost_mesh.triangles(), gi.tip, self.ghost["ok"]
            ghost_col = hex_rgb(C.green if ghost_ok else C.amber)
            hud["ghost_text"] = "target  " + "  ".join(f"{AXES[i]} {self.ghost['pose'][i]:+.1f}" for i in range(6))
        tris = self.robot_mesh.triangles()
        labels = [(point, "FANUC", (24, 25, 28), False) for point in self.robot_mesh.brand_points]
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
                                labels=labels, gizmo=bool(self.settings.get("show_axes", True)), hud=hud)
        rgb = self.renderer.render(w, h, self.cam, frame)
        self._fps_count += 1
        elapsed = clock() - self._fps_time
        if elapsed >= 1.0:
            self.fps = self._fps_count / elapsed
            self._fps_count = 0; self._fps_time = clock()
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def _close_event(self, event):
        portal_workers = [self.capture_init_worker, getattr(self, "button_recognition_worker", None)]
        active_workers = [worker for worker in portal_workers if worker is not None and worker.isRunning()]
        if active_workers and not self._interrupt_requested:
            QtWidgets.QMessageBox.warning(self.window, "Screen share is starting", "Cancel the desktop portal prompt and wait for capture setup to finish before closing R08.")
            event.ignore(); return
        if active_workers:
            for worker in active_workers:
                worker.terminate()
                worker.wait(1000)
        if hasattr(self, "timer"): self.timer.stop()
        if hasattr(self, "interrupt_timer"): self.interrupt_timer.stop()
        self.automated_run = False
        if self.wayland_pointer: self.wayland_pointer.close()
        self.wayland_pointer = None
        try:
            self.cfg.set("settings", self.settings); self.cfg.save(); self.bank.save()
        except Exception: pass
        self.ocr.stop(); self.step_reader.stop(); self.ik.stop()
        for worker in (self.ocr, self.step_reader, self.ik): worker.join(timeout=2)
        self.gpu_monitor.stop(); self.gpu_monitor.join(timeout=2)
        self._close_capture_sources()
        event.accept()

    def run(self):
        previous_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._on_sigint)
        self.interrupt_timer = QtCore.QTimer(self.window)
        self.interrupt_timer.timeout.connect(self._poll_sigint)
        self.interrupt_timer.start(100)
        try:
            result = self.qt.exec()
            return 130 if self._interrupt_requested else result
        finally:
            self.interrupt_timer.stop()
            signal.signal(signal.SIGINT, previous_sigint)

    def _on_sigint(self, _signum, _frame):
        # Do not raise KeyboardInterrupt inside a PySide callback; Qt would
        # print it and keep the event loop alive. The timer closes dialogs safely.
        self._interrupt_requested = True

    def _poll_sigint(self):
        if not self._interrupt_requested:
            return
        self.interrupt_timer.stop()
        modal = self.qt.activeModalWidget()
        if modal is not None:
            modal.close()
        self.window.close()
        if self.window.isVisible():
            self.qt.quit()


def main():
    app = QtApp()
    return app.run()
