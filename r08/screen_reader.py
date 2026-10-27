"""
screen_reader.py - background threads that feed the UI.

    Shared      thread-safe state (pose, banner state, IK results).  The pose passes a SPEED GATE: a jump the real robot
                cannot make (> GATE_* per second) is ignored unless it is read GATE_CONFIRM times in a row.
    OcrWorker   grabs the number table (mss), reads 6 cells with digits.read_cell, confirms each change on two
                consecutive scans, offers the pose to Shared.  Also classifies the IDLE/MOVING banner colour and
                (optionally, needs Tesseract) the 'Step size' text.
    IKWorker    solves inverse kinematics for the accepted pose ("live") and for the Go-to preview ("ghost").

Thread rules: workers only write Shared (under sh.lock) ; the UI thread only calls sh.snapshot().
"""
import re, threading, traceback, os, shutil
from types import SimpleNamespace
import numpy as np
import cv2
from .common import clock, wrap180, new_mss, STEP_LEVELS, TESSERACT_PATH
from . import digits, kinematics as K

try:
    import pytesseract
    import os
    if os.path.exists(TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    HAVE_TESS = True
except Exception:
    pytesseract, HAVE_TESS = None, False

GATE_POS, GATE_POS_RATE = 1.5, 10.0       # inches, inches / second
GATE_ANG, GATE_ANG_RATE = 6.0, 80.0       # degrees, degrees / second
GATE_CONFIRM = 3
SIG_CHANGE_FRAC = 0.012                   # fraction of changed pixels that makes a cell "new"
FORCE_REREAD_S = 3.0
CELL_X_MARGIN, CELL_Y_MARGIN = 0.02, 0.08


class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.region = self.status_region = self.step_region = None
        self.pose, self.seq, self.t_change, self.t_read = None, 0, 0.0, 0.0
        self.state, self.raw = "Waiting for setup", ""
        self.scan_rate = self.ocr_rate = 0.0
        self.busy, self.last_moving_t = None, -1e9      # busy: True MOVING / False IDLE / None unknown
        self.step_level, self.step_t = None, 0.0
        self.step_status = "Step-size region not configured"
        self.ik = {}                                    # "live" / "ghost" -> SimpleNamespace(q, pe, re, pose, t)
        self.nrej, self.last_rej = 0, None
        self._cand, self._candn, self._t_acc = None, 0, clock()

    def snapshot(self):
        with self.lock:
            return SimpleNamespace(
                pose=None if self.pose is None else self.pose.copy(), seq=self.seq, t_change=self.t_change,
                t_read=self.t_read, state=self.state, raw=self.raw, scan_rate=self.scan_rate, ocr_rate=self.ocr_rate,
                busy=self.busy, last_moving_t=self.last_moving_t, step_level=self.step_level, step_t=self.step_t,
                has_status=self.status_region is not None, has_step=self.step_region is not None,
                step_status=self.step_status)

    def set_regions(self, region=None, status=None, step=None, keep=False):
        with self.lock:
            if not keep or region is not None:
                self.region = region
            if not keep or status is not None:
                self.status_region = status
            if not keep or step is not None:
                if step != self.step_region:
                    self.step_level, self.step_t = None, 0.0
                    self.step_status = "Reading step-size region" if step else "Step-size region not configured"
                self.step_region = step

    def offer_pose(self, v):
        """Speed-gated pose update.  -> True when the accepted pose changed."""
        v = np.asarray(v, float)
        now = clock()
        with self.lock:
            old = self.pose
            if old is not None:
                dt = max(0.0, now - self._t_acc)
                dp = float(np.max(np.abs(v[:3] - old[:3])))
                da = float(np.max(np.abs(wrap180(v[3:] - old[3:]))))
                if dp > GATE_POS + GATE_POS_RATE * dt or da > GATE_ANG + GATE_ANG_RATE * dt:
                    c = self._cand
                    if c is not None and np.max(np.abs(v[:3] - c[:3])) < 0.35 and np.max(np.abs(wrap180(v[3:] - c[3:]))) < 2.5:
                        self._candn += 1
                    else:
                        self._cand, self._candn = v.copy(), 1
                    if self._candn < GATE_CONFIRM:
                        self.nrej += 1
                        self.last_rej = (v.copy(), old.copy())
                        return False
                self._cand, self._candn = None, 0
            changed = old is None or bool(np.any(np.abs(old - v) > 1e-9))
            if changed:
                self.pose, self.t_change, self._t_acc = v.copy(), now, now
                self.seq += 1
            self.t_read, self.state = now, "Live"
            return changed


# ------------------------------------------------------------------ helpers
def split_cells(gray):
    """Region grayscale -> six cell crops, row-major (X Y Z / Xa Ya Za)."""
    h, w = gray.shape[:2]
    out = []
    for r in range(2):
        y1, y2 = r * h // 2, (r + 1) * h // 2
        for c in range(3):
            x1, x2 = c * w // 3, (c + 1) * w // 3
            px, py = int((x2 - x1) * CELL_X_MARGIN), int((y2 - y1) * CELL_Y_MARGIN)
            out.append(gray[y1 + py:y2 - py, x1 + px:x2 - px])
    return out


def grab_gray(sct, region):
    bgra = np.asarray(sct.grab(region))
    return bgra, cv2.cvtColor(bgra, cv2.COLOR_BGRA2GRAY)


def classify_banner(bgra):
    """Green banner -> False (IDLE).  Amber banner -> True (MOVING).  Anything else -> None."""
    hsv = cv2.cvtColor(np.ascontiguousarray(bgra[:, :, :3]), cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 90)
    if mask.mean() < 0.12:
        return None
    hue = float(np.median(hsv[:, :, 0][mask]))
    if 36 <= hue <= 95:
        return False
    if 5 <= hue < 36:
        return True
    return None


def parse_step_level(text):
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text.replace(",", "."))]
    lp = {i for i, (p, a) in enumerate(STEP_LEVELS) for n in nums if abs(n - p) < 1e-6}
    la = {i for i, (p, a) in enumerate(STEP_LEVELS) for n in nums if abs(n - a) < 1e-6}
    both = lp & la
    if len(both) == 1:
        return next(iter(both))
    if len(both) == 0 and len(lp | la) == 1 and (bool(lp) != bool(la)):
        return next(iter(lp | la))
    return None


class CellState:
    __slots__ = ("sig", "val", "cand", "fails", "last_read", "text")

    def __init__(self):
        self.sig, self.val, self.cand, self.fails, self.last_read, self.text = None, None, None, 0, 0.0, ""


# ------------------------------------------------------------------ OCR worker
class OcrWorker(threading.Thread):
    def __init__(self, shared, settings, ik, bank):
        super().__init__(daemon=True, name="ocr")
        self.sh, self.settings, self.ik, self.bank = shared, settings, ik, bank
        self.stop_evt = threading.Event()

    def stop(self):
        self.stop_evt.set()

    def run(self):
        try:
            self._loop()
        except Exception as exc:                                  # pragma: no cover
            with self.sh.lock:
                self.sh.state = f"OCR stopped: {exc}"

    def _update_cell(self, st, i, g, t0):
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        changed = st.sig is None or st.sig.shape != b.shape or np.count_nonzero(st.sig != b) > SIG_CHANGE_FRAC * b.size
        if changed:
            st.sig = b
        if not (changed or st.val is None or st.cand is not None or t0 - st.last_read > FORCE_REREAD_S):
            return 0
        r = digits.read_cell(g, i // 3, self.bank)
        st.last_read = t0
        if r is None:
            st.fails += 1
            if st.fails >= 4:
                st.val, st.cand = None, None
            return 1
        st.fails = 0
        v = r.value
        if r.sign_missing:                                        # '+'/'-' glyph not found: pick the sign nearest the last value
            if st.val is None:
                return 1
            cands = [abs(v), -abs(v)]
            d = [abs(c - st.val) if i < 3 else abs(wrap180(c - st.val)) for c in cands]
            k = int(np.argmin(d))
            if d[k] > (2.5 if i < 3 else 8.0):
                return 1
            v = cands[k]
        st.text = r.text
        if st.val is not None and abs(v - st.val) < 1e-6:
            st.cand = None
        elif st.cand is not None and abs(v - st.cand) < 1e-6:     # same new value twice in a row -> accept
            st.val, st.cand = v, None
            if self.settings.get("self_learn", True):
                self.bank.learn_from(r)
        else:
            st.cand = v
        return 1

    def _loop(self):
        sh = self.sh
        sct = new_mss()
        cells = [CellState() for _ in range(6)]
        last_region = None
        scans = calls = 0
        rate_t = clock()
        while not self.stop_evt.is_set():
            t0 = clock()
            with sh.lock:
                region, sreg, preg = sh.region, sh.status_region, sh.step_region
            if region is None:
                self.stop_evt.wait(0.2)
                continue
            if region != last_region:
                cells, last_region = [CellState() for _ in range(6)], region
            try:
                _, gray = grab_gray(sct, region)
            except Exception as exc:
                with sh.lock:
                    sh.state = f"Screen capture failed: {exc}"
                self.stop_evt.wait(0.5)
                continue
            for i, g in enumerate(split_cells(gray)):
                if g.size:
                    calls += self._update_cell(cells[i], i, g, t0)
            busy = None
            if sreg is not None:
                try:
                    busy = classify_banner(np.asarray(sct.grab(sreg)))
                except Exception:
                    busy = None
            now = clock()
            with sh.lock:
                if sreg is None:
                    sh.busy = None
                else:
                    sh.busy = busy
                    if busy:
                        sh.last_moving_t = now
                sh.raw = " | ".join(c.text for c in cells)
                if not all(c.val is not None for c in cells):
                    sh.state = "Reading the table\u2026"
                acc = sh.pose
            if all(c.val is not None for c in cells):
                pose = np.array([c.val for c in cells], float)
                if acc is None or np.any(np.abs(acc - pose) > 1e-9):
                    if sh.offer_pose(pose):
                        self.ik.request("live", pose)
                else:
                    with sh.lock:
                        sh.t_read, sh.state = now, "Live"
            scans += 1
            if now - rate_t >= 1.0:
                with sh.lock:
                    sh.scan_rate, sh.ocr_rate = scans / (now - rate_t), calls / (now - rate_t)
                scans = calls = 0
                rate_t = now
            delay = 1.0 / max(1.0, float(self.settings.get("ocr_fps", 15))) - (clock() - t0)
            if delay > 0:
                self.stop_evt.wait(delay)


class StepReader(threading.Thread):
    """Read the optional panel step-size text continuously, separate from pose OCR."""
    def __init__(self, shared):
        super().__init__(daemon=True, name="step-ocr")
        self.sh = shared
        self.stop_evt = threading.Event()

    def stop(self):
        self.stop_evt.set()

    def run(self):
        sh = self.sh
        if not HAVE_TESS:
            with sh.lock: sh.step_status = "Install pytesseract and the Tesseract executable to read panel step size"
            return
        cmd = getattr(getattr(pytesseract, "pytesseract", None), "tesseract_cmd", "tesseract")
        if not (os.path.isfile(cmd) or shutil.which(cmd)):
            with sh.lock: sh.step_status = "Tesseract executable not found; panel step size is not being read"
            return
        try:
            sct = new_mss()
        except Exception as exc:
            with sh.lock: sh.step_status = f"Step-size capture unavailable: {exc}"
            return
        while not self.stop_evt.is_set():
            with sh.lock: region = sh.step_region
            if region is None:
                with sh.lock:
                    sh.step_level, sh.step_t = None, 0.0
                    sh.step_status = "Step-size region not configured"
                self.stop_evt.wait(0.4)
                continue
            try:
                gray = cv2.cvtColor(np.asarray(sct.grab(region)), cv2.COLOR_BGRA2GRAY)
                enlarged = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                _, binary = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if binary.mean() < 127: binary = 255 - binary
                binary = cv2.copyMakeBorder(binary, 20, 20, 30, 30, cv2.BORDER_CONSTANT, value=255)
                text = pytesseract.image_to_string(binary, config="--oem 3 --psm 7", timeout=0.75)
                level = parse_step_level(text)
                with sh.lock:
                    if level is not None:
                        sh.step_level, sh.step_t = level, clock()
                        sh.step_status = "Step-size OCR current"
                    elif sh.step_t and clock() - sh.step_t > 3.0:
                        sh.step_level = None
                        sh.step_status = "Step-size OCR running; no current match"
                    elif sh.step_t:
                        sh.step_status = "Step-size OCR running; keeping recent match"
                    else:
                        sh.step_status = "Step-size OCR running; waiting for a match"
            except Exception as exc:
                with sh.lock:
                    if sh.step_t and clock() - sh.step_t > 3.0: sh.step_level = None
                    sh.step_status = f"Step-size OCR error: {exc}"
            self.stop_evt.wait(0.5)


# ------------------------------------------------------------------ IK worker
class IKWorker(threading.Thread):
    def __init__(self, shared):
        super().__init__(daemon=True, name="ik")
        self.sh = shared
        self.evt, self.req, self.prev = threading.Event(), {}, {}
        self.stop_evt, self.rlock = threading.Event(), threading.Lock()
        self.last_live = None

    def request(self, key, pose):
        pose = np.asarray(pose, float)
        if key == "live":
            if self.last_live is not None and np.array_equal(self.last_live, pose):
                return
            self.last_live = pose.copy()
        with self.rlock:
            self.req[key] = pose.copy()
        self.evt.set()

    def stop(self):
        self.stop_evt.set()
        self.evt.set()

    def run(self):
        while not self.stop_evt.is_set():
            self.evt.wait(0.5)
            self.evt.clear()
            with self.rlock:
                reqs, self.req = self.req, {}
            for key, pose in reqs.items():
                try:
                    q, pe, re = K.solve_ik(K.table_pose_to_target(pose), self.prev.get(key))
                    self.prev[key] = q
                    res = SimpleNamespace(q=q, pe=pe * 1000.0, re=float(np.degrees(re)), pose=pose, t=clock())
                    with self.sh.lock:
                        self.sh.ik[key] = res
                except Exception:
                    traceback.print_exc()
