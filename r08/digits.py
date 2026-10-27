"""
digits.py - reads the +65.3 / +170 style numbers off the stream WITHOUT Tesseract.

Why: Tesseract confuses 5/9 etc. on tiny video text.  The overlay always uses ONE font, so a small
template classifier trained on that font is far more reliable and ~100x faster.

Pipeline for one table cell (grayscale crop):
    segment(gray)            -> list[Glyph]   (upscale, threshold, delete table lines, connected components,
                                               classify each blob as '+', '-', '.', or 'digit')
    DigitBank.match(feat)    -> (digit, score, margin)   (nearest template; templates = seeds + learned)
    read_cell(gray,row,bank) -> Reading | None  (checks the cell layout, builds the number)

Layout rules (that is what makes it robust):
    row 0 (position):  sign + digits + '.' + ONE decimal       e.g. +65.3
    row 1 (angle):     sign + whole degrees                    e.g. +170
    A cell that does not match the layout returns None (the caller keeps the last good value).

Template bank:
    * seeds   = digits rendered from system fonts (DejaVu/Verdana/Arial/Pillow default...) - never saved
    * learned = real glyphs from YOUR screen - saved in r08_digits.npz
      learned in two ways:  (1) teach(): you type what the stream shows (UI: "Teach digits")  <- most reliable
                            (2) self-learning: very confident reads add new samples (setting self_learn)
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
import cv2

GW, GH = 16, 24                  # template size
MIN_CONTRAST = 55.0              # text must be at least this much brighter than the cell background
ACCEPT_SCORE = 0.45              # every digit must match at least this well
POS_MAX, ANG_MAX = 150.0, 360.0
LEARN_CAP = 40                   # learned samples kept per digit
_K3 = np.ones((3, 3), np.uint8)


@dataclass
class Glyph:
    x: int
    y: int
    w: int
    h: int
    kind: str                    # 'digit' '+' '-' '.'
    feat: Optional[tuple] = None
    label: int = -1
    score: float = 0.0
    margin: float = 0.0


@dataclass
class Reading:
    value: float
    text: str
    sign_missing: bool
    score: float
    glyphs: List[Glyph] = field(default_factory=list)


# ------------------------------------------------------------------ features
def glyph_features(gn, comp):
    """gn: float image 0..1 around ONE glyph, comp: uint8/bool mask of that glyph's core pixels.
    -> (unit vector length GW*GH, aspect ratio w/h) or None.  Used for BOTH seeds and real glyphs."""
    mask = (cv2.dilate(comp.astype(np.uint8), _K3) > 0) & (gn > 0.15)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = (gn * mask)[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    h, w = crop.shape
    if h < 2:
        return None
    small = cv2.resize(crop, (GW, GH), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (3, 3), 0.8)
    v = small.ravel() - small.mean()
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return None
    return (v / n).astype(np.float32), float(w) / h


# ------------------------------------------------------------------ seeds
def _font_makers():
    try:
        from PIL import ImageFont
    except Exception:
        return []
    names = ["DejaVuSans.ttf", "verdana.ttf", "segoeui.ttf", "arial.ttf", "tahoma.ttf", "LiberationSans-Regular.ttf"]
    try:
        import matplotlib
        names.insert(0, os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"))
    except Exception:
        pass
    out = []
    for n in names:
        try:
            ImageFont.truetype(n, 20)
            out.append(lambda s, n=n: ImageFont.truetype(n, s))
        except Exception:
            continue
        if len(out) >= 4:
            break
    try:
        ImageFont.load_default(size=20)
        out.append(lambda s: ImageFont.load_default(size=s))
    except Exception:
        pass
    return out


def _render_char(ch, maker, size):
    if maker is None:                                    # OpenCV Hershey fallback
        img = np.zeros((size * 2, size * 2), np.uint8)
        cv2.putText(img, ch, (size // 3, int(size * 1.4)), cv2.FONT_HERSHEY_SIMPLEX, size / 22.0, 255, max(2, size // 10), cv2.LINE_AA)
        return img.astype(np.float32) / 255.0
    from PIL import Image, ImageDraw
    im = Image.new("L", (size * 2, size * 2), 0)
    ImageDraw.Draw(im).text((size // 2, size // 4), ch, fill=255, font=maker(size))
    return np.asarray(im, np.float32) / 255.0


def build_seeds():
    makers = _font_makers() or [None]
    V, A, L = [], [], []
    for mk in makers:
        for size in (32, 52):
            for sigma in (0.0, 1.3):
                for d in range(10):
                    try:
                        img = _render_char(str(d), mk, size)
                    except Exception:
                        continue
                    if sigma:
                        img = cv2.GaussianBlur(img, (0, 0), sigma)
                    m = float(img.max())
                    if m < 0.2:
                        continue
                    img = img / m
                    f = glyph_features(img, img > 0.5)
                    if f is not None:
                        V.append(f[0]); A.append(f[1]); L.append(d)
    return V, A, L


# ------------------------------------------------------------------ template bank
class DigitBank:
    def __init__(self, path):
        self.path = path
        sv, sa, sl = build_seeds()
        self.seed = (sv, sa, sl)
        self.lv, self.la, self.ll = [], [], []
        self.dirty, self.n_self = False, 0
        self._load()
        self._rebuild()

    def _rebuild(self):
        V = list(self.seed[0]) + self.lv
        self.M = np.array(V, np.float32)
        self.A = np.array(list(self.seed[1]) + self.la, np.float32)
        self.L = np.array(list(self.seed[2]) + self.ll, np.int64)
        self.bonus = np.array([0.0] * len(self.seed[0]) + [0.03] * len(self.lv), np.float32)   # real samples win ties

    def match(self, vec, ar):
        """-> (digit, score, margin).  score = best correlation minus a small aspect-ratio penalty."""
        s = self.M @ vec - 0.4 * np.abs(self.A - ar) + self.bonus
        best = np.full(10, -9.0, np.float32)
        np.maximum.at(best, self.L, s)
        o = np.argsort(best)[::-1]
        return int(o[0]), float(best[o[0]]), float(best[o[0]] - best[o[1]])

    def learn(self, digit, vec, ar, force=False):
        idx = [i for i, l in enumerate(self.ll) if l == digit]
        if not force and idx and max(float(self.lv[i] @ vec) for i in idx) > 0.985:
            return False                                       # already covered
        self.lv.append(vec); self.la.append(ar); self.ll.append(digit)
        idx = [i for i, l in enumerate(self.ll) if l == digit]
        if len(idx) > LEARN_CAP:
            for i in sorted(idx[:len(idx) - LEARN_CAP], reverse=True):
                del self.lv[i], self.la[i], self.ll[i]
        self._rebuild()
        self.dirty = True
        return True

    def learn_from(self, reading):
        """Self-learning from a confidently read cell."""
        n = 0
        for g in reading.glyphs:
            if g.kind == 'digit' and g.feat is not None and g.score >= 0.92 and g.margin >= 0.15:
                if self.learn(g.label, g.feat[0], g.feat[1]):
                    n += 1
        self.n_self += n
        return n

    def reset_learned(self):
        self.lv, self.la, self.ll = [], [], []
        self._rebuild()
        self.dirty = False
        try:
            os.remove(self.path)
        except OSError:
            pass

    def stats(self):
        counts = [self.ll.count(d) for d in range(10)]
        return f"{len(self.ll)} learned samples  ({' '.join(str(c) for c in counts)})   seeds {len(self.seed[0])}"

    def save(self):
        if not self.dirty:
            return
        try:
            np.savez(self.path, v=np.array(self.lv, np.float32).reshape(-1, GW * GH),
                     a=np.array(self.la, np.float32), l=np.array(self.ll, np.int64))
            self.dirty = False
        except Exception as exc:
            print("Could not save digit bank:", exc)

    def _load(self):
        try:
            z = np.load(self.path)
            v, a, l = z["v"], z["a"], z["l"]
            if v.ndim == 2 and v.shape[1] == GW * GH:
                self.lv, self.la, self.ll = [x for x in v], [float(x) for x in a], [int(x) for x in l]
        except Exception:
            pass


# ------------------------------------------------------------------ segmentation
def upscale_factor(h):
    return 3 if h < 80 else (2 if h < 140 else 1)


def segment(gray):
    """Grayscale cell -> list[Glyph] left to right (kinds: '+','-','.','digit') or None."""
    if gray is None or gray.ndim != 2:
        return None
    h0, w0 = gray.shape
    if h0 < 10 or w0 < 24:
        return None
    s = upscale_factor(h0)
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s > 1 else gray
    g = g.astype(np.float32)
    lo, hi = float(np.median(g)), float(np.percentile(g, 99.5))
    if hi - lo < MIN_CONTRAST:
        return None
    gn = np.clip((g - lo) / (hi - lo), 0, 1)
    b = (gn > 0.5).astype(np.uint8)
    H, W = b.shape
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(0.6 * W)), 1))     # table grid lines
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(0.8 * H))))
    lines = cv2.morphologyEx(b, cv2.MORPH_OPEN, kh) | cv2.morphologyEx(b, cv2.MORPH_OPEN, kv)
    if lines.any():
        b[cv2.dilate(lines, _K3) > 0] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    amin = max(3, 2 * s * s)
    comps = []
    for k in range(1, n):
        x, y, w, h, a = (int(v) for v in st[k])
        if a < amin or w > 0.9 * W or h > 0.97 * H:
            continue
        comps.append({"x": x, "y": y, "w": w, "h": h, "ids": [k]})
    if not comps:
        return None
    tall = max(c["h"] for c in comps)
    if tall < 0.2 * H:
        return None
    big = [c for c in comps if c["h"] >= 0.6 * tall]
    top, bot = min(c["y"] for c in big), max(c["y"] + c["h"] for c in big)
    comps = [c for c in comps if top - 0.15 * tall <= c["y"] + c["h"] / 2 <= bot + 0.15 * tall]
    comps.sort(key=lambda c: c["x"])
    merged = []                                                   # re-join glyphs the video compression broke in two
    for c in comps:
        if merged:
            p = merged[-1]
            ov = min(p["x"] + p["w"], c["x"] + c["w"]) - max(p["x"], c["x"])
            if ov > 0.6 * min(p["w"], c["w"]) and min(p["h"], c["h"]) > 0.3 * tall:
                y0, y1 = min(p["y"], c["y"]), max(p["y"] + p["h"], c["y"] + c["h"])
                if y1 - y0 <= 1.2 * tall:
                    x0, x1 = min(p["x"], c["x"]), max(p["x"] + p["w"], c["x"] + c["w"])
                    p.update(x=x0, w=x1 - x0, y=y0, h=y1 - y0, ids=p["ids"] + c["ids"])
                    continue
        merged.append(c)

    def feat_for(x, y, w, h, ids, xcut=None):
        pad = 2
        ya, yb, xa, xb = max(0, y - pad), min(H, y + h + pad), max(0, x - pad), min(W, x + w + pad)
        sub = lab[ya:yb, xa:xb]
        comp = np.isin(sub, ids)
        if xcut is not None:                                      # restrict to a column range (merged digits)
            keep = np.zeros_like(comp)
            keep[:, max(0, xcut[0] - xa):max(0, xcut[1] - xa)] = True
            comp &= keep
        return glyph_features(gn[ya:yb, xa:xb], comp)

    glyphs = []
    for c in merged:
        x, y, w, h = c["x"], c["y"], c["w"], c["h"]
        hh = h / tall
        if hh < 0.3:
            if w >= 1.5 * h and w >= 0.2 * tall:
                glyphs.append(Glyph(x, y, w, h, '-'))
            elif y + h >= bot - 0.25 * tall and w <= 0.4 * tall:
                glyphs.append(Glyph(x, y, w, h, '.'))
        elif hh < 0.86:
            if 0.7 <= w / h <= 1.4:
                glyphs.append(Glyph(x, y, w, h, '+'))
        else:
            pieces = [(x, x + w)]
            if w / h > 1.05:                                      # two digits glued together
                nn = max(2, int(round(w / (0.62 * h))))
                cuts = [x]
                sub = np.isin(lab[y:y + h, x:x + w], c["ids"]).sum(axis=0)
                for j in range(1, nn):
                    e = int(w * j / nn)
                    r = max(1, int(0.15 * w))
                    lo_, hi_ = max(1, e - r), min(w - 1, e + r)
                    cuts.append(x + lo_ + int(np.argmin(sub[lo_:hi_])) if hi_ > lo_ else x + e)
                cuts.append(x + w)
                pieces = list(zip(cuts[:-1], cuts[1:]))
            for (xa, xb) in pieces:
                gl = Glyph(xa, y, xb - xa, h, 'digit')
                gl.feat = feat_for(x, y, w, h, c["ids"], None if len(pieces) == 1 else (xa, xb))
                glyphs.append(gl)
    return glyphs or None


# ------------------------------------------------------------------ reading
def read_cell(gray, row, bank):
    """-> Reading or None.  row 0 = position (one decimal), row 1 = angle (whole degrees)."""
    glyphs = segment(gray)
    if not glyphs:
        return None
    sign, rest = None, glyphs
    if glyphs[0].kind in '+-':
        sign, rest = glyphs[0].kind, glyphs[1:]
    if not rest or any(g.kind in '+-' for g in rest):
        return None
    dots = [i for i, g in enumerate(rest) if g.kind == '.']
    digs = [g for g in rest if g.kind == 'digit']
    if len(dots) > 1 or not digs or any(g.feat is None for g in digs):
        return None
    if dots and (dots[0] == 0 or dots[0] == len(rest) - 1):
        return None
    for g in digs:
        g.label, g.score, g.margin = bank.match(*g.feat)
    score = min(g.score for g in digs)
    if score < ACCEPT_SCORE:
        return None
    s = "".join(str(g.label) if g.kind == 'digit' else '.' for g in rest)
    if '.' in s:
        if s.index('.') != len(s) - 2:                            # exactly one decimal
            return None
    elif row == 0:
        if len(s) < 2:
            return None
        s = s[:-1] + '.' + s[-1:]                                 # tiny dot lost in compression: position has 1 decimal
    if row == 1 and len(s.split('.')[0]) > 3:
        return None
    v = float(s)
    if v > (POS_MAX if row == 0 else ANG_MAX):
        return None
    if sign == '-':
        v = -v
    return Reading(round(v, 1) + 0.0, (sign or '?') + s, sign is None, score, glyphs)


def teach(gray, row, truth, bank):
    """Learn from a cell whose true text you typed (e.g. '+65.3').  -> (samples added, message)."""
    glyphs = segment(gray)
    if not glyphs:
        return 0, "no text found in the cell"
    t = truth.strip().replace("\u2212", "-").replace(" ", "")
    exp = []
    for ch in t:
        if ch in "+-.":
            exp.append(ch)
        elif ch.isdigit():
            exp.append('digit')
        else:
            return 0, f"'{ch}' is not allowed"
    kinds = [g.kind for g in glyphs]
    if kinds != exp:
        return 0, f"found {describe(glyphs)} but you typed {t}"
    n = 0
    for g, ch in zip(glyphs, t):
        if g.kind == 'digit' and g.feat is not None:
            bank.learn(int(ch), g.feat[0], g.feat[1], force=True)
            n += 1
    return n, "ok"


def describe(glyphs):
    return "".join(g.kind if g.kind != 'digit' else (str(g.label) if g.label >= 0 else '#') for g in glyphs) or "-"
