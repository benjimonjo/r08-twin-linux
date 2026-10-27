"""
model3d.py - everything GEOMETRIC for drawing (no OpenGL / no OpenCV drawing of the scene here).

    Camera / View        orbit camera and the pinhole projection used by BOTH renderers and the 2D overlay
    RobotMesh            R-2000iA/200F built from tapered boxes + cylinders; update(q) moves it,
                         triangles() returns a triangle soup  (pos (N,3), normal (N,3), colour (N,3)) float32
    plate_solid / pipe_solid   mapped objects -> triangle soup
    floor_lines()        ground grid + axes as line segments

Triangle soup = what render.py consumes.  Normals always point OUTWARD (flipped using the part centre),
cylinder sides get smooth radial normals.  To change how the robot looks edit PART_DEFS and RobotMesh.update.
Colours in the soup are 0..1 floats.
"""
import math
import numpy as np
import cv2
from .common import C, IN, hex_rgb
from . import kinematics as K

SEG = 16
_ANG = np.linspace(0.0, 2 * np.pi, SEG, endpoint=False)
_COS, _SIN = np.cos(_ANG)[:, None], np.sin(_ANG)[:, None]
PRISM_FACES = [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]]

PLATE_W, PLATE_T = 0.24, 0.05
SPIKE_R, TIP_LEN, TIP_R = 0.0115, 0.055, 0.0015
Y, YD, BK, GR, RD, TIPC = C.yellow, C.yellow_d, C.black, C.grey, C.rob_red, "#d9d2b0"

# ------------------------------------------------------------------ camera
class Camera:
    PRESETS = {"iso": (-52.0, 22.0), "front": (-90.0, 6.0), "side": (0.0, 6.0), "top": (-90.0, 88.0)}

    def __init__(self):
        self.fov = 38.0
        self.reset()

    def reset(self):
        self.azim, self.elev, self.dist = -52.0, 22.0, 8.8
        self.target = np.array([1.2, 0.0, 0.9])

    def preset(self, name):
        self.azim, self.elev = self.PRESETS[name]

    def basis(self):
        az, el = math.radians(self.azim), math.radians(self.elev)
        off = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        eye = self.target + self.dist * off
        f = -off
        right = np.cross(f, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, f)
        return eye, f, right, up

    def orbit(self, dx, dy):
        self.azim -= dx * 0.40
        self.elev = float(np.clip(self.elev + dy * 0.40, -8.0, 89.0))

    def pan(self, dx, dy, h_px):
        _, _, right, up = self.basis()
        k = 2.0 * self.dist * math.tan(math.radians(self.fov) / 2) / max(h_px, 1)
        self.target = self.target - right * dx * k + up * dy * k

    def zoom(self, notches):
        self.dist = float(np.clip(self.dist * (0.88 ** notches), 1.2, 30.0))


class View:
    """Pinhole projection for one frame.  project(P) -> (pixels (N,2), camera depth (N,))."""

    def __init__(self, cam, w, h):
        self.eye, self.f, self.r, self.u = cam.basis()
        self.B = np.stack([self.r, self.u, self.f])
        self.fov, self.w, self.h = cam.fov, w, h
        self.F = (h / 2) / math.tan(math.radians(cam.fov) / 2)
        self.cx, self.cy = w / 2, h / 2
        self.lt = cv2.LINE_AA
        L = -self.f * 0.65 + self.u * 0.65 - self.r * 0.40
        self.light = L / np.linalg.norm(L)

    def project(self, P):
        c = (np.atleast_2d(P) - self.eye) @ self.B.T
        d = c[:, 2]
        dd = np.maximum(d, 1e-4)
        return np.stack([self.cx + c[:, 0] * self.F / dd, self.cy - c[:, 1] * self.F / dd], 1), d

    def mvp(self, near=0.1, far=80.0):
        """OpenGL model-view-projection (row-major 4x4; transpose before uploading)."""
        V = np.eye(4)
        V[0, :3], V[1, :3], V[2, :3] = self.r, self.u, -self.f
        V[:3, 3] = -V[:3, :3] @ self.eye
        t = 1.0 / math.tan(math.radians(self.fov) / 2)
        P = np.zeros((4, 4))
        P[0, 0], P[1, 1] = t / (self.w / self.h), t
        P[2, 2], P[2, 3], P[3, 2] = (far + near) / (near - far), 2 * far * near / (near - far), -1.0
        return P @ V

    def seg(self, img, a, b, col, th):
        a, b = np.asarray(a, float), np.asarray(b, float)
        pts, d = self.project(np.array([a, b]))
        near = 0.12
        if d[0] < near and d[1] < near:
            return
        if d[0] < near or d[1] < near:
            t = (near - d[0]) / (d[1] - d[0])
            p = a + t * (b - a)
            if d[0] < near:
                a = p
            else:
                b = p
            pts, d = self.project(np.array([a, b]))
        cv2.line(img, (int(pts[0, 0]), int(pts[0, 1])), (int(pts[1, 0]), int(pts[1, 1])), col, int(th), self.lt)


# ------------------------------------------------------------------ primitive vertex builders
def _basis(d):
    n = np.linalg.norm(d)
    d = np.array([1.0, 0.0, 0.0]) if n < 1e-9 else d / n
    helper = np.array([0.0, 0.0, 1.0]) if abs(d[2]) <= 0.92 else np.array([0.0, 1.0, 0.0])
    side = np.cross(d, helper)
    side /= np.linalg.norm(side)
    up = np.cross(side, d)
    up /= np.linalg.norm(up)
    return side, up


def box_verts(p1, p2, side, up, w1, h1, w2, h2):
    """Tapered box from p1 to p2 with explicit orientation (side = width axis, up = height axis)."""
    s1, u1, s2, u2 = side * w1 / 2, up * h1 / 2, side * w2 / 2, up * h2 / 2
    return np.array([p1 - s1 - u1, p1 + s1 - u1, p1 + s1 + u1, p1 - s1 + u1,
                     p2 - s2 - u2, p2 + s2 - u2, p2 + s2 + u2, p2 - s2 + u2])


def prism_lat(p1, p2, lat, w1, h1, w2, h2):
    """Tapered box whose WIDTH runs along `lat` (e.g. a joint axis) - keeps its orientation as the arm tilts."""
    d = p2 - p1
    n = np.linalg.norm(d)
    d = d / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    side = lat - d * float(np.dot(lat, d))
    ns = np.linalg.norm(side)
    side = _basis(d)[0] if ns < 1e-6 else side / ns
    up = np.cross(side, d)
    up /= np.linalg.norm(up)
    return box_verts(p1, p2, side, up, w1, h1, w2, h2)


def cyl_verts(p1, p2, r1, r2=None):
    r2 = r1 if r2 is None else r2
    side, up = _basis(p2 - p1)
    radial = _COS * side + _SIN * up
    return np.vstack([p1 + radial * r1, p2 + radial * r2])


def cyl_center(c, axis, r, length):
    a = axis / np.linalg.norm(axis)
    return cyl_verts(c - a * length / 2, c + a * length / 2, r)


# ------------------------------------------------------------------ robot mesh
def _chain_names(name, n=3):
    return [f"{name}_{k}" for k in range(n)]


PART_DEFS = (
    [("base_plate", "prism", GR), ("base_ring", "cyl", BK), ("column", "prism", Y), ("bridge", "prism", Y),
     ("j2", "cyl", YD), ("j2_ring", "cyl", BK)]
    + [(n, "prism", Y) for n in _chain_names("lower_arm")]
    + [("j3", "cyl", YD), ("j3_ring", "cyl", BK), ("elbow", "prism", Y), ("j4_motor", "prism", BK), ("j4_red", "prism", RD),
       ("forearm", "cyl", Y), ("collar", "cyl", YD), ("j4_ring", "cyl", BK),
       ("j5", "cyl", Y), ("j5_ring", "cyl", BK), ("j5_cap", "cyl", YD), ("j6a", "cyl", Y), ("j6b", "cyl", BK),
       ("tool_plate", "prism", Y)]
    + [(f"spike_{k}", "cyl", Y) for k in range(3)] + [(f"spike_{k}_tip", "cyl", TIPC) for k in range(3)]
)


class RobotMesh:
    def __init__(self):
        self.slices, total = {}, 0
        tri, part, smooth, cols = [], [], [], []
        cyl_starts, starts = [], []
        for pid, (name, kind, color) in enumerate(PART_DEFS):
            rgb = np.array(hex_rgb(color), np.float32) / 255.0
            starts.append(total)
            if kind == "prism":
                n = 8
                for f in PRISM_FACES:
                    for a, b, c in ((f[0], f[1], f[2]), (f[0], f[2], f[3])):
                        tri.append([total + a, total + b, total + c]); part.append(pid); smooth.append(False); cols.append(rgb)
            else:
                n = 2 * SEG
                cyl_starts.append(total)
                for i in range(SEG):
                    j = (i + 1) % SEG
                    for t in ((i, j, SEG + j), (i, SEG + j, SEG + i)):
                        tri.append([total + t[0], total + t[1], total + t[2]]); part.append(pid); smooth.append(True); cols.append(rgb)
                for off in (0, SEG):                               # caps (fans)
                    for k in range(1, SEG - 1):
                        tri.append([total + off, total + off + k, total + off + k + 1]); part.append(pid); smooth.append(False); cols.append(rgb)
            self.slices[name] = slice(total, total + n)
            total += n
        self.V = np.zeros((total, 3))
        self.tri = np.array(tri)
        self.tri_part = np.array(part)
        self.tri_smooth = np.array(smooth)
        self.tri_col = np.repeat(np.array(cols, np.float32), 3, axis=0)
        self.starts = np.array(starts)
        self.counts = np.array([8 if k == "prism" else 2 * SEG for _, k, _ in PART_DEFS])
        self.cyl_vidx = np.array(cyl_starts)[:, None] + np.arange(2 * SEG)[None, :]
        self.VN = np.zeros((total, 3))
        self.centers = np.zeros((len(PART_DEFS), 3))
        V, S = self.V, self.slices
        H = K.J2_HEIGHT
        V[S["base_plate"]] = box_verts(np.array([0, 0, 0.0]), np.array([0, 0, 0.19]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), 1.0, 0.92, 0.92, 0.84)
        V[S["base_ring"]] = cyl_verts(np.array([0, 0, 0.18]), np.array([0, 0, 0.30]), 0.42)
        self.update(np.zeros(6))

    # ---- pose
    def update(self, q):
        _, J, A, Tf, tip = K.forward_kinematics(q)
        j1, j2, j3, j4, j5, j6 = J
        fl, Rm = Tf[:3, 3], Tf[:3, :3]
        V, S = self.V, self.slices
        H = K.J2_HEIGHT
        lat1, lat2, lat5 = A[1], A[2], A[4]                       # J2 / J3 / J5 axes in the world
        fd = j5 - j4
        fd = fd / (np.linalg.norm(fd) + 1e-12)
        ta = Rm[:, 0]
        lat = Rm[:, K.TOOL["axis"]] * K.TOOL["sign"]
        thk = Rm[:, K.thick_axis()]
        V[S["column"]] = prism_lat(np.array([0, 0, 0.28]), j1, lat1, 0.66, 0.62, 0.58, 0.56)
        V[S["bridge"]] = prism_lat(j1, j2, lat1, 0.60, 0.70, 0.58, 0.68)
        V[S["j2"]] = cyl_center(j2, lat1, 0.27, 0.70)
        V[S["j2_ring"]] = cyl_center(j2, lat1, 0.20, 0.725)
        W = [0.46, 0.42, 0.36, 0.30]
        Hh = [0.52, 0.44, 0.38, 0.34]
        for k in range(3):
            t0, t1 = k / 3.0, (k + 1) / 3.0
            V[S[f"lower_arm_{k}"]] = prism_lat(j2 + (j3 - j2) * t0, j2 + (j3 - j2) * t1, lat1, W[k], Hh[k], W[k + 1], Hh[k + 1])
        V[S["j3"]] = cyl_center(j3, lat2, 0.24, 0.54)
        V[S["j3_ring"]] = cyl_center(j3, lat2, 0.18, 0.565)
        V[S["elbow"]] = prism_lat(j3, j4, lat2, 0.46, 0.46, 0.38, 0.38)
        V[S["j4_motor"]] = prism_lat(j4 - fd * 0.62, j4 - fd * 0.10, lat2, 0.36, 0.36, 0.32, 0.32)
        V[S["j4_red"]] = prism_lat(j4 - fd * 0.84, j4 - fd * 0.60, lat2, 0.28, 0.28, 0.30, 0.30)
        V[S["forearm"]] = cyl_verts(j4, j5, 0.125, 0.095)
        V[S["collar"]] = cyl_verts(j4 - fd * 0.02, j4 + fd * 0.30, 0.155, 0.14)
        V[S["j4_ring"]] = cyl_verts(j4 + fd * 0.30, j4 + fd * 0.345, 0.15)
        V[S["j5"]] = cyl_center(j5, lat5, 0.175, 0.30)
        V[S["j5_ring"]] = cyl_center(j5, lat5, 0.14, 0.34)
        V[S["j5_cap"]] = cyl_center(j5, lat5, 0.105, 0.37)
        V[S["j6a"]] = cyl_verts(j5, j5 + ta * 0.09, 0.15, 0.14)
        V[S["j6b"]] = cyl_verts(j5 + ta * 0.09, fl, 0.125, 0.12)
        V[S["tool_plate"]] = box_verts(fl - ta * 0.012, fl + ta * 0.016, lat, thk, PLATE_W, PLATE_T, PLATE_W, PLATE_T)
        for k, (L, off) in enumerate(K.SPIKES_IN):
            c = fl + lat * off * IN
            end = c + ta * L * IN
            mid = end - ta * TIP_LEN
            V[S[f"spike_{k}"]] = cyl_verts(c + ta * 0.016, mid, SPIKE_R, SPIKE_R)
            V[S[f"spike_{k}_tip"]] = cyl_verts(mid, end, SPIKE_R, TIP_R)
        return type("Info", (), {"joints": J, "flange": fl, "tip": tip})()

    # ---- triangle soup
    def triangles(self):
        V = self.V
        self.centers = np.add.reduceat(V, self.starts, axis=0) / self.counts[:, None]
        ring = V[self.cyl_vidx].reshape(-1, 2, SEG, 3)               # smooth radial normals for cylinder sides
        c1, c2 = ring[:, 0].mean(axis=1), ring[:, 1].mean(axis=1)
        a = c2 - c1
        an = np.linalg.norm(a, axis=1, keepdims=True)
        a = np.where(an > 1e-9, a / np.maximum(an, 1e-9), np.array([0.0, 0.0, 1.0]))
        rel = ring - c1[:, None, None, :]
        rel = rel - np.einsum("nsij,nj->nsi", rel, a)[..., None] * a[:, None, None, :]
        rel /= np.linalg.norm(rel, axis=3, keepdims=True) + 1e-12
        self.VN[self.cyl_vidx.ravel()] = rel.reshape(-1, 3)
        P = V[self.tri]
        fn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
        flip = np.einsum("ij,ij->i", fn, P.mean(axis=1) - self.centers[self.tri_part]) < 0
        fn[flip] *= -1
        nrm = np.where(self.tri_smooth[:, None, None], self.VN[self.tri], fn[:, None, :])
        return P.reshape(-1, 3).astype(np.float32), nrm.reshape(-1, 3).astype(np.float32), self.tri_col


# ------------------------------------------------------------------ mapped objects (plates / pipes)
def solid_soup(V, polys, color_rgb255, center=None):
    """Triangulate convex polygons (index lists into V) -> soup with outward flat normals."""
    V = np.asarray(V, float)
    ctr = V.mean(axis=0) if center is None else center
    tri = []
    for f in polys:
        for k in range(1, len(f) - 1):
            tri.append([f[0], f[k], f[k + 1]])
    tri = np.array(tri)
    P = V[tri]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    n[np.einsum("ij,ij->i", n, P.mean(axis=1) - ctr) < 0] *= -1
    nn = np.repeat(n[:, None, :], 3, axis=1)
    col = np.tile(np.array(color_rgb255, np.float32) / 255.0, (len(tri) * 3, 1))
    return P.reshape(-1, 3).astype(np.float32), nn.reshape(-1, 3).astype(np.float32), col


def plate_dims(o):
    P = [np.asarray(x, float) for x in o["pts"][:3]]
    u = P[1] - P[0]
    L = float(np.linalg.norm(u))
    if L < 1e-6:
        return None
    ud = u / L
    w = P[2] - P[0]
    w = w - ud * float(np.dot(w, ud))
    W = float(np.linalg.norm(w))
    return None if W < 1e-6 else (L, W)


def plate_solid(o, selected=False):
    if len(o["pts"]) < 3 or plate_dims(o) is None:
        return None
    p0, p1, p2 = [K.table_to_base(p) for p in o["pts"][:3]]
    u = p1 - p0
    ud = u / np.linalg.norm(u)
    w = p2 - p0
    w = w - ud * float(np.dot(w, ud))
    n = np.cross(ud, w / np.linalg.norm(w))
    if n[2] < 0:
        n = -n
    t = float(o.get("t", 0.5)) * IN
    top = [p0, p0 + u, p0 + u + w, p0 + w]
    # The three recorded points describe the plate's bottom face; build its
    # thickness upward from that face in the world coordinate frame.
    V = np.array(top + [p + n * t for p in top])
    col = (150, 160, 175) if selected else (118, 126, 138)
    return solid_soup(V, PRISM_FACES, col), V.mean(axis=0)


def pipe_solid(o, selected=False):
    if len(o["pts"]) < 2:
        return None
    A, B = K.table_to_base(o["pts"][0]), K.table_to_base(o["pts"][1])
    d = B - A
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return None
    dd = d / L
    up = np.array([0.0, 0.0, 1.0])
    n = up - dd * float(np.dot(up, dd))
    n = np.array([1.0, 0.0, 0.0]) if np.linalg.norm(n) < 1e-6 else n / np.linalg.norm(n)
    outer_r = float(o.get("d", 4.0)) * IN / 2
    inner_r = min(max(0.0, float(o.get("id", 0.0)) * IN / 2), max(0.0, outer_r - 0.005 * IN))
    center_a, center_b = A - n * outer_r, B - n * outer_r
    if inner_r <= 1e-8:
        V = cyl_verts(center_a, center_b, outer_r)
        polys = [[i, (i + 1) % SEG, SEG + (i + 1) % SEG, SEG + i] for i in range(SEG)]
        polys += [list(range(SEG)), list(range(SEG, 2 * SEG))]
        soup = solid_soup(V, polys, (190, 175, 150) if selected else (150, 138, 120))
        return soup, V.mean(axis=0)

    # Four rings make an open-ended tube: outer wall, inner wall and annular
    # rims. Preserve each face's winding so the inside normals face the bore.
    radial = np.array([math.cos(2 * math.pi * i / SEG) * n + math.sin(2 * math.pi * i / SEG) * np.cross(dd, n)
                       for i in range(SEG)])
    V = np.concatenate((center_a + outer_r * radial, center_b + outer_r * radial,
                        center_a + inner_r * radial, center_b + inner_r * radial), axis=0)
    polys = []
    for i in range(SEG):
        j = (i + 1) % SEG
        polys.append([i, j, SEG + j, SEG + i])
        polys.append([2 * SEG + j, 2 * SEG + i, 3 * SEG + i, 3 * SEG + j])
        polys.append([i, 2 * SEG + i, 2 * SEG + j, j])
        polys.append([SEG + i, SEG + j, 3 * SEG + j, 3 * SEG + i])
    tri = []
    for face in polys:
        for k in range(1, len(face) - 1):
            tri.append([face[0], face[k], face[k + 1]])
    tri = np.asarray(tri)
    P = V[tri]
    normals = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    color = (190, 175, 150) if selected else (150, 138, 120)
    colors = np.tile(np.array(color, np.float32) / 255.0, (len(tri) * 3, 1))
    soup = (P.reshape(-1, 3).astype(np.float32), np.repeat(normals[:, None, :], 3, axis=1).reshape(-1, 3).astype(np.float32), colors)
    return soup, V.mean(axis=0)


def obj_summary(o):
    try:
        if o["type"] == "plate":
            L, W = plate_dims(o)
            return f"{o['name']}   {L:.1f} x {W:.1f} x {o.get('t', 0.5):g} in"
        L = float(np.linalg.norm(np.asarray(o["pts"][1], float) - np.asarray(o["pts"][0], float)))
        inner = float(o.get("id", 0.0))
        size = f"OD {o.get('d', 4):g}" + (f" / ID {inner:g}" if inner > 0 else "")
        return f"{o['name']}   {size} x {L:.1f} in"
    except Exception:
        return o.get("name", "?")


# ------------------------------------------------------------------ floor
def floor_lines(extent=6.0, step=0.5, include_axes=True):
    """-> (segments (N,2,3) float32, colours (N,3) 0..1).  Grid + coordinate axes at the base and at the WORLD origin."""
    segs, cols = [], []
    minor, major = np.array([29, 33, 41]) / 255.0, np.array([46, 52, 66]) / 255.0
    n = int(extent / step)
    for i in range(-n, n + 1):
        s = i * step
        c = major if i % 2 == 0 else minor
        segs.append([(s, -extent, 0), (s, extent, 0)]); cols.append(c)
        segs.append([(-extent, s, 0), (extent, s, 0)]); cols.append(c)
    if include_axes:
        axc = [(235, 90, 90), (90, 205, 120), (100, 160, 255)]
        for org, ln in ((np.zeros(3), 0.55), (K.TABLE_ORIGIN_IN_BASE_M, 0.30)):
            for k in range(3):
                e = np.zeros(3)
                e[k] = ln
                segs.append([tuple(org), tuple(org + e)]); cols.append(np.array(axc[k]) / 255.0)
    return np.array(segs, np.float32), np.array(cols, np.float32)
