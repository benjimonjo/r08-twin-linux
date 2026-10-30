"""
model3d.py - everything GEOMETRIC for drawing (no OpenGL / no OpenCV drawing of the scene here).

    Camera / View        orbit camera and the pinhole projection used by BOTH renderers and the 2D overlay
    RobotMesh          R-2000iA/165F loaded from the supplied original STL; update(q) updates its pose metadata,
                         triangles() returns a triangle soup  (pos (N,3), normal (N,3), colour (N,3)) float32
    plate_solid / pipe_solid   mapped objects -> triangle soup
    floor_lines()        ground grid + axes as line segments

Triangle soup = what render.py consumes.  Normals always point OUTWARD (flipped using the part centre),
cylinder sides get smooth radial normals.  To change how the robot looks edit PART_DEFS and RobotMesh.update.
Colours in the soup are 0..1 floats.
"""
import math
import os
import json
import numpy as np
import cv2
from .common import C, IN, hex_rgb
from . import kinematics as K
from .step_importer import load_ply_mesh

SEG = 24
_ANG = np.linspace(0.0, 2 * np.pi, SEG, endpoint=False)
_COS, _SIN = np.cos(_ANG)[:, None], np.sin(_ANG)[:, None]
PRISM_FACES = [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]]

PLATE_W, PLATE_T = 0.24, 0.05
SPIKE_R, TIP_LEN, TIP_R = 0.006, 0.055, 0.0015
SPIKE_BODY_LEN = 0.095
SPIKE_BODY_W = 0.028
SPIKE_BODY_H = 0.036
ORIGIN_MARKER_RADIUS = 0.028
ORIGIN_MARKER_HEIGHT = 0.012
# FANUC's characteristic warm yellow, with separate enamel, joint, and cover
# colors so the model reads as cast housings rather than one yellow silhouette.
Y, YD, BK, GR, RD, TIPC = "#f5c900", "#d9a900", "#202329", "#646b73", C.rob_red, "#d9d2b0"
PALE, METAL, PANEL = "#f9d83d", "#aeb4b7", "#343941"

# ------------------------------------------------------------------ camera
class Camera:
    PRESETS = {"iso": (-52.0, 22.0), "front": (-90.0, 6.0), "side": (0.0, 6.0), "top": (-90.0, 88.0)}

    def __init__(self):
        self.fov = 38.0
        self.reset()

    def reset(self):
        # Start from the side-biased view used in FANUC's R-2000 product
        # photography so the shoulder disc and long forearm are easy to read.
        self.azim, self.elev, self.dist = -58.0, 18.0, 6.0
        self.target = np.array([0.75, 0.0, 1.15])

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


def housing_verts(p1, p2, lat, w1, h1, w2, h2):
    """Tapered, bevel-cornered arm casing aligned to a link's joint axis."""
    d = p2 - p1
    d /= np.linalg.norm(d) + 1e-12
    side = lat - d * float(np.dot(lat, d))
    side_len = np.linalg.norm(side)
    side = _basis(d)[0] if side_len < 1e-6 else side / side_len
    up = np.cross(side, d)
    up /= np.linalg.norm(up) + 1e-12
    # Chamfered rectangular outline: broad flat faces, soft-looking corners.
    outline = np.array([[-.31, -.50], [.31, -.50], [.50, -.31], [.50, .31],
                        [.31, .50], [-.31, .50], [-.50, .31], [-.50, -.31]])
    def ring(p, w, h):
        return p + outline[:, :1] * w * side + outline[:, 1:] * h * up
    return np.vstack((ring(p1, w1, h1), ring(p2, w2, h2)))


HOUSING_RINGS = 5
ARM_CROSS = 16


def arm_verts(p1, p2, lat, w1, h1, w2, h2, bend=0.14):
    """Five-section tapered casting with the subtle bow of a FANUC arm link."""
    d = p2 - p1
    d /= np.linalg.norm(d) + 1e-12
    side = lat - d * float(np.dot(lat, d))
    side_len = np.linalg.norm(side)
    side = _basis(d)[0] if side_len < 1e-6 else side / side_len
    up = np.cross(side, d)
    up /= np.linalg.norm(up) + 1e-12
    theta = np.linspace(0.0, 2 * math.pi, ARM_CROSS, endpoint=False)
    c, s = np.cos(theta), np.sin(theta)
    # A superellipse gives the arm a rounded rectangular cast-metal section.
    outline = np.column_stack((np.sign(c) * np.sqrt(np.abs(c)) * .50,
                               np.sign(s) * np.sqrt(np.abs(s)) * .50))
    rings = []
    for t in np.linspace(0.0, 1.0, HOUSING_RINGS):
        center = p1 + (p2 - p1) * t + up * (bend * math.sin(math.pi * t))
        width, height = w1 + (w2 - w1) * t, h1 + (h2 - h1) * t
        rings.append(center + outline[:, :1] * width * side + outline[:, 1:] * height * up)
    return np.vstack(rings)


def cyl_verts(p1, p2, r1, r2=None):
    r2 = r1 if r2 is None else r2
    side, up = _basis(p2 - p1)
    radial = _COS * side + _SIN * up
    return np.vstack([p1 + radial * r1, p2 + radial * r2])


def cyl_center(c, axis, r, length):
    a = axis / np.linalg.norm(axis)
    return cyl_verts(c - a * length / 2, c + a * length / 2, r)


# ------------------------------------------------------------------ legacy procedural helpers
def _chain_names(name, n=3):
    return [f"{name}_{k}" for k in range(n)]


PART_DEFS = (
    [("base_plate", "prism", GR), ("base_ring", "cyl", BK)]
    + [(f"base_foot_{i}", "prism", GR) for i in range(4)]
    + [(f"base_bolt_{i}", "cyl", BK) for i in range(4)]
    + [("base_shroud", "housing", Y),
     ("column", "housing", Y), ("bridge", "housing", Y),
     ("j2", "cyl", Y), ("j2_motor", "cyl", RD)]
    + [(f"j2_bolt_{i}", "cyl", BK) for i in range(16)]
    + [("upper_arm", "arm", Y),
       ("j3", "cyl", Y), ("j3_motor", "cyl", RD),
       ("elbow", "housing", Y), ("j4_motor", "housing", BK), ("j4_red", "housing", RD),
       ("forearm", "housing", Y), ("collar", "cyl", YD), ("j4_ring", "cyl", BK),
       ("wrist_housing", "housing", Y), ("j5", "cyl", Y),
       ("j6a", "cyl", Y), ("j6b", "cyl", BK),
       ("tool_plate", "prism", Y)]
    + [(f"j3_bolt_{i}", "cyl", BK) for i in range(16)]
    + [(f"wrist_bolt_{i}", "cyl", BK) for i in range(16)]
    + [(f"spike_{k}", "cyl", Y) for k in range(3)] + [(f"spike_{k}_tip", "cyl", TIPC) for k in range(3)]
)


class _ProceduralRobotMesh:
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
            elif kind == "housing":
                n = 16
                for i in range(8):
                    j = (i + 1) % 8
                    for t in ((i, j, 8 + j), (i, 8 + j, 8 + i)):
                        tri.append([total + t[0], total + t[1], total + t[2]]); part.append(pid); smooth.append(True); cols.append(rgb)
                for off in (0, 8):
                    for k in range(1, 7):
                        tri.append([total + off, total + off + k, total + off + k + 1]); part.append(pid); smooth.append(False); cols.append(rgb)
            elif kind == "arm":
                n = HOUSING_RINGS * ARM_CROSS
                for ring in range(HOUSING_RINGS - 1):
                    start, nxt = total + ring * ARM_CROSS, total + (ring + 1) * ARM_CROSS
                    for i in range(ARM_CROSS):
                        j = (i + 1) % ARM_CROSS
                        for t in ((start + i, start + j, nxt + j), (start + i, nxt + j, nxt + i)):
                            tri.append(list(t)); part.append(pid); smooth.append(True); cols.append(rgb)
                for off in (total, total + (HOUSING_RINGS - 1) * ARM_CROSS):
                    for k in range(1, ARM_CROSS - 1):
                        tri.append([off, off + k, off + k + 1]); part.append(pid); smooth.append(False); cols.append(rgb)
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
        self.counts = np.array([8 if k == "prism" else 16 if k == "housing" else HOUSING_RINGS * ARM_CROSS if k == "arm" else 2 * SEG for _, k, _ in PART_DEFS])
        self.cyl_vidx = np.array(cyl_starts)[:, None] + np.arange(2 * SEG)[None, :]
        self.VN = np.zeros((total, 3))
        self.centers = np.zeros((len(PART_DEFS), 3))
        V, S = self.V, self.slices
        H = K.J2_HEIGHT
        V[S["base_plate"]] = box_verts(np.array([0, 0, 0.0]), np.array([0, 0, 0.19]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), 1.0, 0.92, 0.92, 0.84)
        V[S["base_ring"]] = cyl_verts(np.array([0, 0, 0.18]), np.array([0, 0, 0.30]), 0.42)
        foot_xy = ((-.37, -.33), (.37, -.33), (.37, .33), (-.37, .33))
        for i, (x, y) in enumerate(foot_xy):
            V[S[f"base_foot_{i}"]] = box_verts(np.array([x, y, .02]), np.array([x, y, .23]),
                                                  np.array([1., 0, 0]), np.array([0., 1, 0]), .24, .22, .22, .20)
            V[S[f"base_bolt_{i}"]] = cyl_verts(np.array([x, y, .21]), np.array([x, y, .25]), .038)
        V[S["base_shroud"]] = housing_verts(np.array([0, 0, .18]), np.array([0, 0, .52]), np.array([0, 0, 1]), .80, .80, .70, .68)
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
        V[S["column"]] = housing_verts(np.array([0, 0, 0.30]), j1, lat1, 0.68, 0.66, 0.58, 0.56)
        V[S["bridge"]] = housing_verts(j1, j2, lat1, 0.58, 0.67, 0.58, 0.64)
        # Large J2 shoulder gearbox and side cover, sized to the prominent
        # circular casting in the R-2000iA reference photographs.
        V[S["j2"]] = cyl_center(j2, lat1, 0.45, 0.78)
        # A single painted joint face with fasteners reads closer to the
        # photographed casting than stacked, coplanar gear-like disks.
        j2_side, j2_up = _basis(lat1)
        V[S["j2_motor"]] = cyl_center(j2 - lat1 * 0.46 + j2_up * .22, lat1, 0.15, 0.12)
        V[S["upper_arm"]] = arm_verts(j2 + np.array([0, 0, .08]), j3, lat1, .62, .63, .34, .38, bend=.18)
        arm_dir = j3 - j2
        arm_dir /= np.linalg.norm(arm_dir) + 1e-12
        arm_side = lat1 - arm_dir * float(np.dot(lat1, arm_dir))
        arm_side = _basis(arm_dir)[0] if np.linalg.norm(arm_side) < 1e-6 else arm_side / np.linalg.norm(arm_side)
        arm_up = np.cross(arm_side, arm_dir)
        self.brand_points = [j2 + (j3 - j2) * .58 - arm_up * .245]
        V[S["j3"]] = cyl_center(j3, lat2, 0.29, 0.64)
        V[S["j3_motor"]] = cyl_center(j3 + lat2 * .40, lat2, .12, .10)
        V[S["elbow"]] = housing_verts(j3, j4, lat2, 0.49, 0.48, 0.39, 0.38)
        V[S["j4_motor"]] = housing_verts(j4 - fd * 0.62, j4 - fd * 0.10, lat2, 0.38, 0.38, 0.34, 0.34)
        V[S["j4_red"]] = housing_verts(j4 - fd * 0.84, j4 - fd * 0.60, lat2, 0.29, 0.28, 0.31, 0.30)
        V[S["forearm"]] = housing_verts(j4 + fd*.08 + thk*.06, j5 - fd*.06 + thk*.04,
                                        lat2, .43, .32, .30, .22)
        V[S["collar"]] = cyl_verts(j4 - fd * 0.02, j4 + fd * 0.30, 0.155, 0.14)
        V[S["j4_ring"]] = cyl_verts(j4 + fd * 0.30, j4 + fd * 0.345, 0.15)
        V[S["wrist_housing"]] = housing_verts(j5 - fd*.16, j5 + fd*.22, lat5, .54, .50, .46, .44)
        V[S["j5"]] = cyl_center(j5, lat5, 0.34, 0.80)
        V[S["j6a"]] = cyl_verts(j5, j5 + ta * 0.09, 0.15, 0.14)
        V[S["j6b"]] = cyl_verts(j5 + ta * 0.09, fl, 0.125, 0.12)
        V[S["tool_plate"]] = box_verts(fl - ta * 0.012, fl + ta * 0.016, lat, thk, PLATE_W, PLATE_T, PLATE_W, PLATE_T)
        for k, (L, off) in enumerate(K.SPIKES_IN):
            c = fl + lat * off * IN
            end = c + ta * L * IN
            mid = end - ta * TIP_LEN
            V[S[f"spike_{k}"]] = cyl_verts(c + ta * 0.016, mid, SPIKE_R, SPIKE_R)
            V[S[f"spike_{k}_tip"]] = cyl_verts(mid, end, SPIKE_R, TIP_R)
        # Recessed dark fasteners around the J2, J3, and wrist side covers.
        for prefix, center, axis, radius, face_offset in (
                ("j2_bolt", j2, lat1, .28, .40),
                ("j3_bolt", j3, lat2, .16, .33),
                ("wrist_bolt", j5, lat5, .19, .41)):
            side, up = _basis(axis)
            for face, sign in enumerate((-1.0, 1.0)):
                for i in range(8):
                    angle = 2 * math.pi * i / 8
                    bolt_center = center + axis * (face_offset * sign) + radius * (math.cos(angle) * side + math.sin(angle) * up)
                    V[S[f"{prefix}_{face * 8 + i}"]] = cyl_center(bolt_center, axis, .020, .025)
        return type("Info", (), {"joints": J, "flange": fl, "tip": tip})()

    # ---- triangle soup
    def triangles(self):
        V = self.V
        self.centers = np.add.reduceat(V, self.starts, axis=0) / self.counts[:, None]
        P = V[self.tri]
        fn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
        flip = np.einsum("ij,ij->i", fn, P.mean(axis=1) - self.centers[self.tri_part]) < 0
        fn[flip] *= -1
        # Average normals only across the sides of cast housings and cylinders;
        # the end faces remain flat, preserving visible panel seams.
        self.VN.fill(0)
        smooth_tri = self.tri[self.tri_smooth]
        smooth_normals = fn[self.tri_smooth]
        for axis in range(3):
            self.VN[:, axis] = np.bincount(smooth_tri.ravel(),
                                           weights=np.repeat(smooth_normals[:, axis], 3),
                                           minlength=len(V))
        self.VN /= np.linalg.norm(self.VN, axis=1, keepdims=True) + 1e-12
        nrm = np.where(self.tri_smooth[:, None, None], self.VN[self.tri], fn[:, None, :])
        return P.reshape(-1, 3).astype(np.float32), nrm.reshape(-1, 3).astype(np.float32), self.tri_col


# ------------------------------------------------------------------ supplied FANUC link meshes
class RobotMesh:
    """Render the supplied RoboDK/URDF link meshes using the shared FK chain."""

    def __init__(self):
        root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "FANUC_R2000iA_200F_model", "meshes")
        self.meshes = [self._load_obj(os.path.join(root, f"link{i}.obj")) for i in range(7)]
        self.colors = [
            np.array(hex_rgb(GR), np.float32) / 255.0,
            *[np.array(hex_rgb(Y), np.float32) / 255.0 for _ in range(5)],
            np.array(hex_rgb(PANEL), np.float32) / 255.0,
        ]
        self.triangles_data = None
        self.brand_points = []
        self.update(np.zeros(6))

    @staticmethod
    def _load_obj(path):
        vertices, faces = [], []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                fields = line.split()
                if not fields:
                    continue
                if fields[0] == "v":
                    vertices.append([float(x) for x in fields[1:4]])
                elif fields[0] == "f":
                    ids = [int(x.split("/")[0]) - 1 for x in fields[1:]]
                    for i in range(1, len(ids) - 1):
                        faces.append((ids[0], ids[i], ids[i + 1]))
        return np.asarray(vertices, float), np.asarray(faces, int)

    @staticmethod
    def _transform(parent, origin, axis, angle):
        T = parent.copy()
        T[:3, 3] += parent[:3, :3] @ origin
        T[:3, :3] = parent[:3, :3] @ K.rot_axis(axis, angle)
        return T

    def update(self, q):
        q = np.asarray(q, float)
        transforms = [np.eye(4)]
        for i in range(6):
            transforms.append(self._transform(transforms[-1], K.JOINT_ORIGINS[i], K.JOINT_AXES[i], q[i]))
        chunks = []
        normals = []
        colors = []
        for i, (vertices, faces) in enumerate(self.meshes):
            T = transforms[i]
            v = np.concatenate((vertices, np.ones((len(vertices), 1))), axis=1) @ T.T
            p = v[faces][:, :, :3]
            n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            chunks.append(p.reshape(-1, 3))
            normals.append(np.repeat(n, 3, axis=0))
            colors.append(np.tile(self.colors[i], (len(p) * 3, 1)))
        self.triangles_data = (
            np.concatenate(chunks).astype(np.float32),
            np.concatenate(normals).astype(np.float32),
            np.concatenate(colors).astype(np.float32),
        )
        self.brand_points = [transforms[2][:3, 3] + transforms[2][:3, :3] @ np.array([0.55, -0.23, 0.0])]
        return type("Info", (), {"joints": K.forward_kinematics(q)[1], "flange": transforms[6][:3, 3],
                                 "tip": K.forward_kinematics(q)[4]})()

    def triangles(self):
        return self.triangles_data


# ------------------------------------------------------------------ active STEP robot model
class RobotMesh:
    """Render the separate robot_files STL links using their DH frames."""

    def __init__(self):
        root = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "robot_files",
        )
        with open(os.path.join(root, "robot.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        self.dh = data["dh"]
        mesh_files = {item["name"]: item["stl"] for item in data["meshes"]}
        self.meshes = [
            (load_ply_mesh(os.path.join(root, mesh_files[f"link{i}" if i else "link0_base"].replace(".stl", ".ply")), axis_order=(0, 1, 2)), i)
            for i in range(7)
        ]
        self.brand_points = []
        self.triangles_data = None
        self.update(np.zeros(6))

    @staticmethod
    def _dh_transform(item, q):
        alpha = float(item["alpha_rad"])
        a = float(item["a_mm"]) * 0.001
        d = float(item["d_mm"]) * 0.001
        theta = float(item["theta_offset_rad"]) + q
        ca, sa = math.cos(alpha), math.sin(alpha)
        ct, st = math.cos(theta), math.sin(theta)
        return np.array([
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ])

    def update(self, q):
        q = np.asarray(q, float)
        transforms = [np.eye(4)]
        for item, angle in zip(self.dh, q):
            transforms.append(transforms[-1] @ self._dh_transform(item, angle))
        chunks, normals, colors = [], [], []
        for (mesh, frame), color in zip(self.meshes, [GR, Y, Y, Y, Y, Y, Y, Y, Y]):
            positions, local_normals, local_colors = mesh
            T = transforms[frame]
            p = positions.reshape(-1, 3)
            world = p @ T[:3, :3].T + T[:3, 3]
            n = local_normals.reshape(-1, 3) @ T[:3, :3].T
            chunks.append(world.astype(np.float32))
            normals.append(n.astype(np.float32))
            colors.append(local_colors)
        # The three probing spikes are defined in the local flange frame:
        # lengths and lateral offsets are measured from the end of the arm.
        flange = transforms[6]
        fl = flange[:3, 3]
        # Spike length follows J6's axis; the row is offset across the
        # perpendicular flange axis.
        axis, lateral, thickness = flange[:3, 2], flange[:3, 0], flange[:3, 1]
        tool_parts = []
        plate = box_verts(
            fl - axis * 0.012, fl + axis * 0.016,
            lateral, thickness, PLATE_W, PLATE_T, PLATE_W, PLATE_T,
        )
        tool_parts.append(solid_soup(plate, PRISM_FACES, hex_rgb(Y)))
        for length_in, offset_in in K.SPIKES_IN:
            center = fl + lateral * offset_in * IN
            end = center + axis * length_in * IN
            body_start = center + axis * 0.016
            body_end = body_start + axis * min(
                SPIKE_BODY_LEN, max(0.0, length_in * IN - 0.025)
            )
            # The photographed probe has a broad painted sleeve at the
            # flange, then a narrow metal shaft with a tapered working tip.
            tool_parts.append(solid_soup(
                box_verts(
                    body_start, body_end, lateral, thickness,
                    SPIKE_BODY_W, SPIKE_BODY_H,
                    SPIKE_BODY_W * 0.78, SPIKE_BODY_H * 0.78,
                ),
                PRISM_FACES,
                hex_rgb(Y),
            ))
            shaft_end = end - axis * TIP_LEN
            tool_parts.append(solid_soup(
                cyl_verts(body_end, shaft_end, SPIKE_R * 0.72, SPIKE_R * 0.60),
                [[i, (i + 1) % SEG, SEG + (i + 1) % SEG, SEG + i]
                 for i in range(SEG)] + [list(range(SEG)), list(range(SEG, 2 * SEG))],
                hex_rgb(METAL),
            ))
            tool_parts.append(solid_soup(
                cyl_verts(shaft_end, end, SPIKE_R * 0.60, TIP_R),
                [[i, (i + 1) % SEG, SEG + (i + 1) % SEG, SEG + i]
                 for i in range(SEG)] + [list(range(SEG)), list(range(SEG, 2 * SEG))],
                hex_rgb(TIPC),
            ))
        for positions, local_normals, color in tool_parts:
            chunks.append(positions)
            normals.append(local_normals)
            colors.append(color)
        # The red floor dot in the reference image marks the robot base
        # origin, so keep it visible independently of the robot pose.
        origin_marker = cyl_verts(
            K.ROBOT_ORIGIN_M + np.array([0.0, 0.0, 0.002]),
            K.ROBOT_ORIGIN_M + np.array([0.0, 0.0, ORIGIN_MARKER_HEIGHT]),
            ORIGIN_MARKER_RADIUS,
        )
        marker_polys = [
            [i, (i + 1) % SEG, SEG + (i + 1) % SEG, SEG + i]
            for i in range(SEG)
        ] + [list(range(SEG)), list(range(SEG, 2 * SEG))]
        marker_positions, marker_normals, marker_colors = solid_soup(
            origin_marker, marker_polys, hex_rgb(RD)
        )
        chunks.append(marker_positions)
        normals.append(marker_normals)
        colors.append(marker_colors)
        self.triangles_data = (
            np.concatenate(chunks),
            np.concatenate(normals),
            np.concatenate(colors),
        )
        positions = self.triangles_data[0]
        self.brand_points = [positions.reshape(-1, 3).mean(axis=0)]
        return type("Info", (), {
            "joints": K.forward_kinematics(q)[1],
            "flange": K.forward_kinematics(q)[3][:3, 3],
            "tip": K.forward_kinematics(q)[4],
        })()

    def triangles(self):
        return self.triangles_data


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
    # The three recorded points describe the plate's top face. Extend the
    # plate down from the underside rather than upward into the robot base.
    V = np.array(top + [p - n * t for p in top])
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
    """-> (segments (N,2,3) float32, colours (N,3) 0..1).

    Grid and coordinate axes use the robot base origin and the WORLD origin.
    The active robot mesh also renders the base origin as a red reference dot.
    """
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
        for org, ln in ((K.ROBOT_ORIGIN_M, 0.55), (K.TABLE_ORIGIN_IN_BASE_M, 0.30)):
            for k in range(3):
                e = np.zeros(3)
                e[k] = ln
                segs.append([tuple(org), tuple(org + e)]); cols.append(np.array(axc[k]) / 255.0)
    return np.array(segs, np.float32), np.array(cols, np.float32)
