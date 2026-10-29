"""
render.py - turns a Frame (triangle soups + overlay data) into an RGB image.

    make_renderer(prefer_gpu) -> Renderer      (GPU via moderngl if installed + working, else CPU)
    Renderer.render(w, h, cam, frame) -> uint8 RGB (h, w, 3)

Two backends draw the SAME inputs:
    GLBackend   OpenGL offscreen (moderngl standalone context), 4x MSAA, depth buffer.  Fast, no clipping artefacts.
    CPUBackend  OpenCV painter's algorithm (triangles sorted far->near).  Slower, tiny artefacts, no dependencies.
Then draw_overlay() paints 2D things on top with OpenCV (trail, tip marker, axis gizmo, labels, HUD text).

Frame fields (SimpleNamespace, built in ui.App._render):
    tris         (pos, nrm, col) soup of robot + mapped objects          (N,3) float32 each
    ghost_tris   same, or None        ghost_col (r,g,b 0..255)          ghost_tip (3,) or None / ghost_ok
    tip (3,)     trail (M,3) or None  labels [(P, text, rgb, dot)]      gizmo bool
    hud dict     keys: tip_text, ghost_text, msg, fps, backend
"""
import math
import numpy as np
import cv2
from .common import C, hex_rgb
from .model3d import View, Camera, floor_lines

BG_TOP, BG_BOT = np.array(hex_rgb(C.view_top), float), np.array(hex_rgb(C.view_bot), float)
FONT = cv2.FONT_HERSHEY_SIMPLEX
AXIS_NAMES = ["X", "Y", "Z", "Xa", "Ya", "Za"]

# ------------------------------------------------------------------ GPU backend
VS = """#version 330
uniform mat4 mvp;
in vec3 in_pos; in vec3 in_nrm; in vec3 in_col;
out vec3 v_pos; out vec3 v_nrm; out vec3 v_col;
void main(){ v_pos=in_pos; v_nrm=in_nrm; v_col=in_col; gl_Position = mvp*vec4(in_pos,1.0); }
"""
FS = """#version 330
uniform vec3 eye; uniform vec3 light; uniform float alpha; uniform vec3 tint; uniform float use_tint;
in vec3 v_pos; in vec3 v_nrm; in vec3 v_col;
out vec4 f_col;
void main(){
  vec3 n = normalize(v_nrm);
  vec3 te = normalize(eye - v_pos);
  if (dot(n, te) < 0.0) n = -n;
  float lam = clamp(dot(n, light), 0.0, 1.0);
  vec3 hv = normalize(light + te);
  float spec = pow(clamp(dot(n, hv), 0.0, 1.0), 26.0) * 0.22;
  float shade = 0.24 + 0.74*lam + 0.16*(0.5 + 0.5*n.z);
  vec3 base = mix(v_col, tint, use_tint);
  f_col = vec4(base*shade + vec3(spec), alpha);
}
"""
LVS = """#version 330
uniform mat4 mvp; in vec3 in_pos; in vec3 in_col; out vec3 v_col;
void main(){ v_col=in_col; gl_Position = mvp*vec4(in_pos,1.0); }
"""
LFS = """#version 330
in vec3 v_col; out vec4 f_col; void main(){ f_col = vec4(v_col,1.0); }
"""
# The imported STEP assembly contains about 480k vertices. Keep the complete
# triangle soup so adjacent faces remain connected; dropping arbitrary faces
# produces the spiky/wireframe appearance of a broken mesh.
MAX_VERTS = 600000
MAX_SPLATS = 120000

SPLAT_VS = """#version 330
uniform mat4 mvp; uniform float focal_px; uniform float splat_radius;
uniform vec3 eye; uniform vec3 light;
in vec3 in_pos; in vec3 in_nrm; in vec3 in_col;
out vec3 v_col;
void main(){
  vec4 clip = mvp * vec4(in_pos, 1.0);
  gl_Position = clip;
  gl_PointSize = clamp(2.0 * splat_radius * focal_px / max(clip.w, 0.05), 2.0, 22.0);
  vec3 n = normalize(in_nrm);
  vec3 to_eye = normalize(eye - in_pos);
  if (dot(n, to_eye) < 0.0) n = -n;
  float lam = clamp(dot(n, light), 0.0, 1.0);
  v_col = in_col * (0.30 + 0.70 * lam);
}
"""
SPLAT_FS = """#version 330
in vec3 v_col; out vec4 f_col;
void main(){
  vec2 p = 2.0 * gl_PointCoord - 1.0;
  float r2 = dot(p, p);
  if (r2 > 1.0) discard;
  float a = 0.52 * exp(-4.0 * r2);
  if (a < 0.015) discard;
  f_col = vec4(v_col, a);
}
"""


def _splat_samples(soup, max_points=MAX_SPLATS, spacing=0.045, tint=None):
    """Sample triangle faces into colored surfel points for the splat view."""
    pos, nrm, col = (np.asarray(part, dtype=np.float32).reshape(-1, 3, 3) for part in soup)
    if not len(pos):
        return np.empty((0, 9), np.float32)
    edges = np.stack((pos[:, 1] - pos[:, 0], pos[:, 2] - pos[:, 1], pos[:, 0] - pos[:, 2]), axis=1)
    levels = np.clip(np.ceil(np.linalg.norm(edges, axis=2).max(axis=1) / spacing).astype(int), 1, 9)
    batches = []
    for level in np.unique(levels):
        indices = np.flatnonzero(levels == level)
        bary = np.asarray([(1.0 - i / level - j / level, i / level, j / level)
                           for i in range(level + 1) for j in range(level + 1 - i)], np.float32)
        points = np.einsum("tvc,bv->tbc", pos[indices], bary)
        normals = np.einsum("tvc,bv->tbc", nrm[indices], bary)
        normals /= np.linalg.norm(normals, axis=2, keepdims=True) + 1e-8
        colors = np.einsum("tvc,bv->tbc", col[indices], bary)
        if tint is not None:
            colors[:] = np.asarray(tint, np.float32)[None, None, :] / 255.0
        batches.append(np.concatenate((points, normals, colors), axis=2).reshape(-1, 9))
    result = np.concatenate(batches, axis=0)
    if len(result) > max_points:
        result = result[::int(math.ceil(len(result) / max_points))][:max_points]
    return np.ascontiguousarray(result, dtype=np.float32)


def _composite_splats(img, view, samples):
    if not len(samples):
        return img
    pixels, depth = view.project(samples[:, :3])
    good = (depth > 0.12) & (pixels[:, 0] > -20) & (pixels[:, 0] < img.shape[1] + 20) & \
           (pixels[:, 1] > -20) & (pixels[:, 1] < img.shape[0] + 20)
    indices = np.flatnonzero(good)
    indices = indices[np.argsort(depth[indices])[::-1]]
    layer = np.zeros_like(img)
    alpha = np.zeros(img.shape[:2], np.uint8)
    for i in indices:
        x, y = np.rint(pixels[i]).astype(int)
        radius = int(np.clip(0.016 * view.F / depth[i], 2, 10))
        color = tuple(int(c) for c in np.clip(samples[i, 6:9] * 255, 0, 255))
        cv2.circle(layer, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(alpha, (x, y), radius, 150, -1, cv2.LINE_AA)
    sigma = 1.2
    layer = cv2.GaussianBlur(layer, (0, 0), sigma)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigma).astype(np.float32)[:, :, None] / 255.0
    return np.clip(img.astype(np.float32) * (1.0 - alpha) + layer.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


class GLBackend:
    name = "GPU (OpenGL)"

    def __init__(self):
        import moderngl
        self.mgl = moderngl
        self.ctx = moderngl.create_standalone_context()
        self.prog = self.ctx.program(vertex_shader=VS, fragment_shader=FS)
        self.lprog = self.ctx.program(vertex_shader=LVS, fragment_shader=LFS)
        self.sprog = self.ctx.program(vertex_shader=SPLAT_VS, fragment_shader=SPLAT_FS)
        self.vbo = self.ctx.buffer(reserve=MAX_VERTS * 36, dynamic=True)
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "3f 3f 3f", "in_pos", "in_nrm", "in_col")])
        self.svbo = self.ctx.buffer(reserve=MAX_SPLATS * 36, dynamic=True)
        self.svao = self.ctx.vertex_array(self.sprog, [(self.svbo, "3f 3f 3f", "in_pos", "in_nrm", "in_col")])
        segs, cols = floor_lines(include_axes=False)
        data = np.hstack([segs.reshape(-1, 3), np.repeat(cols, 2, axis=0)]).astype("f4")
        self.lvbo = self.ctx.buffer(data.tobytes())
        self.lvao = self.ctx.vertex_array(self.lprog, [(self.lvbo, "3f 3f", "in_pos", "in_col")])
        self.nline = len(data)
        all_segs, all_cols = floor_lines()
        axis_data = np.hstack([all_segs[len(segs):].reshape(-1, 3),
                               np.repeat(all_cols[len(cols):], 2, axis=0)]).astype("f4")
        self.avbo = self.ctx.buffer(axis_data.tobytes())
        self.avao = self.ctx.vertex_array(self.lprog, [(self.avbo, "3f 3f", "in_pos", "in_col")])
        self.naxis = len(axis_data)
        self.size, self.fbo, self.fbo_ms = None, None, None
        self._ensure(64, 64)                       # fail early (inside make_renderer) if the driver can't do it
        self.render_base(64, 64, View(Camera(), 64, 64),
                         (np.zeros((3, 3), "f4"), np.ones((3, 3), "f4"), np.ones((3, 3), "f4")), None, (0, 255, 0))

    def _ensure(self, w, h):
        if self.size == (w, h):
            return
        for f in (self.fbo, self.fbo_ms):
            if f is not None:
                f.release()
        ctx = self.ctx
        self.fbo = ctx.framebuffer(ctx.renderbuffer((w, h)), ctx.depth_renderbuffer((w, h)))
        try:
            self.fbo_ms = ctx.framebuffer(ctx.renderbuffer((w, h), samples=4), ctx.depth_renderbuffer((w, h), samples=4))
        except Exception:
            self.fbo_ms = None
        self.size = (w, h)

    def render_base(self, w, h, view, tris, ghost, ghost_col, show_axes=True, gaussian_splats=False):
        mgl, ctx = self.mgl, self.ctx
        self._ensure(w, h)
        target = self.fbo_ms or self.fbo
        target.use()
        target.clear(BG_BOT[0] / 255 * 1.2, BG_BOT[1] / 255 * 1.2, BG_BOT[2] / 255 * 1.3, 1.0)
        ctx.enable(mgl.DEPTH_TEST)
        ctx.disable(mgl.BLEND)
        mvp = view.mvp().T.astype("f4").tobytes()
        self.lprog["mvp"].write(mvp)
        self.lvao.render(mgl.LINES, vertices=self.nline)
        if show_axes:
            self.avao.render(mgl.LINES, vertices=self.naxis)
        if gaussian_splats:
            parts = [_splat_samples(tris)]
            if ghost is not None:
                parts.append(_splat_samples(ghost, max_points=24000, tint=ghost_col))
            data = np.vstack(parts)[:MAX_SPLATS]
            if len(data):
                self.svbo.write(data.tobytes())
                self.sprog["mvp"].write(mvp)
                self.sprog["eye"].value = tuple(float(x) for x in view.eye)
                self.sprog["light"].value = tuple(float(x) for x in view.light)
                self.sprog["focal_px"].value = float(view.F)
                self.sprog["splat_radius"].value = 0.016
                ctx.enable(mgl.BLEND)
                ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
                self.svao.render(mgl.POINTS, vertices=len(data))
                ctx.disable(mgl.BLEND)
        else:
            parts = [np.hstack(tris)]
            n1 = len(parts[0])
            if ghost is not None:
                parts.append(np.hstack(ghost))
            data = np.ascontiguousarray(np.vstack(parts), dtype="f4")[:MAX_VERTS]
            self.vbo.write(data.tobytes())
            self.prog["mvp"].write(mvp)
            self.prog["eye"].value = tuple(float(x) for x in view.eye)
            self.prog["light"].value = tuple(float(x) for x in view.light)
            self.prog["alpha"].value = 1.0
            self.prog["use_tint"].value = 0.0
            self.prog["tint"].value = (0.0, 0.0, 0.0)
            self.vao.render(mgl.TRIANGLES, vertices=min(n1, len(data)))
        if not gaussian_splats and ghost is not None and len(data) > n1:
            ctx.enable(mgl.BLEND)
            ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
            self.prog["alpha"].value = 0.45
            self.prog["use_tint"].value = 1.0
            self.prog["tint"].value = tuple(c / 255.0 for c in ghost_col)
            self.vao.render(mgl.TRIANGLES, vertices=len(data) - n1, first=n1)
            ctx.disable(mgl.BLEND)
        if self.fbo_ms is not None:
            ctx.copy_framebuffer(self.fbo, self.fbo_ms)
        raw = self.fbo.read(components=3, alignment=1)
        return np.ascontiguousarray(np.frombuffer(raw, np.uint8).reshape(h, w, 3)[::-1])


# ------------------------------------------------------------------ CPU backend
class CPUBackend:
    name = "CPU (OpenCV)"

    def __init__(self):
        self._bg = {}
        self._floor, self._floor_cols = floor_lines(include_axes=False)
        all_floor, all_cols = floor_lines()
        self._axes, self._axis_cols = all_floor[len(self._floor):], all_cols[len(self._floor_cols):]

    def _background(self, W, H):
        if (W, H) not in self._bg:
            self._bg.clear()
            t = np.linspace(0, 1, H)[:, None]
            col = (BG_TOP + (BG_BOT - BG_TOP) * t).astype(np.uint8)
            img = np.empty((H, W, 3), np.uint8)
            img[:] = col[:, None, :]
            self._bg[(W, H)] = img
        return self._bg[(W, H)].copy()

    @staticmethod
    def _paint(img, view, soup, tint=None):
        pos, nrm, col = soup
        nt = len(pos) // 3
        if nt == 0:
            return
        P = pos.reshape(nt, 3, 3)
        scr, depth = view.project(pos)
        scr = np.rint(scr).astype(np.int32).reshape(nt, 3, 2)
        depth = depth.reshape(nt, 3)
        cen = P.mean(axis=1)
        N = nrm.reshape(nt, 3, 3).mean(axis=1)
        N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
        to_eye = view.eye - cen
        # STEP solids do not necessarily share a consistent winding order.
        # Render both sides and orient normals toward the camera for shading,
        # as the OpenGL fragment shader does.
        facing = np.einsum("ij,ij->i", N, to_eye) < 0
        N[facing] *= -1
        ok = depth.min(axis=1) > 0.12
        idx = np.nonzero(ok)[0]
        if len(idx) == 0:
            return
        idx = idx[np.argsort(-depth[idx].mean(axis=1))]
        dist = np.linalg.norm(to_eye[idx], axis=1, keepdims=True) + 1e-12
        lam = np.clip(N[idx] @ view.light, 0, 1)
        hv = view.light + to_eye[idx] / dist
        hv /= np.linalg.norm(hv, axis=1, keepdims=True) + 1e-12
        spec = np.clip(np.einsum("ij,ij->i", N[idx], hv), 0, 1) ** 26 * 0.22
        shade = 0.24 + 0.74 * lam + 0.16 * (0.5 + 0.5 * N[idx, 2])
        base = col.reshape(nt, 3, 3)[idx, 0] * 255.0 if tint is None else np.tile(np.asarray(tint, float), (len(idx), 1))
        fill = np.clip(base * shade[:, None] + 255.0 * spec[:, None], 0, 255).astype(np.uint8).tolist()
        for k, i in enumerate(idx.tolist()):
            cv2.fillConvexPoly(img, scr[i], tuple(fill[k]))

    def render_base(self, w, h, view, tris, ghost, ghost_col, show_axes=True, gaussian_splats=False):
        img = self._background(w, h)
        for (a, b), c in zip(self._floor, self._floor_cols):
            view.seg(img, a, b, tuple(int(x * 255) for x in c), 1)
        if show_axes:
            for (a, b), c in zip(self._axes, self._axis_cols):
                view.seg(img, a, b, tuple(int(x * 255) for x in c), 1)
        if gaussian_splats:
            parts = [_splat_samples(tris, max_points=18000)]
            if ghost is not None:
                parts.append(_splat_samples(ghost, max_points=6000, tint=ghost_col))
            img = _composite_splats(img, view, np.vstack(parts))
        else:
            self._paint(img, view, tris)
            if ghost is not None:
                layer = img.copy()
                self._paint(layer, view, ghost, tint=ghost_col)
                m = (layer != img).any(axis=2)
                img[m] = (img[m] * 0.55 + layer[m] * 0.45).astype(np.uint8)
        return img


# ------------------------------------------------------------------ 2D overlay
def put_text(img, text, org, scale=0.48, color=(232, 233, 237), th=1):
    cv2.putText(img, text, org, FONT, scale, (12, 14, 18), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, th, cv2.LINE_AA)


def _gizmo(img, v, tip, labels):
    tip = np.asarray(tip, float)
    dist = float(np.linalg.norm(tip - v.eye))
    if dist < 0.3:
        return
    s = 0.075 * dist
    cols = [(235, 90, 90), (90, 205, 120), (100, 160, 255)]
    E = np.eye(3)
    for k in range(3):
        e, col = E[k], cols[k]
        end = tip + e * s
        side = E[(k + 1) % 3]
        v.seg(img, tip, end, col, 2)
        for sg in (1, -1):
            v.seg(img, end, end - e * s * 0.16 + side * sg * s * 0.07, col, 2)
        labels.append((end + e * s * 0.06, AXIS_NAMES[k] + "+", col, False))
        u, w = E[(k + 1) % 3], E[(k + 2) % 3]              # u x w = axis -> right-hand rotation = angle +
        r = s * 0.5
        th = np.radians(np.linspace(25, 300, 22))
        pts = [tip + r * (math.cos(t) * u + math.sin(t) * w) for t in th]
        dim = tuple(int(c * 0.85) for c in col)
        for a, c2 in zip(pts[:-1], pts[1:]):
            v.seg(img, a, c2, dim, 1)
        tang = -math.sin(th[-1]) * u + math.cos(th[-1]) * w
        rad = math.cos(th[-1]) * u + math.sin(th[-1]) * w
        for sg in (1, -1):
            v.seg(img, pts[-1], pts[-1] - tang * s * 0.15 + rad * sg * s * 0.07, dim, 1)
        labels.append((pts[-1] + rad * s * 0.14 + tang * s * 0.06, AXIS_NAMES[k + 3] + "+", col, False))


def _projected_camera_texture(img, view, tris, camera_view, calibration):
    if camera_view is None or not camera_view.size or not calibration:
        return
    try:
        matrix = np.asarray(calibration["camera_matrix"], np.float64)
        rvec = np.asarray(calibration["rvec"], np.float64).reshape(3, 1)
        tvec = np.asarray(calibration["tvec"], np.float64).reshape(3, 1)
        points = np.asarray(tris[0], np.float32).reshape(-1, 3)[::3]
        if not len(points):
            return
        uv, _ = cv2.projectPoints(points, rvec, tvec, matrix, np.zeros(5))
        uv = uv.reshape(-1, 2)
        screen, depth = view.project(points)
        h, w = camera_view.shape[:2]
        valid = ((depth > 0.12) & (uv[:, 0] >= 0) & (uv[:, 0] < w) &
                 (uv[:, 1] >= 0) & (uv[:, 1] < h) &
                 (screen[:, 0] >= 0) & (screen[:, 0] < img.shape[1]) &
                 (screen[:, 1] >= 0) & (screen[:, 1] < img.shape[0]))
        indices = np.flatnonzero(valid)
        indices = indices[np.argsort(depth[indices])[::-1]]
        layer = img.copy()
        for i in indices:
            x, y = np.rint(screen[i]).astype(int)
            sx, sy = np.rint(uv[i]).astype(int)
            color = tuple(int(c) for c in camera_view[sy, sx])
            cv2.circle(layer, (x, y), 3, color, -1, cv2.LINE_AA)
        cv2.addWeighted(layer, 0.72, img, 0.28, 0, img)
    except (KeyError, ValueError, cv2.error):
        return


def draw_overlay(img, v, fr):
    h, w = img.shape[:2]
    labels = list(fr.labels or [])
    if fr.trail is not None and len(fr.trail) > 1:
        p, d = v.project(fr.trail)
        n = len(fr.trail)
        for k in range(1, n):
            if d[k] < 0.12 or d[k - 1] < 0.12:
                continue
            t = k / n
            col = (int(30 + 60 * t), int(60 + 110 * t), int(110 + 140 * t))
            cv2.line(img, (int(p[k - 1, 0]), int(p[k - 1, 1])), (int(p[k, 0]), int(p[k, 1])), col, max(1, int(1 + 1.5 * t)), v.lt)
    if fr.ghost_tip is not None:
        g = np.asarray(fr.ghost_tip, float)
        col = hex_rgb(C.green if fr.ghost_ok else C.amber)
        p, d = v.project(g)
        if d[0] > 0.12:
            c = (int(p[0, 0]), int(p[0, 1]))
            cv2.circle(img, c, 11, col, 1, v.lt)
            cv2.line(img, (c[0] - 16, c[1]), (c[0] + 16, c[1]), col, 1, v.lt)
            cv2.line(img, (c[0], c[1] - 16), (c[0], c[1] + 16), col, 1, v.lt)
        v.seg(img, g, (g[0], g[1], 0.0), tuple(int(x * 0.55) for x in col), 1)
    tp, td = v.project(fr.tip)
    if td[0] > 0.12:
        c = (int(tp[0, 0]), int(tp[0, 1]))
        cv2.circle(img, c, 7, (20, 40, 80), -1, v.lt)
        cv2.circle(img, c, 5, hex_rgb(C.blue), -1, v.lt)
        cv2.circle(img, c, 7, (235, 242, 255), 1, v.lt)
    if fr.gizmo:
        _gizmo(img, v, fr.tip, labels)
    hud = fr.hud or {}
    camera_overlay = hud.get("camera_overlay")
    if camera_overlay is not None and camera_overlay.size:
        try:
            overlay = cv2.resize(camera_overlay, (w, h), interpolation=cv2.INTER_AREA)
            alpha = float(np.clip(hud.get("camera_overlay_alpha", 0.35), 0.0, 1.0))
            cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
        except cv2.error:
            pass
    if hud.get("camera_texture"):
        _projected_camera_texture(img, v, fr.tris, hud.get("camera_view"), hud["camera_texture"])
    if hud.get("camera_landmarks"):
        for index, point in enumerate(hud["camera_landmarks"]):
            pixels, depth = v.project(np.asarray(point, float))
            if depth[0] <= 0.12:
                continue
            x, y = np.rint(pixels[0]).astype(int)
            if 0 <= x < w and 0 <= y < h:
                cv2.drawMarker(img, (x, y), (80, 235, 120), cv2.MARKER_CROSS, 12, 2)
                put_text(img, f"C{index}", (x + 7, y - 7), 0.38, (80, 235, 120))
    for P, text, col, dot in labels:
        p, d = v.project(np.asarray(P, float))
        if d[0] <= 0.12:
            continue
        x, y = int(p[0, 0]), int(p[0, 1])
        if not (0 <= x < w and 0 <= y < h):
            continue
        if dot:
            cv2.circle(img, (x, y), 5, (15, 15, 15), -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), 4, col, -1, cv2.LINE_AA)
        if text:
            put_text(img, text, (x + 7, y - 7), 0.42, col)
    if hud.get("axes", True):
        org, L = (54, h - 54), 32
        axes = [((1, 0, 0), (235, 90, 90), "X"), ((0, 1, 0), (90, 205, 120), "Y"), ((0, 0, 1), (100, 160, 255), "Z")]
        axes.sort(key=lambda a: -float(np.dot(a[0], v.f)))
        cv2.circle(img, org, L + 12, (24, 27, 33), -1, cv2.LINE_AA)
        for vec, col, name in axes:
            x, y = float(np.dot(vec, v.r)), float(np.dot(vec, v.u))
            e = (int(org[0] + x * L), int(org[1] - y * L))
            cv2.line(img, org, e, col, 2, cv2.LINE_AA)
            cv2.circle(img, e, 9, col, -1, cv2.LINE_AA)
            cv2.putText(img, name, (e[0] - 4, e[1] + 4), FONT, 0.38, (18, 20, 24), 1, cv2.LINE_AA)
    if hud.get("tip_text"):
        tp, td = v.project(fr.tip)
        if td[0] > 0.12:
            x, y = int(tp[0, 0]) + 14, int(tp[0, 1]) - 12
            tw = cv2.getTextSize(hud["tip_text"], FONT, 0.46, 1)[0][0]
            put_text(img, hud["tip_text"], (int(np.clip(x, 6, w - tw - 8)), int(np.clip(y, 18, h - 8))), 0.46)
    if hud.get("ghost_text"):
        put_text(img, hud["ghost_text"], (14, 26), 0.5, hex_rgb(C.green))
    camera_view = hud.get("camera_view") if hud.get("gaussian_splats") else None
    if camera_view is not None and camera_view.size:
        # A single stream frame has no depth, so present it honestly as a
        # camera-facing Gaussian point image alongside the reconstructed model.
        panel_w = min(340, max(180, w // 3))
        panel_h = max(80, int(panel_w * camera_view.shape[0] / max(1, camera_view.shape[1])))
        title_h, margin = 24, 8
        if panel_h + title_h + margin * 2 < h:
            small = cv2.resize(camera_view, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
            cloud = np.zeros_like(small)
            spacing = 4
            ys = np.arange(spacing // 2, panel_h, spacing)
            xs = np.arange(spacing // 2, panel_w, spacing)
            cloud[ys[:, None], xs[None, :]] = small[ys[:, None], xs[None, :]]
            cloud = cv2.GaussianBlur(cloud, (0, 0), 1.5)
            x, y = w - panel_w - margin, title_h + margin
            x2, y2 = x + panel_w, y + panel_h + title_h
            img[y:y2, x:x2] = (img[y:y2, x:x2].astype(np.float32) * 0.28).astype(np.uint8)
            img[y + title_h:y2, x:x2] = np.clip(
                img[y + title_h:y2, x:x2].astype(np.float32) * 0.25 + cloud.astype(np.float32) * 0.75,
                0, 255).astype(np.uint8)
            cv2.rectangle(img, (x, y), (x2 - 1, y2 - 1), (80, 90, 105), 1, cv2.LINE_AA)
            put_text(img, hud.get("camera_name", "Discord camera"), (x + 6, y + 17), 0.42, (225, 230, 240))
    if hud.get("msg"):
        tw = cv2.getTextSize(hud["msg"], FONT, 0.7, 2)[0][0]
        put_text(img, hud["msg"], ((w - tw) // 2, h // 2), 0.7, (200, 205, 215), 2)
    foot = []
    if hud.get("fps") is not None:
        foot.append(f"{hud['fps']:.0f} fps")
    if "gpu" in hud:
        usage = hud["gpu"]
        foot.append(f"GPU {usage:.0f}%" if usage is not None else "GPU n/a")
    if hud.get("show_backend", True) and hud.get("backend"):
        foot.append(hud["backend"])
    if foot:
        t = "  ".join(foot)
        tw = cv2.getTextSize(t, FONT, 0.4, 1)[0][0]
        put_text(img, t, (w - tw - 10, h - 12), 0.4, (120, 128, 140))
    return img


# ------------------------------------------------------------------ facade
class Renderer:
    def __init__(self, backend):
        self.backend = backend
        self.name = backend.name

    def render(self, w, h, cam, fr):
        view = View(cam, w, h)
        hud = fr.hud or {}
        img = self.backend.render_base(w, h, view, fr.tris, fr.ghost_tris, fr.ghost_col,
                                       bool(hud.get("axes", True)),
                                       bool(hud.get("gaussian_splats", False)))
        return draw_overlay(img, view, fr)


def make_renderer(prefer_gpu=True):
    """-> (Renderer, note).  Never raises: falls back to the CPU backend and says why."""
    if prefer_gpu:
        try:
            return Renderer(GLBackend()), ""
        except Exception as exc:
            return Renderer(CPUBackend()), f"GPU renderer unavailable ({type(exc).__name__}: {exc}); using CPU renderer. pip install moderngl for smooth 3D."
    return Renderer(CPUBackend()), ""
