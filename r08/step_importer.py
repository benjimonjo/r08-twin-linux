"""Load the supplied FANUC CAD and mesh assets."""
import os
import struct
import numpy as np


def load_step_mesh(path, tolerance_mm=10.0):
    """Return a tessellated STEP triangle soup in metres."""
    try:
        import cadquery as cq
    except ImportError as exc:
        raise RuntimeError(
            "STEP model support requires cadquery (OpenCascade). Install requirements.txt."
        ) from exc
    if not os.path.isfile(path):
        raise FileNotFoundError(f"STEP robot model not found: {path}")
    try:
        shape = cq.importers.importStep(path).val()
        vertices, faces = shape.tessellate(float(tolerance_mm))
    except Exception as exc:
        raise RuntimeError(f"Could not import STEP robot model {path}: {exc}") from exc
    if not vertices or not faces:
        raise RuntimeError(f"STEP robot model contains no tessellated faces: {path}")
    raw = np.asarray([[p.x, p.y, p.z] for p in vertices], dtype=float)
    points = raw[:, (0, 2, 1)] * 0.001
    triangles = points[np.asarray(faces, dtype=np.int64)]
    positions = triangles.reshape(-1, 3)
    return _triangle_soup(positions)


def load_stl_mesh(path):
    """Return an STL triangle soup in metres, oriented with Z as vertical."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"STL robot model not found: {path}")
    with open(path, "rb") as fh:
        header = fh.read(80)
        count_data = fh.read(4)
        if len(header) != 80 or len(count_data) != 4:
            raise RuntimeError(f"Invalid binary STL header: {path}")
        count = struct.unpack("<I", count_data)[0]
        expected = 84 + count * 50
        if os.path.getsize(path) != expected:
            raise RuntimeError(f"Only binary STL files are supported: {path}")
        records = np.frombuffer(fh.read(count * 50), dtype=np.uint8).reshape(count, 50)
    raw = records[:, 12:48].view("<f4").reshape(count, 9)
    raw = raw.reshape(count, 3, 3)
    # The supplied CAD uses Y as vertical. Convert millimetres to metres and
    # map CAD (X, Y, Z) to application (X, Z, Y).
    positions = raw[:, :, (0, 2, 1)].reshape(-1, 3) * 0.001
    return _triangle_soup(positions)


def _triangle_soup(positions):
    normals = np.cross(
        positions[1::3] - positions[0::3],
        positions[2::3] - positions[0::3],
    )
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    normals = np.repeat(normals, 3, axis=0)
    color = np.tile(np.array([0.96, 0.79, 0.0], dtype=np.float32), (len(positions), 1))
    return positions.astype(np.float32), normals.astype(np.float32), color
