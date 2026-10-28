"""
kinematics.py - FANUC R-2000iA/200F forward / inverse kinematics.   Units: metres, radians.

Robot data (datasheet: reach 2483 mm, J2 offset 312 mm, lower arm 1075 mm; forearm length chosen so that
312 + 1075 + 1096 = 2483 mm = J5-centre reach).  EDIT HERE if you measure better numbers.

JOINT_ORIGINS[i] = position of joint i relative to joint i-1, expressed in joint i-1's frame
JOINT_AXES[i]    = rotation axis of joint i in its own frame
At q = 0: lower arm points straight up, forearm points forward (+X).

TOOL: three round spikes on a flat plate.  SPIKES_IN = (length from flange face in inches, sideways offset in inches).
TCP (the X Y Z the panel shows) = tip of spike TCP_SPIKE.  TOOL["axis"/"sign"] choose which flange axis the spike
row runs along; the UI has buttons to flip/turn it if the model disagrees with the photos.

Public API:
    forward_kinematics(q) -> (T_tool 4x4, joint_pos (6,3), joint_axes (6,3), T_flange 4x4, tip (3,))
    table_pose_to_target(pose) -> 4x4 target in base frame
    solve_ik(target, previous_q) -> (q, pos_err_m, rot_err_rad)
    table_to_base(p_inches) -> point in base frame (metres)
"""
import math
import os
import json
import numpy as np
from .common import IN, wrap180

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "FANUC_R2000iA_200F_model")
ROBOT_FILES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "robot_files")
_MODEL_DATA = {
    "nominal_joint_origins_m": {
        "J1": [0.0, 0.0, 0.9], "J2": [0.0, 0.0, 0.175],
        "J3": [1.11, 0.0, 0.0], "J4": [1.075, 0.0, 0.0],
        "J5": [0.298, 0.0, 0.0], "J6": [0.2, 0.0, 0.0],
    },
    "joint_limits": [
        {"range_deg": [-360.0, 360.0], "axis": [0, 0, 1]},
        {"range_deg": [-67.5, 67.5], "axis": [0, 1, 0]},
        {"range_deg": [-183.7, 183.7], "axis": [0, 1, 0]},
        {"range_deg": [-360.0, 360.0], "axis": [1, 0, 0]},
        {"range_deg": [-125.0, 125.0], "axis": [0, 1, 0]},
        {"range_deg": [-360.0, 360.0], "axis": [1, 0, 0]},
    ],
}
_model_data_path = os.path.join(MODEL_DIR, "kinematics.json")
if os.path.isfile(_model_data_path):
    with open(_model_data_path, encoding="utf-8") as _fh:
        _MODEL_DATA = json.load(_fh)
_robot_json_path = os.path.join(ROBOT_FILES_DIR, "robot.json")
_ROBOT_DH = None
if os.path.isfile(_robot_json_path):
    with open(_robot_json_path, encoding="utf-8") as _fh:
        _ROBOT_DH = json.load(_fh)
if _ROBOT_DH is not None:
    _MODEL_DATA["nominal_joint_origins_m"] = {
        f"J{i + 1}": [0.0, 0.0, float(item["d_mm"]) * 0.001]
        for i, item in enumerate(_ROBOT_DH["dh"])
    }
    _MODEL_DATA["joint_limits"] = [
        {"range_deg": [_ROBOT_DH["limits"]["min_deg"][i], _ROBOT_DH["limits"]["max_deg"][i]],
         "axis": [0.0, 0.0, 1.0]}
        for i in range(6)
    ]
J2_HEIGHT = float(_MODEL_DATA["nominal_joint_origins_m"]["J1"][2])
JOINT_ORIGINS = np.array([_MODEL_DATA["nominal_joint_origins_m"][f"J{i}"] for i in range(1, 7)], float)
JOINT_AXES = np.array([item["axis"] for item in _MODEL_DATA["joint_limits"]], float)
FLANGE_OFFSET_M = np.zeros(3)
# wide on purpose: the real robot decides, the twin only has to follow
INFINITE_JOINTS = (3, 5)
JOINT_LIMITS_DEG = np.array([item["range_deg"] for item in _MODEL_DATA["joint_limits"]], float)
JOINT_LIMITS_RAD = np.radians(JOINT_LIMITS_DEG)

POSITION_UNIT_TO_METERS = 0.0254
TABLE_ORIGIN_IN_BASE_M = np.array([0.0, 0.0, J2_HEIGHT])
TABLE_FRAME_TO_ROBOT_R = np.eye(3)
TOOL_FRAME_Y_DEG = 90.0

SPIKES_IN = [(10.8, -1.5), (7.2, 0.0), (14.4, 1.5)]       # medium, short, long
TCP_SPIKE = 2
TOOL = {"axis": 2, "sign": 1.0}

IK_MAX_ITERATIONS = 120
IK_DAMPING = 0.025
IK_ORIENTATION_WEIGHT_M = 0.35
IK_MAX_STEP_DEG = 10.0
GOOD_POSITION_ERROR_MM = 8.0
GOOD_ORIENTATION_ERROR_DEG = 3.0


def rot_axis(axis, ang):
    c, s = math.cos(ang), math.sin(ang)
    x, y, z = axis
    k = 1.0 - c
    return np.array([[c + x * x * k, x * y * k - z * s, x * z * k + y * s],
                     [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
                     [z * x * k - y * s, z * y * k + x * s, c + z * z * k]])


def rot_x(a): return rot_axis((1, 0, 0), a)
def rot_y(a): return rot_axis((0, 1, 0), a)
def rot_z(a): return rot_axis((0, 0, 1), a)


TOOL_FRAME_ROTATION = rot_y(math.radians(TOOL_FRAME_Y_DEG))


def thick_axis():
    return 1 if TOOL["axis"] == 2 else 2


def tool_local_tip():
    length_in, offset_in = SPIKES_IN[TCP_SPIKE]
    return np.array([offset_in, 0.0, length_in]) * IN


def forward_kinematics(q):
    if _ROBOT_DH is not None:
        return _forward_kinematics_dh(q)
    Rm = np.eye(3)
    p = np.zeros(3)
    jp = np.empty((6, 3))
    ja = np.empty((6, 3))
    for i in range(6):
        p = p + Rm @ JOINT_ORIGINS[i]
        jp[i] = p
        ja[i] = Rm @ JOINT_AXES[i]
        Rm = Rm @ rot_axis(JOINT_AXES[i], q[i])
    p_fl = p + Rm @ FLANGE_OFFSET_M
    tip = p_fl + Rm @ tool_local_tip()
    T = np.eye(4)
    T[:3, :3] = Rm @ TOOL_FRAME_ROTATION
    T[:3, 3] = tip
    Tf = np.eye(4)
    Tf[:3, :3] = Rm
    Tf[:3, 3] = p_fl
    return T, jp, ja, Tf, tip


def _dh_matrix(item, q):
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


def _forward_kinematics_dh(q):
    q = np.asarray(q, float)
    cur = np.eye(4)
    jp = np.empty((6, 3))
    ja = np.empty((6, 3))
    for i, (item, angle) in enumerate(zip(_ROBOT_DH["dh"], q)):
        alpha = float(item["alpha_rad"])
        a = float(item["a_mm"]) * 0.001
        ca, sa = math.cos(alpha), math.sin(alpha)
        pre = cur @ np.array([
            [1.0, 0.0, 0.0, a],
            [0.0, ca, -sa, 0.0],
            [0.0, sa, ca, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        jp[i] = pre[:3, 3]
        ja[i] = pre[:3, :3] @ np.array([0.0, 0.0, 1.0])
        rotation = np.eye(4)
        rotation[:3, :3] = rot_axis(
            (0.0, 0.0, 1.0), float(item["theta_offset_rad"]) + angle
        )
        cur = pre @ rotation
        cur[:3, 3] += cur[:3, :3] @ np.array([0.0, 0.0, float(item["d_mm"]) * 0.001])
    Tf = cur.copy()
    tip = Tf[:3, 3] + Tf[:3, :3] @ tool_local_tip()
    T = cur.copy()
    T[:3, 3] = tip
    return T, jp, ja, Tf, tip


def table_to_base(p_in):
    return TABLE_FRAME_TO_ROBOT_R @ (np.asarray(p_in, float) * POSITION_UNIT_TO_METERS) + TABLE_ORIGIN_IN_BASE_M


def table_pose_to_target(pose):
    pos = np.asarray(pose[:3], float)
    x, y, z = np.radians(np.asarray(pose[3:], float))
    T = np.eye(4)
    T[:3, 3] = table_to_base(pos)
    T[:3, :3] = TABLE_FRAME_TO_ROBOT_R @ (rot_z(z) @ rot_y(y) @ rot_x(x))
    return T


def rotation_error_vector(cur, tgt):
    R = tgt @ cur.T
    ang = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    if ang < 1e-9:
        return np.zeros(3)
    if abs(math.pi - ang) < 1e-5:
        vals, vecs = np.linalg.eig(R)
        axis = np.real(vecs[:, int(np.argmin(np.abs(vals - 1.0)))])
        n = np.linalg.norm(axis)
        return axis / n * ang if n > 1e-9 else np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(ang))
    return axis * ang


def clamp_q(q):
    return np.clip(np.asarray(q, float), JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])


def workspace_bounds(sample_count=20000, seed=8):
    """Estimate the TCP workspace from uniformly sampled joint-limit poses.

    The result is an estimate, not a collision or singularity boundary. Increase
    sample_count when tighter bounds are needed.
    """
    rng = np.random.default_rng(seed)
    samples = rng.uniform(JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1],
                          size=(int(sample_count), 6))
    points = np.empty((len(samples), 3), float)
    for index, q in enumerate(samples):
        points[index] = forward_kinematics(q)[4]
    return {
        "min_m": points.min(axis=0),
        "max_m": points.max(axis=0),
        "max_radius_m": float(np.linalg.norm(points, axis=1).max()),
        "samples": len(points),
    }


def unwrap_q_to_reference(q, reference):
    q = np.asarray(q, float).copy()
    if reference is None:
        for i in INFINITE_JOINTS:
            q[i] = math.radians(wrap180(math.degrees(q[i])))
        return q
    reference = np.asarray(reference, float)
    for i in INFINITE_JOINTS:
        q[i] = reference[i] + math.radians(wrap180(math.degrees(q[i] - reference[i])))
    return q


def solve_ik_once(target, seed):
    q = clamp_q(seed)
    max_step = math.radians(IK_MAX_STEP_DEG)
    lam = (IK_DAMPING ** 2) * np.eye(6)
    for _ in range(IK_MAX_ITERATIONS):
        cur, jp, ja, _, _ = forward_kinematics(q)
        pe = target[:3, 3] - cur[:3, 3]
        re = rotation_error_vector(cur[:3, :3], target[:3, :3])
        if np.linalg.norm(pe) < 0.0002 and np.linalg.norm(re) < math.radians(0.2):
            break
        J = np.empty((6, 6))
        J[:3, :] = np.cross(ja, cur[:3, 3] - jp).T
        J[3:, :] = (ja * IK_ORIENTATION_WEIGHT_M).T
        err = np.concatenate([pe, re * IK_ORIENTATION_WEIGHT_M])
        try:
            dq = J.T @ np.linalg.solve(J @ J.T + lam, err)
        except np.linalg.LinAlgError:
            break
        big = float(np.max(np.abs(dq)))
        if big > max_step:
            dq *= max_step / big
        q = clamp_q(q + dq)
    cur = forward_kinematics(q)[0]
    return (q, float(np.linalg.norm(target[:3, 3] - cur[:3, 3])),
            float(np.linalg.norm(rotation_error_vector(cur[:3, :3], target[:3, :3]))))


def make_ik_seeds(target, previous):
    seeds = [np.asarray(previous, float)] if previous is not None else []
    p = target[:3, 3] - TABLE_ORIGIN_IN_BASE_M
    j1 = math.degrees(math.atan2(p[1], p[0]))
    for g in ([20, 20, -170, 90, 0], [50, 40, -170, 80, 0], [50, 40, 10, -80, -180], [0, 0, 0, 0, 0],
              [-30, 60, 0, 30, 0], [80, -30, 0, -30, 0], [120, 30, 180, 60, 180], [20, -40, -180, -60, 0]):
        seeds.append(np.radians([j1] + g))
    return seeds


def joint_distance(a, b):
    if a is None or b is None:
        return 0.0
    a, b = np.degrees(a), np.degrees(b)
    total = 0.0
    for i in range(6):
        d = a[i] - b[i]
        if i in (3, 5):
            d = (d + 180.0) % 360.0 - 180.0
        total += abs(d)
    return total


def solve_ik(target, previous=None):
    """Try the previous solution first (cheap, continuous); fall back to a multi-seed search."""
    if previous is not None:
        q, pe, re = solve_ik_once(target, previous)
        q = unwrap_q_to_reference(q, previous)
        if pe * 1000 <= GOOD_POSITION_ERROR_MM and math.degrees(re) <= GOOD_ORIENTATION_ERROR_DEG:
            return q, pe, re
    best = None
    for seed in make_ik_seeds(target, previous):
        q, pe, re = solve_ik_once(target, seed)
        q = unwrap_q_to_reference(q, previous)
        score = pe * 1000 + math.degrees(re) * 2 + joint_distance(q, previous) * 0.01
        if best is None or score < best[0]:
            best = (score, q, pe, re)
    return best[1], best[2], best[3]
