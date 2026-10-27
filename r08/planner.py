"""
planner.py - fewest jog-panel button presses needed to move the read-out from one pose to another.

The panel moves by fixed step sizes (STEP_LEVELS).  Positions live on a 0.05 in lattice, angles on 0.1 deg.
plan_presses(cur, tgt, panel_level, pos_tol, ang_tol, plus_larger) -> list[Press]
Press.kind is "jog" (move an axis) or "step" (change the panel step size).  No GUI code here.
"""
import math
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from .common import STEP_LEVELS, AXES, UNITS, MINUS, POS_READ_RES, ANG_READ_RES, wrap180, nice_btn

POS_UNIT, ANG_UNIT = 0.05, 0.1
POS_COINS = [round(p / POS_UNIT) for p, _ in STEP_LEVELS]
ANG_COINS = [round(a / ANG_UNIT) for _, a in STEP_LEVELS]
N_LEVELS = len(STEP_LEVELS)


def step_value(level, axis):
    return STEP_LEVELS[level][0 if axis < 3 else 1]


def read_res(axis):
    return POS_READ_RES if axis < 3 else ANG_READ_RES


def fmt_step(level):
    p, a = STEP_LEVELS[level]
    return f"{p:g} in / {a:g}\u00b0"


@dataclass
class Press:
    btn: str                  # panel button name, e.g. "X+", "Step-"
    kind: str = "jog"         # "jog" or "step"
    axis: int = -1
    sign: int = 0
    level: int = 0            # jog: level the panel must be at.  step: level AFTER the press
    expect: float = 0.0       # jog: expected change of the read-out on `axis`
    verifiable: bool = True   # big enough for the read-out to show it
    label: str = ""


@lru_cache(maxsize=8192)
def min_press_counts(D, coins):
    """Signed press counts (aligned with `coins`) reaching integer distance D with the fewest presses."""
    if D == 0:
        return tuple(0 for _ in coins)
    limit = abs(D) + max(coins)
    prev = {0: None}
    dq = deque([0])
    while dq:
        v = dq.popleft()
        for i, c in enumerate(coins):
            for s in (1, -1):
                w = v + s * c
                if w in prev or abs(w) > limit:
                    continue
                prev[w] = (v, i, s)
                if w == D:
                    counts = [0] * len(coins)
                    while prev[w] is not None:
                        pv, ci, sg = prev[w]
                        counts[ci] += sg
                        w = pv
                    return tuple(counts)
                dq.append(w)
    return None


def step_button_for(to_smaller, plus_larger):
    plus = (not to_smaller) if plus_larger else to_smaller
    return "Step+" if plus else "Step-"


def step_presses(a, b, plus_larger=True):
    out, cur = [], a
    while cur != b:
        to_smaller = b > cur
        cur += 1 if to_smaller else -1
        out.append(Press(btn=step_button_for(to_smaller, plus_larger), kind="step", level=cur,
                         label=f"step size \u2192 {fmt_step(cur)}"))
    return out


def jog_press(axis, sign, level):
    exp = sign * step_value(level, axis)
    return Press(btn=f"{AXES[axis]}{'+' if sign > 0 else '-'}", kind="jog", axis=axis, sign=sign, level=level,
                 expect=exp, verifiable=abs(exp) >= 2.5 * read_res(axis),
                 label=f"{AXES[axis]} {'+' if sign > 0 else MINUS}{step_value(level, axis):g}{UNITS[axis]}")


def plan_presses(cur, tgt, panel_level, pos_tol, ang_tol, plus_larger=True):
    """Press list taking the read-out from `cur` to `tgt` (NaN / None = don't care)."""
    deltas = []
    for ax in range(6):
        t = tgt[ax]
        if t is None or not math.isfinite(t):
            continue
        d = t - cur[ax]
        if ax >= 3:
            d = wrap180(d)
        if abs(d) <= (pos_tol if ax < 3 else ang_tol):
            continue
        deltas.append((ax, int(round(d / (POS_UNIT if ax < 3 else ANG_UNIT)))))
    deltas = [(a, D) for a, D in deltas if D != 0]
    if not deltas:
        return []
    best = None
    for cutoff in range(N_LEVELS):
        counts, ok = {}, True
        for ax, D in deltas:
            c = min_press_counts(D, tuple((POS_COINS if ax < 3 else ANG_COINS)[cutoff:]))
            if c is None:
                ok = False
                break
            counts[ax] = c
        if not ok:
            continue
        used = sorted({cutoff + k for c in counts.values() for k, n in enumerate(c) if n})
        n_press = sum(abs(n) for c in counts.values() for n in c)
        lo, hi = used[0], used[-1]
        fwd = abs(panel_level - lo) + (hi - lo)       # coarse -> fine
        rev = abs(panel_level - hi) + (hi - lo)       # fine -> coarse
        cost = n_press + min(fwd, rev)
        if best is None or cost < best[0]:
            best = (cost, cutoff, counts, used, fwd <= rev)
    _, cutoff, counts, used, forward = best
    plan, level = [], panel_level
    for lv in (used if forward else used[::-1]):
        plan += step_presses(level, lv, plus_larger)
        level = lv
        jogs = []
        for ax, c in counts.items():
            n = c[lv - cutoff]
            if n:
                jogs.append((ax, 1 if n > 0 else -1, abs(n)))
        # lift before moving sideways, lower afterwards
        jogs.sort(key=lambda j: (0 if (j[0] == 2 and j[1] > 0) else 2 if (j[0] == 2 and j[1] < 0) else 1, j[0]))
        for ax, sg, n in jogs:
            plan += [jog_press(ax, sg, lv) for _ in range(n)]
    return plan


def describe_plan(plan):
    groups = []
    for p in plan:
        k = (p.btn, p.level if p.kind == "jog" else -1, p.axis)
        if groups and groups[-1][0] == k:
            groups[-1][1] += 1
        else:
            groups.append([k, 1])
    out = []
    for (btn, lv, ax), n in groups:
        s = nice_btn(btn) + (f" \u00d7{n}" if n > 1 else "")
        if lv >= 0:
            s += f" @{step_value(lv, ax):g}"
        out.append(s)
    return "  \u203a  ".join(out)
