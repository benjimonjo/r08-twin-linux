"""
guide.py - the guided-job state machine:  glide to a button -> wait for the USER's click -> check the new pose
-> fix mistakes by re-planning.  It never clicks.

    Guide(env).start(target, plan, title, now)   env = ui.App (see "env API" below)
    Guide.tick(now)                              call every UI tick
States: idle | aim (waiting for the click) | after (checking the effect) | wrong (user pressed another button) | final

env API used here (all implemented by ui.App):
    env.log(msg, tag)  env.banner(kind,title,detail,progress)  env.shared  env.settings  env.buttons{name:(x,y)}
    env.cursor  env.beacon  env.clicks  env.pop_stop()  env.hit_test(x,y)  env.passive_click(c)
    env.panel_level  env.set_panel_level(level, why)  env.apply_step_click(name)  env.flip_step_direction()
    env.point_in_own_window(x,y)  env.set_target(vec)  env.job_finished(ok)
"""
import math
import numpy as np
from .common import AXES, UNITS, MINUS, wrap180, nice_btn
from .planner import (plan_presses, describe_plan, fmt_step, read_res, step_value, N_LEVELS)

MAX_RETRIES = 2            # same press repeated when the robot did not respond
MAX_ROUNDS = 8             # corrective re-plans per job (user slips don't count)
MAX_CLICKS = 90            # safety stop for runaway jobs
MIN_PRESS_INTERVAL = 0.7   # pause after a press whose effect is too small to read
STEP_ASSUME_DELAY = 0.5
STEP_VERIFY_TIMEOUT = 3.5
SETTLE_MAX = 30.0
FINAL_MIN_WAIT = 0.6


class Guide:
    def __init__(self, env):
        self.env = env
        self.state = "idle"
        self.reset()

    def reset(self):
        self.plan, self.i, self.target, self.title = [], 0, None, ""
        self.rounds = self.clicks = self.retries = self.extra = 0
        self.press, self.t_click, self.pose_before, self.seq_before = None, 0.0, None, 0
        self.level_before, self.t_start, self.warned = 0, 0.0, set()

    @property
    def active(self):
        return self.state != "idle"

    # ---------------- starting / ending ----------------
    def start(self, target, plan, title, now):
        env = self.env
        if self.active:
            env.log("A job is already running \u2013 stop it first (Esc).", "warn")
            return False
        snap = env.shared.snapshot()
        if snap.pose is None:
            env.log("No pose reading yet \u2013 the number table has to be readable first.", "warn")
            return False
        need = {p.btn for p in plan}
        missing = [b for b in need if not env.buttons.get(b)]
        if missing:
            env.log("Button positions missing for " + ", ".join(nice_btn(b) for b in sorted(missing)) +
                    " \u2013 run Setup \u203a Panel buttons.", "err")
            return False
        if snap.has_step and snap.step_level is not None and snap.step_level != env.panel_level:
            env.set_panel_level(snap.step_level, "read from the screen")
        self.reset()
        self.target = np.array([np.nan if t is None else float(t) for t in target], float)
        self.plan, self.title, self.t_start = list(plan), title, now
        env.set_target(self.target)
        env.log(f"\u25b6 {title}: {len(plan)} click{'s' if len(plan) != 1 else ''}"
                + (f"   {describe_plan(plan)}" if plan else "   (already there)"), "act")
        if plan:
            self._aim(now)
        else:
            self._enter_final(now)
        return True

    def cancel(self, reason="Stopped"):
        if not self.active:
            return
        env = self.env
        self.state = "idle"
        env.cursor.stop()
        env.beacon.hide()
        env.log(f"\u25a0 {reason}", "warn")
        env.banner("idle", "Stopped", reason, None)
        env.job_finished(False)

    def _fail(self, why):
        env = self.env
        self.state = "idle"
        env.cursor.stop()
        env.beacon.hide()
        env.log(f"\u2716 {why}", "err")
        env.banner("fail", "Needs attention", why, None)
        env.job_finished(False)

    def _done(self, now, snap):
        env = self.env
        self.state = "idle"
        env.beacon.hide()
        parts = [f"{AXES[ax]} {snap.pose[ax]:+.1f}" for ax in range(6) if math.isfinite(self.target[ax])]
        msg = f"{self.clicks} click{'s' if self.clicks != 1 else ''}, {now - self.t_start:.0f}s \u2013 " + "  ".join(parts)
        env.log(f"\u2714 {self.title}: verified.  {msg}", "ok")
        env.banner("done", "Verified", msg, None)
        env.job_finished(True)

    # ---------------- states ----------------
    def _aim(self, now):
        env, p = self.env, self.plan[self.i]
        self.press, self.state = p, "aim"
        self.level_before = env.panel_level
        x, y = env.buttons[p.btn]
        env.cursor.glide(x, y)
        n = len(self.plan)
        env.beacon.show(x, y, f"{nice_btn(p.btn)}   {self.i + 1}/{n}", "aim")
        env.banner("aim", f"Click {nice_btn(p.btn)}", f"{self.i + 1} of {n}  \u00b7  {p.label}", (self.i, n))
        if "own" not in self.warned and env.point_in_own_window(x, y):
            self.warned.add("own")
            env.log("The panel button is hidden behind this window \u2013 move the R08 window aside.", "warn")

    def _enter_final(self, now):
        self.state, self.t_click = "final", now
        self.env.beacon.hide()
        self.env.cursor.stop()
        self.env.banner("check", "Checking the pose", "comparing the read-out with the target", None)

    def _settled(self, now, snap):
        if now - snap.t_change < self.env.settings["settle"]:
            return False
        if snap.has_status and (snap.busy or now - snap.last_moving_t < 0.35):
            return False
        return True

    def tick(self, now):
        env = self.env
        clicks = env.clicks.drain()
        if not self.active:
            for c in clicks:
                env.passive_click(c)
            return
        if env.pop_stop():
            self.cancel("Stopped (Esc)")
            return
        snap = env.shared.snapshot()
        if snap.pose is None:
            return
        for (_, x, y) in clicks:
            name = env.hit_test(x, y)
            if name is None:
                continue
            if name == "Abort":
                self.cancel("Panel Abort was pressed")
                return
            if self.state == "aim":
                if name == self.press.btn:
                    self._clicked(now, snap)
                else:
                    self._wrong(name, now, snap)
                break
            self.extra += 1
        if self.state == "after":
            self._poll_after(now, snap)
        elif self.state == "wrong":
            if now - self.t_click >= 1.2 and self._settled(now, snap):
                self._replan(now, snap, "different button pressed", count=False)
        elif self.state == "final":
            self._poll_final(now, snap)

    def _clicked(self, now, snap):
        env = self.env
        self.clicks += 1
        if self.clicks > MAX_CLICKS:
            self._fail("Stopped after too many clicks \u2013 something is not responding as expected.")
            return
        self.t_click, self.pose_before, self.seq_before, self.extra = now, snap.pose.copy(), snap.seq, 0
        self.state = "after"
        env.beacon.set_mode("check")
        env.banner("check", "Checking\u2026", f"waiting for the robot after {nice_btn(self.press.btn)}",
                   (self.i + 1, len(self.plan)))

    def _wrong(self, name, now, snap):
        env, p = self.env, self.press
        env.log(f"You clicked {nice_btn(name)} but {nice_btn(p.btn)} was next \u2013 I'll re-plan from where the robot ends up.", "warn")
        if name.startswith("Step"):
            env.apply_step_click(name)
        self.state, self.t_click = "wrong", now
        env.beacon.set_mode("check")
        env.banner("check", "Checking\u2026", f"you pressed {nice_btn(name)}", None)

    def _poll_after(self, now, snap):
        env, p, S = self.env, self.press, self.env.settings
        dt = now - self.t_click
        if p.kind == "step":
            self._poll_step(now, snap, p, dt)
            return
        if not S["verify_each"]:
            if dt >= 0.35:
                self._advance(now)
            return
        saw_moving = snap.last_moving_t >= self.t_click
        if not p.verifiable:                                   # too small for the read-out to show
            if snap.has_status:
                if dt < 0.6:
                    return
                if saw_moving and not self._settled(now, snap) and dt < SETTLE_MAX:
                    return
                if not saw_moving and dt < S["motion_wait"]:
                    return
                if not saw_moving and "tiny" not in self.warned:
                    self.warned.add("tiny")
                    env.log("No MOVING banner seen for a tiny step \u2013 the final check will confirm.", "dim")
            elif dt < MIN_PRESS_INTERVAL:
                return
            self._advance(now)
            return
        moved = snap.seq != self.seq_before and snap.t_change >= self.t_click
        if not moved:                      # banner may already say MOVING while the numbers lag behind
            if dt > S["motion_wait"]:
                self._no_motion(now)
            return
        if self._settled(now, snap) or dt > SETTLE_MAX:
            self._evaluate(now, snap)

    def _poll_step(self, now, snap, p, dt):
        env = self.env
        want, before = p.level, self.level_before
        direction = 1 if want > before else -1
        if snap.has_step and snap.step_level is not None and snap.step_t >= self.t_click:
            lv = snap.step_level
            if lv == want:
                env.set_panel_level(want, None)
                env.log(f"\u2714 step size now {fmt_step(want)}", "ok")
                self._advance(now)
                return
            if lv == before - direction:                       # went the OTHER way: Step+/Step- are swapped
                env.flip_step_direction()
                env.set_panel_level(lv, None)
                self._replan(now, snap, "Step+/Step\u2212 work the other way round on your panel \u2013 learned")
                return
            env.set_panel_level(lv, "read from the screen")
            self._replan(now, snap, "step size differs from plan")
            return
        if snap.has_step:
            if dt > STEP_VERIFY_TIMEOUT:
                if snap.step_level is not None and snap.step_level != want:
                    self._retry(now, f"{nice_btn(p.btn)} had no effect")
                else:
                    env.set_panel_level(want, None)
                    self._advance(now)
            return
        if dt >= STEP_ASSUME_DELAY:
            env.set_panel_level(want, None)
            self._advance(now)

    def _evaluate(self, now, snap):
        env, p = self.env, self.press
        ax, res = p.axis, read_res(p.axis)
        d = snap.pose[ax] - self.pose_before[ax]
        if ax >= 3:
            d = wrap180(d)
        tol = max(abs(p.expect) * 0.3, res * 1.6)
        if abs(d - p.expect) <= tol:
            env.log(f"\u2714 {nice_btn(p.btn)}  {d:+.2f}{UNITS[ax]}", "ok")
            self._side_effects(snap)
            self.retries = 0
            self._advance(now)
            return
        if abs(d) <= res * 1.2:
            self._no_motion(now)
            return
        best = None
        for lv in range(N_LEVELS):
            sv = step_value(lv, ax)
            if d * p.expect > 0 and abs(abs(d) - sv) <= max(0.3 * sv, res * 1.6):
                if best is None or abs(abs(d) - sv) < best[0]:
                    best = (abs(abs(d) - sv), lv)
        env.log(f"\u26a0 {nice_btn(p.btn)} moved {d:+.2f}{UNITS[ax]}, expected {p.expect:+.2f}", "warn")
        if best is not None and best[1] != p.level:
            env.set_panel_level(best[1], "from the measured move")
        self._replan(now, snap, "unexpected move size")

    def _side_effects(self, snap):
        p = self.press
        for ax in range(6):
            if ax == p.axis:
                continue
            d = snap.pose[ax] - self.pose_before[ax]
            if ax >= 3:
                d = wrap180(d)
            if abs(d) > max(read_res(ax) * 2.5, 0.25 * abs(p.expect)) and ("side%d" % ax) not in self.warned:
                self.warned.add("side%d" % ax)
                self.env.log(f"Note: {nice_btn(p.btn)} also changed {AXES[ax]} by {d:+.1f} (coupled angles?). "
                             "The final check will correct it if you asked for that axis.", "dim")

    def _no_motion(self, now):
        self._retry(now, f"No motion after {nice_btn(self.press.btn)}")

    def _retry(self, now, why):
        env = self.env
        self.retries += 1
        if self.retries > MAX_RETRIES:
            self._fail(f"{why} after {MAX_RETRIES + 1} tries \u2013 the robot may be at a limit, the panel disconnected, "
                       "or its step size is much smaller than I think (set it with \u2018Panel is here\u2019).")
            return
        env.log(f"\u26a0 {why} \u2013 click {nice_btn(self.press.btn)} again (try {self.retries + 1}/{MAX_RETRIES + 1}).", "warn")
        self._aim(now)

    def _advance(self, now):
        self.i += 1
        self.retries = 0
        if self.i >= len(self.plan):
            self._enter_final(now)
        else:
            self._aim(now)

    def _residual(self, pose):
        S, out = self.env.settings, []
        for ax in range(6):
            t = self.target[ax]
            if not math.isfinite(t):
                continue
            e = t - pose[ax]
            if ax >= 3:
                e = wrap180(e)
            out.append((ax, e, abs(e) <= (S["pos_tol"] if ax < 3 else S["ang_tol"])))
        return out

    def _poll_final(self, now, snap):
        dt = now - self.t_click
        if dt < FINAL_MIN_WAIT or (not self._settled(now, snap) and dt < SETTLE_MAX):
            return
        res = self._residual(snap.pose)
        if all(ok for _, _, ok in res):
            self._done(now, snap)
            return
        worst = max((r for r in res if not r[2]), key=lambda r: abs(r[1]))
        self._replan(now, snap, f"{AXES[worst[0]]} is {worst[1]:+.2f}{UNITS[worst[0]]} off")

    def _replan(self, now, snap, reason, count=True):
        env, S = self.env, self.env.settings
        self.rounds += 1 if count else 0
        if self.rounds > MAX_ROUNDS:
            self._fail("Couldn't reach the target within tolerance \u2013 the robot may be at a limit or the target is out of reach.")
            return
        plan = plan_presses(snap.pose, self.target, env.panel_level, S["pos_tol"], S["ang_tol"], S["step_plus_larger"])
        env.log(f"\u21bb Re-planning ({reason}): {len(plan)} click{'s' if len(plan) != 1 else ''}"
                + (f"   {describe_plan(plan)}" if plan else ""), "warn")
        self.plan, self.i, self.retries = plan, 0, 0
        if plan:
            self._aim(now)
        else:
            self._enter_final(now)
