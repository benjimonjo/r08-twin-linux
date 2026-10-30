"""Discord-backed jog panel for R08.

Sends the robot-panel button presses over Discord's REST API (via the
`robotcord` package) instead of moving the mouse and clicking the calibrated
panel buttons.  Presses run on a background thread so the UI never blocks;
Abort always jumps the queue.

Usage (e.g. in the controller "direct mode" path):

    from r08.discord_panel import DiscordPanel
    panel = DiscordPanel.from_env()          # None if no token configured
    if panel:
        panel.press("X+")
        panel.press("Camera Step-")
        panel.abort()
"""
from __future__ import annotations

import logging
import os
import queue
import threading
from pathlib import Path
from typing import Callable, Optional

from robotcord import RobotClient

log = logging.getLogger("r08.discord_panel")

# R08's panel names -> robotcord labels. Anything not listed is passed through.
ALIASES = {
    "backup": "Back Up",
    "back up": "Back Up",
    "safe j456": "Safe J1456",
    "safe j1456": "Safe J1456",
    "safe j23": "Safe J23",
    "safe abort": "Safe Abort",
    "abort": "Abort",
    "camera up": "Cam Up", "camera down": "Cam Down",
    "camera left": "Cam Left", "camera right": "Cam Right",
    "camera step+": "Cam Step+", "camera step-": "Cam Step-",
    "robot step+": "Robot Step+", "robot step-": "Robot Step-",
    "reset camera": "Reset Camera", "switch camera": "Switch Camera",
    "zoom+": "Zoom+", "zoom-": "Zoom-",
}
_CANON = None


def _canonical(name: str) -> str:
    global _CANON
    if _CANON is None:
        from robotcord import config
        _CANON = {k.lower(): k for k in config.CUSTOM_IDS}
    key = name.strip().lower()
    if key in ALIASES:
        return ALIASES[key]
    if key in _CANON:
        return _CANON[key]
    raise KeyError(f"Unknown panel button {name!r}")


def load_token() -> Optional[str]:
    """DISCORD_TOKEN env var, else ~/.config/r08/discord_token (kept out of the repo)."""
    tok = os.environ.get("DISCORD_TOKEN")
    if tok:
        return tok
    p = Path.home() / ".config" / "r08" / "discord_token"
    if p.is_file():
        return p.read_text().strip()
    return None


class DiscordPanel:
    def __init__(self, client: RobotClient,
                 on_result: Optional[Callable[[str, object], None]] = None):
        self.client = client
        self.on_result = on_result
        self._q: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name="discord-panel", daemon=True)
        self._t.start()

    @classmethod
    def from_env(cls, dry_run: bool = False, verify_role: bool = False,
                 role_name: str = "Stream Controller", **kw) -> Optional["DiscordPanel"]:
        """verify_role=True raises PermissionError if the account lacks the role
        (an undeterminable check only logs a warning)."""
        tok = load_token()
        if not tok and not dry_run:
            return None
        client = RobotClient(tok or "", dry_run=dry_run, **kw)
        if verify_role and not dry_run:
            rc = client.check_role(role_name)
            if rc.has_role is False:
                raise PermissionError(f"Missing the {role_name!r} role: {rc.detail}")
            if rc.has_role is None:
                log.warning("Could not verify %r role: %s", role_name, rc.detail)
        return cls(client)

    # -- API ----------------------------------------------------------------
    def press(self, name: str) -> None:
        """Queue a press (non-blocking). Raises KeyError for unknown names."""
        self._q.put(_canonical(name))

    def abort(self) -> None:
        """Send Abort immediately, discarding any queued presses."""
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        threading.Thread(target=self._send, args=("Abort",), daemon=True).start()

    def close(self) -> None:
        self._stop.set()
        self._q.put("")  # wake worker
        self.client.close()

    # -- internals ----------------------------------------------------------
    def _send(self, label: str) -> None:
        res = self.client.press(label)
        if self.on_result:
            try:
                self.on_result(label, res)
            except Exception:
                log.exception("on_result callback failed")

    def _run(self) -> None:
        while not self._stop.is_set():
            label = self._q.get()
            if label:
                self._send(label)
