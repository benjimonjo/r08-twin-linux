"""Discord interaction client: clicks message buttons via POST /interactions."""
from __future__ import annotations

import logging
import random
import string
import threading
import time
from dataclasses import dataclass
from typing import Optional

import requests

from . import config
from .token import extract_token

log = logging.getLogger("robotcord")

_DISCORD_EPOCH_MS = 1420070400000


def make_nonce() -> str:
    """Discord snowflake-style nonce for the current time."""
    return str((int(time.time() * 1000) - _DISCORD_EPOCH_MS) << 22)


def make_session_id() -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(32))


@dataclass
class RoleCheck:
    has_role: Optional[bool]   # True / False, or None if it could not be determined
    role_name: str
    role_id: Optional[str] = None
    detail: str = ""


@dataclass
class Result:
    ok: bool
    status: Optional[int] = None
    body: str = ""
    skipped: Optional[str] = None  # "cooldown" | "dry_run"


class RobotClient:
    def __init__(
        self,
        token: str,
        *,
        guild_id: str = config.GUILD_ID,
        channel_id: str = config.CHANNEL_ID,
        application_id: str = config.APPLICATION_ID,
        message_id: str = config.MESSAGE_ID,
        safe_message_id: str = config.SAFE_MESSAGE_ID,
        cooldown_s: float = config.COOLDOWN_S,
        dry_run: bool = False,
        api_base: str = config.API_BASE,
        timeout: float = 10.0,
    ):
        self.token = extract_token(token)
        if not self.token and not dry_run:
            raise ValueError("A Discord token is required (or use dry_run=True)")
        self.guild_id = guild_id
        self.channel_id = channel_id
        self.application_id = application_id
        self.message_id = message_id
        self.safe_message_id = safe_message_id
        self.cooldown_s = cooldown_s
        self.dry_run = dry_run
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.session_id = make_session_id()
        self._last_success = 0.0
        self._lock = threading.Lock()
        self._http = requests.Session()

    # -- public API ---------------------------------------------------------
    @property
    def labels(self):
        return list(config.CUSTOM_IDS)

    def press(self, label: str) -> Result:
        """Press a named button (e.g. "X+", "Abort"). Thread-safe."""
        if label not in config.CUSTOM_IDS:
            raise KeyError(f"Unknown button {label!r}; see RobotClient.labels")
        custom_id = config.CUSTOM_IDS[label]
        msg_id = (
            self.safe_message_id
            if label in config.SAFE_MESSAGE_BUTTONS
            else self.message_id
        )

        with self._lock:
            if label not in config.NO_COOLDOWN:
                remaining = self.cooldown_s - (time.monotonic() - self._last_success)
                if remaining > 0:
                    log.info("[COOLDOWN] %s ignored (%.2fs left)", label, remaining)
                    return Result(False, skipped="cooldown")

            if self.dry_run:
                self._last_success = time.monotonic()
                log.info("[DRY RUN] %s (custom_id=%s, msg=%s)", label, custom_id, msg_id)
                return Result(True, skipped="dry_run")

            res = self._send_component(custom_id, msg_id)
            if not res.ok and res.status in (400, 404):
                try:
                    self.refresh_message_ids()
                except requests.RequestException as exc:
                    log.error("[ERROR] panel refresh failed: %s", exc)
                else:
                    msg_id = (self.safe_message_id
                              if label in config.SAFE_MESSAGE_BUTTONS else self.message_id)
                    res = self._send_component(custom_id, msg_id)
            return res

    def abort(self) -> Result:
        return self.press("Abort")

    def check_role(self, role_name: str = "Stream Controller") -> RoleCheck:
        """Does the logged-in user hold `role_name` in the guild? (needed to summon panels)

        Uses the user's own member object (/users/@me/guilds/{id}/member) and the
        guild's role list; role names are matched case-insensitively.
        """
        hdr = {"Authorization": self.token}
        try:
            m = self._http.get(f"{self.api_base}/users/@me/guilds/{self.guild_id}/member",
                               headers=hdr, timeout=self.timeout)
            if m.status_code in (403, 404):
                return RoleCheck(False, role_name, detail="not a member of this server")
            m.raise_for_status()
            my_roles = set(m.json().get("roles", []))
            r = self._http.get(f"{self.api_base}/guilds/{self.guild_id}/roles",
                               headers=hdr, timeout=self.timeout)
            r.raise_for_status()
        except requests.RequestException as exc:
            return RoleCheck(None, role_name, detail=f"lookup failed: {exc}")
        match = [x for x in r.json() if x.get("name", "").lower() == role_name.lower()]
        if not match:
            return RoleCheck(None, role_name, detail="no role with that name in the server")
        rid = match[0]["id"]
        return RoleCheck(rid in my_roles, role_name, rid,
                         "" if rid in my_roles else "role not assigned to this account")

    def refresh_message_ids(self, limit: int = 50) -> dict:
        """Find the newest live panel messages in the channel and use them.

        Panels are recognised by the custom_ids of their buttons, so this works
        even though the bot deletes old panels when it spawns a new one.
        Returns {"main": id|None, "safe": id|None}.
        """
        resp = self._http.get(
            f"{self.api_base}/channels/{self.channel_id}/messages",
            params={"limit": limit},
            headers={"Authorization": self.token},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        main_ids = {v for k, v in config.CUSTOM_IDS.items() if k not in config.SAFE_MESSAGE_BUTTONS}
        safe_ids = {v for k, v in config.CUSTOM_IDS.items() if k in config.SAFE_MESSAGE_BUTTONS}
        found = {"main": None, "safe": None}
        for msg in resp.json():                      # newest first
            ids = {c.get("custom_id")
                   for row in msg.get("components", [])
                   for c in row.get("components", [])}
            if found["main"] is None and ids & main_ids:
                found["main"] = msg["id"]
            if found["safe"] is None and ids & safe_ids:
                found["safe"] = msg["id"]
        if found["main"]:
            self.message_id = found["main"]
        if found["safe"]:
            self.safe_message_id = found["safe"]
        log.info("[PANELS] %s", found)
        return found

    # -- internals ----------------------------------------------------------
    def _send_component(self, custom_id: str, message_id: str) -> Result:
        payload = {
            "type": 3,
            "nonce": make_nonce(),
            "guild_id": self.guild_id,
            "channel_id": self.channel_id,
            "message_flags": 32768,
            "message_id": message_id,
            "application_id": self.application_id,
            "session_id": self.session_id,
            "data": {"component_type": 2, "custom_id": custom_id},
        }
        try:
            resp = self._http.post(
                f"{self.api_base}/interactions",
                json=payload,
                headers={"Authorization": self.token},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            log.error("[ERROR] %s", exc)
            return Result(False, body=str(exc))

        log.info("[HTTP] %s %s", resp.status_code, resp.reason)
        if resp.status_code in (200, 204):
            self._last_success = time.monotonic()
            return Result(True, status=resp.status_code)
        body = resp.text[:300]
        log.warning("[BODY] %s", body)
        return Result(False, status=resp.status_code, body=body)

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
