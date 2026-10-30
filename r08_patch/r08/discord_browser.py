"""Playwright-driven Discord panel for R08 (no token needed).

Opens a real, persistent Chromium profile on discord.com. Log in ONCE by hand;
presses are real DOM clicks on the bot's button messages, so Discord's own
JavaScript sends the interaction.

Panels are located by the buttons they contain (newest matching message wins),
not by fixed message IDs, so a panel you just summoned with /controls or
/robotpanel works immediately.

Same interface as DiscordPanel: press(name), abort(), close(), plus
summon() / ensure_panels() / list_buttons().

Discord does not put the button custom_id in the DOM, so buttons are found by
visible label. Run `python -m r08.discord_browser --list` once and edit
BUTTON_TEXT (or pass button_text=/text_overrides_file=) if a label differs.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from robotcord import config

log = logging.getLogger("r08.discord_browser")

BUTTON_TEXT: Dict[str, str] = {k: k for k in config.CUSTOM_IDS}

# Normal text channel that hosts the same control bot (not the voice-chat channel).
TEXT_CHANNEL_ID = "1191491623159332976"

# Which slash command posts which panel. ASSUMPTION: /controls posts the
# movement/camera panel and /robotpanel posts the safe-position panel (the
# "robot_panel:*" buttons). Swap here if your bot does it the other way round.
PANEL_COMMANDS = {"main": "/controls", "safe": "/robotpanel"}
# A button that only exists on that panel, used to recognise it in the chat.
PANEL_PROBE = {"main": "X+", "safe": "Safe J23"}

_STOP = object()


class BrowserPanel:
    def __init__(
        self,
        profile_dir: str | Path = "~/.config/r08/discord_profile",
        *,
        guild_id: str = config.GUILD_ID,
        channel_id: str = TEXT_CHANNEL_ID,
        button_text: Optional[Dict[str, str]] = None,
        text_overrides_file: Optional[str | Path] = None,
        panel_commands: Optional[Dict[str, str]] = None,
        role_name: str = "Stream Controller",
        # The bot answers non-controllers with a private message such as
        # "Access error: You do not have access to use the controls command.
        #  (Required permission: Stream Controller)"
        denial_text: str = "Access error",
        url: Optional[str] = None,             # override (used by tests)
        headless: bool = False,                # keep False for first login
        summon_on_start: bool = True,          # post panels if none are visible
        cooldown_s: float = config.COOLDOWN_S,
        click_timeout_ms: int = 3000,
        command_delay_ms: int = 900,           # wait for slash autocomplete
        panel_wait_ms: int = 6000,             # wait for bot to post panel
        on_result: Optional[Callable[[str, bool, str], None]] = None,
    ):
        self.profile_dir = str(Path(profile_dir).expanduser())
        self.channel_id = channel_id
        self.url = url or f"https://discord.com/channels/{guild_id}/{channel_id}"
        self.headless = headless
        self.summon_on_start = summon_on_start
        self.cooldown_s = cooldown_s
        self.click_timeout_ms = click_timeout_ms
        self.command_delay_ms = command_delay_ms
        self.panel_wait_ms = panel_wait_ms
        self.on_result = on_result
        self.role_name = role_name
        self.denial_text = denial_text
        self.last_denial: Optional[str] = None   # text of the most recent denial
        self.commands = {**PANEL_COMMANDS, **(panel_commands or {})}
        self.text = dict(BUTTON_TEXT)
        if text_overrides_file and Path(text_overrides_file).is_file():
            self.text.update(json.loads(Path(text_overrides_file).read_text()))
        if button_text:
            self.text.update(button_text)

        self._q: "queue.Queue" = queue.Queue()
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None
        self._last_ok = 0.0
        self._t = threading.Thread(target=self._run, name="discord-browser", daemon=True)
        self._t.start()

    # -- public -------------------------------------------------------------
    def wait_ready(self, timeout: float = 180.0) -> bool:
        """Block until the browser is open and (optionally) panels are posted."""
        ok = self._ready.wait(timeout)
        if self._error:
            raise self._error
        return ok

    def press(self, name: str) -> None:
        self._q.put(("press", self._canon(name)))

    def abort(self) -> None:
        """Highest priority: drop queued presses and click Abort next."""
        self._drain()
        self._q.put(("press", "Abort"))

    def summon(self, which: str = "both") -> None:
        """Post a fresh panel: which = 'main' (/controls), 'safe' (/robotpanel) or 'both'."""
        self._q.put(("summon", which))

    def ensure_panels(self) -> None:
        """Summon only the panels that are not currently in the chat."""
        self._q.put(("ensure", None))

    def check_role(self) -> Optional[bool]:
        """Send the main summon command and read the bot's verdict.

        True  = panel was posted (you have the role; note this DOES post a panel)
        False = the bot replied with its "Access error ... Stream Controller" message
        None  = nothing recognisable appeared in time
        """
        out: "queue.Queue" = queue.Queue()
        self._q.put(("check", out))
        return out.get(timeout=self.panel_wait_ms / 1000 + 30)

    def list_buttons(self) -> Dict[str, List[str]]:
        out: "queue.Queue" = queue.Queue()
        self._q.put(("list", out))
        return out.get(timeout=20)

    def close(self) -> None:
        self._q.put(_STOP)
        self._t.join(timeout=10)

    # -- internals ----------------------------------------------------------
    def _drain(self) -> None:
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def _canon(self, name: str) -> str:
        key = name.strip().lower()
        for k in config.CUSTOM_IDS:
            if k.lower() == key:
                return k
        aliases = {"backup": "Back Up", "safe j456": "Safe J1456",
                   "camera up": "Cam Up", "camera down": "Cam Down",
                   "camera left": "Cam Left", "camera right": "Cam Right",
                   "camera step+": "Cam Step+", "camera step-": "Cam Step-"}
        if key in aliases:
            return aliases[key]
        raise KeyError(f"Unknown panel button {name!r}")

    def _panel(self, page, kind: str):
        """Newest chat message in this channel that contains the panel's probe button."""
        probe = self.text[PANEL_PROBE[kind]]
        return (page.locator(f'[id^="chat-messages-{self.channel_id}-"]')
                .filter(has=page.get_by_role("button", name=probe, exact=True))
                .last)

    def _panel_present(self, page, kind: str) -> bool:
        return self._panel(page, kind).count() > 0

    def _chat(self, page):
        return page.locator(f'[id^="chat-messages-{self.channel_id}-"]')

    def _denials(self, page):
        return self._chat(page).filter(has_text=self.denial_text)

    def _panel_id(self, page, kind: str) -> Optional[str]:
        loc = self._panel(page, kind)
        return loc.get_attribute("id", timeout=500) if loc.count() else None

    def _summon_one(self, page, kind: str) -> Optional[bool]:
        """Send the slash command and wait for either a new panel or the bot's
        private "Access error" reply.  True = panel posted, False = denied
        (missing role) or no panel appeared, so callers can stop."""
        cmd = self.commands[kind]
        before_panel = self._panel_id(page, kind)
        before_denials = self._denials(page).count()

        box = page.locator('div[role="textbox"]').first
        box.click(timeout=self.click_timeout_ms)
        page.keyboard.type(cmd, delay=40)
        page.wait_for_timeout(self.command_delay_ms)   # slash autocomplete appears
        page.keyboard.press("Enter")                   # select the command
        page.wait_for_timeout(250)
        page.keyboard.press("Enter")                   # submit it

        deadline = time.monotonic() + self.panel_wait_ms / 1000
        while time.monotonic() < deadline:
            denials = self._denials(page)
            if denials.count() > before_denials:
                text = " ".join(denials.last.inner_text().split())
                self.last_denial = text
                self._report(f"summon {cmd}", False, f"access denied by bot: {text}")
                return False
            now = self._panel_id(page, kind)
            if now and now != before_panel:
                self.last_denial = None
                self._report(f"summon {cmd}", True, "")
                return True
            page.wait_for_timeout(150)
        self._report(f"summon {cmd}", False, "no panel or access error appeared (timed out)")
        return None

    def _summon(self, page, which: str) -> None:
        kinds = ["main", "safe"] if which == "both" else [which]
        for kind in kinds:
            if self._summon_one(page, kind) is False and self.last_denial:
                break   # missing role: the next command would be denied too

    def _ensure(self, page) -> None:
        for kind in ("main", "safe"):
            if not self._panel_present(page, kind):
                if self._summon_one(page, kind) is False and self.last_denial:
                    break

    def _do_press(self, page, label: str) -> None:
        if label not in config.NO_COOLDOWN:
            if time.monotonic() - self._last_ok < self.cooldown_s:
                self._report(label, False, "cooldown")
                return
        kind = "safe" if label in config.SAFE_MESSAGE_BUTTONS else "main"
        err = ""
        # The bot deletes the old panel when a new one spawns. The locator is
        # re-resolved on every attempt, so one retry picks up the replacement.
        for attempt in range(2):
            try:
                btn = self._panel(page, kind).get_by_role("button", name=self.text[label], exact=True)
                btn.click(timeout=self.click_timeout_ms)
                self._last_ok = time.monotonic()
                self._report(label, True, "")
                return
            except Exception as exc:  # noqa: BLE001
                err = str(exc).splitlines()[0]
                if attempt == 0:
                    page.wait_for_timeout(400)
        self._report(label, False, err)

    def _report(self, label: str, ok: bool, msg: str) -> None:
        log.info("[%s] %s %s", "OK" if ok else "FAIL", label, msg)
        if self.on_result:
            try:
                self.on_result(label, ok, msg)
            except Exception:
                log.exception("on_result failed")

    def _run(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            self._error = exc
            self._ready.set()
            return
        try:
            with sync_playwright() as pw:
                ctx = pw.chromium.launch_persistent_context(
                    self.profile_dir, headless=self.headless,
                    viewport={"width": 1280, "height": 800})
                page = ctx.pages[0] if ctx.pages else ctx.new_page()
                page.goto(self.url)
                if self.summon_on_start:
                    try:
                        # generous: first run you may still be logging in
                        page.locator('div[role="textbox"]').first.wait_for(timeout=120_000)
                        self._ensure(page)
                    except Exception as exc:  # noqa: BLE001
                        self._report("startup", False, str(exc).splitlines()[0])
                self._ready.set()
                while True:
                    item = self._q.get()
                    if item is _STOP:
                        break
                    kind, arg = item
                    if kind == "press":
                        self._do_press(page, arg)
                    elif kind == "summon":
                        self._summon(page, arg)
                    elif kind == "ensure":
                        self._ensure(page)
                    elif kind == "check":
                        r = self._summon_one(page, "main")
                        arg.put(None if r is None else (True if r else (False if self.last_denial else None)))
                    elif kind == "list":
                        res = {}
                        for k in ("main", "safe"):
                            loc = self._panel(page, k).get_by_role("button")
                            res[k] = [(loc.nth(i).inner_text()
                                       or loc.nth(i).get_attribute("aria-label") or "").strip()
                                      for i in range(loc.count())]
                        arg.put(res)
                ctx.close()
        except BaseException as exc:  # noqa: BLE001
            self._error = exc
            self._ready.set()
            log.exception("browser thread died")


if __name__ == "__main__":
    import argparse
    import pprint
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default=TEXT_CHANNEL_ID, help="text channel ID")
    ap.add_argument("--list", action="store_true", help="print button texts on the panels")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    p = BrowserPanel(channel_id=args.channel)
    print("Log in to Discord in the opened window if needed.")
    input("Press Enter once the channel is loaded... ")
    p.ensure_panels()
    if args.list:
        pprint.pprint(p.list_buttons())
    input("Enter to quit... ")
    p.close()
