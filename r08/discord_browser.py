"""Playwright-driven Robocord panel for R08 (no token needed).

Opens a real, persistent Chromium profile on discord.com, you log in ONCE by
hand, and presses are real DOM clicks on the bot's button messages. Discord's
own JavaScript sends the interaction, so headers/session/fingerprint are the
genuine client's.

Same interface as DiscordPanel: press(name), abort(), close().

NOTE: Discord does not put the button custom_id in the DOM, so buttons are
found by their visible label. Run `python -m r08.discord_browser --list`
once and fill in BUTTON_TEXT (or a JSON override file) if a label differs.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import robotcord_config as config

log = logging.getLogger("r08.discord_browser")

# robotcord label -> visible button text. Defaults assume the text equals the
# label; EDIT after running --list. A "{n}" style is not needed: exact match.
BUTTON_TEXT: Dict[str, str] = {k: k for k in config.CUSTOM_IDS}
TEXT_CHANNEL_ID = "1191491623159332976"
PANEL_COMMANDS = {"main": "/controls", "safe": "/robotpanel"}
PANEL_PROBE = {"main": "X+", "safe": "Safe J23"}
BROWSER_WIDTH = 1280
BROWSER_HEIGHT = 1400

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
        url: Optional[str] = None,          # override (used by tests)
        headless: bool = False,             # retained for API compatibility; profile controls mode
        summon_on_start: bool = True,
        debug_visible: bool = False,
        cooldown_s: float = config.COOLDOWN_S,
        click_timeout_ms: int = 3000,
        on_result: Optional[Callable[[str, bool, str], None]] = None,
    ):
        self.profile_dir = str(Path(profile_dir).expanduser())
        self.login_marker = Path(self.profile_dir) / ".r08_logged_in"
        self.channel_id = channel_id
        self.url = url or f"https://discord.com/channels/{guild_id}/{channel_id}"
        self.headless = not debug_visible and not self.login_marker.exists()
        self.summon_on_start = summon_on_start
        self.cooldown_s = cooldown_s
        self.click_timeout_ms = click_timeout_ms
        self.on_result = on_result
        self.text = dict(BUTTON_TEXT)
        if text_overrides_file and Path(text_overrides_file).is_file():
            self.text.update(json.loads(Path(text_overrides_file).read_text()))
        if button_text:
            self.text.update(button_text)

        self._q: "queue.Queue" = queue.Queue()
        self._abort = threading.Event()
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None
        self._last_ok = 0.0
        self._t = threading.Thread(target=self._run, name="discord-browser", daemon=True)
        self._t.start()

    # -- public -------------------------------------------------------------
    def wait_ready(self, timeout: float = 120.0) -> bool:
        """Block until the browser is open (log in manually the first time)."""
        ok = self._ready.wait(timeout)
        if self._error:
            raise self._error
        return ok

    def press(self, name: str) -> None:
        self._q.put(("press", self._canon(name)))

    def summon(self, which: str = "both") -> None:
        self._q.put(("summon", which))

    def ensure_panels(self) -> None:
        self._q.put(("ensure", None))

    def abort(self) -> None:
        """Highest priority: drop queued presses and click Abort next."""
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._q.put(("press", "Abort"))

    def list_buttons(self) -> Dict[str, List[str]]:
        """Return visible button texts for each panel message (for setup)."""
        out: "queue.Queue" = queue.Queue()
        self._q.put(("list", out))
        return out.get(timeout=15)

    def status(self, timeout: float = 0.25) -> Dict[str, object]:
        """Return browser visibility, URL, panel presence, and visible buttons."""
        out: "queue.Queue" = queue.Queue()
        self._q.put(("status", out))
        try:
            return out.get(timeout=timeout)
        except queue.Empty:
            return {"ready": False, "error": "browser status unavailable"}

    def close(self) -> None:
        self._q.put(_STOP)
        self._t.join(timeout=10)

    # -- internals ----------------------------------------------------------
    def _canon(self, name: str) -> str:
        key = name.strip().lower()
        for k in config.CUSTOM_IDS:
            if k.lower() == key:
                return k
        aliases = {"backup": "Back Up", "safe j456": "Safe J1456",
                   "step+": "Robot Step+", "step-": "Robot Step-",
                   "camera up": "Cam Up", "camera down": "Cam Down",
                   "camera left": "Cam Left", "camera right": "Cam Right",
                   "camera step+": "Cam Step+", "camera step-": "Cam Step-"}
        if key in aliases:
            return aliases[key]
        raise KeyError(f"Unknown panel button {name!r}")

    def _panel(self, page, kind: str):
        probe = self.text[PANEL_PROBE[kind]]
        return (page.locator(f'[id^="chat-messages-{self.channel_id}-"]')
                .filter(has=page.get_by_role("button", name=probe, exact=True))
                .last)

    def _panel_present(self, page, kind: str) -> bool:
        return self._panel(page, kind).count() > 0

    def _panel_id(self, page, kind: str):
        panel = self._panel(page, kind)
        return panel.get_attribute("id") if panel.count() else None

    def _summon_one(self, page, kind: str):
        before = self._panel_id(page, kind)
        box = page.locator('div[role="textbox"]').first
        box.click(timeout=self.click_timeout_ms)
        command = PANEL_COMMANDS[kind]
        page.keyboard.type(command, delay=40)
        suggestion = page.locator('[role="option"]:visible').filter(
            has_text=command.removeprefix("/")
        ).first
        try:
            suggestion.wait_for(timeout=3000)
        except Exception:
            box.press("Escape")
            log.warning("Discord did not show the autocomplete option for %s.", command)
            return False
        try:
            suggestion.click(timeout=self.click_timeout_ms)
        except Exception as exc:
            box.press("Escape")
            log.warning("Could not select Discord command %s: %s", command, exc)
            return False
        page.wait_for_timeout(150)
        box.press("Enter")
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            current = self._panel_id(page, kind)
            if current and current != before:
                return True
            page.wait_for_timeout(150)
        return False

    def _ensure(self, page):
        for kind in ("main", "safe"):
            if not self._panel_present(page, kind):
                if not self._summon_one(page, kind):
                    log.warning("Discord did not spawn the %s panel.", kind)

    def _do_press(self, page, label: str) -> None:
        if label not in config.NO_COOLDOWN:
            if time.monotonic() - self._last_ok < self.cooldown_s:
                self._report(label, False, "cooldown")
                return
        safe = label in config.SAFE_MESSAGE_BUTTONS
        text = self.text[label]
        try:
            btn = self._panel(page, "safe" if safe else "main").get_by_role("button", name=text, exact=True)
            btn.click(timeout=self.click_timeout_ms)
            self._last_ok = time.monotonic()
            self._report(label, True, "")
        except Exception as exc:  # noqa: BLE001
            self._report(label, False, str(exc).splitlines()[0])

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
                ctx = self._launch_context(pw, self.headless)
                page = ctx.pages[0] if ctx.pages else ctx.new_page()
                page.goto(self.url)
                if self.headless and "/login" in page.url:
                    ctx.close()
                    self.headless = False
                    ctx = self._launch_context(pw, False)
                    page = ctx.pages[0] if ctx.pages else ctx.new_page()
                    page.goto(self.url)
                if self.summon_on_start:
                    self._wait_for_login(page)
                    self.login_marker.parent.mkdir(parents=True, exist_ok=True)
                    self.login_marker.touch()
                    self._ensure(page)
                self._ready.set()
                while True:
                    item = self._q.get()
                    if item is _STOP:
                        break
                    kind, arg = item
                    if kind == "press":
                        self._do_press(page, arg)
                    elif kind == "summon":
                        for panel_kind in ("main", "safe") if arg == "both" else (arg,):
                            self._summon_one(page, panel_kind)
                    elif kind == "ensure":
                        self._ensure(page)
                    elif kind == "list":
                        res = {}
                        for safe in (False, True):
                            panel = self._panel(page, "safe" if safe else "main")
                            loc = panel.get_by_role("button")
                            res["safe" if safe else "main"] = [
                                (loc.nth(i).inner_text() or loc.nth(i).get_attribute("aria-label") or "").strip()
                                for i in range(loc.count())]
                        arg.put(res)
                    elif kind == "status":
                        panels = {}
                        buttons = {}
                        for panel_kind in ("main", "safe"):
                            panel = self._panel(page, panel_kind)
                            present = panel.count() > 0
                            panels[panel_kind] = present
                            if present:
                                loc = panel.get_by_role("button")
                                buttons[panel_kind] = [
                                    (loc.nth(i).inner_text() or
                                     loc.nth(i).get_attribute("aria-label") or "").strip()
                                    for i in range(loc.count())
                                ]
                            else:
                                buttons[panel_kind] = []
                        arg.put({
                            "ready": True,
                            "visible": not self.headless,
                            "url": page.url,
                            "panels": panels,
                            "buttons": buttons,
                        })
                ctx.close()
        except BaseException as exc:  # noqa: BLE001
            self._error = exc
            self._ready.set()
            log.exception("browser thread died")

    def _launch_context(self, playwright, headless):
        return playwright.chromium.launch_persistent_context(
            self.profile_dir,
            headless=headless,
            args=[f"--window-size={BROWSER_WIDTH},{BROWSER_HEIGHT}"],
            viewport={"width": BROWSER_WIDTH, "height": BROWSER_HEIGHT})

    def _wait_for_login(self, page):
        if "/login" in page.url:
            page.wait_for_url("**/channels/**", timeout=120_000)
        page.locator('div[role="textbox"]').first.wait_for(timeout=30_000)


if __name__ == "__main__":
    import argparse
    import pprint
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print button texts on the panel messages")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    p = BrowserPanel()
    print("Log in to Discord in the opened window if needed, open the robot channel,")
    input("then press Enter here... ")
    if args.list:
        pprint.pprint(p.list_buttons())
    p.close()
