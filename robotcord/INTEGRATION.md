# Adding Discord panel control to r08-twin-linux

1. Copy `robotcord/` (repo root) and `r08/discord_panel.py` into the project.
2. Add `requests>=2.28` to `requirements.txt`.
3. Provide the token outside the repo: `export DISCORD_TOKEN=...` or put it in
   `~/.config/r08/discord_token` (chmod 600). Never put it in `r08_config.json`.
4. In the code that performs a calibrated panel click (direct mode), replace the
   pointer-click with:

       panel = DiscordPanel.from_env()           # create once at startup
       ...
       if panel: panel.press(button_name)         # else fall back to pointer click

   and route Abort to `panel.abort()`.
5. Keep your OCR verify step: Discord presses are fire-and-forget (HTTP 204 only
   means Discord accepted it, not that the robot moved).

## Option B: browser-driven panel (no token)

`r08/discord_browser.py` provides `BrowserPanel` with the same `press()` / `abort()` / `close()`
interface as `DiscordPanel`, but clicks the real buttons in a persistent Chromium profile.

    pip install playwright && playwright install chromium
    python -m r08.discord_browser --list

    panel = BrowserPanel()   # defaults to https://discord.com/channels/990738288765698169/1191491623159332976
    panel.wait_ready()
    panel.press("X+")

- On startup it posts any missing panel with `/controls` (movement/camera) and `/robotpanel`
  (safe positions). ASSUMPTION: that is which command makes which panel; swap via
  `panel_commands={"main": "/robotpanel", "safe": "/controls"}` if not.
  `summon()` / `ensure_panels()` can be called any time (e.g. after the panel's buttons expire).
- Panels are found by their buttons (newest wins), not by message ID.
- Buttons are matched by visible label (Discord hides `custom_id`). If a label differs, pass
  `button_text={"X+": "X +"}` or `text_overrides_file=`.
- Set `summon_on_start=False` to avoid posting into the channel automatically.

## Role check ("Stream Controller")

Summoning panels needs the `Stream Controller` role.

- Token client (authoritative): `RobotClient(token, ...).check_role()` -> `RoleCheck(has_role, role_id, detail)`;
  `DiscordPanel.from_env(verify_role=True)` raises `PermissionError` if the role is definitely missing.
- Browser (no token): commands are visible to everyone, so `BrowserPanel` sends the command and reads the
  bot's private reply. A message containing `denial_text` (default `"Access error"`, i.e. "Access error: You do
  not have access to use the controls command. (Required permission: Stream Controller)") means no role.
  The denial is reported through `on_result`, stored in `panel.last_denial`, and further summons in that batch
  are skipped. `panel.check_role()` -> True (panel posted) / False (denied) / None (nothing recognisable);
  note that a successful check posts a panel.
