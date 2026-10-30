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
    python -m r08.discord_browser --list     # log in once, then see the button texts

Buttons are found by visible label (Discord does not expose `custom_id` in the DOM). If a label
differs from the default, pass `button_text={"X+": "X +"}` or a JSON file via
`text_overrides_file=`. The robot channel/message must be rendered on screen.
