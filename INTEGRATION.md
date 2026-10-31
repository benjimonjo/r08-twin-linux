# Discord panel integration

R08 integrates the browser-driven panel as the supported Robocord integration.
It uses Discord's normal visible web UI and never accepts or stores a Discord
user token.

## Browser-driven panel (Option B)

`r08/discord_browser.py` provides `BrowserPanel` with the same `press()` / `abort()` / `close()`
interface as `DiscordPanel`, but clicks the real buttons in a persistent Chromium profile.

    pip install playwright && playwright install chromium
    python -m r08.discord_browser --list     # log in once, then see the button texts

On startup, the adapter opens the text channel and posts `/controls` and
`/robotpanel` when the corresponding panels are not already visible. It then
finds the newest rendered panel by its visible probe button, so it does not
depend on stale message IDs. The account must have the bot's required
controller role.

Buttons are found by visible label (Discord does not expose `custom_id` in the DOM). If a label
differs from the default, pass `button_text={"X+": "X +"}` or a JSON file via
`text_overrides_file=`.

In R08, select **Controller → Set controller input mode → Directly press panel
buttons**, then select **Discord browser (Option B)**. The browser opens when
direct mode is selected, remains open while direct mode is active, and closes
when direct mode or controller automation is disabled.

The first launch is headed so the user can log in manually. After the login
session is confirmed, R08 records it in the persistent profile and later
launches headlessly. If Discord redirects the saved profile to login, R08
reopens a headed browser for re-authentication.

The token-based REST/self-bot client is intentionally not integrated. It would
automate a Discord user account outside the supported application/bot flow.
