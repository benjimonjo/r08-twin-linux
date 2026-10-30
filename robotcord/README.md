# robotcord

Python port of the Discord side of the robot control web page: it clicks the
buttons on the bot's control-panel messages by POSTing component interactions
to `https://discord.com/api/v9/interactions`.

```python
from robotcord import RobotClient

with RobotClient(token="PASTE_TOKEN_OR_CURL_HERE") as bot:
    bot.press("X+")          # move
    bot.press("Cam Left")    # camera
    bot.press("Safe J23")    # goes to the safe-panel message automatically
    bot.abort()              # never rate-limited by the cooldown
```

- `dry_run=True` logs instead of sending.
- `cooldown_s` (default 1.0) applies to everything except `Abort` / `Safe Abort`.
- `press()` returns `Result(ok, status, body, skipped)`.
- Pasted cURL/fetch text is accepted as `token`; the Authorization value is extracted.
- Keep the token out of source control (e.g. `os.environ["DISCORD_TOKEN"]`).
