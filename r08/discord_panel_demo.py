"""python -m r08.discord_panel_demo   (dry run, nothing is sent)"""
import logging, time
from r08.discord_panel import DiscordPanel

logging.basicConfig(level=logging.INFO)
p = DiscordPanel.from_env(dry_run=True)
for b in ("X+", "Camera Step-", "Backup", "Safe J456"):
    p.press(b); time.sleep(1.1)
p.abort(); time.sleep(0.3)
