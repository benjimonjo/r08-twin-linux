"""Safe Discord panel identifiers used by the browser integration.

This module contains public panel metadata only. Browser mode authenticates
through the user-visible Discord session and never accepts or stores a token.
"""

GUILD_ID = "990738288765698169"
CHANNEL_ID = "1191491623159332976"
MESSAGE_ID = "1565517444611702884"
SAFE_MESSAGE_ID = "1565517456003563540"

CUSTOM_IDS = {
    "X+": "move:x_plus", "Y+": "move:y_plus", "Z+": "move:z_plus",
    "Xa+": "angle:xa_plus", "Ya+": "angle:ya_plus", "Za+": "angle:za_plus",
    "X-": "move:x_minus", "Y-": "move:y_minus", "Z-": "move:z_minus",
    "Xa-": "angle:xa_minus", "Ya-": "angle:ya_minus", "Za-": "angle:za_minus",
    "Robot Step+": "robot_step:plus", "Robot Step-": "robot_step:minus",
    "Abort": "robot:abort",
    "Cam Up": "move:up", "Cam Down": "move:down",
    "Cam Left": "move:left", "Cam Right": "move:right",
    "Zoom+": "zoom:in", "Zoom-": "zoom:out",
    "Reset Camera": "reset_camera", "Switch Camera": "switch_camera",
    "Cam Step+": "cam_step:minus", "Cam Step-": "cam_step:plus",
    "Safe Abort": "robot_panel:abort",
    "Back Up": "robot_panel:collision_backup",
    "Safe J23": "robot_panel:go_to_safe_j23",
    "Safe J1456": "robot_panel:go_to_safe_j1456",
}
SAFE_MESSAGE_BUTTONS = frozenset({"Safe Abort", "Back Up", "Safe J23", "Safe J1456"})
NO_COOLDOWN = frozenset({"Abort"})
COOLDOWN_S = 1.0
