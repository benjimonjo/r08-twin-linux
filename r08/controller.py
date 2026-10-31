"""Optional SDL game-controller polling for the jog target controls."""
import time

try:
    import pygame
except Exception:
    pygame = None


CONTROLLER_ACTIONS = (
    "X+", "X-", "Y+", "Y-", "Z+", "Z-",
    "Xa+", "Xa-", "Ya+", "Ya-", "Za+", "Za-",
    "Step+", "Step-",
)
ABORT_ACTION = "Abort"
VIEW_MODIFIER = "View modifier"
CAMERA_MODIFIER = VIEW_MODIFIER
ANGLE_MODIFIER = "Angle modifier"
VIEW_AXIS_ACTIONS = (
    "View orbit horizontal", "View orbit vertical",
    "View pan horizontal", "View pan vertical",
)
VIEW_BUTTON_ACTIONS = ("View zoom in", "View zoom out")
VIEW_BUTTON_MAPPING_DEFAULTS = {
    "View zoom in": "axis4:+",
    "View zoom out": "axis5:+",
}
VIEW_MAPPING_DEFAULTS = {
    "View orbit horizontal": "axis0:+",
    "View orbit vertical": "axis1:+",
    "View pan horizontal": "axis2:+",
    "View pan vertical": "axis3:+",
}
STICK_MAPPING_DEFAULTS = {
    "Position X+": "axis1:+", "Position X-": "axis1:-",
    "Position Y+": "axis0:+", "Position Y-": "axis0:-",
    "Position Z+": "axis3:-", "Position Z-": "axis3:+",
    "Angle Xa+": "axis1:+", "Angle Xa-": "axis1:-",
    "Angle Ya+": "axis0:+", "Angle Ya-": "axis0:-",
    "Angle Za+": "axis3:-", "Angle Za-": "axis3:+",
    "View orbit horizontal": "axis0:+", "View orbit vertical": "axis1:+",
    "View pan horizontal": "axis2:+", "View pan vertical": "axis3:+",
}
MODIFIER_MAPPING_DEFAULTS = {ANGLE_MODIFIER: 4, CAMERA_MODIFIER: 5}
DEFAULT_MAPPING = {action: index for index, action in enumerate(CONTROLLER_ACTIONS)}
AXIS_DEADZONE = 0.55
TRIGGER_AXIS_START = 4


class Controller:
    @staticmethod
    def normalize_axis_value(axis, value):
        """Normalize SDL trigger axes from -1..1 to 0..1."""
        value = float(value)
        if int(axis) >= TRIGGER_AXIS_START:
            return max(0.0, min(1.0, (value + 1.0) / 2.0))
        return max(-1.0, min(1.0, value))

    def __init__(self, mapping=None, device=0, repeat_delay=0.12):
        self.mapping = dict(DEFAULT_MAPPING)
        self.mapping.update(mapping or {})
        for action, control in VIEW_MAPPING_DEFAULTS.items():
            self.mapping.setdefault(action, control)
        for action, control in VIEW_BUTTON_MAPPING_DEFAULTS.items():
            self.mapping.setdefault(action, control)
        for action, control in STICK_MAPPING_DEFAULTS.items():
            self.mapping.setdefault(action, control)
        for action, control in MODIFIER_MAPPING_DEFAULTS.items():
            self.mapping.setdefault(action, control)
        self.mapping.setdefault(ABORT_ACTION, None)
        self.device_index = int(device)
        self.joystick = None
        self.error = None
        self._held_buttons = set()
        self._hat_value = (0, 0)
        self._axis_values = {}
        self._next_button_repeat = 0.0
        self.repeat_delay = max(0.01, float(repeat_delay))
        if pygame is not None:
            try:
                pygame.init()
                pygame.joystick.init()
                self._open()
            except Exception as exc:
                self.error = str(exc)
        else:
            self.error = "pygame is not installed"

    def set_repeat_delay(self, delay):
        self.repeat_delay = max(0.01, float(delay))

    @staticmethod
    def available():
        if pygame is None:
            return []
        try:
            pygame.joystick.init()
            return [pygame.joystick.Joystick(i).get_name()
                    for i in range(pygame.joystick.get_count())]
        except Exception:
            return []

    @staticmethod
    def button_options(device_index=0):
        """Return selectable button and D-pad controls for a controller."""
        if pygame is None:
            return []
        try:
            joystick = pygame.joystick.Joystick(int(device_index))
            joystick.init()
            name = joystick.get_name()
            count = joystick.get_numbuttons()
            lowered = name.lower()
            if any(token in lowered for token in ("dualshock", "dualsense", "playstation", "ps4", "ps5")):
                labels = (
                    "Triangle", "Circle", "Cross", "Square",
                    "L1", "R1", "L2", "R2", "Create", "Options",
                    "L3 click", "R3 click", "PS", "Touchpad", "Mute",
                )
            elif any(token in lowered for token in ("xbox", "x-input", "xinput")):
                labels = (
                    "X", "Y", "B", "A", "LB", "RB", "View", "Menu",
                    "Xbox", "L3 click", "R3 click",
                )
            elif any(token in lowered for token in ("switch", "nintendo", "joy-con", "pro controller")):
                labels = ("B", "A", "Y", "X", "L", "R", "ZL", "ZR", "Minus", "Plus")
            else:
                labels = ()
            options = [
                (index, f"{labels[index]} (button {index})" if index < len(labels)
                 else f"Button {index}")
                for index in range(count)
            ] + [
                (f"hat0:{direction}", f"D-pad {label} (hat 0)")
                for direction, label in (
                    ("up", "Up"), ("down", "Down"), ("left", "Left"), ("right", "Right")
                )
            ]
            axis_labels = (
                "LX", "LY", "RX", "RY",
                "L2 axis" if "playstation" in lowered or "dual" in lowered or "ps" in lowered else "LT axis",
                "R2 axis" if "playstation" in lowered or "dual" in lowered or "ps" in lowered else "RT axis",
                "Trigger 6", "Trigger 7",
            )
            options.extend(
                (f"axis{axis}:{sign}", f"{axis_labels[axis]} {'+' if sign == '+' else '-'}")
                for axis in range(joystick.get_numaxes())
                for sign in ("+", "-")
            )
            return [(None, "Unassigned")] + options
        except Exception:
            return []

    def _open(self):
        if pygame.joystick.get_count() <= self.device_index:
            self.error = "No controller detected"
            return
        self.joystick = pygame.joystick.Joystick(self.device_index)
        self.joystick.init()
        self.error = None

    def set_device(self, index):
        self.device_index = int(index)
        self.joystick = None
        self._held_buttons.clear()
        self._hat_value = (0, 0)
        self._axis_values.clear()
        self._next_button_repeat = 0.0
        self._open()

    @staticmethod
    def _runtime_action(action):
        return {
            "Position X+": "X+", "Position X-": "X-",
            "Position Y+": "Y+", "Position Y-": "Y-",
            "Position Z+": "Z+", "Position Z-": "Z-",
            "Angle Xa+": "Xa+", "Angle Xa-": "Xa-",
            "Angle Ya+": "Ya+", "Angle Ya-": "Ya-",
            "Angle Za+": "Za+", "Angle Za-": "Za-",
        }.get(action, action)

    def _binding(self, action):
        if action in self.mapping:
            return self.mapping[action]
        aliases = {
            "Position X+": "X+", "Position X-": "X-",
            "Position Y+": "Y+", "Position Y-": "Y-",
            "Position Z+": "Z+", "Position Z-": "Z-",
            "Angle Xa+": "Xa+", "Angle Xa-": "Xa-",
            "Angle Ya+": "Ya+", "Angle Ya-": "Ya-",
            "Angle Za+": "Za+", "Angle Za-": "Za-",
        }
        return self.mapping.get(aliases.get(action))

    def _held(self, control):
        if isinstance(control, int):
            return control in self._held_buttons
        if not isinstance(control, str):
            return False
        if control.startswith("hat0:"):
            direction = control.removeprefix("hat0:")
            return {
                "left": self._hat_value[0] < 0, "right": self._hat_value[0] > 0,
                "up": self._hat_value[1] > 0, "down": self._hat_value[1] < 0,
            }.get(direction, False)
        return False

    def _active_bindings(self):
        if self._held(self.mapping.get(CAMERA_MODIFIER)):
            actions = VIEW_AXIS_ACTIONS
        elif self._held(self.mapping.get(ANGLE_MODIFIER)):
            actions = (
                "Angle Xa+", "Angle Xa-", "Angle Ya+", "Angle Ya-",
                "Angle Za+", "Angle Za-",
            )
        else:
            actions = (
                "Position X+", "Position X-", "Position Y+", "Position Y-",
                "Position Z+", "Position Z-",
            )
        return {action: self._binding(action) for action in actions}

    def _poll_buttons(self, bindings, pressed_buttons, now):
        if now < self._next_button_repeat:
            return []
        actions = []
        for action, control in bindings.items():
            if self._held(control):
                output = self._runtime_action(action)
                if action in VIEW_AXIS_ACTIONS and self._held(self.mapping.get(CAMERA_MODIFIER)):
                    output = f"{VIEW_MODIFIER}:{action}:+"
                actions.append((output, 1.0))
        for action in ("Step+", "Step-", ABORT_ACTION):
            if self._held(self.mapping.get(action)):
                actions.append((action, 1.0))
        if self.mapping.get(ABORT_ACTION) in pressed_buttons:
            actions.append((ABORT_ACTION, 1.0))
        self._next_button_repeat = now + self.repeat_delay
        return actions

    def _poll_axes(self, bindings, camera_held):
        actions = []
        for action, control in bindings.items():
            if not isinstance(control, str) or not control.startswith("axis"):
                continue
            try:
                axis_text, sign = control.removeprefix("axis").split(":")
                value = float(self._axis_values.get(int(axis_text), 0.0))
            except (TypeError, ValueError):
                continue
            oriented = value if sign == "+" else -value
            if camera_held:
                active = abs(oriented) >= AXIS_DEADZONE
            else:
                active = oriented >= AXIS_DEADZONE
            if not active:
                continue
            magnitude = (abs(oriented) if camera_held else oriented) - AXIS_DEADZONE
            magnitude /= 1.0 - AXIS_DEADZONE
            direction = "+" if oriented >= 0 else "-"
            output = self._runtime_action(action)
            if camera_held:
                output = f"{VIEW_MODIFIER}:{action}:{direction}"
            actions.append((output, max(0.0, min(1.0, magnitude))))
        return actions

    def capture_control(self):
        """Return the next button, hat direction, or joystick direction pressed."""
        if pygame is None:
            return None
        try:
            pygame.event.pump()
            for event in pygame.event.get():
                if event.joy != self.device_index:
                    continue
                if event.type == pygame.JOYBUTTONDOWN:
                    return event.button
                if event.type == pygame.JOYHATMOTION:
                    hat_x, hat_y = event.value
                    if hat_x < 0:
                        return "hat0:left"
                    if hat_x > 0:
                        return "hat0:right"
                    if hat_y > 0:
                        return "hat0:up"
                    if hat_y < 0:
                        return "hat0:down"
                if event.type == pygame.JOYAXISMOTION:
                    value = self.normalize_axis_value(event.axis, event.value)
                    if value >= AXIS_DEADZONE:
                        return f"axis{event.axis}:+"
                    if event.axis < TRIGGER_AXIS_START and value <= -AXIS_DEADZONE:
                        return f"axis{event.axis}:-"
        except Exception as exc:
            self.error = str(exc)
        return None

    def poll(self):
        if pygame is None or self.joystick is None:
            return []
        now = time.monotonic()
        out = []
        pressed_buttons = set()
        try:
            pygame.event.pump()
            for event in pygame.event.get():
                if getattr(event, "joy", self.device_index) != self.device_index:
                    continue
                if event.type == pygame.JOYBUTTONDOWN:
                    self._held_buttons.add(event.button)
                    pressed_buttons.add(event.button)
                elif event.type == pygame.JOYBUTTONUP:
                    self._held_buttons.discard(event.button)
                elif event.type == pygame.JOYHATMOTION and event.hat == 0:
                    self._hat_value = event.value
                elif event.type == pygame.JOYAXISMOTION:
                    self._axis_values[event.axis] = self.normalize_axis_value(
                        event.axis, event.value)
            # Some SDL backends do not emit axis events until the window has
            # focus. Read the current state as well so jogging remains reliable.
            for axis in range(self.joystick.get_numaxes()):
                self._axis_values[axis] = self.normalize_axis_value(
                    axis, self.joystick.get_axis(axis))
            if hasattr(self.joystick, "get_button"):
                self._held_buttons = {
                    button for button in range(self.joystick.get_numbuttons())
                    if self.joystick.get_button(button)
                }
            if hasattr(self.joystick, "get_hat") and self.joystick.get_numhats():
                self._hat_value = self.joystick.get_hat(0)

            bindings = self._active_bindings()
            camera_held = self._held(self.mapping.get(CAMERA_MODIFIER))
            out.extend(self._poll_buttons(bindings, pressed_buttons, now))
            out.extend(self._poll_axes(bindings, camera_held))
            for action in VIEW_BUTTON_ACTIONS:
                control = self.mapping.get(action)
                if isinstance(control, str) and control.startswith("axis"):
                    try:
                        axis_text, sign = control.removeprefix("axis").split(":")
                        value = self._axis_values.get(int(axis_text), 0.0)
                    except (TypeError, ValueError):
                        continue
                    if sign == "+" and value >= AXIS_DEADZONE:
                        out.append((action, (value - AXIS_DEADZONE) / (1.0 - AXIS_DEADZONE)))
        except Exception as exc:
            self.error = str(exc)
        return out

    def state(self):
        """Return a snapshot for the controller diagnostics readout."""
        if self.joystick is None:
            return {"name": self.error or "No controller detected", "buttons": [], "axes": [], "hat": (0, 0)}
        return {
            "name": self.joystick.get_name(),
            "buttons": [i in self._held_buttons for i in range(self.joystick.get_numbuttons())],
            "axes": [float(self._axis_values.get(i, 0.0)) for i in range(self.joystick.get_numaxes())],
            "hat": tuple(self._hat_value),
        }

    def close(self):
        if pygame is not None:
            try:
                pygame.joystick.quit()
                pygame.quit()
            except Exception:
                pass
