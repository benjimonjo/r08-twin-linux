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
DEFAULT_MAPPING = {action: index for index, action in enumerate(CONTROLLER_ACTIONS)}
AXIS_DEADZONE = 0.55


class Controller:
    def __init__(self, mapping=None, device=0):
        self.mapping = dict(DEFAULT_MAPPING)
        self.mapping.update(mapping or {})
        self.device_index = int(device)
        self.joystick = None
        self.error = None
        self._held_buttons = set()
        self._hat_value = (0, 0)
        self._axis_values = {}
        self._next_button_repeat = 0.0
        if pygame is not None:
            try:
                pygame.init()
                pygame.joystick.init()
                self._open()
            except Exception as exc:
                self.error = str(exc)
        else:
            self.error = "pygame is not installed"

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
                    "Cross", "Circle", "Square", "Triangle",
                    "L1", "R1", "L2", "R2", "Share", "Options",
                    "L3", "R3", "PS", "Touchpad",
                )
            elif any(token in lowered for token in ("xbox", "x-input", "xinput")):
                labels = (
                    "A", "B", "X", "Y", "LB", "RB", "Back", "Start",
                    "Guide", "LS", "RS",
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
                "Left stick horizontal", "Left stick vertical",
                "Right stick horizontal", "Right stick vertical",
                "Axis 4", "Axis 5", "Axis 6", "Axis 7",
            )
            options.extend(
                (f"axis{axis}:{sign}", f"{axis_labels[axis]} {'+' if sign == '+' else '-'}")
                for axis in range(joystick.get_numaxes())
                for sign in ("+", "-")
            )
            return options
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
                if event.type == pygame.JOYAXISMOTION and abs(event.value) >= AXIS_DEADZONE:
                    return f"axis{event.axis}:{'+' if event.value > 0 else '-'}"
        except Exception as exc:
            self.error = str(exc)
        return None

    def poll(self):
        if pygame is None or self.joystick is None:
            return []
        now = time.monotonic()
        out = []
        try:
            pygame.event.pump()
            for event in pygame.event.get():
                if getattr(event, "joy", self.device_index) != self.device_index:
                    continue
                if event.type == pygame.JOYBUTTONDOWN:
                    self._held_buttons.add(event.button)
                elif event.type == pygame.JOYBUTTONUP:
                    self._held_buttons.discard(event.button)
                elif event.type == pygame.JOYHATMOTION and event.hat == 0:
                    self._hat_value = event.value
                elif event.type == pygame.JOYAXISMOTION:
                    self._axis_values[event.axis] = event.value

            if now >= self._next_button_repeat:
                for action, control in self.mapping.items():
                    if control in self._held_buttons:
                        out.append((action, 1.0))
                    elif isinstance(control, str) and control.startswith("hat0:"):
                        hat_x, hat_y = self._hat_value
                        directions = {
                            "left": hat_x < 0, "right": hat_x > 0,
                            "up": hat_y > 0, "down": hat_y < 0,
                        }
                        if directions.get(control.removeprefix("hat0:"), False):
                            out.append((action, 1.0))
                self._next_button_repeat = now + 0.12

            for action, control in self.mapping.items():
                if not (isinstance(control, str) and control.startswith("axis")):
                    continue
                try:
                    axis, sign = control.removeprefix("axis").split(":")
                    value = float(self._axis_values.get(int(axis), 0.0))
                except (TypeError, ValueError):
                    continue
                if (sign == "+" and value >= AXIS_DEADZONE) or (sign == "-" and value <= -AXIS_DEADZONE):
                    magnitude = (abs(value) - AXIS_DEADZONE) / (1.0 - AXIS_DEADZONE)
                    out.append((action, max(0.0, min(1.0, magnitude))))
        except Exception as exc:
            self.error = str(exc)
        return out

    def close(self):
        if pygame is not None:
            try:
                pygame.joystick.quit()
                pygame.quit()
            except Exception:
                pass
