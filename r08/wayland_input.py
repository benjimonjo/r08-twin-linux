"""Opt-in pointer input through the XDG RemoteDesktop portal on Wayland."""
import threading


class WaylandPointer:
    """Request portal consent, map calibrated coordinates, and send pointer clicks."""

    def __init__(self, capture_monitor, ready_callback, output_type="monitor"):
        self.capture_monitor = dict(capture_monitor)
        self.ready_callback = ready_callback
        self.output_type = str(output_type).lower()
        self.session = self.loop = self.GLib = self.Xdp = None
        self.stream_id = None
        self.stream_position = None
        self.stream_size = None
        threading.Thread(target=self._start, daemon=True).start()

    def _start(self):
        try:
            import gi
            gi.require_version("Xdp", "1.0")
            from gi.repository import GLib, Xdp
            self.GLib, self.Xdp = GLib, Xdp
            self.loop = GLib.MainLoop()
            self.portal = Xdp.Portal.new()
            output = (Xdp.OutputType.WINDOW if self.output_type == "window"
                      else Xdp.OutputType.MONITOR)
            self.portal.create_remote_desktop_session(
                Xdp.DeviceType.POINTER, output,
                Xdp.RemoteDesktopFlags(0), Xdp.CursorMode.EMBEDDED,
                None, self._created, None)
            self.loop.run()
        except Exception as exc:
            self._ready(False, str(exc))

    def _created(self, portal, result, _data=None):
        try:
            self.session = portal.create_remote_desktop_session_finish(result)
            if self.session is None:
                raise RuntimeError("The desktop portal did not create a pointer session.")
            self.session.start(None, None, self._started, None)
        except Exception as exc:
            try:
                if self.session: self.session.close()
            except Exception:
                pass
            self._ready(False, str(exc))
            self.loop.quit()

    def _started(self, session, result, _data=None):
        try:
            if not session.start_finish(result):
                raise RuntimeError("The desktop portal did not grant pointer access.")
            streams = session.get_streams()
            streams = streams.unpack() if streams is not None else []
            if not streams:
                raise RuntimeError("The portal granted pointer access but returned no screen stream for coordinates.")
            self.stream_id, props = streams[0]
            self.stream_position = tuple(props.get("position", (0, 0)))
            self.stream_size = tuple(props.get("size", ()))
            if len(self.stream_size) != 2 or min(self.stream_size) <= 0:
                raise RuntimeError("The portal did not provide the screen dimensions needed for pointer mapping.")
            cap = self.capture_monitor
            cap_left = float(cap.get("left", 0))
            cap_top = float(cap.get("top", 0))
            cap_right = cap_left + float(cap["width"])
            cap_bottom = cap_top + float(cap["height"])
            stream_left, stream_top = map(float, self.stream_position)
            stream_right = stream_left + float(self.stream_size[0])
            stream_bottom = stream_top + float(self.stream_size[1])
            if self.output_type == "window":
                if abs(stream_left - cap_left) > 2 or abs(stream_top - cap_top) > 2:
                    raise RuntimeError(
                        "KDE selected a different window for pointer input. "
                        f"Expected the calibrated panel window near {cap_left:.0f},{cap_top:.0f}, "
                        f"got {stream_left:.0f},{stream_top:.0f}.")
            else:
                panel = self.capture_monitor
                panel_center = (
                    float(panel["left"]) + float(panel["width"]) / 2,
                    float(panel["top"]) + float(panel["height"]) / 2,
                )
                if not (stream_left <= panel_center[0] < stream_right and
                        stream_top <= panel_center[1] < stream_bottom):
                    raise RuntimeError(
                        "KDE selected a different monitor for pointer input. "
                        f"The calibrated panel monitor is centered near "
                        f"{panel_center[0]:.0f},{panel_center[1]:.0f}, but the selected "
                        f"stream is {self.stream_position[0]},{self.stream_position[1]} "
                        f"({self.stream_size[0]}×{self.stream_size[1]}).")
                overlap_w = max(0.0, min(cap_right, stream_right) - max(cap_left, stream_left))
                overlap_h = max(0.0, min(cap_bottom, stream_bottom) - max(cap_top, stream_top))
                if overlap_w < min(float(cap["width"]), float(self.stream_size[0])) * 0.5 or \
                        overlap_h < min(float(cap["height"]), float(self.stream_size[1])) * 0.5:
                    raise RuntimeError(
                        "KDE selected a different monitor for pointer input. "
                        f"Expected screen near {cap_left:.0f},{cap_top:.0f} "
                        f"({cap['width']}×{cap['height']}), got "
                        f"{self.stream_position[0]},{self.stream_position[1]} "
                        f"({self.stream_size[0]}×{self.stream_size[1]}). "
                        "Restart automated input and select the monitor containing the robot panel.")
            self._ready(
                True,
                f"Pointer stream {self.output_type}: "
                f"{self.stream_size[0]}×{self.stream_size[1]} at "
                f"{self.stream_position[0]},{self.stream_position[1]}.")
        except Exception as exc:
            try:
                if self.session: self.session.close()
            except Exception:
                pass
            self._ready(False, str(exc))
            self.loop.quit()

    def _ready(self, ok, message):
        self.ready_callback(bool(ok), str(message))

    def map_point(self, x, y):
        """Map a calibrated desktop point into the selected pointer stream.

        Monitor captures and the RemoteDesktop monitor stream use the same
        desktop coordinate space, even when the stream spans the full virtual
        workspace. Window captures are local to the selected window and need
        the selected window origin restored.
        """
        m = self.capture_monitor
        x, y = float(x), float(y)
        capture_left, capture_top = float(m["left"]), float(m["top"])
        capture_right = capture_left + float(m["width"])
        capture_bottom = capture_top + float(m["height"])
        if not (capture_left <= x < capture_right and
                capture_top <= y < capture_bottom):
            raise RuntimeError(
                f"A calibrated panel button is outside the selected input stream "
                f"({self.stream_size[0]}×{self.stream_size[1]} at "
                f"{self.stream_position[0]},{self.stream_position[1]}). "
                "Select the monitor containing the robot panel.")
        if self.output_type == "window":
            # KDE's window RemoteDesktop backend interprets pointer coordinates
            # in desktop space even though the selected video surface is
            # window-local. Apply the selected window's desktop origin once.
            width, height = float(m["width"]), float(m["height"])
            px = (x - capture_left) / width
            py = (y - capture_top) / height
            sw, sh = map(float, self.stream_size)
            mapped_x, mapped_y = px * sw, py * sh
            stream_left, stream_top = map(float, self.stream_position)
            return stream_left + mapped_x, stream_top + mapped_y
        return x, y

    def click(self, x, y):
        """Queue one left click on the portal main loop; returns immediately."""
        if not self.session or not self.loop or not self.GLib:
            raise RuntimeError("Wayland pointer permission is not active.")
        sx, sy = self.map_point(x, y)
        print(
            "R08 pointer map: "
            f"input=({float(x):.1f},{float(y):.1f}) "
            f"capture=({self.capture_monitor.get('left', 0)},"
            f"{self.capture_monitor.get('top', 0)},"
            f"{self.capture_monitor.get('width')},"
            f"{self.capture_monitor.get('height')}) "
            f"stream=({self.stream_position[0]},{self.stream_position[1]},"
            f"{self.stream_size[0]},{self.stream_size[1]}) "
            f"output=({sx:.1f},{sy:.1f})",
            flush=True)
        self.GLib.idle_add(self._press_at, sx, sy)

    def _press_at(self, x, y):
        if not self.session:
            return False
        try:
            self.session.pointer_position(self.stream_id, x, y)
            self.session.pointer_button(272, self.Xdp.ButtonState.PRESSED)  # BTN_LEFT
            self.GLib.timeout_add(70, self._release)
        except Exception:
            self._release()
        return False

    def _release(self):
        try:
            self.session.pointer_button(272, self.Xdp.ButtonState.RELEASED)
        except Exception:
            pass
        return False

    def close(self):
        if self.GLib and self.loop and self.loop.is_running():
            self.GLib.idle_add(self._close_on_loop)

    def _close_on_loop(self):
        try:
            if self.session:
                self.session.close()
        except Exception:
            pass
        self.loop.quit()
        return False
