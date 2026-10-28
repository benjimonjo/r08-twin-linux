"""Opt-in pointer input through the XDG RemoteDesktop portal on Wayland."""
import threading


class WaylandPointer:
    """Request portal consent, map calibrated coordinates, and send pointer clicks."""

    def __init__(self, capture_monitor, ready_callback):
        self.capture_monitor = dict(capture_monitor)
        self.ready_callback = ready_callback
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
            self.portal.create_remote_desktop_session(
                Xdp.DeviceType.POINTER, Xdp.OutputType.MONITOR,
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
            capture_ratio = float(cap["width"]) / max(1.0, float(cap["height"]))
            stream_ratio = float(self.stream_size[0]) / float(self.stream_size[1])
            if abs(capture_ratio - stream_ratio) / max(capture_ratio, stream_ratio) > 0.08:
                raise RuntimeError(
                    "The input portal selected a screen with a different shape from the calibrated capture. "
                    "Select the monitor that contains the robot panel."
                )
            self._ready(True, "")
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
        """Map capture pixels proportionally into the selected portal stream.

        The portals may report different compositor origins for the same output;
        pointer coordinates are stream-local, so only the position within the
        captured monitor and the stream's logical size matter here.
        """
        m = self.capture_monitor
        width, height = float(m["width"]), float(m["height"])
        px = (float(x) - float(m["left"])) / width
        py = (float(y) - float(m["top"])) / height
        if not (0.0 <= px < 1.0 and 0.0 <= py < 1.0):
            raise RuntimeError("A calibrated panel button is outside the selected screen stream.")
        sw, sh = self.stream_size
        return px * sw, py * sh

    def click(self, x, y):
        """Queue one left click on the portal main loop; returns immediately."""
        if not self.session or not self.loop or not self.GLib:
            raise RuntimeError("Wayland pointer permission is not active.")
        sx, sy = self.map_point(x, y)
        self.GLib.idle_add(self._press_at, sx, sy)

    def _press_at(self, x, y):
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
