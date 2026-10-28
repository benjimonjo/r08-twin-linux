"""XDG ScreenCast portal capture for Linux Wayland desktops.

The portal mediates user permission and exposes the selected monitor through
PipeWire. This module presents the small subset of the mss API used by R08.
"""
import threading
import os

import numpy as np


class _Shot:
    def __init__(self, frame):
        self._frame = np.ascontiguousarray(frame)
        self.size = (self._frame.shape[1], self._frame.shape[0])
        self.bgra = self._frame.tobytes()

    def __array__(self, dtype=None, copy=None):
        if dtype is not None:
            return self._frame.astype(dtype, copy=copy is not False)
        return self._frame.copy() if copy else self._frame


class WaylandCapture:
    """ScreenCast portal + PipeWire capture, with mss-compatible grab()."""
    def __init__(self, source="monitor", restore_token=None):
        try:
            import gi
            gi.require_version("Gst", "1.0")
            gi.require_version("Xdp", "1.0")
            from gi.repository import Gst
            from gi.repository import Gio, GLib, Xdp
        except Exception as exc:
            raise RuntimeError(
                "Wayland capture needs PyGObject, libportal's Xdp introspection "
                "bindings, and GStreamer's PipeWire plugin (pipewiresrc). Install "
                "the corresponding system packages for your Linux distribution."
            ) from exc
        self.Gst, self.Gio, self.GLib, self.Xdp = Gst, Gio, GLib, Xdp
        self.source = str(source).lower()
        self.restore_token = restore_token
        self.new_restore_token = None
        if self.source not in ("monitor", "window"):
            raise ValueError("Wayland capture source must be 'monitor' or 'window'.")
        Gst.init(None)
        self._lock = threading.Lock()
        self._frame = None
        self.monitors = [{"left": 0, "top": 0, "width": 1, "height": 1}]
        try:
            self._start_portal()
        except Exception:
            self.close()
            raise

    def _start_portal(self):
        Gst = self.Gst
        # Portal setup runs in a capture QThread. Give its Gio async callbacks
        # a private GLib context instead of sharing the process-wide default
        # context with Qt's event dispatcher.
        context = self.GLib.MainContext.new()
        context.push_thread_default()
        loop = self.GLib.MainLoop.new(context, False)
        failure = []
        cancellable = self.Gio.Cancellable.new()

        def started(session, result, _data=None):
            try:
                if not session.start_finish(result):
                    raise RuntimeError("The desktop portal could not start screen sharing.")
                streams = session.get_streams().unpack()
                if not streams:
                    raise RuntimeError("The portal did not return a selected screen stream.")
                self._node_id, props = streams[0]
                # Window streams may omit compositor position; their regions
                # are addressed from the window's top-left corner.
                self._monitor_left, self._monitor_top = props.get("position", (0, 0))
                self._pw_fd = session.open_pipewire_remote()
                if self._pw_fd < 0:
                    raise RuntimeError("The portal could not open its PipeWire connection.")
                self._session = session
                try:
                    self.new_restore_token = session.get_restore_token()
                except (AttributeError, TypeError):
                    self.new_restore_token = None
            except Exception as exc:
                failure.append(exc)
            loop.quit()

        def created(portal_obj, result, _data=None):
            try:
                session = portal_obj.create_screencast_session_finish(result)
                if session is None:
                    raise RuntimeError("The desktop portal did not create a screen sharing session.")
                session.start(None, cancellable, started, None)
            except Exception as exc:
                failure.append(exc)
                loop.quit()

        try:
            portal = self.Xdp.Portal.new()
            # Ask KDE for the standard chooser with both tabs available. Some
            # portal backends suppress the chooser or provide an empty list
            # when only WINDOW is requested; the monitor-only path stays
            # explicit for normal screen capture.
            output_type = (self.Xdp.OutputType.MONITOR | self.Xdp.OutputType.WINDOW
                           if self.source == "window" else self.Xdp.OutputType.MONITOR)

            timeout_fired = []
            def portal_timeout():
                timeout_fired.append(True)
                failure.append(RuntimeError(
                "The desktop portal did not answer. Its screen/window picker may be hidden or unavailable; "
                    "check that xdg-desktop-portal-kde is running, then retry."
                ))
                cancellable.cancel()
                loop.quit()
                return False

            timeout_id = self.GLib.timeout_add_seconds(90, portal_timeout)
            persist_mode = getattr(self.Xdp.PersistMode, "PERMANENT",
                                   getattr(self.Xdp.PersistMode, "PERSISTENT",
                                           self.Xdp.PersistMode.TRANSIENT))
            portal.create_screencast_session(
                output_type,
                self.Xdp.ScreencastFlags(0),
                self.Xdp.CursorMode.EMBEDDED,
                persist_mode, self.restore_token, cancellable, created, None)
            loop.run()
        except Exception as exc:
            raise RuntimeError(f"Could not open the desktop screen sharing portal: {exc}") from exc
        finally:
            if "timeout_id" in locals() and not timeout_fired:
                self.GLib.source_remove(timeout_id)
            context.pop_thread_default()
        if failure:
            raise RuntimeError(f"Wayland screen capture could not start: {failure[0]}") from failure[0]

        if Gst.ElementFactory.find("pipewiresrc") is None:
            raise RuntimeError(
                "GStreamer has no pipewiresrc element. Install the system package "
                "that provides the GStreamer PipeWire plugin (for Debian/Ubuntu: "
                "gstreamer1.0-pipewire)."
            )
        # fd is the portal-restricted PipeWire connection, not a desktop-wide
        # PipeWire connection. The portal-selected stream is the only source.
        self._pipeline = Gst.parse_launch(
            f"pipewiresrc fd={self._pw_fd} path={self._node_id} do-timestamp=true "
            "! videoconvert ! video/x-raw,format=BGRA ! appsink name=framesink "
            "emit-signals=true max-buffers=2 drop=true sync=false"
        )
        sink = self._pipeline.get_by_name("framesink")
        sink.connect("new-sample", self._on_sample)
        if self._pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("GStreamer could not start the portal PipeWire stream (check gstreamer1.0-pipewire).")
        # Wait for the negotiated dimensions and set the monitor geometry.
        for _ in range(100):
            with self._lock:
                shape = None if self._frame is None else self._frame.shape
            if shape:
                self.monitors = [{"left": self._monitor_left, "top": self._monitor_top,
                                  "width": shape[1], "height": shape[0], "source": self.source}]
                return
            threading.Event().wait(0.05)
        raise RuntimeError("The portal granted screen sharing but no PipeWire video frames arrived.")

    def _on_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return self.Gst.FlowReturn.ERROR
        caps = sample.get_caps().get_structure(0)
        width, height = caps.get_value("width"), caps.get_value("height")
        buffer = sample.get_buffer()
        ok, info = buffer.map(self.Gst.MapFlags.READ)
        if ok:
            try:
                rows = np.frombuffer(info.data, np.uint8).reshape(height, -1)
                frame = rows[:, :width * 4].reshape(height, width, 4).copy()
                with self._lock:
                    self._frame = frame
            finally:
                buffer.unmap(info)
        return self.Gst.FlowReturn.OK

    def grab(self, region):
        with self._lock:
            if self._frame is None:
                raise RuntimeError("No frame has arrived from KDE's screen sharing portal yet.")
            frame = self._frame.copy()
        x = max(0, int(region.get("left", self._monitor_left)) - self._monitor_left)
        y = max(0, int(region.get("top", self._monitor_top)) - self._monitor_top)
        w, h = int(region.get("width", frame.shape[1])), int(region.get("height", frame.shape[0]))
        return _Shot(frame[y:y + h, x:x + w])

    def close(self):
        if getattr(self, "_pipeline", None):
            self._pipeline.set_state(self.Gst.State.NULL)
        if getattr(self, "_session", None):
            self._session.close()
        if getattr(self, "_pw_fd", -1) >= 0:
            os.close(self._pw_fd)
            self._pw_fd = -1
