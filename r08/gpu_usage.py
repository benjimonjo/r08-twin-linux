"""Small, optional GPU utilization sampler for the renderer HUD."""
import glob
import shutil
import subprocess
import threading


def read_gpu_usage():
    """Return utilization percent where a system interface is available."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2, check=True,
            )
            values = [float(line.strip().replace("%", "")) for line in result.stdout.splitlines()
                      if line.strip().replace("%", "").replace(".", "", 1).isdigit()]
            if values:
                return sum(values) / len(values)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass

    # Linux DRM exposes this counter on AMD GPUs without extra packages.
    for path in sorted(glob.glob("/sys/class/drm/card[0-9]*/device/gpu_busy_percent")):
        try:
            value = float(open(path, encoding="ascii").read().strip())
            if 0 <= value <= 100:
                return value
        except (OSError, ValueError):
            continue
    return None


class GpuUsageMonitor(threading.Thread):
    """Poll GPU utilization off the UI thread; ``latest`` is percent or None."""
    def __init__(self, interval=2.0):
        super().__init__(daemon=True, name="gpu-usage")
        self.interval = interval
        self.latest = None
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            self.latest = read_gpu_usage()
            self.stop_event.wait(self.interval)

    def stop(self):
        self.stop_event.set()
