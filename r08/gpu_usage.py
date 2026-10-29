"""Application GPU utilization sampler for the renderer HUD."""
import os
import threading

try:
    import pynvml
except ImportError:
    pynvml = None

def read_gpu_usage(nvml=pynvml, pid=None):
    """Return this process's NVIDIA SM utilization, or ``None`` if unavailable."""
    if nvml is None:
        return None
    try:
        process_id = int(os.getpid() if pid is None else pid)
        values = []
        for index in range(nvml.nvmlDeviceGetCount()):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            samples = nvml.nvmlDeviceGetProcessUtilization(handle, process_id, 0)
            values.extend(float(sample.smUtil) for sample in samples)
        if values:
            return min(100.0, max(0.0, sum(values) / len(values)))
    except (AttributeError, OSError, TypeError, ValueError, nvml.NVMLError):
        return None
    return None


class GpuUsageMonitor(threading.Thread):
    """Poll GPU utilization off the UI thread; ``latest`` is percent or None."""
    def __init__(self, interval=2.0):
        super().__init__(daemon=True, name="gpu-usage")
        self.interval = interval
        self.latest = None
        self.stop_event = threading.Event()

    def run(self):
        initialized = False
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                initialized = True
            except pynvml.NVMLError:
                pass
        while not self.stop_event.is_set():
            self.latest = read_gpu_usage()
            self.stop_event.wait(self.interval)
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass

    def stop(self):
        self.stop_event.set()
