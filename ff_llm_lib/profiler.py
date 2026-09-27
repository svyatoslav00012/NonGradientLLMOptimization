import threading
import time

import psutil
import torch


def _accelerator_current_memory_mb(device):
    """Currently allocated accelerator memory in MB (the monitor tracks the max itself)."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1024 ** 2
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1024 ** 2
    return None


def _accelerator_utilization_pct(device):
    """GPU utilization %, CUDA only (torch exposes no utilization API on MPS)."""
    if device.type == "cuda":
        try:
            return torch.cuda.utilization(device)
        except Exception:
            return None
    return None


class ResourceMonitor:
    """Samples CPU, RAM and accelerator usage on a background thread while the block runs.

        with ResourceMonitor(device) as mon:
            history, elapsed = train(...)
        summary = mon.summary()
    """

    def __init__(self, device, interval=0.2):
        self.device = device
        self.interval = interval
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread = None
        self._samples = {
            "process_cpu_pct_of_one_core": [],
            "system_cpu_pct": [],
            "process_rss_mb": [],
            "accelerator_mem_mb": [],
            "accelerator_util_pct": [],
        }

    def _sample_loop(self):
        # the first cpu_percent() call returns a meaningless value; discard it
        self._proc.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)

        while not self._stop.wait(self.interval):
            self._samples["process_cpu_pct_of_one_core"].append(self._proc.cpu_percent(interval=None))
            self._samples["system_cpu_pct"].append(psutil.cpu_percent(interval=None))
            self._samples["process_rss_mb"].append(self._proc.memory_info().rss / 1024 ** 2)

            mem = _accelerator_current_memory_mb(self.device)
            if mem is not None:
                self._samples["accelerator_mem_mb"].append(mem)

            util = _accelerator_utilization_pct(self.device)
            if util is not None:
                self._samples["accelerator_util_pct"].append(util)

    def __enter__(self):
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop.set()
        self._thread.join(timeout=self.interval * 5)
        self.wall_time_sec = time.time() - self._t0

    def summary(self):
        n_cores = psutil.cpu_count(logical=True)

        def stats(key):
            vals = self._samples[key]
            if not vals:
                return None
            return {"avg": sum(vals) / len(vals), "max": max(vals), "n_samples": len(vals)}

        cpu_one_core = stats("process_cpu_pct_of_one_core")
        return {
            "sample_interval_sec": self.interval,
            "n_cpu_cores": n_cores,
            # psutil reports % of one core; divide by core count for % of the machine
            "process_cpu_pct_of_one_core": cpu_one_core,
            "process_cpu_pct_of_machine": (
                {"avg": cpu_one_core["avg"] / n_cores, "max": cpu_one_core["max"] / n_cores,
                 "n_samples": cpu_one_core["n_samples"]}
                if cpu_one_core else None
            ),
            "system_cpu_pct_of_machine": stats("system_cpu_pct"),
            "process_rss_mb": stats("process_rss_mb"),
            "accelerator_mem_mb": stats("accelerator_mem_mb"),
            "accelerator_util_pct": stats("accelerator_util_pct"),
        }
