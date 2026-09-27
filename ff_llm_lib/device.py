import platform

import psutil
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_generator(seed, device=None):
    """Seeded generator; device=None gives a CPU generator (used for batching)."""
    gen = torch.Generator(device=device) if device is not None else torch.Generator()
    gen.manual_seed(seed)
    return gen


def device_memory_mb(device):
    """Accelerator memory in MB, if the backend exposes it."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024 ** 2
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1024 ** 2
    return None


def reset_memory_stats(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def system_snapshot(device):
    proc = psutil.Process()
    return {
        "device": str(device),
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(logical=True),
        "process_cpu_percent": proc.cpu_percent(interval=0.1),
        "process_rss_mb": proc.memory_info().rss / 1024 ** 2,
        "system_ram_used_gb": psutil.virtual_memory().used / 1024 ** 3,
        "system_ram_total_gb": psutil.virtual_memory().total / 1024 ** 3,
        "accelerator_mem_mb": device_memory_mb(device),
    }
