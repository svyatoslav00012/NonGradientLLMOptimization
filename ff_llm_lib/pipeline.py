import matplotlib.pyplot as plt
import torch

from ff_llm_lib.device import get_device, make_generator, system_snapshot
from ff_llm_lib.experiment import list_runs, load_run, plot_history, save_run
from ff_llm_lib.profiler import ResourceMonitor

__all__ = ["setup", "run_training", "show_loss", "finalize_run", "save_run", "list_runs", "load_run"]


def setup(seed=42):
    """Seed torch, pick the device, return (device, cpu_generator, device_generator)."""
    torch.manual_seed(seed)
    device = get_device()
    cpu_generator = make_generator(seed)
    sample_generator = make_generator(seed, device=device)
    print(f"device: {device}")
    print(system_snapshot(device))
    return device, cpu_generator, sample_generator


def run_training(train_call, device):
    """Run a zero-arg training closure under ResourceMonitor and print a usage summary."""
    with ResourceMonitor(device, interval=0.2) as mon:
        history, elapsed = train_call()
    resource_summary = mon.summary()

    n_steps = history["step"][-1] + 1
    print(f"\ntraining time: {elapsed:.1f}s for {n_steps} steps ({elapsed / n_steps * 1000:.2f} ms/step)")
    print(f"cpu: avg {resource_summary['process_cpu_pct_of_machine']['avg']:.1f}% "
          f"/ max {resource_summary['process_cpu_pct_of_machine']['max']:.1f}% of {resource_summary['n_cpu_cores']} cores "
          f"(this process; system-wide avg {resource_summary['system_cpu_pct_of_machine']['avg']:.1f}%)")
    print(f"ram (process RSS): avg {resource_summary['process_rss_mb']['avg']:.0f} MB "
          f"/ peak {resource_summary['process_rss_mb']['max']:.0f} MB")
    if resource_summary["accelerator_mem_mb"]:
        print(f"accelerator memory: avg {resource_summary['accelerator_mem_mb']['avg']:.1f} MB "
              f"/ peak {resource_summary['accelerator_mem_mb']['max']:.1f} MB")
    if resource_summary["accelerator_util_pct"]:
        print(f"accelerator utilization: avg {resource_summary['accelerator_util_pct']['avg']:.1f}%")
    elif device.type == "mps":
        print("accelerator utilization: not available on MPS")

    return history, elapsed, resource_summary


def show_loss(history, title=""):
    fig = plot_history(history, title)
    plt.show()


def finalize_run(runs_dir, tag, config, history, elapsed, device, model,
                  resource_summary, samples=None, extra_metrics=None):
    """save_run() followed by list_runs()."""
    run_dir = save_run(runs_dir, tag, config, history, elapsed, device, model.num_params(),
                        resource_summary=resource_summary, samples=samples, extra_metrics=extra_metrics)
    list_runs(runs_dir)
    return run_dir
