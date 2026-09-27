import csv
import dataclasses
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt


def plot_history(history, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["step"], history["train_loss"], label="train loss")
    axes[0].plot(history["step"], history["dev_loss"], label="dev/val loss")
    if history.get("cross_entropy"):
        # next-token CE, comparable across methods (their train losses are not)
        axes[0].plot(history["step"], history["cross_entropy"], label="cross-entropy", linestyle="--", color="tab:red")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].set_title(title)
    axes[0].legend()

    axes[1].plot(history["step"], history["lr"], color="tab:green")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("learning rate")
    axes[1].set_title("LR schedule")

    fig.tight_layout()
    return fig


def save_run(runs_dir, tag, config, history, elapsed, device, num_params,
             resource_summary=None, samples=None, extra_metrics=None):
    """Save one run to runs_dir/<timestamp>_<tag>/ and append a row to runs_dir/manifest.csv.

    Writes config.json, history.json, metrics.json, loss.png and (optionally) samples.txt.
    """
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / f"{timestamp}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=False)

    config_dict = dataclasses.asdict(config)
    (run_dir / "config.json").write_text(json.dumps(config_dict, indent=2))
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    n_steps = history["step"][-1] + 1 if history["step"] else 0
    metrics = {
        "tag": tag,
        "timestamp": timestamp,
        "device": str(device),
        "num_params": num_params,
        "steps": n_steps,
        "elapsed_sec": elapsed,
        "ms_per_step": (elapsed / n_steps * 1000) if n_steps else None,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_dev_loss": history["dev_loss"][-1] if history["dev_loss"] else None,
        "resource_summary": resource_summary,
        **(extra_metrics or {}),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    fig = plot_history(history, tag)
    fig.savefig(run_dir / "loss.png", dpi=120)
    plt.close(fig)

    if samples is not None:
        text = "\n".join(samples) if isinstance(samples, (list, tuple)) else str(samples)
        (run_dir / "samples.txt").write_text(text)

    _append_manifest_row(runs_dir, run_dir.name, config_dict, metrics)

    print(f"saved run to {run_dir}")
    return run_dir


_MANIFEST_FIELDS = [
    "run", "tag", "timestamp", "device", "num_params", "n_embd", "n_head", "n_layer",
    "block_size", "dropout", "use_rope", "use_rmsnorm", "use_swiglu",
    "steps", "elapsed_sec", "ms_per_step", "final_train_loss", "final_dev_loss",
    "avg_cpu_pct_of_machine", "max_rss_mb", "max_accelerator_mem_mb", "avg_accelerator_util_pct",
]


def _round(x, n=4):
    return round(x, n) if isinstance(x, (int, float)) else x


def _get(d, *keys, default=None):
    for k in keys:
        if d is None:
            return default
        d = d.get(k)
    return d if d is not None else default


def _append_manifest_row(runs_dir, run_name, config_dict, metrics):
    manifest_path = Path(runs_dir) / "manifest.csv"
    rs = metrics.get("resource_summary")
    row = {
        "run": run_name,
        "tag": metrics["tag"],
        "timestamp": metrics["timestamp"],
        "device": metrics["device"],
        "num_params": metrics["num_params"],
        "steps": metrics["steps"],
        "elapsed_sec": _round(metrics["elapsed_sec"], 2),
        "ms_per_step": _round(metrics["ms_per_step"], 2),
        "final_train_loss": _round(metrics["final_train_loss"]),
        "final_dev_loss": _round(metrics["final_dev_loss"]),
        "avg_cpu_pct_of_machine": _round(_get(rs, "process_cpu_pct_of_machine", "avg")),
        "max_rss_mb": _round(_get(rs, "process_rss_mb", "max"), 1),
        "max_accelerator_mem_mb": _round(_get(rs, "accelerator_mem_mb", "max"), 1),
        "avg_accelerator_util_pct": _round(_get(rs, "accelerator_util_pct", "avg")),
        **{k: config_dict.get(k) for k in
           ["n_embd", "n_head", "n_layer", "block_size", "dropout", "use_rope", "use_rmsnorm", "use_swiglu"]},
    }
    write_header = not manifest_path.exists()
    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def list_runs(runs_dir):
    """Print runs_dir/manifest.csv as a table."""
    manifest_path = Path(runs_dir) / "manifest.csv"
    if not manifest_path.exists():
        print(f"no runs logged yet under {runs_dir}")
        return []
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return rows
    widths = {k: max(len(k), max(len(str(r[k])) for r in rows)) for k in rows[0]}
    header = "  ".join(k.ljust(widths[k]) for k in rows[0])
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r[k]).ljust(widths[k]) for k in r))
    return rows


def load_run(runs_dir, run_name):
    """Reload config, history and metrics of one saved run."""
    run_dir = Path(runs_dir) / run_name
    return {
        "config": json.loads((run_dir / "config.json").read_text()),
        "history": json.loads((run_dir / "history.json").read_text()),
        "metrics": json.loads((run_dir / "metrics.json").read_text()),
    }
