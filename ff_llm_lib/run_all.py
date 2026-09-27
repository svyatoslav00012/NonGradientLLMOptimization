"""Train several methods sequentially under identical conditions and compare them.

Each method gets the same seed (so the same init and training batches), the
same final eval windows, and a freed accelerator cache. A failing method is
logged and skipped.
"""
import gc
import json
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ff_llm_lib.device import make_generator
from ff_llm_lib.experiment import save_run
from ff_llm_lib.local_learning import eval_ce_all_positions
from ff_llm_lib.model import GPT, GPTConfig
from ff_llm_lib.pipeline import run_training

RUNS_DIRS = {
    "backprop": "runs_tinystories",
    "sigprop": "runs_tinystories_sigprop",
    "dfa": "runs_tinystories_dfa",
    "sharp": "runs_tinystories_sharp",
    "ff": "runs_tinystories_ff",
}


def _free(device):
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _build(kind, config, model_kwargs):
    if kind == "backprop":
        return GPT(config)
    if kind == "sigprop":
        from ff_llm_lib.sigprop import SigPropModel
        return SigPropModel(config)
    if kind == "dfa":
        from ff_llm_lib.dfa import DFAModel
        return DFAModel(config, **model_kwargs)
    if kind == "sharp":
        from ff_llm_lib.sharp import SharpModel
        return SharpModel(config, **model_kwargs)
    if kind == "ff":
        from ff_llm_lib.forward_forward import FFModel
        return FFModel(config, **model_kwargs)
    raise ValueError(f"unknown method kind {kind!r}")


def _train_fn(kind):
    if kind == "backprop":
        from ff_llm_lib.train import train_lm
        return train_lm
    if kind == "sigprop":
        from ff_llm_lib.sigprop import train_sigprop
        return train_sigprop
    if kind == "dfa":
        from ff_llm_lib.dfa import train_dfa
        return train_dfa
    if kind == "sharp":
        from ff_llm_lib.sharp import train_sharp
        return train_sharp
    from ff_llm_lib.forward_forward import train_ff
    return train_ff


def _final_eval(kind, model, val_data, block_size, device, n_windows, seed, train_kwargs):
    gen = torch.Generator().manual_seed(seed)
    if kind == "backprop":
        model.eval()
        ce, acc = eval_ce_all_positions(lambda x: model(x)[0], val_data, block_size, device,
                                        n_windows=n_windows, generator=gen)
        model.train()
        return {"cross_entropy": ce, "accuracy": acc}
    if kind == "sigprop":
        from ff_llm_lib.sigprop import evaluate
        ev = evaluate(model, val_data, block_size, device, n_windows=n_windows, generator=gen)
        ev["ce_per_exit"] = [ev[f"ce_block_{i}"] for i in range(model.config.n_layer)]
        return ev
    if kind == "dfa":
        from ff_llm_lib.dfa import evaluate
        return evaluate(model, val_data, block_size, device, n_windows=n_windows, generator=gen)
    if kind == "sharp":
        from ff_llm_lib.sharp import evaluate
        ev = evaluate(model, val_data, block_size, device, n_windows=n_windows, generator=gen)
        ev["ce_per_exit"] = [ev[f"ce_exit_{g}"] for g in range(model.n_exits)]
        return ev
    from ff_llm_lib.forward_forward import ff_evaluate
    return ff_evaluate(model, val_data, block_size, device, n_windows=n_windows,
                       loss_kind=train_kwargs.get("loss_kind", "nce"),
                       threshold=train_kwargs.get("threshold", 2.0), generator=gen)


def _generate(kind, model, tokenizer, device, prompt, max_new_tokens, generator, train_kwargs):
    kw = dict(prompt=prompt, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40, generator=generator)
    if kind == "backprop":
        from ff_llm_lib.generate import generate_from_tokenizer
        return generate_from_tokenizer(model, tokenizer, device, **kw)
    if kind == "sigprop":
        from ff_llm_lib.sigprop import generate_sigprop
        return generate_sigprop(model, tokenizer, device, **kw)
    if kind == "dfa":
        from ff_llm_lib.dfa import generate_dfa
        return generate_dfa(model, tokenizer, device, **kw)
    if kind == "sharp":
        from ff_llm_lib.sharp import generate_sharp
        return generate_sharp(model, tokenizer, device, **kw)
    from ff_llm_lib.forward_forward import generate_ff
    return generate_ff(model, tokenizer, device, loss_kind=train_kwargs.get("loss_kind", "nce"), **kw)


def _sharp_inference_extras(model, tokenizer, val_data, block_size, device, seed, sample_generator):
    from ff_llm_lib.sharp import early_exit_eval, generate_sharp, speculative_generate
    exits = early_exit_eval(model, val_data, block_size, device, thresholds=(0.3, 0.5, 0.7, 0.9),
                            n_windows=64, generator=torch.Generator().manual_seed(seed))
    prompt, n_tok = "Once upon a time", 200
    t0 = time.time()
    generate_sharp(model, tokenizer, device, prompt=prompt, max_new_tokens=n_tok, temperature=0.8, top_k=40,
                   generator=sample_generator)
    plain_sec = time.time() - t0
    spec = []
    for draft_exit in range(model.n_exits - 1):
        for n_draft in (2, 4):
            _, st = speculative_generate(model, tokenizer, device, prompt=prompt, max_new_tokens=n_tok,
                                         draft_exit=draft_exit, n_draft=n_draft, temperature=0.8, top_k=40,
                                         generator=sample_generator)
            st.update(draft_exit=draft_exit, n_draft=n_draft, speedup=plain_sec / st["seconds"])
            spec.append(st)
    return {"early_exit": exits, "speculative": spec, "plain_generation_sec": plain_sec}


def run_method(spec, config, train_data, val_data, tokenizer, device, common_train, seed,
               final_eval_windows, batch_dir, prompts=("Once upon a time", "One day, a little"),
               sample_tokens=150):
    """
    spec: {"name": str, "kind": backprop|sigprop|dfa|sharp|ff,
           "model": {...model kwargs}, "train": {...train kwarg overrides},
           "final_eval_windows": optional override}
    Returns a result dict (also written to batch_dir/<name>.json).
    """
    name, kind = spec["name"], spec["kind"]
    model_kwargs = spec.get("model", {})
    train_kwargs = {**common_train, **spec.get("train", {})}
    print(f"\n{'=' * 100}\n{name}  ({kind})  model={model_kwargs}  train={train_kwargs}\n{'=' * 100}")

    _free(device)
    torch.manual_seed(seed)
    generator = make_generator(seed)
    sample_generator = make_generator(seed, device=device)
    model = _build(kind, config, model_kwargs).to(device)
    n_params = model.num_params() if hasattr(model, "num_params") else sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")

    train = _train_fn(kind)
    history, elapsed, resource_summary = run_training(
        lambda: train(model, train_data, val_data, config.block_size, device, generator=generator,
                      eval_seed=seed + 1, **train_kwargs),
        device,
    )
    history.setdefault("cross_entropy", history["dev_loss"])  # backprop: dev loss is the CE

    n_final = spec.get("final_eval_windows", final_eval_windows)
    ev = _final_eval(kind, model, val_data, config.block_size, device, n_final, seed + 2, train_kwargs)
    print(f"final val CE {ev['cross_entropy']:.4f} | acc {ev['accuracy']:.1%} ({n_final} windows)")

    samples = [_generate(kind, model, tokenizer, device, p, sample_tokens, sample_generator, train_kwargs)
               for p in prompts]
    for s in samples:
        print(f"--- {s!r}")

    extras = {}
    if kind == "sharp":
        extras = _sharp_inference_extras(model, tokenizer, val_data, config.block_size, device, seed + 3,
                                         sample_generator)
        for r in extras["early_exit"]:
            print(f"early exit thr {r['threshold']:.1f}: CE {r['cross_entropy']:.4f} "
                  f"| compute {r['compute_fraction']:.0%}")
        best = max(extras["speculative"], key=lambda s: s["speedup"])
        print(f"best speculative: draft exit {best['draft_exit']}, n_draft {best['n_draft']}: "
              f"accept {best['acceptance_rate']:.0%}, speedup {best['speedup']:.2f}x")

    run_dir = save_run(
        RUNS_DIRS[kind], name, config, history, elapsed, device, n_params, resource_summary=resource_summary,
        samples=samples,
        extra_metrics={"algorithm": kind, "method_name": name, "model_kwargs": model_kwargs,
                       "train_config": train_kwargs, "val_cross_entropy": ev["cross_entropy"],
                       "val_accuracy": ev["accuracy"], "final_eval_windows": n_final,
                       "ce_per_exit": ev.get("ce_per_exit"), "batch_dir": str(batch_dir), **extras},
    )

    peak = (resource_summary.get("accelerator_mem_mb") or {}).get("max")
    result = {
        "name": name, "kind": kind, "params": n_params, "val_ce": ev["cross_entropy"], "val_acc": ev["accuracy"],
        "train_sec": elapsed, "ms_per_step": elapsed / (history["step"][-1] + 1) * 1000,
        "peak_mem_mb": peak, "ce_per_exit": ev.get("ce_per_exit"), "history": history,
        "run_dir": str(run_dir), "samples": samples, **extras,
    }
    Path(batch_dir, f"{name}.json").write_text(json.dumps(result, indent=2, default=str))

    del model
    _free(device)
    return result


def run_all(methods, config_dict, train_data, val_data, tokenizer, device, common_train, seed=42,
            final_eval_windows=256, batch_root="runs_all"):
    """Run every enabled method in order; returns (results, batch_dir)."""
    config = GPTConfig(**config_dict)
    batch_dir = Path(batch_root) / time.strftime("%Y%m%d_%H%M%S")
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "plan.json").write_text(json.dumps(
        {"config": config_dict, "common_train": common_train, "methods": methods, "seed": seed}, indent=2))
    print(f"batch dir: {batch_dir}")

    results = []
    t0 = time.time()
    for spec in methods:
        if not spec.get("enabled", True):
            continue
        try:
            results.append(run_method(spec, config, train_data, val_data, tokenizer, device, common_train, seed,
                                      final_eval_windows, batch_dir))
        except Exception:
            tb = traceback.format_exc()
            print(f"!!! {spec['name']} FAILED -- skipping\n{tb}")
            (batch_dir / f"{spec['name']}.FAILED.txt").write_text(tb)
            _free(device)
    print(f"\nall done in {(time.time() - t0) / 60:.1f} min -> {batch_dir}")
    return results, batch_dir


def load_batch(batch_dir):
    """Reload results of a finished (or partially finished) batch, in run order."""
    batch_dir = Path(batch_dir)
    order = [m["name"] for m in json.loads((batch_dir / "plan.json").read_text())["methods"]]
    results = [json.loads(p.read_text()) for p in batch_dir.glob("*.json") if p.name != "plan.json"]
    return sorted(results, key=lambda r: order.index(r["name"]) if r["name"] in order else len(order))


def summary_table(results):
    cols = ["name", "params", "val_ce", "val_acc", "train_sec", "ms_per_step", "peak_mem_mb"]
    best = min(r["val_ce"] for r in results)
    bp = next((r for r in results if r["kind"] == "backprop"), None)
    rows = []
    for r in results:
        row = {c: r.get(c) for c in cols}
        row["gap_vs_bp"] = (f"{(r['val_ce'] / bp['val_ce'] - 1):+.1%}" if bp else "")
        row["mem_vs_bp"] = (f"{r['peak_mem_mb'] / bp['peak_mem_mb']:.2f}x"
                            if bp and r.get("peak_mem_mb") and bp.get("peak_mem_mb") else "")
        row["time_vs_bp"] = f"{r['train_sec'] / bp['train_sec']:.2f}x" if bp else ""
        row["name"] = ("* " if r["val_ce"] == best else "  ") + r["name"]
        rows.append(row)

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}" if v < 20 else f"{v:,.0f}"
        if isinstance(v, int):
            return f"{v:,}"
        return "" if v is None else str(v)

    keys = list(rows[0])
    w = {k: max(len(k), *(len(fmt(r[k])) for r in rows)) for k in keys}
    print("  ".join(k.ljust(w[k]) for k in keys))
    print("  ".join("-" * w[k] for k in keys))
    for r in rows:
        print("  ".join(fmt(r[k]).ljust(w[k]) for k in keys))
    print("\n* = lowest val CE. gap/mem/time relative to backprop. val_acc is a fraction.")


def plot_comparison(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for r in results:
        h = r["history"]
        axes[0].plot(h["step"], h["cross_entropy"], marker="o", label=r["name"])
        axes[1].plot(h["elapsed"], h["cross_entropy"], marker="o", label=r["name"])
    axes[0].set(xlabel="step", ylabel="val cross-entropy", title="quality vs. steps")
    axes[1].set(xlabel="training seconds (eval excluded)", title="quality vs. wall-clock")
    names = [r["name"] for r in results]
    axes[2].barh(names, [r["peak_mem_mb"] or 0 for r in results], color="tab:gray")
    axes[2].set(xlabel="peak accelerator memory (MB)", title="memory")
    axes[2].invert_yaxis()
    for ax in axes[:2]:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_exits(results):
    """Per-exit val CE for methods that have exits (SigProp, SHARP)."""
    with_exits = [r for r in results if r.get("ce_per_exit")]
    if not with_exits:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in with_exits:
        ax.plot(range(1, len(r["ce_per_exit"]) + 1), r["ce_per_exit"], marker="o", label=r["name"])
    bp = next((r for r in results if r["kind"] == "backprop"), None)
    if bp:
        ax.axhline(bp["val_ce"], color="k", linestyle="--", label="backprop (full depth)")
    ax.set(xlabel="exit (block group)", ylabel="val cross-entropy", title="quality at each exit")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
