from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch

from benchmarks.benchmark_gpu import match_eidos_variant
from eidos.baselines import ConvStripedLM, TransformerLM, match_ffn_hidden
from eidos.model import EidosConfig, EidosLM


def main() -> None:
    p = argparse.ArgumentParser(description="CPU diagnostic only; never substitute for the RTX benchmark")
    p.add_argument("--config", default="configs/eidos_glyph_10m.json")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--out", default="results/cpu_diagnostic.json")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    cfg = EidosConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    cfg = replace(cfg, gradient_checkpointing=False, mtp_weight=0.0)
    target = EidosLM(cfg).parameter_count()

    specs = []
    for name, pattern, rank in [
        ("eidos_core", "CMCA", cfg.memory_rank),
        ("eidos_wide_memory", "CMCA", min(cfg.dim, cfg.memory_rank * 2)),
        ("eidos_dual_anchor", "CAMA", cfg.memory_rank),
        ("eidos_pulse", "MCMA", cfg.memory_rank),
    ]:
        matched, _ = match_eidos_variant(cfg, pattern, target, rank)
        specs.append((name, EidosLM, matched))
    for name, factory in [("transformer_full", TransformerLM), ("conv_striped", ConvStripedLM)]:
        matched, _ = match_ffn_hidden(cfg, target, factory)
        specs.append((name, factory, matched))

    rows = []
    for name, factory, matched in specs:
        torch.manual_seed(1337)
        model = factory(matched).train()
        x = torch.randint(0, matched.vocab_size, (1, args.seq_len))
        y = torch.randint(0, matched.vocab_size, (1, args.seq_len))
        times = []
        for _ in range(args.steps):
            model.zero_grad(set_to_none=True)
            start = time.perf_counter()
            model(x, y)["loss"].backward()
            times.append(time.perf_counter() - start)
        measured = times[2:] if len(times) > 2 else times
        median = statistics.median(measured)
        rows.append({
            "name": name,
            "parameters": model.parameter_count(),
            "ffn_hidden": matched.ffn_hidden,
            "memory_rank": matched.memory_rank,
            "median_step_seconds": median,
            "tokens_per_second": args.seq_len / median,
            "all_step_seconds": times,
        })
    transformer = next(r for r in rows if r["name"] == "transformer_full")
    for row in rows:
        row["ratio_vs_transformer"] = row["tokens_per_second"] / transformer["tokens_per_second"]
    payload = {
        "device": "CPU diagnostic sandbox",
        "threads": args.threads,
        "seq_len": args.seq_len,
        "results": rows,
        "warning": "CPU results are implementation diagnostics, not RTX 5070 evidence.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
