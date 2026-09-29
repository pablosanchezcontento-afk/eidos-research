from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import torch

from eidos.baselines import (
    ConvStripedLM,
    RecurrentStripedLM,
    TransformerLM,
    match_ffn_hidden,
)
from eidos.model import EidosConfig, EidosLM


def make_optimizer(model: torch.nn.Module, kind: str, lr: float):
    if kind == "adafactor":
        return torch.optim.Adafactor(model.parameters(), lr=lr, weight_decay=0.1)
    if kind == "bnb8":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError("Install bitsandbytes for --optimizer bnb8") from exc
        return bnb.optim.PagedAdamW8bit(
            model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
        )
    return torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
    )


def match_eidos_variant(
    base_cfg: EidosConfig,
    macrocycle: str,
    target_params: int,
    memory_rank: int | None = None,
    max_hidden: int = 32768,
) -> tuple[EidosConfig, int]:
    """Match an EIDOS variant without constructing hundreds of models."""

    rank = memory_rank or base_cfg.memory_rank
    def make(hidden: int) -> EidosConfig:
        return replace(
            base_cfg, macrocycle=macrocycle, memory_rank=rank,
            ffn_hidden=hidden, mtp_weight=0.0
        )
    p8 = EidosLM(make(8)).parameter_count()
    p16 = EidosLM(make(16)).parameter_count()
    slope = (p16 - p8) / 8.0
    estimate = int(round(8 + (target_params - p8) / slope))
    center = max(8, min(max_hidden, int(round(estimate / 8)) * 8))
    candidates = sorted({max(8, min(max_hidden, center + d)) for d in (-16, -8, 0, 8, 16)})
    best = None
    for hidden in candidates:
        cfg = make(hidden)
        params = EidosLM(cfg).parameter_count()
        error = abs(params - target_params)
        if best is None or error < best[0]:
            best = (error, cfg, params)
    if best is None:
        raise RuntimeError("could not parameter-match EIDOS variant")
    return best[1], best[2]


def benchmark(
    name: str,
    model: torch.nn.Module,
    cfg: EidosConfig,
    args,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = model.to(device=device, dtype=dtype).train()
    optimizer = make_optimizer(model, args.optimizer, args.lr)
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode, fullgraph=False)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")

    # Fixed batches remove host-side RNG and allocation noise from measured steps.
    x = torch.randint(0, cfg.vocab_size, (args.micro_batch, args.seq_len), device=device)
    y = torch.randint(0, cfg.vocab_size, (args.micro_batch, args.seq_len), device=device)
    times: list[float] = []
    losses: list[float] = []
    for step in range(args.warmup + args.steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=dtype):
            loss = model(x, y)["loss"]
        if args.precision == "fp16":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        if step >= args.warmup:
            times.append(time.perf_counter() - start)
            losses.append(float(loss.detach()))

    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    mean = sum(times) / len(times)
    tokens = args.micro_batch * args.seq_len
    return {
        "name": name,
        "parameters": raw.parameter_count(),
        "ffn_hidden": cfg.ffn_hidden,
        "macrocycle": getattr(cfg, "macrocycle", ""),
        "mean_step_seconds": mean,
        "training_tokens_per_second": tokens / mean,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "random_batch_loss_not_quality": sum(losses) / len(losses),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Parameter-matched EIDOS and strong baselines on one CUDA GPU"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--optimizer", choices=["adafactor", "bnb8", "adamw"], default="adafactor")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--compile", action="store_true")
    p.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
    )
    p.add_argument("--out", default="")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Run this on the RTX 5070 machine.")
    base_cfg = EidosConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    base_cfg = replace(base_cfg, gradient_checkpointing=False, mtp_weight=0.0)
    if args.seq_len > base_cfg.max_seq_len:
        raise SystemExit("seq-len exceeds config max_seq_len")

    target_params = EidosLM(base_cfg).parameter_count()
    candidates: list[tuple[str, EidosConfig, Callable[[EidosConfig], torch.nn.Module]]] = []

    variant_specs = [
        ("eidos_core", "CMCA", base_cfg.memory_rank),
        ("eidos_wide_memory", "CMCA", min(base_cfg.dim, base_cfg.memory_rank * 2)),
        ("eidos_dual_anchor", "CAMA", base_cfg.memory_rank),
        ("eidos_pulse", "MCMA", base_cfg.memory_rank),
    ]
    for name, pattern, rank in variant_specs:
        cfg, params = match_eidos_variant(base_cfg, pattern, target_params, memory_rank=rank)
        gap = 100.0 * abs(params - target_params) / target_params
        if gap > 0.5:
            raise SystemExit(f"parameter matching failed for {name}: {gap:.4f}%")
        candidates.append((name, cfg, EidosLM))

    for name, factory in [
        ("transformer_full", TransformerLM),
        ("conv_striped", ConvStripedLM),
        ("recurrent_striped", RecurrentStripedLM),
    ]:
        cfg, params = match_ffn_hidden(base_cfg, target_params, factory)
        cfg = replace(cfg, mtp_weight=0.0, gradient_checkpointing=False)
        gap = 100.0 * abs(params - target_params) / target_params
        if gap > 0.5:
            raise SystemExit(f"parameter matching failed for {name}: {gap:.4f}%")
        candidates.append((name, cfg, factory))

    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    rows = []
    for name, cfg, factory in candidates:
        print(f"\n=== {name} ({cfg.macrocycle}, ffn={cfg.ffn_hidden}) ===", flush=True)
        rows.append(benchmark(name, factory(cfg), cfg, args, device, dtype))

    transformer = next(row for row in rows if row["name"] == "transformer_full")
    transformer_speed = transformer["training_tokens_per_second"]
    for row in rows:
        row["throughput_ratio_vs_transformer"] = (
            row["training_tokens_per_second"] / transformer_speed
        )
        row["efficiency_gate_1p05x"] = row["throughput_ratio_vs_transformer"] >= 1.05

    payload = {
        "schema_version": 1,
        "architecture": "EIDOS v0.4",
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "seq_len": args.seq_len,
        "micro_batch": args.micro_batch,
        "precision": args.precision,
        "optimizer": args.optimizer,
        "compile": args.compile,
        "compile_mode": args.compile_mode if args.compile else None,
        "target_parameters": target_params,
        "results": rows,
        "gates": {
            "efficiency_definition": "candidate throughput must be at least 1.05x Transformer",
            "quality_not_measured_here": True,
        },
        "note": "Random-batch loss is deliberately not interpreted as quality.",
    }
    text = json.dumps(payload, indent=2)
    print("\n" + text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
