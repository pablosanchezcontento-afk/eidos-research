from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from eidos.baselines import ConvStripedLM, RecurrentStripedLM, TransformerLM, match_ffn_hidden
from eidos.data import RandomPackedLoader
from eidos.model import EidosConfig, EidosLM


def load_config(path: str) -> EidosConfig:
    return EidosConfig(**json.loads(Path(path).read_text(encoding="utf-8")))


def architecture_factory(name: str, base_cfg: EidosConfig):
    target = EidosLM(replace(base_cfg, macrocycle="CMCA", mtp_weight=0.0)).parameter_count()
    if name.startswith("eidos_"):
        specs = {
            "eidos_core": ("CMCA", base_cfg.memory_rank),
            "eidos_wide_memory": ("CMCA", min(base_cfg.dim, base_cfg.memory_rank * 2)),
            "eidos_dual_anchor": ("CAMA", base_cfg.memory_rank),
            "eidos_pulse": ("MCMA", base_cfg.memory_rank),
        }
        if name not in specs:
            raise ValueError(f"unknown architecture {name}")
        pattern, memory_rank = specs[name]
        def make(hidden: int) -> EidosConfig:
            return replace(
                base_cfg, macrocycle=pattern, memory_rank=memory_rank,
                ffn_hidden=hidden, mtp_weight=0.0
            )
        p8 = EidosLM(make(8)).parameter_count()
        p16 = EidosLM(make(16)).parameter_count()
        slope = (p16 - p8) / 8.0
        estimate = int(round(8 + (target - p8) / slope))
        center = max(8, min(32768, int(round(estimate / 8)) * 8))
        best = None
        for hidden in sorted({max(8, min(32768, center + d)) for d in (-16, -8, 0, 8, 16)}):
            cfg = make(hidden)
            params = EidosLM(cfg).parameter_count()
            err = abs(params - target)
            if best is None or err < best[0]:
                best = (err, cfg, params)
        assert best is not None
        cfg, params = best[1], best[2]
        return EidosLM(cfg), cfg, params, target

    factories = {
        "transformer_full": TransformerLM,
        "conv_striped": ConvStripedLM,
        "recurrent_striped": RecurrentStripedLM,
    }
    if name not in factories:
        raise ValueError(f"unknown architecture {name}")
    factory = factories[name]
    cfg, params = match_ffn_hidden(base_cfg, target, factory)
    cfg = replace(cfg, mtp_weight=0.0)
    return factory(cfg), cfg, params, target


def cosine_lr(step: int, total: int, warmup: int, peak: float, minimum: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def make_optimizer(model: torch.nn.Module, kind: str, lr: float, weight_decay: float):
    if kind == "adafactor":
        return torch.optim.Adafactor(model.parameters(), lr=lr, weight_decay=weight_decay)
    if kind == "bnb8":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError("Install bitsandbytes or use adafactor/adamw") from exc
        return bnb.optim.PagedAdamW8bit(
            model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay
        )
    return torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay
    )


@torch.no_grad()
def evaluate(model, loader, batches, autocast_context) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = loader.next()
        with autocast_context():
            losses.append(float(model(x, y)["loss"]))
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    p = argparse.ArgumentParser(description="Controlled EIDOS/baseline pretraining run")
    p.add_argument("--architecture", required=True, choices=[
        "eidos_core", "eidos_wide_memory", "eidos_dual_anchor", "eidos_pulse",
        "transformer_full", "conv_striped", "recurrent_striped",
    ])
    p.add_argument("--config", required=True)
    p.add_argument("--train-bin", required=True)
    p.add_argument("--val-bin", required=True)
    p.add_argument("--test-bin", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--token-dtype", choices=["uint16", "uint32"], default="uint16")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--optimizer", choices=["adamw", "adafactor", "bnb8"], default="adafactor")
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="default")
    p.add_argument("--minimum-tokens-per-param", type=float, default=20.0)
    p.add_argument("--allow-undertrained", action="store_true")
    p.add_argument("--uniform-margin-bits", type=float, default=0.05)
    args = p.parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    base_cfg = load_config(args.config)
    base_cfg = replace(base_cfg, gradient_checkpointing=False, mtp_weight=0.0)
    if args.seq_len > base_cfg.max_seq_len:
        raise SystemExit("seq-len exceeds max_seq_len")
    model, cfg, params, target = architecture_factory(args.architecture, base_cfg)
    parameter_gap_pct = 100.0 * abs(params - target) / target
    if parameter_gap_pct > 0.5:
        raise SystemExit(f"parameter matching failed: {parameter_gap_pct:.4f}%")

    planned_tokens = args.steps * args.micro_batch * args.grad_accum * args.seq_len
    tokens_per_param = planned_tokens / params
    if tokens_per_param < args.minimum_tokens_per_param and not args.allow_undertrained:
        raise SystemExit(
            f"planned run has {tokens_per_param:.3f} tokens/parameter; "
            f"minimum is {args.minimum_tokens_per_param}. Use --allow-undertrained only for smoke tests."
        )

    if args.precision == "bf16":
        amp_dtype = torch.bfloat16
    elif args.precision == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
    parameter_dtype = amp_dtype if device.type == "cuda" and args.precision != "fp32" else torch.float32
    model = model.to(device=device, dtype=parameter_dtype).train()
    optimizer = make_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")
    raw = model
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode, fullgraph=False)

    def autocast_context():
        if device.type == "cuda" and args.precision != "fp32":
            return torch.autocast("cuda", dtype=amp_dtype)
        return nullcontext()

    train = RandomPackedLoader(args.train_bin, args.micro_batch, args.seq_len, device, args.seed, args.token_dtype)
    val = RandomPackedLoader(args.val_bin, args.micro_batch, args.seq_len, device, args.seed + 1, args.token_dtype)
    test = (
        RandomPackedLoader(args.test_bin, args.micro_batch, args.seq_len, device, args.seed + 2, args.token_dtype)
        if args.test_bin else None
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    wall_start = time.perf_counter()
    tokens_seen = 0
    best_val = float("inf")
    best_step = -1
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in range(args.steps):
        lr = cosine_lr(step, args.steps, args.warmup, args.lr, args.lr * args.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        step_start = time.perf_counter()
        for _ in range(args.grad_accum):
            x, y = train.next()
            with autocast_context():
                loss = model(x, y)["loss"] / args.grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += float(loss.detach())
            tokens_seen += x.numel()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), args.clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        step_seconds = time.perf_counter() - step_start

        record = {
            "step": step,
            "train_loss_nats": total_loss,
            "train_bits_per_token": total_loss / math.log(2.0),
            "lr": lr,
            "grad_norm": float(grad_norm),
            "tokens_seen": tokens_seen,
            "tokens_per_second": args.micro_batch * args.grad_accum * args.seq_len / max(step_seconds, 1e-9),
            "wall_seconds": time.perf_counter() - wall_start,
        }
        if step % 10 == 0 or step == args.steps - 1:
            print(json.dumps(record), flush=True)
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            val_loss = evaluate(model, val, args.eval_batches, autocast_context)
            if val_loss < best_val:
                best_val, best_step = val_loss, step
            val_record = {"step": step, "val_loss_nats": val_loss, "val_bits_per_token": val_loss / math.log(2.0)}
            print(json.dumps(val_record), flush=True)

    final_val = evaluate(model, val, args.eval_batches, autocast_context)
    final_test = evaluate(model, test, args.eval_batches, autocast_context) if test else None
    uniform_bits = math.log2(cfg.vocab_size)
    val_bits = final_val / math.log(2.0)
    test_bits = final_test / math.log(2.0) if final_test is not None else None
    learned_gate = val_bits < uniform_bits - args.uniform_margin_bits
    if test_bits is not None:
        learned_gate = learned_gate and test_bits < uniform_bits - args.uniform_margin_bits

    summary = {
        "schema_version": 1,
        "architecture": args.architecture,
        "config": cfg.to_dict(),
        "parameters": params,
        "parameter_gap_pct": parameter_gap_pct,
        "seed": args.seed,
        "lr": args.lr,
        "steps": args.steps,
        "planned_tokens": planned_tokens,
        "tokens_per_parameter": tokens_per_param,
        "undertrained_override": args.allow_undertrained,
        "final_val_loss_nats": final_val,
        "final_val_bits_per_token": val_bits,
        "final_test_loss_nats": final_test,
        "final_test_bits_per_token": test_bits,
        "uniform_bits_per_token": uniform_bits,
        "learned_better_than_uniform_gate": learned_gate,
        "best_observed_val_loss_nats": best_val if math.isfinite(best_val) else None,
        "best_observed_step": best_step,
        "wall_seconds": time.perf_counter() - wall_start,
        "mean_training_tokens_per_second": planned_tokens / max(time.perf_counter() - wall_start, 1e-9),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
        "test_was_sealed_until_final_evaluation": bool(test),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save({"model": raw.state_dict(), "config": cfg.to_dict(), "summary": summary}, out / "checkpoint_final.pt")
    print(json.dumps(summary, indent=2))
    if not learned_gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
