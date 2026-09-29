from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from train import architecture_factory, load_config


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a reproducible EIDOS training matrix PowerShell script")
    p.add_argument("--config", required=True)
    p.add_argument("--train-bin", required=True)
    p.add_argument("--val-bin", required=True)
    p.add_argument("--test-bin", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--script-out", required=True)
    p.add_argument("--phase", choices=["calibration", "confirmation"], default="calibration")
    p.add_argument("--architectures", nargs="+", default=[
        "eidos_core", "eidos_wide_memory", "eidos_dual_anchor",
        "transformer_full", "conv_striped", "recurrent_striped",
    ])
    p.add_argument("--lrs", nargs="+", type=float, default=[3e-4, 6e-4, 1.2e-3, 2.5e-3])
    p.add_argument("--seeds", nargs="+", type=int, default=[1337])
    p.add_argument("--tokens-per-param", type=float, default=2.0)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--precision", default="bf16")
    p.add_argument("--optimizer", default="adafactor")
    p.add_argument("--compile-mode", default="default")
    args = p.parse_args()

    cfg = load_config(args.config)
    rows = []
    lines = [
        "$ErrorActionPreference = 'Continue'",
        "$env:PYTHONPATH = (Get-Location).Path",
        "",
    ]
    for architecture in args.architectures:
        _, matched_cfg, params, target = architecture_factory(architecture, cfg)
        gap = 100.0 * abs(params - target) / target
        if gap > 0.5:
            raise SystemExit(f"parameter gap for {architecture}: {gap:.4f}%")
        steps = math.ceil(
            args.tokens_per_param * params /
            (args.micro_batch * args.grad_accum * args.seq_len)
        )
        for lr in args.lrs:
            for seed in args.seeds:
                run_name = f"{architecture}_lr{lr:.8g}_seed{seed}"
                run_dir = str(Path(args.out_dir) / run_name)
                command = [
                    "python", ".\\train.py",
                    "--architecture", architecture,
                    "--config", args.config,
                    "--train-bin", args.train_bin,
                    "--val-bin", args.val_bin,
                    "--out", run_dir,
                    "--steps", str(steps),
                    "--micro-batch", str(args.micro_batch),
                    "--grad-accum", str(args.grad_accum),
                    "--seq-len", str(args.seq_len),
                    "--lr", str(lr),
                    "--warmup", str(max(100, int(steps * 0.02))),
                    "--precision", args.precision,
                    "--optimizer", args.optimizer,
                    "--compile", "--compile-mode", args.compile_mode,
                    "--seed", str(seed),
                    "--eval-every", str(max(100, steps // 20)),
                    "--eval-batches", "20",
                    "--minimum-tokens-per-param", "20",
                ]
                if args.phase == "calibration" and args.tokens_per_param < 20:
                    command.append("--allow-undertrained")
                if args.phase == "confirmation":
                    if not args.test_bin:
                        raise SystemExit("confirmation requires --test-bin")
                    command.extend(["--test-bin", args.test_bin])
                rendered = " ".join(ps_quote(x) if any(c in x for c in " \")'") else x for x in command)
                lines.extend([
                    f"Write-Host \"=== {run_name} ===\" -ForegroundColor Cyan",
                    rendered,
                    "",
                ])
                rows.append({
                    "architecture": architecture,
                    "lr": lr,
                    "seed": seed,
                    "parameters": params,
                    "parameter_gap_pct": gap,
                    "steps": steps,
                    "tokens_per_parameter": args.tokens_per_param,
                    "run_dir": run_dir,
                })

    script = Path(args.script_out)
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("\n".join(lines), encoding="utf-8")
    manifest = script.with_suffix(".json")
    manifest.write_text(json.dumps({
        "phase": args.phase,
        "config": args.config,
        "lr_grid": args.lrs,
        "rows": rows,
        "warning": "Calibration is optimizer tuning only; it is not architectural evidence.",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"script": str(script), "manifest": str(manifest), "runs": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
