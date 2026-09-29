from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_gate(
    speed: dict,
    summaries: list[dict],
    minimum_seeds: int = 3,
    quality_margin_pct: float = 0.5,
    throughput_ratio: float = 1.05,
) -> dict:
    """Pure gate logic: aggregate sealed-test summaries and decide which EIDOS candidate (if any) is promoted."""
    speed_rows = {row["name"]: row for row in speed["results"]}
    if "transformer_full" not in speed_rows:
        raise ValueError("speed JSON has no transformer_full row")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in summaries:
        grouped[row["architecture"]].append(row)

    baseline_names = ["transformer_full", "conv_striped", "recurrent_striped"]
    missing = [name for name in baseline_names if name not in grouped]
    if missing:
        raise ValueError(f"missing quality summaries for baselines: {missing}")

    aggregate = {}
    for name, rows in grouped.items():
        test_values = [r["final_test_bits_per_token"] for r in rows]
        if any(v is None for v in test_values):
            raise ValueError(f"{name} contains a run without sealed test evaluation")
        aggregate[name] = {
            "seeds": sorted(r["seed"] for r in rows),
            "n": len(rows),
            "mean_test_bits": statistics.mean(test_values),
            "stdev_test_bits": statistics.stdev(test_values) if len(test_values) > 1 else None,
            "all_better_than_uniform": all(r["learned_better_than_uniform_gate"] for r in rows),
            "max_parameter_gap_pct": max(r["parameter_gap_pct"] for r in rows),
            "minimum_tokens_per_parameter": min(r["tokens_per_parameter"] for r in rows),
        }

    baseline_best_name = min(baseline_names, key=lambda name: aggregate[name]["mean_test_bits"])
    baseline_best_bits = aggregate[baseline_best_name]["mean_test_bits"]
    transformer_speed = speed_rows["transformer_full"]["training_tokens_per_second"]

    candidates = {}
    for name, metrics in aggregate.items():
        if not name.startswith("eidos_"):
            continue
        speed_row = speed_rows.get(name)
        ratio = speed_row["training_tokens_per_second"] / transformer_speed if speed_row else 0.0
        quality_gain_pct = 100.0 * (baseline_best_bits - metrics["mean_test_bits"]) / baseline_best_bits
        checks = {
            "enough_seeds": metrics["n"] >= minimum_seeds,
            "all_runs_learned": metrics["all_better_than_uniform"],
            "parameters_matched": metrics["max_parameter_gap_pct"] <= 0.5,
            "tokens_per_parameter_at_least_20": metrics["minimum_tokens_per_parameter"] >= 20.0,
            "beats_best_baseline_by_margin": quality_gain_pct >= quality_margin_pct,
            "faster_than_transformer_by_margin": ratio >= throughput_ratio,
        }
        candidates[name] = {
            **metrics,
            "throughput_ratio_vs_transformer": ratio,
            "quality_gain_pct_vs_best_baseline": quality_gain_pct,
            "best_baseline": baseline_best_name,
            "checks": checks,
            "promotion_pass": all(checks.values()),
        }

    passing = [name for name, row in candidates.items() if row["promotion_pass"]]
    winner = min(passing, key=lambda name: candidates[name]["mean_test_bits"]) if passing else None
    return {
        "schema_version": 1,
        "gate": "EIDOS Mythos promotion gate",
        "baseline_aggregates": {name: aggregate[name] for name in baseline_names},
        "best_baseline": baseline_best_name,
        "candidates": candidates,
        "winner": winner,
        "mythos_unlocked": winner is not None,
        "interpretation": (
            "A model is promoted only if the same EIDOS candidate beats the strongest measured baseline "
            "in sealed test quality and exceeds Transformer throughput."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Apply the non-negotiable EIDOS promotion gate")
    p.add_argument("--speed-json", required=True)
    p.add_argument("--summaries", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--minimum-seeds", type=int, default=3)
    p.add_argument("--quality-margin-pct", type=float, default=0.5)
    p.add_argument("--throughput-ratio", type=float, default=1.05)
    args = p.parse_args()

    try:
        payload = build_gate(
            load_json(args.speed_json),
            [load_json(path) for path in args.summaries],
            args.minimum_seeds,
            args.quality_margin_pct,
            args.throughput_ratio,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["winner"] is None:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
