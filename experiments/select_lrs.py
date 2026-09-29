from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Select interior learning rates from calibration summaries")
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--lr-grid", nargs="+", type=float, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    grouped = defaultdict(list)
    for summary_path in Path(args.runs_dir).rglob("summary.json"):
        row = json.loads(summary_path.read_text(encoding="utf-8"))
        grouped[row["architecture"]].append({**row, "path": str(summary_path)})
    if not grouped:
        raise SystemExit("no summary.json files found")

    grid = sorted(args.lr_grid)
    selected = {}
    unresolved = {}
    for architecture, rows in grouped.items():
        valid = [r for r in rows if r["learned_better_than_uniform_gate"]]
        if not valid:
            unresolved[architecture] = {"reason": "all runs failed the uniform baseline"}
            continue
        best = min(valid, key=lambda r: r["final_val_bits_per_token"])
        lr = float(best["lr"])
        if lr == grid[-1]:
            unresolved[architecture] = {
                "reason": "best LR is at upper boundary",
                "suggested_grid": [grid[-1], grid[-1] * 2, grid[-1] * 4],
            }
        elif lr == grid[0]:
            unresolved[architecture] = {
                "reason": "best LR is at lower boundary",
                "suggested_grid": [grid[0] / 4, grid[0] / 2, grid[0]],
            }
        else:
            selected[architecture] = {
                "lr": lr,
                "val_bits": best["final_val_bits_per_token"],
                "summary": best["path"],
            }

    payload = {
        "selected": selected,
        "unresolved": unresolved,
        "all_architectures_resolved": bool(selected) and not unresolved,
        "rule": "No architecture advances while its best learning rate touches a grid boundary.",
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if unresolved:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
