from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream an auditable open corpus to local JSONL")
    parser.add_argument("--preset", choices=["web", "spanish_web", "code", "math_sft", "code_sft"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-docs", type=int, default=100000)
    parser.add_argument("--max-gb", type=float, default=8.0)
    args = parser.parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("pip install datasets") from exc

    if args.preset == "web":
        repo, name, split = "HuggingFaceFW/fineweb-edu", "sample-10BT", "train"
    elif args.preset == "spanish_web":
        repo, name, split = "HuggingFaceFW/fineweb-2", "spa_Latn", "train"
    elif args.preset == "code":
        repo, name, split = "common-pile/stackv2_edu_filtered", None, "train"
    elif args.preset == "math_sft":
        repo, name, split = "nvidia/OpenMathInstruct-1", None, "train"
    else:
        repo, name, split = "nvidia/OpenCodeInstruct", None, "train"

    ds = load_dataset(repo, name=name, split=split, streaming=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    max_bytes = int(args.max_gb * 1024**3)
    written = 0
    docs = 0
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            if args.preset in {"web", "spanish_web", "code"}:
                text = row.get("text") or row.get("content") or row.get("code")
            elif args.preset == "math_sft":
                text = f"<user>\n{row.get('question','')}\n<assistant>\n{row.get('generated_solution','')}"
            else:
                question = row.get("input") or row.get("instruction") or row.get("problem") or ""
                answer = row.get("output") or row.get("response") or row.get("solution") or ""
                text = f"<user>\n{question}\n<assistant>\n{answer}"
            if not text or len(text) < 80:
                continue
            payload = json.dumps({"text": text}, ensure_ascii=False) + "\n"
            f.write(payload)
            written += len(payload.encode("utf-8"))
            docs += 1
            if docs >= args.max_docs or written >= max_bytes:
                break
    print(json.dumps({"preset": args.preset, "repo": repo, "docs": docs, "bytes": written, "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
