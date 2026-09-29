from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def iter_documents(paths: list[str]):
    for raw in paths:
        root = Path(raw)
        files = sorted(root.rglob("*")) if root.is_dir() else [root]
        for path in files:
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix == ".jsonl":
                for line in path.open("r", encoding="utf-8", errors="ignore"):
                    try:
                        row = json.loads(line)
                        text = row.get("text") or row.get("content") or row.get("code")
                    except Exception:
                        text = line
                    if text:
                        yield str(text)
            elif suffix in {".txt", ".md", ".py", ".json", ".js", ".ts", ".java", ".php", ".html", ".css", ".sql"}:
                text = path.read_text(encoding="utf-8", errors="ignore")
                if text.strip():
                    yield text


def main() -> None:
    p = argparse.ArgumentParser(description="Deduplicate and pack sealed train/val/test token streams")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--train-out", default="train.bin")
    p.add_argument("--val-out", default="val.bin")
    p.add_argument("--test-out", default="test.bin")
    p.add_argument("--val-per-mille", type=int, default=10)
    p.add_argument("--test-per-mille", type=int, default=10)
    p.add_argument("--manifest-out", default="data_manifest.json")
    args = p.parse_args()
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit("pip install tokenizers") from exc

    if args.val_per_mille + args.test_per_mille >= 1000:
        raise SystemExit("validation + test fractions must be below 1000 per mille")
    tok = Tokenizer.from_file(args.tokenizer)
    dtype = np.uint16 if tok.get_vocab_size() <= 65535 else np.uint32
    counts = {f"{split}_{kind}": 0 for split in ("train", "val", "test") for kind in ("docs", "tokens")}
    duplicates = 0
    seen: set[bytes] = set()
    outputs = {
        "train": open(args.train_out, "wb"),
        "val": open(args.val_out, "wb"),
        "test": open(args.test_out, "wb"),
    }
    try:
        for text in iter_documents(args.inputs):
            normalized = " ".join(text.split())
            fingerprint = hashlib.blake2b(normalized.encode("utf-8", "ignore"), digest_size=16).digest()
            if fingerprint in seen:
                duplicates += 1
                continue
            seen.add(fingerprint)
            ids = tok.encode(text).ids
            if len(ids) < 4:
                continue
            bucket = int.from_bytes(fingerprint[:8], "little") % 1000
            if bucket < args.test_per_mille:
                split = "test"
            elif bucket < args.test_per_mille + args.val_per_mille:
                split = "val"
            else:
                split = "train"
            np.asarray(ids, dtype=dtype).tofile(outputs[split])
            counts[f"{split}_docs"] += 1
            counts[f"{split}_tokens"] += len(ids)
    finally:
        for handle in outputs.values():
            handle.close()

    manifest = {
        "tokenizer": args.tokenizer,
        "vocab_size": tok.get_vocab_size(),
        "dtype": str(dtype),
        "duplicates_removed": duplicates,
        "split_method": "blake2b document hash before tokenization",
        **counts,
    }
    Path(args.manifest_out).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
