from __future__ import annotations

import argparse
from pathlib import Path


def iter_files(paths: list[str]):
    for raw in paths:
        p = Path(raw)
        files = sorted(p.rglob("*")) if p.is_dir() else [p]
        for f in files:
            if f.is_file() and f.suffix.lower() in {".txt", ".md", ".py", ".jsonl", ".json"}:
                yield str(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer from scratch")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--out", default="tokenizer.json")
    args = parser.parse_args()
    try:
        from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
    except ImportError as exc:
        raise SystemExit("pip install tokenizers") from exc

    special = ["<pad>", "<bos>", "<eos>", "<unk>", "<system>", "<user>", "<assistant>", "<tool>"]
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=special,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    files = list(iter_files(args.inputs))
    if not files:
        raise SystemExit("No text/code files found")
    tokenizer.train(files, trainer)
    eos = tokenizer.token_to_id("<eos>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A <eos>",
        special_tokens=[("<eos>", eos)],
    )
    tokenizer.save(args.out)
    print(f"saved {args.out} with vocab={tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
