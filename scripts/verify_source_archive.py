"""Reassemble source_archives/*.b64.p* and check the SHA-256 of the preserved EIDOS v0.4 snapshot.

    python scripts/verify_source_archive.py [--extract DIR]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = ROOT / "source_archives"
EXPECTED_SHA256 = "784872f11473cd120aa2eda69db9eae687565d2220d788fcefece14441c97fc9"


def reassemble() -> bytes:
    parts = sorted(ARCHIVE_DIR.glob("eidos-v0.4-source-complete.tar.gz.b64.p*"))
    if not parts:
        raise FileNotFoundError("no archive chunks found")
    encoded = "".join(part.read_text(encoding="ascii") for part in parts)
    return base64.b64decode("".join(encoded.split()), validate=True)


def safe_extract(data: bytes, target: Path) -> list[str]:
    target = target.resolve()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            destination = (target / member.name).resolve()
            if not destination.is_relative_to(target) or member.issym() or member.islnk():
                raise ValueError(f"unsafe archive member: {member.name}")
        archive.extractall(target, filter="data")
        return [member.name for member in members if member.isfile()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--extract", type=Path)
    args = parser.parse_args(argv)
    data = reassemble()
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SHA256:
        print(f"MISMATCH: {digest}", file=sys.stderr)
        return 1
    print(f"OK {digest} ({len(data)} bytes)")
    if args.extract:
        files = safe_extract(data, args.extract)
        print(f"extracted {len(files)} files to {args.extract}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
