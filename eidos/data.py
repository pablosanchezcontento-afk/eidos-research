from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class PackedTokenFile:
    path: str
    dtype: str = "uint16"

    def open(self) -> np.memmap:
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(p)
        dt = np.uint16 if self.dtype == "uint16" else np.uint32
        return np.memmap(p, mode="r", dtype=dt)


class RandomPackedLoader:
    def __init__(
        self,
        path: str,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        seed: int = 1337,
        dtype: str = "uint16",
    ) -> None:
        self.data = PackedTokenFile(path, dtype).open()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.rng = np.random.default_rng(seed)
        if len(self.data) < seq_len + 2:
            raise ValueError(f"token file is too small for sequence length {seq_len}")

    def set_seq_len(self, seq_len: int) -> None:
        if len(self.data) < seq_len + 2:
            raise ValueError(f"token file is too small for sequence length {seq_len}")
        self.seq_len = seq_len

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        starts = self.rng.integers(0, len(self.data) - self.seq_len - 1, size=self.batch_size)
        x = np.stack([np.asarray(self.data[s:s + self.seq_len], dtype=np.int64) for s in starts])
        y = np.stack([np.asarray(self.data[s + 1:s + self.seq_len + 1], dtype=np.int64) for s in starts])
        return (
            torch.from_numpy(x).to(self.device, non_blocking=True),
            torch.from_numpy(y).to(self.device, non_blocking=True),
        )
