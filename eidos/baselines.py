from __future__ import annotations

import math
from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .model import (
    CausalConvMixer,
    EidosConfig,
    GQAttention,
    RMSNorm,
    SwiGLU,
    chunked_fixed_decay_scan,
)


class GatedStateMixer(nn.Module):
    """Low-rank diagonal recurrent baseline without predictive innovation."""

    def __init__(self, cfg: EidosConfig):
        super().__init__()
        d, r, k = cfg.dim, cfg.memory_rank, cfg.memory_bands
        self.rank, self.bands, self.chunk = r, k, cfg.memory_chunk
        self.in_proj = nn.Linear(d, 2 * r + k, bias=False)
        self.read = nn.Linear(r, k, bias=False)
        self.out = nn.Linear(r, d, bias=False)
        self.gate = nn.Linear(d, d, bias=True)
        self.decay_logits = nn.Parameter(torch.zeros(k))
        self.register_buffer("decay_lo", torch.tensor([0.85, 0.965, 0.995]))
        self.register_buffer("decay_hi", torch.tensor([0.97, 0.995, 0.9997]))
        with torch.no_grad():
            self.gate.weight.zero_()
            self.gate.bias.fill_(-1.5)

    def decays(self) -> torch.Tensor:
        return self.decay_lo + (self.decay_hi - self.decay_lo) * torch.sigmoid(self.decay_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, control, write_logits = torch.split(
            self.in_proj(x), [self.rank, self.rank, self.bands], dim=-1
        )
        write = write_logits.softmax(-1)
        decay = self.decays()
        injection = (1.0 - decay[None, None, :, None]) * write[:, :, :, None] * torch.tanh(value)[:, :, None]
        state = chunked_fixed_decay_scan(decay, injection, self.chunk)
        read = self.read(control).softmax(-1)
        mixed = (read[:, :, :, None].float() * state).sum(2).to(x.dtype)
        return self.out(mixed) * torch.sigmoid(self.gate(x))

    def forward_step(self, x, cache, position):
        del position
        value, control, write_logits = torch.split(
            self.in_proj(x), [self.rank, self.rank, self.bands], dim=-1
        )
        write = write_logits.softmax(-1)
        decay = self.decays()
        if cache is None:
            state = torch.zeros(
                (x.shape[0], self.bands, self.rank), device=x.device, dtype=torch.float32
            )
        else:
            state = cache["state"]
        injection = (1.0 - decay[None, :, None]) * write[:, 0, :, None] * torch.tanh(value[:, 0])[:, None]
        state = decay[None, :, None].float() * state + injection.float()
        read = self.read(control).softmax(-1)
        mixed = (read[:, 0, :, None].float() * state).sum(1).to(x.dtype).unsqueeze(1)
        return self.out(mixed) * torch.sigmoid(self.gate(x)), {"state": state}


class BaselineBlock(nn.Module):
    def __init__(self, cfg: EidosConfig, kind: str, residual_scale: float):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        if kind == "A":
            self.mixer = GQAttention(cfg)
        elif kind == "C":
            self.mixer = CausalConvMixer(cfg)
        elif kind == "R":
            self.mixer = GatedStateMixer(cfg)
        else:
            raise ValueError(kind)
        self.norm2 = RMSNorm(cfg.dim)
        self.ffn = SwiGLU(cfg.dim, cfg.ffn_hidden, cfg.dropout)
        self.scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.scale * self.mixer(self.norm1(x))
        x = x + self.scale * self.ffn(self.norm2(x))
        return x


class BaselineLM(nn.Module):
    def __init__(self, cfg: EidosConfig, pattern: str, name: str):
        super().__init__()
        cfg.validate()
        if not pattern or any(ch not in "ACR" for ch in pattern):
            raise ValueError("baseline pattern must contain A, C or R")
        self.cfg = cfg
        self.pattern = pattern
        self.name = name
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        scale = 1.0 / math.sqrt(2.0 * cfg.n_layers)
        kinds = [pattern[i % len(pattern)] for i in range(cfg.n_layers)]
        self.blocks = nn.ModuleList([BaselineBlock(cfg, kind, scale) for kind in kinds])
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            if self.cfg.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        logits = self.lm_head(self.norm(x))
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits.flatten(0, 1), labels.flatten(), ignore_index=-100
            )
        return result

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TransformerLM(BaselineLM):
    def __init__(self, cfg: EidosConfig):
        super().__init__(cfg, "A", "transformer_full")


class ConvStripedLM(BaselineLM):
    def __init__(self, cfg: EidosConfig):
        super().__init__(cfg, "CCCA", "conv_striped")


class RecurrentStripedLM(BaselineLM):
    def __init__(self, cfg: EidosConfig):
        super().__init__(cfg, "RCRA", "recurrent_striped")


def match_ffn_hidden(
    cfg: EidosConfig,
    target_params: int,
    model_factory,
    max_hidden: int = 32768,
) -> tuple[EidosConfig, int]:
    """Match parameters using the exact linear dependence on FFN width."""

    probe8 = replace(cfg, ffn_hidden=8)
    probe16 = replace(cfg, ffn_hidden=16)
    p8 = model_factory(probe8).parameter_count()
    p16 = model_factory(probe16).parameter_count()
    slope_per_unit = (p16 - p8) / 8.0
    estimate = int(round(8 + (target_params - p8) / slope_per_unit))
    estimate = max(8, min(max_hidden, estimate))
    center = max(8, int(round(estimate / 8)) * 8)
    candidates = sorted({max(8, min(max_hidden, center + delta)) for delta in (-16, -8, 0, 8, 16)})
    best = None
    for hidden in candidates:
        candidate = replace(cfg, ffn_hidden=hidden)
        params = model_factory(candidate).parameter_count()
        error = abs(params - target_params)
        if best is None or error < best[0]:
            best = (error, candidate, params)
    if best is None:
        raise RuntimeError("could not parameter-match baseline")
    return best[1], best[2]
