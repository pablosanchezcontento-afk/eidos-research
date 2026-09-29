from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class EidosConfig:
    """Configuration for EIDOS.

    EIDOS = Efficient Innovation-Driven Orthogonal State.

    The model is a hybrid rather than an attention sidecar. Most layers are
    linear-time causal mixers, while periodic attention layers act as exact
    retrieval anchors. The predictive memory is low-rank and uses fixed,
    learnable timescales so the full-sequence path can be expressed with two
    cumulative operations per chunk instead of a logarithmic scan tree.
    """

    vocab_size: int = 32768
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2
    ffn_hidden: int = 640
    max_seq_len: int = 2048
    conv_kernel: int = 7
    memory_rank: int = 64
    memory_bands: int = 3
    memory_chunk: int = 64
    rope_base: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    gradient_checkpointing: bool = False
    mtp_weight: float = 0.0
    macrocycle: str = "CMCA"  # C=causal conv, M=predictive memory, A=global attention

    def validate(self) -> None:
        if self.dim <= 0 or self.n_layers <= 0:
            raise ValueError("dim and n_layers must be positive")
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.dim // self.n_heads) % 2:
            raise ValueError("head dimension must be even for RoPE")
        if not self.macrocycle or any(x not in "CMA" for x in self.macrocycle):
            raise ValueError("macrocycle must contain only C, M and A")
        if "A" not in self.macrocycle:
            raise ValueError("macrocycle must contain at least one attention anchor A")
        if self.memory_bands != 3:
            raise ValueError("the reference implementation currently supports three memory bands")
        if not 8 <= self.memory_rank <= self.dim:
            raise ValueError("memory_rank must be between 8 and dim")
        if self.memory_chunk < 1:
            raise ValueError("memory_chunk must be positive")
        if self.conv_kernel < 2:
            raise ValueError("conv_kernel must be at least 2")
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return x * inv.to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Linear(dim, hidden * 2, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.up(x).chunk(2, dim=-1)
        return self.dropout(self.down(F.silu(gate) * value))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rope_angles(length: int, dim: int, base: float, device: torch.device) -> torch.Tensor:
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    pos = torch.arange(length, device=device, dtype=torch.float32)
    freq = torch.outer(pos, inv)
    return torch.cat((freq, freq), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    angles = _rope_angles(q.shape[-2], q.shape[-1], base, q.device).to(q.dtype)[None, None]
    c, s = angles.cos(), angles.sin()
    return q * c + rotate_half(q) * s, k * c + rotate_half(k) * s


def apply_rope_at(q: torch.Tensor, k: torch.Tensor, position: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    d = q.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, d, 2, device=q.device, dtype=torch.float32) / d))
    freq = float(position) * inv
    angles = torch.cat((freq, freq), dim=-1).to(q.dtype)[None, None, None]
    c, s = angles.cos(), angles.sin()
    return q * c + rotate_half(q) * s, k * c + rotate_half(k) * s


class GQAttention(nn.Module):
    """Flash-SDPA compatible grouped-query causal attention."""

    def __init__(self, cfg: EidosConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.rope_base = cfg.rope_base
        self.q = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.k = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.dropout = cfg.dropout

    @staticmethod
    def _qk_norm(x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q = self.q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self._qk_norm(q), self._qk_norm(k)
        q, k = apply_rope(q, k, self.rope_base)
        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        return self.o(y.transpose(1, 2).contiguous().view(b, t, d))

    def forward_step(
        self,
        x: torch.Tensor,
        cache: dict[str, torch.Tensor] | None,
        position: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, t, d = x.shape
        if t != 1:
            raise ValueError("forward_step expects one token")
        q = self.q(x).view(b, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, 1, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, 1, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self._qk_norm(q), self._qk_norm(k)
        q, k = apply_rope_at(q, k, position, self.rope_base)
        if cache is not None:
            k = torch.cat((cache["k"], k), dim=2)
            v = torch.cat((cache["v"], v), dim=2)
        new_cache = {"k": k, "v": v}
        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k_attn = k.repeat_interleave(repeat, dim=1)
            v_attn = v.repeat_interleave(repeat, dim=1)
        else:
            k_attn, v_attn = k, v
        y = F.scaled_dot_product_attention(q, k_attn, v_attn, dropout_p=0.0)
        return self.o(y.transpose(1, 2).contiguous().view(b, 1, d)), new_cache


class CausalConvMixer(nn.Module):
    """Gated depthwise-separable causal convolution.

    It replaces several attention layers with a linear-time local mixer. The
    dense input/output projections preserve channel interaction; only the
    temporal operation is depthwise.
    """

    def __init__(self, cfg: EidosConfig):
        super().__init__()
        d = cfg.dim
        self.kernel = cfg.conv_kernel
        self.in_proj = nn.Linear(d, 2 * d, bias=False)
        self.depthwise = nn.Conv1d(d, d, self.kernel, groups=d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = self.depthwise(F.pad(u.transpose(1, 2), (self.kernel - 1, 0))).transpose(1, 2)
        return self.dropout(self.out_proj(F.silu(u) * torch.sigmoid(gate)))

    def forward_step(
        self,
        x: torch.Tensor,
        cache: dict[str, torch.Tensor] | None,
        position: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del position
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        b, _, d = u.shape
        if cache is None:
            history = torch.zeros((b, d, self.kernel - 1), device=x.device, dtype=x.dtype)
        else:
            history = cache["history"]
        current = u.transpose(1, 2)
        conv = F.conv1d(torch.cat((history, current), dim=2), self.depthwise.weight, groups=d)
        out = self.out_proj(F.silu(conv.transpose(1, 2)) * torch.sigmoid(gate))
        next_history = torch.cat((history[:, :, 1:], current), dim=2) if self.kernel > 2 else current
        return out, {"history": next_history}


def sequential_fixed_decay_scan(decays: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
    """Float32 oracle for h_t = a*h_(t-1)+b_t with one decay per band."""

    if decays.ndim != 1 or injection.ndim != 4:
        raise ValueError("decays must be [K] and injection [B,T,K,R]")
    if injection.shape[2] != decays.numel():
        raise ValueError("band count mismatch")
    b, t, k, r = injection.shape
    state = torch.zeros((b, k, r), device=injection.device, dtype=torch.float32)
    outputs: list[torch.Tensor] = []
    a = decays.float()[None, :, None]
    for i in range(t):
        state = a * state + injection[:, i].float()
        outputs.append(state)
    return torch.stack(outputs, dim=1)


def chunked_fixed_decay_scan(
    decays: torch.Tensor,
    injection: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Parallel fixed-decay scan using chunk-local renormalized cumsums.

    For a fixed decay a and chunk start state h0:
      h_i = a^(i+1) * (h0 + sum_{j<=i} b_j / a^(j+1)).

    Unlike the old v0.1 scan, this routine does not clamp a cumulative product.
    The allowed decay ranges and chunk-size validation keep the inverse powers
    finite in float32. Chunk boundaries carry the exact last state.
    """

    if decays.ndim != 1 or injection.ndim != 4:
        raise ValueError("decays must be [K] and injection [B,T,K,R]")
    if injection.shape[2] != decays.numel():
        raise ValueError("band count mismatch")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    # EIDOS constrains learnable decays to >=0.85. At chunk_size<=128 the
    # smallest renormalization power remains above 1e-9 in float32. Keep the
    # guard static so torch.compile does not introduce a GPU-to-CPU graph break.
    if chunk_size > 128:
        raise ValueError("chunk_size > 128 is not supported by the fast scan")

    b, t, k, r = injection.shape
    state = torch.zeros((b, k, r), device=injection.device, dtype=torch.float32)
    a = decays.float()[None, None, :, None]
    pieces: list[torch.Tensor] = []
    for start in range(0, t, chunk_size):
        z = injection[:, start : start + chunk_size].float()
        length = z.shape[1]
        exponents = torch.arange(1, length + 1, device=z.device, dtype=torch.float32)
        powers = a.pow(exponents[None, :, None, None])
        local = powers * (state[:, None] + torch.cumsum(z / powers, dim=1))
        pieces.append(local)
        state = local[:, -1]
    return torch.cat(pieces, dim=1)


class PulseMemory(nn.Module):
    """Low-rank predictive memory with adaptive write and fixed timescales.

    The expensive full-width, token-dependent decay tensor from SAPPHO is
    removed. Content lives in a low-rank state; write/read routing is per token,
    and three learnable scalar timescales are shared across rank channels. This
    deliberately trades some flexibility for a much cheaper, compiler-friendly
    training path while preserving constant-state autoregressive inference.
    """

    def __init__(self, cfg: EidosConfig):
        super().__init__()
        d, r, k = cfg.dim, cfg.memory_rank, cfg.memory_bands
        self.rank = r
        self.bands = k
        self.chunk = cfg.memory_chunk
        self.pre = nn.Conv1d(d, d, 3, groups=d, bias=False)
        self.in_proj = nn.Linear(d, 3 * r, bias=False)
        self.write_router = nn.Linear(r, k, bias=True)
        self.read_router = nn.Linear(r, k, bias=True)
        self.budget = nn.Linear(r, 1, bias=True)
        self.surprise_gain = nn.Parameter(torch.tensor(0.25))
        self.decay_logits = nn.Parameter(torch.zeros(k))
        self.out_gate = nn.Linear(d, d, bias=True)
        self.out_proj = nn.Linear(r, d, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer("decay_lo", torch.tensor([0.85, 0.965, 0.995]))
        self.register_buffer("decay_hi", torch.tensor([0.97, 0.995, 0.9997]))
        self.reset_special_initialization()

    def reset_special_initialization(self) -> None:
        with torch.no_grad():
            self.write_router.bias.zero_()
            self.read_router.bias.zero_()
            self.budget.bias.fill_(-1.0)
            self.out_gate.weight.zero_()
            self.out_gate.bias.fill_(-1.5)

    def decays(self) -> torch.Tensor:
        return self.decay_lo + (self.decay_hi - self.decay_lo) * torch.sigmoid(self.decay_logits)

    def _signals(self, z: torch.Tensor, previous_prediction: torch.Tensor):
        value, prediction, control = self.in_proj(z).chunk(3, dim=-1)
        innovation_raw = value - previous_prediction
        rms = innovation_raw.float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        innovation = innovation_raw / (1.0 + rms.to(innovation_raw.dtype))
        surprise = torch.log1p(rms).to(control.dtype)
        budget = torch.sigmoid(self.budget(control) + self.surprise_gain * surprise)
        write = self.write_router(control).softmax(dim=-1) * budget
        read = self.read_router(control + innovation).softmax(dim=-1)
        return innovation, prediction, write, read, surprise

    def forward(self, x: torch.Tensor, return_diagnostics: bool = False):
        z = self.pre(F.pad(x.transpose(1, 2), (2, 0))).transpose(1, 2)
        projected = self.in_proj(z)
        value, prediction, control = projected.chunk(3, dim=-1)
        previous_prediction = torch.cat((torch.zeros_like(prediction[:, :1]), prediction[:, :-1]), dim=1)
        innovation_raw = value - previous_prediction
        rms = innovation_raw.float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        innovation = innovation_raw / (1.0 + rms.to(innovation_raw.dtype))
        surprise = torch.log1p(rms).to(control.dtype)
        budget = torch.sigmoid(self.budget(control) + self.surprise_gain * surprise)
        write = self.write_router(control).softmax(dim=-1) * budget
        read = self.read_router(control + innovation).softmax(dim=-1)
        decay = self.decays()
        injection = (1.0 - decay[None, None, :, None]) * write[:, :, :, None] * innovation[:, :, None, :]
        state = chunked_fixed_decay_scan(decay, injection, self.chunk)
        mixed = (read[:, :, :, None].float() * state).sum(dim=2).to(x.dtype)
        output = self.dropout(self.out_proj(mixed) * torch.sigmoid(self.out_gate(z)))
        if not return_diagnostics:
            return output
        diagnostics = {
            "mean_budget": budget.float().mean().detach(),
            "mean_surprise": surprise.float().mean().detach(),
            "decays": decay.float().detach(),
            "write_entropy": (-(write.clamp_min(1e-8).log() * write).sum(-1).mean()).detach(),
            "read_entropy": (-(read.clamp_min(1e-8).log() * read).sum(-1).mean()).detach(),
        }
        return output, diagnostics

    def forward_step(
        self,
        x: torch.Tensor,
        cache: dict[str, torch.Tensor] | None,
        position: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del position
        b, t, d = x.shape
        if t != 1:
            raise ValueError("forward_step expects one token")
        if cache is None:
            history = torch.zeros((b, d, 2), device=x.device, dtype=x.dtype)
            previous_prediction = torch.zeros((b, 1, self.rank), device=x.device, dtype=x.dtype)
            state = torch.zeros((b, self.bands, self.rank), device=x.device, dtype=torch.float32)
        else:
            history = cache["history"]
            previous_prediction = cache["previous_prediction"]
            state = cache["state"]
        current = x.transpose(1, 2)
        z = F.conv1d(torch.cat((history, current), dim=2), self.pre.weight, groups=d).transpose(1, 2)
        innovation, prediction, write, read, _ = self._signals(z, previous_prediction)
        decay = self.decays()
        injection = (1.0 - decay[None, :, None]) * write[:, 0, :, None] * innovation[:, 0, None, :]
        state = decay[None, :, None].float() * state + injection.float()
        mixed = (read[:, 0, :, None].float() * state).sum(dim=1).to(x.dtype).unsqueeze(1)
        output = self.out_proj(mixed) * torch.sigmoid(self.out_gate(z))
        return output, {
            "history": torch.cat((history[:, :, 1:], current), dim=2),
            "previous_prediction": prediction,
            "state": state,
        }


class EidosBlock(nn.Module):
    def __init__(self, cfg: EidosConfig, kind: str, residual_scale: float):
        super().__init__()
        self.kind = kind
        self.norm1 = RMSNorm(cfg.dim)
        if kind == "C":
            self.mixer = CausalConvMixer(cfg)
        elif kind == "M":
            self.mixer = PulseMemory(cfg)
        elif kind == "A":
            self.mixer = GQAttention(cfg)
        else:
            raise ValueError(kind)
        self.norm2 = RMSNorm(cfg.dim)
        self.ffn = SwiGLU(cfg.dim, cfg.ffn_hidden, cfg.dropout)
        self.scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.scale * self.mixer(self.norm1(x))
        x = x + self.scale * self.ffn(self.norm2(x))
        return x

    def forward_step(
        self,
        x: torch.Tensor,
        cache: dict[str, torch.Tensor] | None,
        position: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mixed, new_cache = self.mixer.forward_step(self.norm1(x), cache, position)
        x = x + self.scale * mixed
        x = x + self.scale * self.ffn(self.norm2(x))
        return x, new_cache


class EidosLM(nn.Module):
    def __init__(self, cfg: EidosConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        scale = 1.0 / math.sqrt(2.0 * cfg.n_layers)
        kinds = [cfg.macrocycle[i % len(cfg.macrocycle)] for i in range(cfg.n_layers)]
        # Exact retrieval must be available at the output even for partial cycles.
        kinds[-1] = "A"
        self.kinds = kinds
        self.blocks = nn.ModuleList([EidosBlock(cfg, kind, scale) for kind in kinds])
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        if cfg.mtp_weight > 0:
            self.mtp_norm: RMSNorm | None = RMSNorm(cfg.dim)
            self.mtp_proj: nn.Linear | None = nn.Linear(cfg.dim, cfg.dim, bias=False)
        else:
            self.mtp_norm = None
            self.mtp_proj = None
        self.apply(self._init_weights)
        # Generic initialization above would reopen the memory gate. Restore
        # the intentionally conservative start after all Linear layers exist.
        for module in self.modules():
            if isinstance(module, PulseMemory):
                module.reset_special_initialization()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _run_block(self, block: EidosBlock, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.gradient_checkpointing and self.training:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = self._run_block(block, x)
        hidden = self.norm(x)
        logits = self.lm_head(hidden)
        result: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            main = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
            loss = main
            result["main_loss"] = main.detach()
            if self.cfg.mtp_weight > 0 and labels.shape[1] > 1:
                assert self.mtp_norm is not None and self.mtp_proj is not None
                mtp_hidden = self.mtp_norm(self.mtp_proj(hidden[:, :-1]))
                mtp_logits = self.lm_head(mtp_hidden)
                mtp = F.cross_entropy(
                    mtp_logits.flatten(0, 1), labels[:, 1:].flatten(), ignore_index=-100
                )
                loss = loss + self.cfg.mtp_weight * mtp
                result["mtp_loss"] = mtp.detach()
            result["loss"] = loss
        return result

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward_step(
        self,
        input_ids: torch.Tensor,
        caches: list[dict[str, torch.Tensor] | None] | None = None,
        position: int = 0,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("forward_step expects [batch,1] token ids")
        if caches is None:
            caches = [None] * len(self.blocks)
        if len(caches) != len(self.blocks):
            raise ValueError(f"expected {len(self.blocks)} caches, got {len(caches)}")
        x = self.token_embedding(input_ids)
        next_caches: list[dict[str, torch.Tensor]] = []
        for block, cache in zip(self.blocks, caches, strict=True):
            x, cache = block.forward_step(x, cache, position)
            next_caches.append(cache)
        logits = self.lm_head(self.norm(x))
        return logits, next_caches


def config_with_macrocycle(cfg: EidosConfig, macrocycle: str) -> EidosConfig:
    return replace(cfg, macrocycle=macrocycle)
