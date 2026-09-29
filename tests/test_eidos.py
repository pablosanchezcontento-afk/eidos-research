from __future__ import annotations

import math

import pytest
import torch

from eidos.baselines import ConvStripedLM, TransformerLM, match_ffn_hidden
from eidos.model import (
    EidosConfig,
    EidosLM,
    PulseMemory,
    chunked_fixed_decay_scan,
    sequential_fixed_decay_scan,
)


def tiny_cfg(**kwargs) -> EidosConfig:
    base = dict(
        vocab_size=128,
        dim=32,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=64,
        max_seq_len=64,
        conv_kernel=5,
        memory_rank=12,
        memory_bands=3,
        memory_chunk=16,
        gradient_checkpointing=False,
        mtp_weight=0.0,
        macrocycle="CMCA",
    )
    base.update(kwargs)
    return EidosConfig(**base)


@pytest.mark.parametrize("chunk", [1, 4, 16, 32, 64, 128])
def test_fixed_scan_matches_sequential(chunk: int) -> None:
    torch.manual_seed(1)
    decays = torch.tensor([0.85, 0.982, 0.9992])
    injection = torch.randn(2, 97, 3, 11) * 0.03
    actual = chunked_fixed_decay_scan(decays, injection, chunk)
    expected = sequential_fixed_decay_scan(decays, injection)
    assert torch.max(torch.abs(actual - expected)).item() < 2e-5


def test_fixed_scan_backward_is_finite() -> None:
    decays = torch.tensor([0.91, 0.982, 0.999], requires_grad=True)
    injection = torch.randn(2, 41, 3, 7, requires_grad=True)
    out = chunked_fixed_decay_scan(decays, injection, 16)
    out.square().mean().backward()
    assert torch.isfinite(decays.grad).all()
    assert torch.isfinite(injection.grad).all()


def test_scan_rejects_unsafe_chunk() -> None:
    with pytest.raises(ValueError):
        chunked_fixed_decay_scan(torch.tensor([0.9, 0.99, 0.999]), torch.zeros(1, 4, 3, 2), 256)


def test_model_is_causal() -> None:
    torch.manual_seed(2)
    model = EidosLM(tiny_cfg()).eval()
    x = torch.randint(0, 128, (1, 24))
    y = x.clone()
    y[:, 13:] = torch.randint(0, 128, y[:, 13:].shape)
    with torch.no_grad():
        a = model(x)["logits"]
        b = model(y)["logits"]
    assert torch.max(torch.abs(a[:, :13] - b[:, :13])).item() < 1e-5


def test_cache_matches_full_forward() -> None:
    torch.manual_seed(3)
    model = EidosLM(tiny_cfg()).eval()
    ids = torch.randint(0, 128, (2, 18))
    with torch.no_grad():
        full = model(ids)["logits"]
        caches = None
        pieces = []
        for pos in range(ids.shape[1]):
            logits, caches = model.forward_step(ids[:, pos : pos + 1], caches, pos)
            pieces.append(logits)
        cached = torch.cat(pieces, dim=1)
    assert torch.max(torch.abs(full - cached)).item() < 3e-5


def test_all_parameters_receive_gradients() -> None:
    torch.manual_seed(4)
    model = EidosLM(tiny_cfg()).train()
    x = torch.randint(0, 128, (2, 20))
    y = torch.randint(0, 128, (2, 20))
    model(x, y)["loss"].backward()
    missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_memory_gate_starts_closed() -> None:
    model = EidosLM(tiny_cfg())
    memories = [m for m in model.modules() if isinstance(m, PulseMemory)]
    assert memories
    for memory in memories:
        assert torch.count_nonzero(memory.out_gate.weight).item() == 0
        assert torch.allclose(memory.out_gate.bias, torch.full_like(memory.out_gate.bias, -1.5))


def test_parameter_matching_within_half_percent() -> None:
    cfg = tiny_cfg(vocab_size=1024)
    target = EidosLM(cfg).parameter_count()
    matched, params = match_ffn_hidden(cfg, target, TransformerLM, max_hidden=1024)
    assert 100 * abs(params - target) / target <= 0.5
    assert matched.ffn_hidden % 8 == 0


def test_eidos_uses_fewer_attention_layers_than_transformer() -> None:
    model = EidosLM(tiny_cfg(n_layers=8))
    assert model.kinds.count("A") == 2
    assert model.kinds.count("M") == 2
    assert model.kinds.count("C") == 4


def test_uniform_bits_reference() -> None:
    assert math.log2(1024) == 10.0
    assert math.log2(256) == 8.0


def test_baselines_forward() -> None:
    cfg = tiny_cfg()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    for model in [TransformerLM(cfg), ConvStripedLM(cfg)]:
        result = model(x, y)
        assert result["logits"].shape == (2, 16, cfg.vocab_size)
        assert torch.isfinite(result["loss"])
