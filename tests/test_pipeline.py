"""Training pipeline, data loading, promotion gate and evidence integrity."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from eidos.data import PackedTokenFile, RandomPackedLoader
from eidos.model import EidosLM

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "eidos_smoke.json"


def write_tokens(path: Path, tokens: np.ndarray, dtype=np.uint16) -> Path:
    memmap = np.memmap(path, mode="w+", dtype=dtype, shape=tokens.shape)
    memmap[:] = tokens
    memmap.flush()
    return path


# --- data -----------------------------------------------------------------------------------------------------------

def test_loader_returns_shifted_targets(tmp_path: Path) -> None:
    path = write_tokens(tmp_path / "t.bin", np.arange(1000) % 128)
    loader = RandomPackedLoader(str(path), batch_size=4, seq_len=16, device=torch.device("cpu"), seed=0)
    x, y = loader.next()
    assert x.shape == y.shape == (4, 16)
    assert x.dtype == torch.int64
    assert torch.equal((x + 1) % 128, y)


def test_loader_is_deterministic_per_seed(tmp_path: Path) -> None:
    path = write_tokens(tmp_path / "t.bin", np.random.default_rng(0).integers(0, 128, 5000))
    a = RandomPackedLoader(str(path), 2, 8, torch.device("cpu"), seed=7).next()
    b = RandomPackedLoader(str(path), 2, 8, torch.device("cpu"), seed=7).next()
    assert all(torch.equal(i, j) for i, j in zip(a, b, strict=True))


def test_loader_rejects_short_files_and_missing_paths(tmp_path: Path) -> None:
    path = write_tokens(tmp_path / "short.bin", np.arange(10))
    with pytest.raises(ValueError):
        RandomPackedLoader(str(path), 1, 16, torch.device("cpu"))
    loader = RandomPackedLoader(str(path), 1, 4, torch.device("cpu"))
    with pytest.raises(ValueError):
        loader.set_seq_len(64)
    with pytest.raises(FileNotFoundError):
        PackedTokenFile(str(tmp_path / "missing.bin")).open()


def test_uint32_tokens(tmp_path: Path) -> None:
    path = write_tokens(tmp_path / "wide.bin", np.arange(100, dtype=np.uint32) + 70000, dtype=np.uint32)
    x, _ = RandomPackedLoader(str(path), 1, 8, torch.device("cpu"), dtype="uint32").next()
    assert int(x.min()) >= 70000


# --- training utilities ---------------------------------------------------------------------------------------------

def test_cosine_schedule_warms_up_and_decays_to_minimum() -> None:
    from train import cosine_lr

    assert cosine_lr(0, 100, 10, 1e-3, 1e-4) == pytest.approx(1e-4)
    assert cosine_lr(9, 100, 10, 1e-3, 1e-4) == pytest.approx(1e-3)
    assert cosine_lr(10, 100, 10, 1e-3, 1e-4) == pytest.approx(1e-3)
    assert cosine_lr(100, 100, 10, 1e-3, 1e-4) == pytest.approx(1e-4)
    values = [cosine_lr(step, 100, 10, 1e-3, 1e-4) for step in range(10, 101)]
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))


@pytest.mark.parametrize("name", [
    "eidos_core", "eidos_wide_memory", "eidos_dual_anchor", "eidos_pulse",
    "transformer_full", "conv_striped", "recurrent_striped",
])
def test_every_architecture_is_parameter_matched(name: str) -> None:
    from train import architecture_factory, load_config

    base = replace(load_config(str(ROOT / "configs" / "eidos_glyph_10m.json")), vocab_size=512, max_seq_len=128)
    model, _, params, target = architecture_factory(name, base)
    assert abs(params - target) / target <= 0.005
    logits = model(torch.randint(0, 512, (1, 16)))["logits"]
    assert logits.shape == (1, 16, 512)


def test_unknown_architecture_is_rejected() -> None:
    from train import architecture_factory, load_config

    with pytest.raises(ValueError):
        architecture_factory("eidos_imaginary", load_config(str(SMOKE_CONFIG)))


def test_configs_are_valid_and_build() -> None:
    from train import load_config

    for path in sorted((ROOT / "configs").glob("*.json")):
        cfg = load_config(str(path))
        assert cfg.dim % cfg.n_heads == 0, path.name
        if cfg.dim <= 64:
            assert EidosLM(cfg).parameter_count() > 0


def test_end_to_end_cpu_training_learns_a_pattern(tmp_path: Path) -> None:
    """Real training run through train.py on a learnable synthetic stream (CPU, fp32)."""
    pattern = np.tile(np.arange(32), 400) % 128
    train = write_tokens(tmp_path / "train.bin", pattern)
    val = write_tokens(tmp_path / "val.bin", pattern[:2000])
    test = write_tokens(tmp_path / "test.bin", pattern[:2000])
    out = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, str(ROOT / "train.py"), "--architecture", "eidos_core", "--config", str(SMOKE_CONFIG),
         "--train-bin", str(train), "--val-bin", str(val), "--test-bin", str(test), "--out", str(out),
         "--steps", "60", "--micro-batch", "8", "--grad-accum", "1", "--seq-len", "32", "--lr", "3e-3",
         "--warmup", "5", "--optimizer", "adamw", "--precision", "fp32", "--eval-every", "30",
         "--eval-batches", "4", "--allow-undertrained"],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["learned_better_than_uniform_gate"] is True
    assert summary["final_test_bits_per_token"] < math.log2(128) - 1
    assert summary["test_was_sealed_until_final_evaluation"] is True
    assert (out / "checkpoint_final.pt").exists()
    assert len((out / "metrics.jsonl").read_text().splitlines()) >= 6


def test_training_refuses_undertrained_runs_without_override(tmp_path: Path) -> None:
    data = write_tokens(tmp_path / "d.bin", np.arange(500) % 128)
    result = subprocess.run(
        [sys.executable, str(ROOT / "train.py"), "--architecture", "eidos_core", "--config", str(SMOKE_CONFIG),
         "--train-bin", str(data), "--val-bin", str(data), "--out", str(tmp_path / "o"), "--steps", "1",
         "--seq-len", "32", "--lr", "1e-3", "--precision", "fp32"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode != 0
    assert "tokens/parameter" in result.stderr


# --- promotion gate -------------------------------------------------------------------------------------------------

def _summary(arch: str, seed: int, bits: float, **overrides) -> dict:
    row = {"architecture": arch, "seed": seed, "final_test_bits_per_token": bits,
           "learned_better_than_uniform_gate": True, "parameter_gap_pct": 0.1, "tokens_per_parameter": 20.0}
    row.update(overrides)
    return row


def _speed(eidos_tps: float) -> dict:
    return {"results": [{"name": "transformer_full", "training_tokens_per_second": 1000.0},
                        {"name": "eidos_core", "training_tokens_per_second": eidos_tps}]}


def _runs(eidos_bits: float, **overrides) -> list[dict]:
    rows = []
    for seed in (1, 2, 3):
        rows += [_summary("transformer_full", seed, 4.00), _summary("conv_striped", seed, 4.10),
                 _summary("recurrent_striped", seed, 4.20), _summary("eidos_core", seed, eidos_bits, **overrides)]
    return rows


def test_gate_promotes_only_a_faster_and_better_candidate() -> None:
    from evaluate_gate import build_gate

    payload = build_gate(_speed(1100), _runs(3.95))
    assert payload["winner"] == "eidos_core"
    assert payload["mythos_unlocked"] is True
    assert payload["best_baseline"] == "transformer_full"


@pytest.mark.parametrize(("speed", "bits", "overrides", "failed"), [
    (1040, 3.95, {}, "faster_than_transformer_by_margin"),
    (1100, 3.99, {}, "beats_best_baseline_by_margin"),
    (1100, 3.95, {"tokens_per_parameter": 5.0}, "tokens_per_parameter_at_least_20"),
    (1100, 3.95, {"parameter_gap_pct": 0.9}, "parameters_matched"),
    (1100, 3.95, {"learned_better_than_uniform_gate": False}, "all_runs_learned"),
])
def test_gate_blocks_candidates_that_miss_any_check(speed, bits, overrides, failed) -> None:
    from evaluate_gate import build_gate

    payload = build_gate(_speed(speed), _runs(bits, **overrides))
    assert payload["winner"] is None
    assert payload["candidates"]["eidos_core"]["checks"][failed] is False


def test_gate_requires_sealed_tests_and_all_baselines() -> None:
    from evaluate_gate import build_gate

    runs = _runs(3.9)
    runs[3]["final_test_bits_per_token"] = None
    with pytest.raises(ValueError, match="sealed"):
        build_gate(_speed(1100), runs)
    with pytest.raises(ValueError, match="baselines"):
        build_gate(_speed(1100), [r for r in _runs(3.9) if r["architecture"] != "conv_striped"])
    with pytest.raises(ValueError, match="transformer_full"):
        build_gate({"results": []}, _runs(3.9))


# --- evidence integrity ---------------------------------------------------------------------------------------------

def test_source_archive_checksum_matches_the_published_value() -> None:
    import verify_source_archive

    assert verify_source_archive.main([]) == 0


def test_evidence_summary_matches_preserved_raw_benchmarks() -> None:
    evidence = json.loads((ROOT / "results/summaries/evidence_status.json").read_text())["directly_preserved"]
    for key, name in (("sappho_eager_ratio", "sappho_vs_transformer_10m_5070"),
                      ("sappho_compiled_ratio", "sappho_vs_transformer_10m_5070_compile")):
        rows = {r["name"]: r for r in json.loads((ROOT / f"results/raw/{name}.json").read_text())["results"]}
        tokens_per_second = {name: row["training_tokens_per_second"] for name, row in rows.items()}
        ratio = tokens_per_second["sappho_lattice"] / tokens_per_second["transformer_full"]
        assert round(ratio, 3) == evidence[key]
