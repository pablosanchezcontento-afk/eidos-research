# Reproducibility and source status

## Source-complete snapshots

The preserved local release contains historical snapshots for NOVA v0.1, ISADORA v0.2, SAPPHO v0.3 and EIDOS v0.4, including source, tests, configurations and available artifacts.

## EIDOS v0.4 in this repository

The EIDOS v0.4 source (`eidos/`, `train.py`, `benchmarks/`, `experiments/`, `tests/`, `configs/`, `reports/`) is
checked in as plain files, extracted from the preserved archive in `source_archives/`
(SHA-256 `784872f1…7fc9`, verified by `python scripts/verify_source_archive.py` and by the test suite).
Changes since the archive are limited to maintenance and are visible in Git history: a guard for `--steps 0`
in `train.py`, an explicit cache-count check in `EidosLM.forward_step`, the promotion gate refactored into a
testable `build_gate()` function, lint fixes and new tests. The archive itself is unchanged.

Reproduce on any machine (CPU is enough for the tests):

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest                     # scan equivalence, model shapes, parameter matching, a real CPU training run, gate logic
ruff check .
```

GPU benchmarks: `scripts/benchmark_eidos_5070.ps1` (Windows, RTX 5070) or `python benchmarks/benchmark_gpu.py --help`.

## Current source gap

The strongest later results refer to a post-v0.4 branch containing features described in the execution record:

- Sparse16 macrocycle;
- learned token shift;
- fused Pulse controls;
- FlexAttention causal backend;
- symmetric gradient checkpointing;
- isolated-process memory measurement;
- 1.093B configuration;
- extended quality runs.

The exact working tree, raw JSON outputs and selected checkpoints for this branch were not present in the supplied archive. Reconstructing code from prose would create a new implementation rather than preserve the measured implementation. This repository therefore labels those results as **recorded but not source-reproducible**.

## Required recovery before a formal release

1. Recover the exact post-v0.4 Git working tree or patch set.
2. Recover all raw GPU benchmark JSON files.
3. Recover quality `summary.json` files and checkpoint-selection metadata.
4. Re-run tests from a clean environment.
5. Run both architectures in isolated fresh processes.
6. Freeze the evaluation set and complete the 2.0 tpp comparison.
7. Publish environment information, package lock file and GPU driver details.
8. Obtain independent reproduction on Linux CUDA.

## Preserved benchmark limitations

The SAPPHO JSON files use random batches and are suitable for throughput and memory analysis only. Their loss values must not be interpreted as language-model quality.
