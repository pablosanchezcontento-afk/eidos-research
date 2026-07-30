# Reproducibility and source status

## Source-complete snapshots

The preserved local release contains historical snapshots for NOVA v0.1, ISADORA v0.2, SAPPHO v0.3 and EIDOS v0.4, including source, tests, configurations and available artifacts.

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
