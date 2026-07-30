# Roadmap

## Phase 0 - provenance recovery

- recover the exact Sparse16 source branch and raw result files;
- tag immutable historical releases;
- generate machine-readable manifests and environment locks.

## Phase 1 - close the quality question

- evaluate selected 2.0 tpp checkpoints on one fixed large validation set;
- repeat across three paired seeds;
- use the sealed test set once after freezing architecture and hyperparameters;
- report confidence intervals, not only point estimates.

## Phase 2 - decisive ablations

- remove Pulse Memory;
- remove token shift;
- compare Sparse8, Sparse12 and Sparse16;
- sweep memory rank and timescales;
- match parameters, tokens, optimizer, learning-rate search and wall-clock budget.

## Phase 3 - capability validation

- long-context retrieval;
- associative recall and state tracking;
- selective copying;
- code and SQL evaluation;
- reasoning with verifiable answers;
- Spanish and English language quality.

## Phase 4 - systems reproduction

- Linux CUDA reproduction;
- Ampere, Ada and data-center GPU comparison;
- PyTorch eager, compile and custom Triton paths;
- throughput, energy, memory and time-to-quality reporting.

## Phase 5 - model release gate

A public pretrained checkpoint should only be trained if the same candidate:

- retains an efficiency advantage on at least two hardware platforms;
- matches or improves quality under a meaningful token budget;
- survives mechanism ablations;
- has source-complete reproducibility;
- is independently replicated.
