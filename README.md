# EIDOS Research Program

**A falsification-first research project exploring GPU-efficient alternatives to all-attention language models.**

EIDOS is an experimental autoregressive architecture built around a sparse hybrid principle:

- most layers perform cheap local causal mixing;
- periodic layers update a low-rank predictive state;
- a small number of layers perform global causal attention.

The project is led by **Pablo Sánchez Contento** as an independent AI research and engineering program. AI coding agents are used for implementation, profiling, test generation and documentation. Architectural decisions, experiment design, evidence thresholds and public claims remain human responsibilities.

> **Current verdict:** an engineering efficiency advantage has been measured on one RTX 5070 configuration. General quality superiority over Transformers has **not** been established.

## Headline evidence

In the strongest preserved project log, a parameter-matched **1.093B** EIDOS Sparse16 profile at context length **4096**, BF16 and symmetric gradient checkpointing reached:

| Model | Training throughput | Peak allocated | Peak reserved |
|---|---:|---:|---:|
| EIDOS Sparse16 | 5,229 tok/s | 4.591 GiB | 5.084 GiB |
| Matched Transformer | 4,049 tok/s | 4.604 GiB | 5.279 GiB |
| Ratio / difference | **1.291x** | -0.3% | **-3.7%** |

These measurements are currently documented from the execution record but the exact post-v0.4 working tree and all raw JSON files were not present in the preserved release archive. They must therefore be treated as **project evidence awaiting full source recovery and external reproduction**, not as a completed scientific claim.

Quality evidence became much narrower as training increased:

- 0.1 tokens/parameter, three paired seeds: EIDOS led by 0.166 bits/token on average.
- 0.5 tokens/parameter, three paired seeds: EIDOS led by only 0.0172 bits/token.
- 2.0 tokens/parameter extension: final fixed-set comparison was not completed in the preserved record; a Transformer checkpoint produced a stronger momentary validation point.

## Start here

- [External project brief](docs/EIDOS_Project_Brief_v0.7.pdf) — concise company and collaborator presentation.
- [Technical preprint](docs/EIDOS_Technical_Preprint_v0.7.pdf) — architecture, methodology, evidence, limitations and roadmap.
- [Research deck](docs/EIDOS_Research_Deck_v0.7.pptx)
- [Architecture](docs/ARCHITECTURE.md)
- [Experimental status](docs/EXPERIMENTAL_STATUS.md)
- [Negative results](docs/NEGATIVE_RESULTS.md)
- [Reproducibility and source status](docs/REPRODUCIBILITY.md)
- [Roadmap](docs/ROADMAP.md)
- [Complete preserved release archive](releases/EIDOS-Research-Public-v0.7.zip)

## Repository map

```text
.
├── docs/                       public-facing papers and technical documentation
├── source/eidos-v0.4/          latest source-complete EIDOS snapshot
├── history/                    documented research lineage and preserved archive
├── results/                    preserved benchmark evidence and summaries
├── releases/                   complete v0.7 archive with all preserved artifacts
├── archive/                    historical audit material
└── governance/                 contribution, security and release policies
```

## What this repository does not claim

This repository does **not** claim that EIDOS replaces Transformers, establishes state-of-the-art language modeling, or has been independently replicated. It publishes the full preserved research path, including failed designs, benchmark defects, invalid comparisons and unresolved questions.

## Source-completeness warning

The exact executable source for the later **Sparse16 + token-shift + FlexAttention** branch described in the v0.7 papers is not present in the supplied archives. The latest source-complete snapshot is published under `source/eidos-v0.4/`. See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## License status

No open-source license has been granted yet. The material is published for review and provenance while a license is selected. See `LICENSE_PENDING.md`.
