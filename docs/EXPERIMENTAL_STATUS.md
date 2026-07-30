# Experimental status

## Evidence classes

This project separates results into three classes:

1. **Directly preserved raw evidence** - raw JSON or source files exist in this repository.
2. **Preserved execution record** - numerical result exists in the technical log, but raw output or exact source is missing.
3. **Hypothesis / planned experiment** - not yet measured.

## Directly preserved raw evidence

### SAPPHO 10M on RTX 5070

| Mode | SAPPHO | Transformer | Ratio |
|---|---:|---:|---:|
| eager | 4,131 tok/s | 11,714 tok/s | 0.353x |
| compiled | 12,141 tok/s | 25,171 tok/s | 0.482x |

Parameter difference was approximately 0.0496%. Compiled allocated VRAM was 0.238 GiB for SAPPHO and 0.220 GiB for the Transformer. The random-batch loss in these files is not a quality metric.

Raw files:

- `results/raw/sappho_vs_transformer_10m_5070.json`
- `results/raw/sappho_vs_transformer_10m_5070_compile.json`

## Results recorded in the preserved execution log

### Engineering profile: 1.093B / context 4096

The log reports isolated-process measurements:

| Model | tok/s | allocated GiB | reserved GiB |
|---|---:|---:|---:|
| EIDOS Sparse16 | 5,229 | 4.591 | 5.084 |
| Transformer | 4,049 | 4.604 | 5.279 |

Throughput ratio: **1.291x**. Reserved-memory difference: **-3.7%**.

This is the strongest engineering result, but it requires recovery of the exact post-v0.4 source and raw outputs before it can be called fully reproducible.

### Quality calibration

| Token budget | Seeds | Recorded result | Status |
|---|---:|---|---|
| 0.1 tokens/parameter | 3 paired | EIDOS lower by 0.166 bits/token on average | promising, heavily undertrained |
| 0.5 tokens/parameter | 3 paired | 11.5330 vs 11.5502 bits/token | narrow 0.0172 advantage |
| 2.0 tokens/parameter | 1 extension | final common fixed-set result absent; Transformer had a stronger momentary validation point | unresolved |

## Current claim boundary

Supported:

- SAPPHO's sidecar topology was inefficient on the tested GPU.
- Sparse hybrid computation is a credible engineering direction.
- The preserved log contains a strong 1.1B throughput result worth reproducing.

Not supported:

- general quality superiority over Transformers;
- state-of-the-art performance;
- hardware-independent speedup;
- novelty of every individual component;
- readiness for commercial deployment.
