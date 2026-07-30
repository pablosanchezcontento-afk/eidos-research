# EIDOS: a practical attempt to reduce all-attention language-model computation

**Pablo Sánchez Contento — Independent research project — v0.7, July 2026**

## Abstract

Modern language models usually rely on Transformer blocks, where self-attention repeatedly allows every token to read information from earlier tokens. This is powerful, but global attention can become expensive as sequences grow. EIDOS investigates a different allocation of computation: most layers use cheap causal local processing, occasional layers update a compact predictive memory, and only a small number of layers perform global attention.

The project does not claim that Transformers have been replaced. It reports a transparent research path containing failed architectures, invalid comparisons, implementation bugs, corrected benchmarks and encouraging but incomplete evidence. The strongest preserved project record reports a 1.291x training-throughput ratio over a parameter-matched Transformer at 1.093 billion parameters and context length 4096 on one RTX 5070 configuration. General language-model quality superiority remains unproven.

## 1. Why investigate alternatives?

A Transformer predicts the next token by repeatedly combining:

1. token embeddings;
2. causal self-attention;
3. feed-forward channel mixing;
4. residual connections and normalization.

Global attention is useful because distant tokens can interact directly. Its cost, however, grows quickly with context length and it is repeated throughout the network. EIDOS asks whether the same global operation must appear in nearly every layer.

## 2. The EIDOS idea

The latest conceptual design uses a sparse sixteen-layer cycle:

```text
Local × 7 → Memory → Local × 7 → Global attention
```

The local layers capture nearby patterns and propagate information causally. Pulse Memory maintains a reduced persistent state and attempts to write only information that was not already predicted. The global anchor periodically reconnects the full sequence.

The hypothesis is not that attention is unnecessary. It is that attention may be used more selectively.

## 3. What failed first

The project began with memory-heavy architectures. NOVA produced promising tiny-model curves, but the baseline learning-rate search was weaker and some ablations were not parameter matched. ISADORA corrected the scan and methodology but did not establish superiority. SAPPHO attached predictive memory beside attention; on an RTX 5070 it achieved only 0.482x Transformer throughput after compilation. This showed that an expensive sidecar was the wrong topology.

EIDOS therefore moved memory and attention into separate scheduled layer roles instead of calculating both at every depth.

## 4. Current evidence

### Directly preserved SAPPHO result

| Model configuration | Throughput ratio vs Transformer |
|---|---:|
| SAPPHO eager | 0.353x |
| SAPPHO compiled | 0.482x |

This negative result is fully preserved as raw JSON.

### Strongest later engineering record

| Model | Throughput | Reserved VRAM |
|---|---:|---:|
| EIDOS Sparse16, 1.093B | 5,229 tok/s | 5.084 GiB |
| Matched Transformer | 4,049 tok/s | 5.279 GiB |

The recorded ratio is 1.291x. The exact later source tree and all raw outputs were not recovered, so external reproduction is still required.

### Quality status

Small undertrained runs initially favored EIDOS, but the margin narrowed from 0.166 bits/token at 0.1 tokens per parameter to 0.0172 bits/token at 0.5 tokens per parameter. A longer run was not completed with a final common evaluation. Therefore the project does not claim a quality win.

## 5. What would count as success?

A meaningful success requires the same frozen EIDOS candidate to:

- maintain an efficiency advantage on more than one hardware/software stack;
- match or improve quality with equal parameters, tokens and hyperparameter-search effort;
- survive multiple paired random seeds;
- pass long-context and capability benchmarks;
- be independently reproduced from complete source and raw results.

## 6. Why publish failures?

Architecture research is vulnerable to accidental advantages: unmatched learning rates, different objectives, inefficient baselines, graph breaks, memory paging or lucky seeds. Publishing rejected results makes the remaining evidence more useful and allows other researchers to avoid repeating the same mistakes.

## Conclusion

EIDOS is best understood as a serious open research program, not a finished replacement for the Transformer. It has produced one credible engineering direction: local causal computation with sparse predictive memory and sparse global attention. The next decisive step is source recovery and a larger, fixed, multi-seed quality comparison.
