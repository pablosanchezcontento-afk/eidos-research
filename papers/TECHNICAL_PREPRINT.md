# EIDOS: Sparse Global Attention with Local Causal Mixing and Predictive Low-Rank Memory

**Pablo Sánchez Contento**  
Independent research preprint, version 0.7 — July 2026

## Abstract

We investigate whether autoregressive language models can reduce the frequency of global self-attention without removing direct long-range communication. EIDOS assigns different computational roles across depth: inexpensive local causal mixers perform most token processing, periodic Pulse Memory layers update a reduced predictive state, and sparse global-attention anchors reconnect the complete sequence. The work follows a falsification-first progression through NOVA, ISADORA, SAPPHO and EIDOS. Several early conclusions were invalidated by unequal learning-rate search, parameter-mismatched ablations, a silent chunked-scan numerical error, duplicated vocabulary projection work, optimizer fragmentation and unfair attention backends.

Directly preserved RTX 5070 measurements show that SAPPHO's attention-plus-memory sidecar reached only 0.482x the compiled throughput of a parameter-matched Transformer. A later execution record reports that EIDOS Sparse16 at approximately 1.093B parameters and sequence length 4096 achieved 5,229 tokens/s versus 4,049 tokens/s for a matched Transformer, a ratio of 1.291x, while reserving 3.7% less GPU memory. The exact post-v0.4 working tree and all associated raw outputs were not recovered; this result is therefore treated as evidence requiring reproduction. Quality experiments favored EIDOS by 0.166 bits/token at 0.1 tokens/parameter but by only 0.0172 bits/token at 0.5 tokens/parameter. A longer fixed-set evaluation remains unresolved. We conclude that sparse hybrid computation is an engineering direction worth reproducing, but general quality superiority is not established.

## 1. Introduction

Transformer language models repeatedly apply global causal self-attention and position-wise feed-forward transformations. Their strong optimization behavior and direct token-to-token communication make them a demanding baseline. Nevertheless, repeating global attention throughout depth may allocate expensive computation more frequently than necessary, especially for long sequences.

EIDOS tests the hypothesis that sequence processing can be decomposed into three timescales:

1. **local causal mixing** for frequent short-range computation;
2. **predictive persistent state** for compressed cross-token continuity;
3. **global causal attention** for occasional unrestricted communication.

The intended contribution is not the novelty of every operator in isolation. It is the architecture schedule, engineering implementation and experimental framework used to determine whether the combination is competitive.

## 2. Research lineage

### 2.1 NOVA v0.1

NOVA introduced Adaptive Budget Innovation Memory. Tiny byte-level experiments appeared to show a scaling crossover, but the architecture received a broader learning-rate sweep than the convolutional baseline. Learning-rate sensitivity exceeded the reported architecture margin, invalidating the central conclusion. Several ablations also differed by roughly 3% in parameter count.

### 2.2 ISADORA v0.2

ISADORA hardened the associative scan, tokenizer pipeline, parameter matching and benchmark protocol. It explicitly rejected the claim of Transformer superiority because the available evidence did not support it. A silent scan issue was identified: clamping a denominator inside a chunked diagonal recurrence changed the recurrence rather than merely stabilizing it, producing large errors at longer chunks.

### 2.3 SAPPHO v0.3

SAPPHO retained attention as the backbone and attached a gated full-width predictive-memory residual branch. LayerScale and a nearly closed initial gate protected early optimization. The topology nevertheless duplicated costly work. On an RTX 5070, directly preserved measurements were:

| Mode | SAPPHO tok/s | Transformer tok/s | Ratio |
|---|---:|---:|---:|
| eager | 4,131 | 11,714 | 0.353x |
| compiled | 12,141 | 25,171 | 0.482x |

The parameter difference was approximately 0.0496%. The result established that the sidecar topology was structurally inefficient on the tested stack.

### 2.4 EIDOS v0.4 and later Sparse16 branch

EIDOS changed the macro-architecture from simultaneous attention and memory to scheduled specialized layers. The source-complete v0.4 snapshot used a convolution–memory–convolution–attention cycle. The later recorded Sparse16 branch further reduced the frequency of memory and attention, fused small parameter groups, added a learned token-shift path and used compiled FlexAttention for the tested Windows stack.

## 3. Architecture

### 3.1 Sparse16 schedule

The later conceptual macrocycle is:

```text
L L L L L L L M L L L L L L L A
```

where `L` denotes a local causal mixer, `M` a Pulse Memory layer and `A` a global causal-attention anchor. For a 48-layer model, this yields 42 local mixers, three memory layers and three global anchors.

### 3.2 Local causal mixer

The local mixer combines a depthwise causal convolution, direct one-token propagation, channel mixing, normalization and residual connections. For hidden states `x_t`, a simplified representation is:

```text
u_t = DWConv_causal(Norm(x))_t
s_t = alpha ⊙ x_{t-1}
h_t = x_t + Proj(u_t + s_t)
y_t = h_t + FFN(Norm(h_t))
```

The learned diagonal coefficient `alpha` creates a minimal recurrent causal route without a full sequential matrix recurrence. During autoregressive generation, it requires only the previous token state for each local layer.

### 3.3 Pulse Memory

Pulse Memory maintains a low-rank state `m_t ∈ R^r`, with `r << d`. The intended update is innovation driven:

```text
p_t = Predict(m_{t-1})
e_t = Compress(x_t) - p_t
w_t = WriteGate(e_t, surprise_t)
m_t = decay ⊙ m_{t-1} + w_t ⊙ e_t
r_t = Read(m_t, e_t)
y_t = x_t + Expand(r_t)
```

Multiple ordered decay bands represent different timescales. Read, write and budget controls were later fused to reduce small parameter tensors and optimizer overhead. The state size is independent of generated sequence length.

### 3.4 Global attention anchor

Global anchors use causal grouped-query attention. Their role is to restore direct arbitrary-distance communication that local and compressed-state paths may not reproduce. EIDOS therefore does not assume attention is unnecessary; it tests whether attention frequency can be reduced across depth.

## 4. Systems methodology

### 4.1 Parameter matching

Comparisons enforce a maximum parameter gap, typically below 0.5%. Feed-forward width is solved to match a target parameter budget rather than selected informally.

### 4.2 Comparable training objectives

Backbones must use the same next-token objective. Auxiliary multi-token prediction heads are disabled or matched in throughput comparisons. Random-batch loss is explicitly labeled as non-quality evidence.

### 4.3 GPU measurement

Reported engineering metrics include:

- tokens per training second;
- mean step time after warm-up;
- peak allocated and reserved VRAM;
- eager and compiled modes;
- forward/backward and optimizer decomposition;
- isolated-process memory measurements;
- reversed execution order to detect thermal or cache bias.

### 4.4 Quality protocol

The intended quality protocol uses:

- deterministic train, validation and sealed test splits;
- equal tokenizer and corpus;
- equal parameter and token budgets;
- equal learning-rate search effort;
- at least three paired random seeds;
- best-checkpoint selection from validation only;
- one final sealed-test evaluation after freezing decisions.

## 5. Engineering results

### 5.1 Rejected measurements

An initial 427M/context-2048 result appeared to show approximately 23x speedup and a dramatic memory advantage. It was rejected because the Transformer attention path likely materialized an inefficient backend and checkpointing was not symmetric. This demonstrates why backend parity is part of architecture evaluation.

### 5.2 Source-recorded 1.093B result

The strongest preserved execution record reports isolated fresh-process measurements at approximately 1.093B parameters, sequence length 4096, BF16, Adafactor and symmetric gradient checkpointing:

| Metric | EIDOS Sparse16 | Transformer |
|---|---:|---:|
| Throughput | 5,229 tok/s | 4,049 tok/s |
| Peak allocated | 4.591 GiB | 4.604 GiB |
| Peak reserved | 5.084 GiB | 5.279 GiB |

The recorded throughput ratio is 1.291x. Reserved memory is 3.7% lower. The exact post-v0.4 source and raw JSON files are absent, so this result is not yet independently reproducible from the repository.

### 5.3 Practical VRAM boundary

The same 1.1B scale technically fit without checkpointing near the 12GB limit, but Windows WDDM paging reduced throughput to approximately 265 tok/s. Symmetric checkpointing reduced pressure and raised useful throughput above 5,000 tok/s. This distinguishes nominal fit from practical trainability.

## 6. Quality results

At short budgets with learning rate 1e-2, EIDOS showed stable gains across three paired seeds. The margin, however, decreased substantially with more training:

| Budget | EIDOS result relative to Transformer |
|---|---|
| 0.1 tokens/parameter | −0.166 bits/token average |
| 0.5 tokens/parameter | 11.5330 vs 11.5502, a −0.0172 advantage |
| 2.0 tokens/parameter | unresolved final fixed-set evaluation |

A Transformer checkpoint produced a stronger momentary validation observation during the 2.0-token-per-parameter extension. No quality-superiority claim is made.

## 7. Threats to validity

1. Measurements were performed primarily on one consumer GPU and Windows software stack.
2. The strongest Sparse16 source revision is missing.
3. Small token budgets do not predict large-scale pretraining reliably.
4. A bilingual corpus may favor different inductive biases than standard benchmark corpora.
5. Throughput can depend on compiler version, attention backend and optimizer implementation.
6. Sparse global attention may underperform on capabilities requiring frequent exact retrieval.
7. Architectural novelty has not undergone formal literature or patent review.

## 8. Required experiments

The next frozen candidate must undergo:

- source recovery and checksum-tagged release;
- three-seed evaluation at a meaningful token budget;
- one fixed large validation evaluation and sealed test;
- local-only, memory-free and attention-frequency ablations;
- long-context retrieval and associative-recall tasks;
- code, SQL and verifiable reasoning evaluation;
- Linux CUDA reproduction on at least one additional GPU generation;
- energy and wall-clock-to-quality measurement.

## 9. Conclusion

The project falsified its original memory-sidecar design and produced a more plausible sparse hybrid architecture. Direct evidence proves that SAPPHO was substantially slower than a matched Transformer. The later EIDOS record suggests that local causal mixing plus sparse predictive memory and sparse global attention may provide a meaningful long-context engineering advantage. Because source recovery and conclusive quality evaluation remain incomplete, EIDOS should be viewed as a reproducible-research target and collaboration opportunity, not a replacement for the Transformer.
