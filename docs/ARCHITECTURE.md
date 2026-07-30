# Architecture

## Research hypothesis

Dense Transformers use global causal self-attention in many layers. EIDOS tests whether global communication can be made **periodic rather than ubiquitous**, while cheaper local and stateful operators perform most sequence processing.

The current conceptual topology is **Sparse16**:

```text
L L L L L L L M L L L L L L L A
```

where:

- `L` - local causal mixer;
- `M` - predictive Pulse Memory;
- `A` - global causal attention anchor.

Fourteen of every sixteen layers are local, one updates memory and one reconnects the complete sequence.

## Local causal mixer

The later experimental branch combined:

1. depthwise causal convolution for short-range pattern extraction;
2. a learned one-token shift path for direct causal propagation;
3. a channel-mixing feed-forward network;
4. residual connections and normalization.

This path is intended to scale approximately linearly with sequence length and map well to compiled GPU kernels.

## Pulse Memory

Pulse Memory is a low-rank persistent state. Instead of copying the full hidden activation into memory, it attempts to write an **innovation signal**: information not already predicted by the current state.

The design uses:

- reduced state rank;
- multiple learned timescales;
- surprise-conditioned writing;
- independent read and write controls;
- causal state updates;
- constant-size state during autoregressive generation.

The memory is sparse in depth: it replaces an ordinary layer at controlled intervals rather than running as an expensive sidecar beside attention.

## Global attention anchor

Global causal attention remains in the architecture as a protected capability. It provides direct long-range token-to-token communication, but only at scheduled anchor layers.

The later implementation record reports use of compiled FlexAttention on Windows because the available PyTorch build did not expose the expected FlashAttention route for the tested GQA configuration.

## Historical evolution

### NOVA v0.1

Adaptive Budget Innovation Memory appeared promising in sub-million-parameter byte-level experiments, but the baseline learning-rate sweep was not matched. The claimed scaling crossover was therefore not established.

### ISADORA v0.2

The scan implementation and experimental protocol were hardened. The scientific conclusion remained negative: there was insufficient evidence of superiority.

### SAPPHO v0.3

Attention remained the backbone and predictive memory was attached as a gated residual sidecar. Real RTX 5070 measurements showed the topology was structurally too expensive: only 0.482x Transformer throughput after compilation.

### EIDOS v0.4

The architecture changed from “attention plus memory” to specialized layer types, initially using a `C-M-C-A` macrocycle. This is the latest source-complete EIDOS snapshot preserved in this repository.

### Post-v0.4 Sparse16 branch

The execution record describes further changes: sparser memory and attention, token shift, parameter fusion, FlexAttention, isolated-process memory measurement and 1.093B profiling. The exact working tree was not included in the provided archive, so this branch is documented but not presented as source-reproducible.
