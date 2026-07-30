# Negative results and invalidated conclusions

Negative evidence is a first-class project artifact.

## 1. NOVA scaling crossover was not demonstrated

NOVA received a broader learning-rate sweep than the convolutional baseline. Learning-rate variation inside NOVA was two to four times larger than the reported architectural margin. The apparent crossover could therefore be hyperparameter selection rather than architecture.

## 2. Early ablations were parameter-mismatched

The full NOVA variant used roughly 3% more parameters than several ablations, while the claimed loss improvement was small and based on one seed. The mechanism-level conclusion was invalid.

## 3. Chunked scan had a silent numerical bug

A `clamp_min(1e-12)` inside the chunked diagonal scan corrupted state for larger chunks. Approximate errors rose from ~4e-7 at chunk 32 to more than 3 at chunks 64 and 128 without producing NaNs or warnings.

## 4. Byte-level evidence did not transfer automatically to BPE

The original curve used byte-level tokens while the intended large-model pipeline used a 32k BPE vocabulary. Redundancy exploited by memory at byte level might disappear after tokenization compression.

## 5. SAPPHO's topology was structurally slow

The compiled model reached only 0.482x the throughput of a parameter-matched Transformer. Profiling showed the full-width memory layer was more expensive than attention and the architecture also performed duplicated vocabulary work in some benchmark paths.

## 6. Strided Pulse Memory did not improve the full model as expected

Stride 4 improved the isolated Pulse component but did not produce the same improvement in full-model training. Grouping, repetition and compiler partition costs erased much of the local gain.

## 7. Optimizer fragmentation mattered

A large part of the EIDOS penalty came from many small parameter tensors in Adafactor. Fusing control projections and state parameters reduced optimizer time substantially. This showed that architecture benchmarks can be dominated by parameter organization, not only FLOPs.

## 8. An apparent 23x long-context win was rejected

A context-2048 comparison initially produced an enormous gap, but the Transformer path had likely fallen outside an efficient attention backend and checkpointing was not applied symmetrically. The result was explicitly discarded.

## 9. A 1.1B model technically fit without checkpointing but entered WDDM paging

Both models approached the 12 GB limit. EIDOS reported ~265 tok/s despite fitting. Symmetric checkpointing reduced memory pressure and increased useful throughput dramatically. “Fits in VRAM” was not treated as equivalent to “practically trainable.”

## 10. Sparse12 did not recover quality

Increasing memory and attention frequency from Sparse16 to Sparse12 did not materially improve the small-model validation result. More global layers were not a sufficient explanation for the quality gap.

## 11. Early quality margins shrank with training

A large short-run advantage narrowed to 0.0172 bits/token at 0.5 tokens/parameter. During the 2.0-token-per-parameter extension, the Transformer achieved a stronger momentary validation point. The project therefore does not claim a quality win.
