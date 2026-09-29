1) Run benchmark_eidos_5070.ps1 first.
2) Only EIDOS candidates at >=1.05x Transformer throughput remain eligible.
3) Build the LR calibration matrix with build_glyph_calibration.ps1.
4) Execute the generated runs/run_glyph_calibration.ps1.
5) Run select_glyph_lrs.ps1. If any LR lands on a boundary, extend and rerun.
6) Confirm surviving candidates with 20 tokens/parameter, three seeds, and a sealed test split.
7) evaluate_gate.py is the only script allowed to unlock Mythos.
