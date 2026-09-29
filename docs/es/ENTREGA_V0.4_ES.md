# Entrega EIDOS v0.4

Empieza por `reports/EIDOS_v0.4_Explicado_Facil.pdf`.

Despues ejecuta:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\benchmark_eidos_5070.ps1
```

Archivos clave:

- `eidos/model.py`: arquitectura y scan.
- `eidos/baselines.py`: Transformer, Conv Striped y recurrente.
- `benchmarks/benchmark_gpu.py`: comparacion real en RTX.
- `train.py`: entrenamiento controlado y test sellado.
- `experiments/build_matrix.py`: generacion de matrices.
- `experiments/select_lrs.py`: bloqueo de LR en borde.
- `experiments/evaluate_gate.py`: puerta final.
- `ACTION_LOG.md`: registro de todo lo realizado.

No entrenes Verse, Epic o Mythos hasta analizar `results/gpu/eidos_glyph_5070_compile.json`.
