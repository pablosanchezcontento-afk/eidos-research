# Programa de investigación EIDOS

EIDOS es un proyecto experimental dirigido por **Pablo Sánchez Contento** para investigar modelos de lenguaje más eficientes en GPU mediante una combinación de mezcla causal local, memoria predictiva de bajo rango y atención global escasa.

El repositorio publica tanto los avances como los fallos: comparaciones injustas, bugs del scan, diseños lentos, resultados invalidados y márgenes de calidad que se redujeron al entrenar durante más tiempo.

**Conclusión actual:** existe una señal de ventaja de ingeniería en una RTX 5070, pero todavía no está demostrada una superioridad general de calidad frente a Transformers.

La documentación principal está redactada en inglés para revisión internacional:

- `docs/EIDOS_Project_Brief_v0.7.html`
- `docs/EIDOS_Technical_Preprint_v0.7.html`
- `docs/ARCHITECTURE.md`
- `docs/EXPERIMENTAL_STATUS.md`
- `docs/NEGATIVE_RESULTS.md`
