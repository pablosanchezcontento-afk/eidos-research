# EIDOS - Preprint técnico resumido

## Resumen

EIDOS es una arquitectura autoregresiva híbrida que reemplaza muchas capas de atención global por mezcla causal local y memoria predictiva de bajo rango. La versión Sparse16 usa aproximadamente 14 capas locales, una capa de memoria y una ancla global por ciclo de 16 capas.

## Arquitectura

```text
L L L L L L L M L L L L L L L A
```

Donde:

- `L`: mezclador causal local con convolución depthwise, token shift y FFN;
- `M`: Pulse Memory, estado predictivo de bajo rango escrito desde innovación;
- `A`: ancla de atención causal global.

## Resultado de ingeniería

Perfil medido: 1.093B parámetros, contexto 4096, BF16, Adafactor, checkpointing simétrico, RTX 5070.

| Modelo | Throughput | Memoria reservada |
|---|---:|---:|
| EIDOS Sparse16 | 5.229 tok/s | 5,084 GiB |
| Transformer emparejado | 4.049 tok/s | 5,279 GiB |

Ratio: **1,291x**.

## Resultado de calidad

La calidad no está resuelta. El margen inicial fue positivo pero se estrechó:

- 0,1 tpp: +0,166 bits/token favorable a EIDOS;
- 0,5 tpp: +0,0172 bits/token favorable a EIDOS;
- 2,0 tpp: comparación final pendiente.

## Limitaciones

- Una sola plataforma principal.
- Presupuesto de entrenamiento muy inferior al pretraining moderno.
- Sin réplica externa.
- Código exacto de la rama posterior a v0.4 incompleto en el archivo conservado.
- Faltan benchmarks downstream.

## Conclusión

EIDOS demuestra una dirección de ingeniería prometedora, pero no prueba todavía superioridad general frente a Transformer.
