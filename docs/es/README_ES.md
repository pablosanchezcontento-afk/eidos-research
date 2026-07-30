# Programa de Investigación EIDOS

**Proyecto de investigación falsable para explorar alternativas eficientes en GPU a modelos de lenguaje basados solo en atención.**

EIDOS es una arquitectura autoregresiva experimental basada en tres ideas:

- la mayoría de capas realizan mezcla causal local barata;
- algunas capas actualizan un estado predictivo de bajo rango;
- unas pocas capas usan atención causal global como ancla.

El proyecto está liderado por **Pablo Sánchez Contento** como programa independiente de investigación e ingeniería en IA. Se usan agentes de programación para acelerar implementación, perfilado, tests y documentación, pero las decisiones arquitectónicas, el diseño experimental y las afirmaciones públicas se mantienen bajo criterio humano.

> **Veredicto actual:** hay una ventaja de ingeniería medida en una RTX 5070, pero la superioridad general de calidad frente a Transformers **no está demostrada**.

## Evidencia principal

En el registro preservado más fuerte, un perfil **EIDOS Sparse16** de **1.093B** parámetros, contexto **4096**, BF16 y checkpointing simétrico obtuvo:

| Modelo | Throughput | Pico asignado | Pico reservado |
|---|---:|---:|---:|
| EIDOS Sparse16 | 5.229 tok/s | 4,591 GiB | 5,084 GiB |
| Transformer emparejado | 4.049 tok/s | 4,604 GiB | 5,279 GiB |
| Diferencia | **1,291x** | -0,3% | **-3,7%** |

Estos datos son evidencia de proyecto pendiente de reproducción externa. La rama exacta posterior a v0.4 con Sparse16 + token shift + FlexAttention no está completa en el archivo preservado.

## Calidad

- 0,1 tokens/parámetro, tres semillas: EIDOS lideró por 0,166 bits/token.
- 0,5 tokens/parámetro, tres semillas: la ventaja bajó a 0,0172 bits/token.
- 2,0 tokens/parámetro: evaluación final común pendiente; un punto momentáneo favoreció al Transformer.

## Qué no se afirma

Este repositorio no afirma que EIDOS sustituya a los Transformers, ni que sea estado del arte, ni que haya sido reproducido externamente. Publica la trayectoria completa, incluyendo fallos, bugs, comparaciones inválidas y preguntas abiertas.
