# Programa de investigación EIDOS

**Un proyecto de investigación independiente, orientado a falsar hipótesis, que explora alternativas más eficientes en GPU a los modelos de lenguaje basados casi por completo en atención.**

EIDOS es una arquitectura autorregresiva experimental basada en un principio híbrido y escaso:

- la mayoría de las capas realizan mezcla causal local barata;
- algunas capas actualizan un estado predictivo de bajo rango;
- unas pocas capas ejecutan atención causal global.

El proyecto está dirigido por **Pablo Sánchez Contento** como programa independiente de investigación e ingeniería en inteligencia artificial. Se utilizan agentes de IA para implementar, perfilar, generar pruebas y documentar. Las decisiones arquitectónicas, el diseño experimental, los criterios de evidencia y las afirmaciones públicas siguen siendo responsabilidad humana.

> **Veredicto actual:** se ha medido una ventaja de eficiencia de ingeniería en una configuración concreta con una RTX 5070. Todavía no se ha demostrado una superioridad general de calidad frente a los Transformers.

## Qué intenta resolver EIDOS

Los Transformers modernos dependen en gran medida de la atención global. Ese mecanismo es muy expresivo, pero su coste crece rápidamente con la longitud del contexto y obliga a realizar operaciones globales en muchas capas.

EIDOS investiga si un modelo puede conservar capacidad de razonamiento y contexto usando una combinación de:

1. **Mezcla local causal**, para relaciones cercanas y procesamiento barato.
2. **Memoria predictiva**, para resumir información del pasado en un estado compacto.
3. **Atención global escasa**, para volver a conectar toda la secuencia solo cuando resulte necesario.

La hipótesis no es que la atención deba desaparecer, sino que quizá se utiliza con demasiada frecuencia.

## Evolución del proyecto

El repositorio conserva tanto los avances como los errores:

- **NOVA v0.1:** primera exploración de mecanismos recurrentes y memoria.
- **ISADORA v0.2:** corrección del scan, mejoras de metodología y primeras comparaciones más justas.
- **SAPPHO v0.3:** atención como columna vertebral y memoria predictiva como rama lateral.
- **EIDOS v0.4:** rediseño estructural para que la memoria sustituya trabajo caro en lugar de duplicarlo.
- **Sparse16 + token shift + FlexAttention:** rama posterior descrita en los registros experimentales, pero cuyo árbol de código exacto no quedó conservado.

## Evidencia principal conservada

En el registro experimental más fuerte disponible, un perfil EIDOS Sparse16 de aproximadamente **1.093 millones de parámetros** —en realidad 1.093 mil millones—, igualado en parámetros con un Transformer, contexto **4096**, BF16 y checkpointing simétrico obtuvo:

| Modelo | Rendimiento de entrenamiento | Pico asignado | Pico reservado |
|---|---:|---:|---:|
| EIDOS Sparse16 | 5.229 tok/s | 4,591 GiB | 5,084 GiB |
| Transformer igualado | 4.049 tok/s | 4,604 GiB | 5,279 GiB |
| Diferencia | **1,291x** | -0,3 % | **-3,7 %** |

Este resultado es prometedor, pero debe interpretarse con cautela: el código exacto de esa rama posterior y todos los JSON originales no estaban dentro del archivo fuente conservado. Por tanto, se publica como **evidencia del proyecto pendiente de recuperación y reproducción externa**, no como afirmación científica cerrada.

## Resultados de calidad

La ventaja observada en calidad se redujo al aumentar el entrenamiento:

- **0,1 tokens por parámetro, 3 semillas:** EIDOS aventajó en 0,166 bits/token de media.
- **0,5 tokens por parámetro, 3 semillas:** la ventaja cayó a solo 0,0172 bits/token.
- **2,0 tokens por parámetro:** la comparación final sobre conjunto fijo no quedó completada; durante la extensión, el Transformer produjo un mejor punto puntual de validación.

Eso significa que hoy no existe evidencia suficiente para afirmar que EIDOS aprende mejor de forma general.

## Resultados negativos importantes

Este proyecto publica también lo que salió mal:

- SAPPHO fue aproximadamente dos veces más lenta que el Transformer incluso con compilación.
- Algunos benchmarks iniciales mezclaban coste de backbone y objetivos auxiliares.
- Hubo barridos de learning rate insuficientes y comparaciones que favorecían a una arquitectura.
- Algunas variantes mejoraban componentes aislados, pero no el paso de entrenamiento completo.
- La ventaja de calidad temprana se estrechó al aumentar el presupuesto de tokens.
- La rama posterior más prometedora no quedó preservada de forma completamente reproducible.

## Qué no se afirma

Este repositorio **no** afirma que EIDOS:

- sustituya a los Transformers;
- sea estado del arte;
- haya sido reproducido por terceros;
- mantenga su ventaja al escalar a múltiples GPUs o a corpus mucho mayores;
- tenga ya una ventaja general en calidad.

## Qué sí aporta hoy

- Una arquitectura híbrida propia y documentada.
- Un historial experimental completo, incluidos fallos.
- Comparaciones emparejadas por parámetros.
- Benchmarks en hardware de consumo.
- Un protocolo de validación más riguroso que al inicio.
- Una dirección plausible para modelos locales grandes con menos atención global.

## Próximos pasos

1. Recuperar o reconstruir exactamente Sparse16 + token shift + FlexAttention.
2. Repetir los benchmarks en procesos aislados y guardar todos los JSON.
3. Ejecutar entrenamientos de calidad más largos con varias semillas.
4. Evaluar en tareas externas, no solo pérdida de lenguaje.
5. Probar en Linux, otras GPUs y otros tamaños de modelo.
6. Publicar una licencia abierta cuando se decida el marco adecuado.

## Documentación

- [Paper simplificado en español](papers/PAPER_SIMPLIFICADO_ES.md)
- [Preprint técnico en español](papers/PREPRINT_TECNICO_ES.md)
- [Arquitectura](docs/ARCHITECTURE.md)
- [Estado experimental](docs/EXPERIMENTAL_STATUS.md)
- [Resultados negativos](docs/NEGATIVE_RESULTS.md)
- [Reproducibilidad](docs/REPRODUCIBILITY.md)
- [Hoja de ruta](docs/ROADMAP.md)

## Conclusión

EIDOS no se presenta como un sustituto demostrado del Transformer, sino como un programa de investigación abierto que intenta responder una pregunta concreta:

> **¿Puede un modelo de lenguaje usar atención global con mucha menos frecuencia y conservar —o mejorar— su capacidad, mientras entrena más rápido en hardware accesible?**

La respuesta todavía no está cerrada. El valor del proyecto está en que publica la hipótesis, el código conservado, los resultados positivos, los fallos y los criterios necesarios para demostrar o refutar la idea.