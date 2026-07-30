# EIDOS explicado: una arquitectura experimental para modelos de lenguaje más eficientes

## Idea principal

Los modelos de lenguaje actuales suelen estar basados en Transformers. Funcionan muy bien porque la atención permite que cada token mire a todos los tokens anteriores. El problema es que esa operación puede ser cara, especialmente cuando el contexto crece.

EIDOS explora otra idea: no usar atención global todo el tiempo. En su lugar, combina:

- capas locales rápidas;
- una memoria predictiva comprimida;
- algunas capas de atención global colocadas estratégicamente.

La pregunta no es si los Transformers son malos. La pregunta es si parte de su trabajo puede hacerse de forma más barata sin perder calidad.

## Cómo funciona un Transformer

Un Transformer procesa texto en pasos:

1. Convierte el texto en tokens.
2. Transforma cada token en un vector.
3. Usa atención para comparar tokens entre sí.
4. Usa una red feed-forward para transformar la información.
5. Repite muchas capas.
6. Predice el siguiente token.

La atención es potente porque permite relaciones directas a largo alcance. Pero también consume mucha memoria y cómputo.

## Qué intenta hacer EIDOS

EIDOS intenta repartir el trabajo:

### 1. Mezcla local

La mayoría de capas procesan información cercana. Esto es rápido y se adapta bien a GPU.

### 2. Memoria predictiva

Algunas capas guardan una versión comprimida de lo importante del pasado. La memoria no intenta guardar todo, sino la sorpresa o innovación: lo que el estado anterior no esperaba bien.

### 3. Atención global escasa

La atención global no desaparece. Aparece cada cierto número de capas para reconectar toda la secuencia.

## Por qué es interesante

Si EIDOS funcionase de forma reproducible, podría permitir:

- entrenar modelos más grandes en GPUs domésticas;
- usar contextos más largos;
- gastar menos VRAM;
- obtener mejor velocidad por token;
- construir modelos locales especializados en código, SQL, sistemas y razonamiento verificable.

## Qué se ha aprendido

El proyecto no empezó con EIDOS. Hubo varias versiones:

- **NOVA:** idea inicial de memoria adaptativa, pero con problemas metodológicos.
- **ISADORA:** corrigió el scan y endureció tests.
- **SAPPHO:** mantuvo atención como base, pero fue demasiado lenta.
- **EIDOS:** cambió el diseño a capas especializadas y escasas.

Lo importante es que las versiones malas no se ocultaron. Se documentaron como parte del aprendizaje.

## Estado actual

Hay una señal positiva de ingeniería: una versión EIDOS Sparse16 registrada internamente fue más rápida que un Transformer emparejado en un perfil de aproximadamente 1.093B parámetros y contexto 4096.

Pero la calidad no está cerrada. En entrenamientos más largos, la ventaja se volvió muy pequeña. Además, el código exacto de la rama más avanzada no está completamente preservado en el repositorio.

Por tanto, la conclusión correcta es:

> EIDOS es una hipótesis prometedora de arquitectura eficiente, no una prueba de que los Transformers hayan sido superados.

## Qué falta

Para convertirlo en un resultado fuerte hace falta:

- recuperar el código exacto de Sparse16 + token-shift + FlexAttention;
- repetir los benchmarks en otra máquina;
- ejecutar entrenamientos más largos;
- comparar contra Transformers fuertes;
- hacer ablations;
- evaluar código, SQL, contexto largo y razonamiento verificable;
- publicar los JSON crudos.

## Por qué el proyecto puede interesar a una empresa

Aunque no haya una victoria definitiva, el proyecto demuestra habilidades reales:

- diseño experimental;
- ingeniería GPU;
- entrenamiento de modelos;
- depuración numérica;
- evaluación honesta;
- documentación técnica;
- uso de agentes de IA como herramienta de ingeniería.

Eso convierte a EIDOS en un proyecto de investigación serio, especialmente útil como base para colaboración, prácticas, mentoría o evaluación técnica.