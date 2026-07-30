# EIDOS - Resumen ejecutivo para empresas y equipos de investigación

## Qué es EIDOS

EIDOS es un programa independiente de investigación en arquitecturas de modelos de lenguaje. Su objetivo es estudiar si un modelo autoregresivo puede reducir el uso constante de atención global sin perder comunicación de largo alcance.

La propuesta combina tres funciones:

1. **Mezcla causal local** para procesar la mayoría de tokens de forma barata.
2. **Memoria predictiva de bajo rango** para conservar información comprimida del pasado.
3. **Anclas de atención global** usadas solo cada cierto número de capas para reconectar toda la secuencia.

El proyecto debe evaluarse como investigación de arquitectura y sistemas GPU, no como un producto final de modelo fundacional.

## Por qué importa

Los Transformers son una base muy fuerte, pero repetir atención global en casi todas las capas puede ser caro en contextos largos. EIDOS explora una pregunta concreta:

> ¿Puede una arquitectura híbrida usar atención global con menos frecuencia y aun así mantener calidad, contexto y capacidad de entrenamiento?

Si la respuesta fuese positiva de forma reproducible, el resultado podría ser útil para modelos locales, contextos largos, hardware de consumo, investigación académica y entrenamiento especializado.

## Estado actual

El proyecto ha producido evidencia de ingeniería interesante, pero no una prueba definitiva de superioridad general.

- SAPPHO, una versión anterior, fue claramente peor en velocidad que un Transformer emparejado.
- EIDOS cambió el diseño hacia un programa escaso: más capas locales, menos memoria y menos atención global.
- En el registro preservado más fuerte, EIDOS Sparse16 alcanzó una ventaja de throughput de **1,291x** frente a un Transformer emparejado de aproximadamente **1.093B** parámetros y contexto **4096**.
- La evidencia de calidad se estrechó al entrenar más: la ventaja pasó de 0,166 bits/token a 0,0172 bits/token en presupuestos mayores.
- La comparación final extendida no está cerrada.

## Qué demuestra el proyecto

El valor actual no es solo la arquitectura. El proyecto demuestra:

- diseño de hipótesis arquitectónicas;
- implementación de modelos desde cero;
- pruebas de causalidad, caché, backward y gradientes;
- benchmarks en RTX 5070;
- comparación con Transformers emparejados en parámetros;
- identificación y publicación de resultados inválidos;
- documentación técnica y pública;
- trabajo con agentes de programación de IA bajo supervisión humana.

## Qué no se afirma

Este repositorio no afirma que EIDOS haya sustituido a los Transformers. Tampoco afirma estado del arte, superioridad general ni reproducción independiente.

La posición correcta es:

> EIDOS contiene una hipótesis arquitectónica prometedora y evidencia de eficiencia que merece reproducción, pero todavía necesita cerrar calidad, ablations y recuperación completa de la rama Sparse16 + token-shift + FlexAttention.

## Oportunidades de colaboración

Un equipo externo podría aportar valor en:

- reproducción en Linux/CUDA y GPUs de centro de datos;
- kernels Triton/CUDA para memoria predictiva y mezcla local;
- evaluación larga de calidad;
- benchmarks de contexto largo;
- datasets bilingües y de código;
- entrenamiento de 30M, 100M, 180M y 1B bajo protocolo sellado;
- revisión científica y publicación.

## Próximo hito serio

El siguiente hito no es entrenar un modelo gigante por impulso. Es congelar una rama reproducible, repetir la comparación contra Transformer con varias semillas, evaluar en un conjunto fijo y abrir el código exacto de la variante que produjo la ventaja de ingeniería.
