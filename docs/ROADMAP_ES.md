# Roadmap de EIDOS

## Objetivo general

Convertir EIDOS en una arquitectura reproducible, evaluada de forma justa y capaz de demostrar si una combinación de mezcla local, memoria predictiva y atención global escasa puede superar a un Transformer en eficiencia y calidad bajo el mismo presupuesto.

## Fase 1 - Recuperación y congelación de código

Prioridad absoluta:

- recuperar o reconstruir la rama Sparse16 + token-shift + FlexAttention;
- fijar un commit reproducible;
- guardar configuraciones exactas;
- publicar JSON crudos;
- documentar entorno CUDA, PyTorch, Triton y drivers;
- añadir tests de equivalencia, causalidad, caché, backward y memoria.

Criterio de salida: cualquier persona debe poder clonar el repositorio, reconstruir el entorno y ejecutar el benchmark principal.

## Fase 2 - Reproducción de eficiencia

Repetir el resultado de 1.093B con:

- RTX 5070 original;
- otra GPU de consumo;
- preferiblemente Linux CUDA;
- orden invertido de ejecución;
- procesos aislados por arquitectura;
- medición de memoria asignada y reservada;
- comparación contra Transformer con backend fuerte.

Criterio de salida: EIDOS debe mantener al menos 1,20x de throughput en contexto largo sin peor memoria práctica.

## Fase 3 - Calidad pequeña y mediana

Entrenar modelos de 10M-30M con:

- tres semillas;
- mismo tokenizer;
- mismo corpus;
- learning rate search equivalente;
- validación fija;
- test sellado;
- ablations.

Criterio de salida: EIDOS debe igualar o superar a Transformer y Conv Striped en calidad por hora, no solo en pérdida final.

## Fase 4 - Escalado 100M-180M

Entrenar una versión seria con corpus bilingüe técnico y código.

Métricas:

- bits/token;
- tokens/s;
- VRAM;
- tiempo hasta calidad;
- contexto largo;
- recuperación asociativa;
- código, SQL y razonamiento verificable.

Criterio de salida: ventaja reproducible o cierre honesto del proyecto como arquitectura no ganadora.

## Fase 5 - Producto local especializado

Solo si las fases anteriores salen bien:

- continued pretraining en programación, SQL, sistemas y documentación técnica;
- instruction tuning con respuestas verificables;
- integración con herramientas locales;
- evaluación con ejecución de código;
- modelo local usable, no solo checkpoint base.

## Decisión de parada

EIDOS debe simplificarse o abandonarse si:

- no reproduce la ventaja de eficiencia;
- pierde calidad de forma estable;
- solo gana por una configuración débil del baseline;
- la ventaja desaparece al aumentar tokens;
- la memoria predictiva no aporta frente a una variante local/convolucional más simple.

## Principio rector

No escalar por ego. Escalar solo cuando el experimento anterior justifique el siguiente.