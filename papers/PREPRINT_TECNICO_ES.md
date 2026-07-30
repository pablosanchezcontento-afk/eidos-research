# EIDOS: atención global escasa con mezcla causal local y memoria predictiva de bajo rango

**Pablo Sánchez Contento**  
Preprint independiente, versión española 0.7 - julio de 2026

## Resumen

Este trabajo investiga si un modelo de lenguaje autoregresivo puede reducir la frecuencia de atención global sin eliminar la comunicación de largo alcance. EIDOS reparte funciones a través de la profundidad: mezcladores causales locales baratos procesan la mayoría de capas, capas periódicas de Pulse Memory actualizan un estado predictivo reducido y anclas escasas de atención global reconectan toda la secuencia.

El proyecto siguió una progresión falsacionista: NOVA, ISADORA, SAPPHO y EIDOS. Varias conclusiones iniciales fueron invalidadas por barridos de learning rate desiguales, ablations con parámetros mal igualados, un bug silencioso en el scan por chunks, trabajo duplicado de proyección al vocabulario, fragmentación del optimizador y backends de atención no equivalentes.

Las mediciones preservadas de SAPPHO en una RTX 5070 muestran que su diseño de atención más memoria lateral alcanzó solo 0,482x del throughput compilado de un Transformer emparejado. Un registro posterior informa que EIDOS Sparse16, con aproximadamente 1.093B parámetros y contexto 4096, obtuvo 5.229 tok/s frente a 4.049 tok/s de un Transformer emparejado, una razón de 1,291x, reservando 3,7% menos memoria. El árbol exacto posterior a v0.4 y todos los JSON crudos no fueron recuperados, por lo que ese resultado debe tratarse como evidencia pendiente de reproducción.

En calidad, EIDOS lideró por 0,166 bits/token a 0,1 tokens/parámetro, pero solo por 0,0172 bits/token a 0,5 tokens/parámetro. Una evaluación más larga permanece sin cierre definitivo. La conclusión es que la computación híbrida escasa merece reproducción, pero la superioridad general frente a Transformers no está establecida.

## 1. Motivación

Los Transformers aplican atención causal global y redes feed-forward en repetición. Son una línea base extremadamente fuerte porque ofrecen comunicación directa entre tokens y optimización estable. Aun así, repetir atención global en casi todas las capas puede asignar demasiado cómputo caro, especialmente cuando crece el contexto.

EIDOS prueba una descomposición en tres escalas:

1. **mezcla local causal** para computación frecuente y barata;
2. **estado predictivo persistente** para continuidad comprimida;
3. **atención global causal** solo ocasionalmente para comunicación sin restricciones.

La contribución no es afirmar que cada operador aislado sea nuevo. La contribución propuesta está en el calendario de capas, la ingeniería GPU y el marco experimental para decidir si la combinación compite.

## 2. Evolución del proyecto

### 2.1 NOVA v0.1

NOVA introdujo ABIM, una memoria de innovación con presupuesto adaptativo. Los experimentos byte-level muy pequeños parecían mostrar un cruce de escalado. Tras auditar los JSON, la conclusión quedó invalidada porque NOVA había recibido más learning rates que el baseline convolucional. La sensibilidad al learning rate era mayor que la ventaja arquitectónica reportada. Algunas ablations también tenían aproximadamente 3% más parámetros.

### 2.2 ISADORA v0.2

ISADORA corrigió el scan asociativo, endureció el tokenizer, el matching de parámetros y el protocolo. También rechazó explícitamente afirmar superioridad frente a Transformers. Se detectó un bug importante: usar `clamp_min` dentro de una recurrencia diagonal por chunks cambiaba la matemática de la recurrencia y producía errores grandes en chunks largos.

### 2.3 SAPPHO v0.3

SAPPHO mantuvo la atención como columna vertebral y añadió una memoria predictiva residual de ancho completo. La rama empezaba casi cerrada para no romper la optimización inicial. En la RTX 5070, sin embargo, fue demasiado lenta:

| Modo | SAPPHO tok/s | Transformer tok/s | Ratio |
|---|---:|---:|---:|
| eager | 4.131 | 11.714 | 0,353x |
| compilado | 12.141 | 25.171 | 0,482x |

La diferencia de parámetros era de aproximadamente 0,0496%. El resultado demostró que esa topología lateral era ineficiente.

### 2.4 EIDOS v0.4 y rama Sparse16

EIDOS cambió el diseño: en lugar de calcular atención y memoria a la vez, separó funciones por capas. La instantánea fuente-completa v0.4 usaba un ciclo convolución-memoria-convolución-atención. La rama posterior Sparse16 redujo aún más la frecuencia de memoria y atención, fusionó parámetros pequeños, añadió token-shift aprendido y usó FlexAttention compilado en el stack Windows probado.

## 3. Arquitectura

El macrociclo conceptual de Sparse16 es:

```text
L L L L L L L M L L L L L L L A
```

Donde:

- `L` es un mezclador causal local;
- `M` es Pulse Memory;
- `A` es una ancla de atención global.

En un modelo de 48 capas, esto produce 42 mezcladores locales, 3 memorias y 3 anclas globales.

### 3.1 Mezclador causal local

El mezclador local combina convolución depthwise causal, propagación directa del token anterior, mezcla de canales, normalización y residual. Su objetivo es cubrir la mayor parte de la computación sin hacer atención global en cada capa.

### 3.2 Pulse Memory

Pulse Memory mantiene un estado de bajo rango `m_t`, mucho menor que la dimensión completa del modelo. La idea es escribir la innovación: aquello que el estado anterior no predice bien.

Forma simplificada:

```text
p_t = Predict(m_{t-1})
e_t = Compress(x_t) - p_t
m_t = decay * m_{t-1} + write_gate * e_t
y_t = x_t + Expand(Read(m_t))
```

El estado no crece con la longitud de generación. Las bandas de decay representan memoria rápida, media y lenta.

### 3.3 Ancla global

Las anclas de atención global no desaparecen. EIDOS no asume que la atención sea inútil. Prueba si se puede usar con menor frecuencia sin perder demasiado acceso directo a largo alcance.

## 4. Metodología

El proyecto exige:

- matching de parámetros, normalmente con margen inferior al 0,5%;
- mismo tokenizer y corpus;
- mismos presupuestos de tokens;
- mismas semillas emparejadas;
- learning rate buscado con el mismo esfuerzo;
- validación separada del test sellado;
- medición de throughput, VRAM, forward/backward y optimizador;
- rechazo de pérdidas sobre batch aleatorio como métrica de calidad.

## 5. Resultados de ingeniería

Un resultado inicial a 427M y contexto 2048 parecía mostrar una ventaja enorme, pero fue rechazado porque el Transformer probablemente usó un backend ineficiente y porque el checkpointing no era simétrico. Esa decisión es importante: una arquitectura no puede ganar porque el baseline esté roto.

El resultado más fuerte del registro preservado informa:

| Métrica | EIDOS Sparse16 | Transformer |
|---|---:|---:|
| Throughput | 5.229 tok/s | 4.049 tok/s |
| Pico asignado | 4,591 GiB | 4,604 GiB |
| Pico reservado | 5,084 GiB | 5,279 GiB |

Ratio: **1,291x**. Memoria reservada: **3,7% menor**.

Pero el código exacto de esa rama no está completo en el archivo conservado. Por tanto, el resultado se presenta como evidencia interna fuerte, no como prueba pública cerrada.

## 6. Resultados de calidad

La ventaja de calidad se redujo al entrenar más:

| Presupuesto | Resultado relativo |
|---|---|
| 0,1 tokens/parámetro | EIDOS mejor por 0,166 bits/token |
| 0,5 tokens/parámetro | EIDOS mejor por 0,0172 bits/token |
| 2,0 tokens/parámetro | evaluación final no cerrada |

La señal existe, pero no es suficiente para afirmar superioridad general.

## 7. Riesgos y amenazas de validez

- Las mediciones dependen de una RTX 5070 y un stack Windows/PyTorch concreto.
- Falta recuperar el código exacto de la rama Sparse16 avanzada.
- Los presupuestos pequeños no predicen por sí solos el comportamiento a escala grande.
- Un corpus bilingüe puede favorecer unas inductive biases frente a otras.
- El backend de atención puede cambiar totalmente la comparación.
- La reducción de atención global puede fallar en tareas que requieren recuperación exacta frecuente.

## 8. Próximos experimentos necesarios

Antes de afirmar algo fuerte, hacen falta:

1. recuperar o reconstruir la rama Sparse16 + token-shift + FlexAttention;
2. publicar hashes y JSON crudos;
3. repetir tres semillas a mayor presupuesto;
4. evaluar en conjunto fijo y test sellado;
5. hacer ablations: sin memoria, sin token-shift, más/menos atención, solo local;
6. probar long-context, código, SQL y razonamiento verificable;
7. reproducir en Linux y otra GPU;
8. medir energía y tiempo hasta calidad.

## 9. Conclusión

El proyecto invalidó sus primeras versiones y produjo una arquitectura híbrida más plausible. La evidencia demuestra que SAPPHO era peor en velocidad que un Transformer. La evidencia posterior sugiere que EIDOS Sparse16 puede ofrecer ventaja de ingeniería en contexto largo, pero todavía no demuestra superioridad general de calidad.

EIDOS debe presentarse como una oportunidad seria de reproducción e investigación, no como un sustituto demostrado del Transformer.