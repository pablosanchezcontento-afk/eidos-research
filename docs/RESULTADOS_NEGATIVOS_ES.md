# Resultados negativos y errores importantes

Este documento existe para dejar claro que el proyecto no es una presentación de marketing. La investigación avanzó precisamente porque varias conclusiones atractivas fueron rechazadas.

## 1. NOVA v0.1: barrido de learning rate injusto

El resultado inicial parecía indicar que NOVA cruzaba al baseline convolucional al crecer. La auditoría posterior encontró que NOVA tuvo más learning rates probados que el baseline.

Consecuencia: el supuesto cruce no podía considerarse demostrado. La variación causada por el learning rate era mayor que la ventaja arquitectónica medida.

Lección: ningún resultado pequeño debe aceptarse si el baseline recibió menos búsqueda de hiperparámetros.

## 2. Ablations con parámetros no igualados

Algunas ablations tenían alrededor de 3% menos parámetros que la variante completa. La mejora atribuida al mecanismo propio era menor que ese margen y menor que el ruido entre semillas.

Consecuencia: no se podía afirmar que el mecanismo fuese la causa de la mejora.

Lección: las ablations deben igualar parámetros, FLOPs y tiempo, no solo nombres de módulos.

## 3. Bug silencioso del scan por chunks

La formulación antigua del scan usaba un clamp dentro de una división por productos acumulados. Con chunks grandes, el estado se corrompía sin producir NaN ni warning.

Consecuencia: un entrenamiento largo podría haber quedado invalidado sin señal visible.

Lección: los scans recurrentes deben verificarse contra un oráculo secuencial y probar chunks, decays extremos, precisión y backward.

## 4. BPE con modelos peor que uniforme

Algunas matrices BPE tenían pérdidas peores que `log2(vocabulario)`, es decir, peores que una distribución uniforme. Eso no mide inteligencia; mide mala optimización o divergencia.

Consecuencia: varias tablas iniciales no debían usarse para afirmar ranking arquitectónico.

Lección: si un modelo es peor que tirar dados sobre el vocabulario, la corrida debe marcarse como fallida.

## 5. SAPPHO fue demasiado lenta

SAPPHO mejoró con `torch.compile`, pero siguió en 0,482x frente al Transformer compilado. El diseño hacía atención y memoria a la vez, duplicando trabajo.

Consecuencia: SAPPHO quedó descartada como arquitectura eficiente.

Lección: una idea de memoria interesante no vale si destruye calidad por segundo.

## 6. Resultados grandes rechazados por backend injusto

Un resultado intermedio parecía mostrar una ventaja enorme a contexto 2048, pero fue rechazado al detectar que el Transformer probablemente usó un backend de atención ineficiente y checkpointing desigual.

Consecuencia: la cifra no se usó como evidencia.

Lección: una arquitectura no gana porque el baseline esté mal configurado.

## 7. Código incompleto de la rama más prometedora

La rama Sparse16 + token-shift + FlexAttention produjo el registro de eficiencia más fuerte, pero el código exacto posterior a v0.4 no quedó preservado en el paquete fuente.

Consecuencia: los resultados son evidencia interna pendiente de reproducción, no una prueba pública completa.

Lección: todo experimento importante debe guardar commit, JSON crudos, configuración, hash del corpus y entorno.

## Conclusión

Los resultados negativos no hacen peor el proyecto. Lo hacen más creíble. Una línea de investigación que publica sus fallos permite que otros entiendan qué se probó, qué se descartó y dónde puede estar la siguiente mejora real.