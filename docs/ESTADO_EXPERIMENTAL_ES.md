# Estado experimental de EIDOS

## Veredicto corto

EIDOS tiene evidencia de eficiencia interesante, pero no tiene todavía evidencia suficiente de superioridad general frente a Transformers.

La posición honesta es:

> La dirección arquitectónica merece reproducción y más experimentos, pero no debe presentarse como arquitectura ganadora hasta cerrar calidad, ablations y código reproducible.

## Resultados confirmados directamente en archivos preservados

### SAPPHO v0.3

SAPPHO comparó una rama de memoria predictiva con un Transformer emparejado. En una RTX 5070:

| Modo | SAPPHO | Transformer | Ratio |
|---|---:|---:|---:|
| Sin compile | 4.131 tok/s | 11.714 tok/s | 0,353x |
| Con compile | 12.141 tok/s | 25.171 tok/s | 0,482x |

Conclusión: SAPPHO fue descartada como ruta eficiente. La compilación ayudó mucho, pero la arquitectura seguía siendo aproximadamente dos veces más lenta que el Transformer.

## Resultados documentados en registro de ejecución

### EIDOS Sparse16 1.093B

El registro preservado describe una medición aislada con:

- aproximadamente 1.093B parámetros;
- contexto 4096;
- BF16;
- Adafactor;
- checkpointing simétrico;
- comparación contra Transformer emparejado.

Resultado reportado:

| Modelo | Throughput | Pico asignado | Pico reservado |
|---|---:|---:|---:|
| EIDOS Sparse16 | 5.229 tok/s | 4,591 GiB | 5,084 GiB |
| Transformer | 4.049 tok/s | 4,604 GiB | 5,279 GiB |

Conclusión provisional: ventaja de ingeniería de 1,291x en throughput y 3,7% menos memoria reservada.

Limitación: el código exacto y todos los JSON crudos de esa rama no están completos en el archivo fuente preservado. Por tanto, el resultado necesita reproducción.

## Calidad

La comparación de calidad favoreció a EIDOS al principio, pero la ventaja se redujo al entrenar más:

| Presupuesto | Resultado |
|---|---|
| 0,1 tokens/parámetro | EIDOS mejor por 0,166 bits/token |
| 0,5 tokens/parámetro | EIDOS mejor por 0,0172 bits/token |
| 2,0 tokens/parámetro | comparación final no cerrada |

Interpretación: hay señal, pero todavía demasiado estrecha. No justifica afirmar que EIDOS sea más inteligente que un Transformer.

## Qué significa esto

EIDOS no está en estado de producto. Está en estado de investigación prometedora:

- la ruta de eficiencia parece más fuerte que la ruta de calidad;
- la calidad necesita más entrenamiento, más semillas y evaluación fija;
- la prioridad es reproducibilidad antes de escalar;
- un resultado negativo también sería útil y publicable como aprendizaje técnico.

## Próxima puerta de decisión

La próxima versión debe pasar esta puerta:

1. código exacto recuperado o reconstruido;
2. tests verdes;
3. tres semillas;
4. mismo tokenizer, corpus, tokens y optimizador;
5. evaluación fija grande;
6. test sellado al final;
7. JSON crudos publicados;
8. comparación contra Transformer y baseline convolucional fuerte.

Solo si pasa eso se debería lanzar un entrenamiento mayor de 180M, 450M o 1B con ambición de modelo útil.