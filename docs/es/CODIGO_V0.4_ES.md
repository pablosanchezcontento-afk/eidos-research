# EIDOS v0.4

**Efficient Innovation-Driven Ordered State**

EIDOS es un stack de investigacion para comprobar si un hibrido de memoria predictiva, convolucion causal y atencion periodica puede superar a un Transformer bajo el mismo presupuesto.

## Veredicto actual

EIDOS **no esta declarado ganador**. SAPPHO v0.3 alcanzo solo el 48,2% del throughput del Transformer en la RTX 5070 compilada. EIDOS v0.4 cambia la topologia para eliminar ese trabajo duplicado y debe volver a medirse.

## Arquitectura

Macrociclo principal:

```text
Causal Conv -> Pulse Memory -> Causal Conv -> Global Attention Anchor
```

La memoria usa:

- estado de bajo rango;
- innovacion frente a la prediccion anterior;
- presupuesto de escritura condicionado por sorpresa;
- tres escalas aprendibles ordenadas;
- lectura y escritura independientes;
- scan por chunks compatible con `torch.compile`;
- estado constante durante generacion.

## Familia

| Perfil | Parametros | Funcion |
|---|---:|---|
| EIDOS Glyph | 11,05M | instrumento, GPU y primera comparacion |
| EIDOS Verse | 26,22M | repeticion y calidad por tiempo |
| EIDOS Epic | 98,48M | escala, codigo y contexto |
| EIDOS Mythos | 173,81M | bloqueado hasta ganar todas las puertas |

## Primer paso en la RTX 5070

En PowerShell, dentro de esta carpeta:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\benchmark_eidos_5070.ps1
```

Compara:

- `eidos_core`
- `eidos_wide_memory`
- `eidos_dual_anchor`
- `eidos_pulse`
- `transformer_full`
- `conv_striped`
- `recurrent_striped`

El benchmark mide velocidad y VRAM, no inteligencia. El campo de loss aleatorio esta etiquetado como no interpretable.

## Puerta obligatoria

Una variante EIDOS solo avanza si la misma candidata:

1. supera el baseline uniforme;
2. usa al menos 20 tokens por parametro;
3. tiene al menos tres semillas;
4. mantiene gap de parametros <=0,5%;
5. vence al mejor baseline en test sellado por >=0,5%;
6. alcanza >=1,05x el throughput de Transformer en la RTX 5070.

`experiments/evaluate_gate.py` aplica estas reglas y es el unico mecanismo que puede desbloquear Mythos.

## Tests

```powershell
python -m pip install -e .
python -m pytest -q
```

La entrega se construyo con 16 tests superados.

## Informes

- `reports/EIDOS_v0.4_Informe_Tecnico.pdf`
- `reports/EIDOS_v0.4_Explicado_Facil.pdf`

## Limitacion principal

El diagnostico CPU sugiere que EIDOS Core ya no arrastra la penalizacion de SAPPHO, pero la calidad y el rendimiento CUDA aun no estan demostrados. No inicies Mythos antes de ejecutar Glyph y aplicar la puerta.
