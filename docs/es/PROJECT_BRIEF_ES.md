# EIDOS - Resumen público

## Idea

EIDOS intenta mejorar la eficiencia de entrenamiento de modelos de lenguaje reduciendo la frecuencia de atención global. En lugar de usar atención completa en todas las capas, combina:

1. mezcladores causales locales;
2. memoria predictiva de bajo rango;
3. anclas globales de atención.

## Por qué importa

La atención global es potente pero costosa. Si una arquitectura puede usarla menos sin perder calidad, podría permitir modelos más grandes, contextos más largos y entrenamiento local más viable.

## Estado actual

EIDOS ha mostrado una ventaja de ingeniería en una RTX 5070: el perfil 1.093B / 4096 alcanzó 1,291x de throughput frente a un Transformer emparejado.

La calidad sigue abierta. Las ventajas iniciales se estrecharon al aumentar el presupuesto de entrenamiento.

## Para empresas y colaboradores

El valor del proyecto hoy está en:

- exploración arquitectónica real;
- ingeniería GPU y perfilado;
- metodología de falsación;
- publicación de resultados negativos;
- posible camino hacia modelos locales más eficientes.

## Próximo hito

Completar evaluación de calidad con conjunto fijo, ablations, tres semillas y reproducción externa antes de cualquier afirmación fuerte.
