# Comparación verificada entre fase 1 y fase 2

## Estado verificado al 20 de octubre

La fase 2 terminó el 20 de octubre a las 06:52:06 de Bogotá, completó las catorce arquitecturas de la referencia y cinco controles, diecinueve configuraciones con dos épocas adicionales completas y evaluación de ambas ramas en las dos resoluciones, no hay entrenamientos ni evaluaciones de esta cola pendientes y esta revisión solo consulta archivos y corrige documentación en CPU.

El tiempo desde el inicio de la tarea fue 6.330 horas, incluido el análisis previo, y desde el lanzamiento independiente 6.013 horas, el cierre quedó dentro del plazo global de diez horas, los códigos de salida de GPU y consolidación CPU fueron cero, esa salida no ocultó el fallo separado del generador audiovisual que se documenta abajo.

La fase 1 conserva 54 configuraciones cortas y su comparación de catorce arquitecturas, las diecinueve ejecuciones de fase 2 son continuaciones de checkpoints válidos y no diecinueve inicializaciones independientes, los resultados históricos completos se conservan en resultados y el informe anterior se archiva en [docs/INFORME_FASE1_HISTORICO.md](../docs/INFORME_FASE1_HISTORICO.md).

## Qué cambió y qué se mantuvo

Se conservaron los HDF5, sus particiones, el entrenamiento sin identidades, los encoders, las pérdidas CO propias y el protocolo market_documentado con vectores L2, coseno, mAP y Rank-1/5/10, se mantuvo entrada 224 y batch físico ocho, no se incorporó validación al aprendizaje ni se añadieron entradas de texto o etiquetas.

La fase 2 agregó recomputación de la rama online para reducir memoria, una medición de la inicialización aleatoria en validación, pilotos de tasa y selección de época y rama con validación, continuó los pesos de fase 1 y reinició explícitamente optimizador y calendario EMA, por eso los cambios de rendimiento no pueden atribuirse solamente a entrenar más épocas ni solamente a la tasa.

Los pilotos escogieron tasa base CNN 1.6 y Transformer 0.0003, las tasas efectivas de referencia al escalar por batch/256 fueron 0.05 y 0.000009375, los pilotos de 400 pasos se descartaron y no se presentan como exposición de los modelos finales, sus seis ejecuciones sumaron 8.666 minutos, la búsqueda fue común por familia y no un ajuste individual exhaustivo.

Cada continuación recibió 2760 actualizaciones adicionales, sumadas a las 1253 iniciales son 4013 actualizaciones y 32104 visitas de imágenes fuente por configuración, esas visitas no son imágenes distintas, se alcanzaron aproximadamente 2.908 épocas equivalentes en total, el origen de datos sigue teniendo 11043 imágenes y se mantienen batches completos con drop_last.

Se planificaron hasta tres épocas adicionales y se realizaron dos para todas las configuraciones, a las 06:13 se decidió no abrir la tercera ronda porque la estimación conservadora era 204.285 minutos más una reserva de 90, frente a 258.525 minutos restantes, no fue una caída del proceso ni una reducción selectiva para una arquitectura.

## Modelo seleccionado y mejora frente a la fase 1

La selección global por validación fue co_byol_bagnet33, rama objetivo, después de dos épocas adicionales, con mAP media de validación 10.327 por ciento, la prueba se abrió después de congelar esa elección, el resultado de prueba no se usa para cambiar modelo ni rama.

| Referencia | Rama | mAP original % | mAP LR % | R1 original % | R1 LR % |
|---|---|---:|---:|---:|---:|
| Fase 1, selección histórica inicial BagNet33 | online | 1.587 | 1.334 | 4.632 | 3.385 |
| Fase 1, selección ampliada ResNet18 | online | 1.861 | 1.432 | 4.394 | 2.672 |
| Fase 2, selección global BagNet33 | objetivo | 2.167 | 2.014 | 5.909 | 4.662 |

Frente al ResNet18 elegido en la comparación ampliada de fase 1, la nueva elección aumenta mAP original 0.306 puntos porcentuales y mAP LR 0.583 puntos, la caída relativa de BagNet33 de fase 2 es 7.056 por ciento, una mejora relativa no equivale a precisión alta en términos absolutos.

La comparación entre fases incluye una política distinta de elección de rama frente a la tabla original que fijaba online, se conserva esa tabla intacta y se ofrecen también las dos comparaciones auxiliares, una aplica la misma regla de mejor rama por validación a los archivos de cada fase y otra mantiene cada rama fija, son análisis retrospectivos de resultados existentes y no reescriben las selecciones históricas.

Manteniendo la misma rama objetivo de BagNet33, mAP original pasa de 1.369 a 2.167 por ciento y LR de 1.179 a 2.014 por ciento, la mejora tampoco desaparece al mantener fija esa rama, el seleccionado actual acierta en Rank-1 en 199 de 3368 consultas originales y 157 de 3368 reducidas, las figuras equilibradas de cinco aciertos y cinco fallos no representan una tasa de acierto del cincuenta por ciento.

## Las catorce arquitecturas, comparación homogénea de selección

La fase 1 de esta tabla usa la rama con mayor mAP media de validación de cada configuración, la fase 2 usa la época y rama congeladas por validación, todas las continuaciones recibieron dos épocas adicionales aunque el checkpoint elegido pueda ser anterior, delta media se expresa en puntos porcentuales y no demuestra un efecto causal aislado.

| Arquitectura | Rama F1 | mAP O F1 % | mAP LR F1 % | Rama F2 | Época extra elegida | mAP O F2 % | mAP LR F2 % | Delta media pp | Min train F2 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| resnet18 | online | 1.861 | 1.432 | online | 0 | 1.861 | 1.432 | +0.000 | 9.276 |
| resnet50 | objetivo | 0.331 | 0.241 | objetivo | 2 | 0.605 | 0.502 | +0.267 | 7.555 |
| rf29_resnet18 | objetivo | 1.357 | 0.912 | online | 1 | 1.331 | 0.810 | -0.064 | 30.644 |
| rf39_resnet50 | objetivo | 0.805 | 0.252 | online | 2 | 0.788 | 0.363 | +0.047 | 91.151 |
| rf67_resnet50 | objetivo | 0.586 | 0.386 | objetivo | 2 | 0.960 | 0.591 | +0.289 | 16.789 |
| rf99_resnet50 | online | 0.521 | 0.367 | online | 1 | 1.324 | 1.140 | +0.788 | 12.322 |
| rf115_resnet50 | online | 0.542 | 0.294 | objetivo | 2 | 0.747 | 0.321 | +0.116 | 12.032 |
| rf163_resnet50 | online | 0.369 | 0.238 | objetivo | 2 | 0.509 | 0.331 | +0.117 | 11.896 |
| rf195_paper193_resnet50 | online | 0.305 | 0.212 | online | 2 | 0.580 | 0.311 | +0.187 | 12.075 |
| bagnet33 | online | 1.587 | 1.334 | objetivo | 2 | 2.167 | 2.014 | +0.630 | 10.509 |
| vit_s16 | online | 0.630 | 0.581 | online | 2 | 0.736 | 0.676 | +0.101 | 7.237 |
| swin_t | online | 0.689 | 0.616 | online | 2 | 1.296 | 1.193 | +0.592 | 8.983 |
| swin_s | online | 0.682 | 0.599 | objetivo | 2 | 1.142 | 1.086 | +0.473 | 13.822 |
| swin_b | online | 0.587 | 0.522 | objetivo | 2 | 0.850 | 0.802 | +0.271 | 15.171 |

Doce arquitecturas mejoran su mAP media observada, ResNet18 conserva exactamente su checkpoint de fase 1 porque sus nuevas épocas no superaron la validación inicial, RF29 empeora la media de prueba porque su caída en consultas reducidas supera la pequeña mejora en originales, no se eligió otro checkpoint mirando prueba para corregir ese resultado.

Las ocho métricas originales y LR, ambas caídas, configuraciones, tiempos y memoria de las diecinueve ejecuciones están en [resultados_fase2/comparacion_fase2.csv](comparacion_fase2.csv), las dos ramas del checkpoint elegido están en comparacion_ramas_fase2.csv, los deltas y la comparación manteniendo rama fija están en comparacion_fase1_fase2.csv y comparacion_fase1_fase2_misma_rama.csv.

## Calidad frente a coste computacional

BagNet33 ofrece la mayor mAP media entre las catorce referencias elegidas por validación y el mayor cociente descriptivo de mAP media por minuto de entrenamiento acumulado de ambas fases, recibió 10.509 minutos adicionales y 13.366 minutos acumulados, con 2.586 GiB de memoria máxima asignada por PyTorch durante fase 2, el coste de búsqueda compartido se informa aparte, ese cociente no mide energía.

RF39 requirió 91.151 minutos adicionales y 132.632 minutos acumulados con mAP original 0.788 y LR 0.363, ese mayor coste no se tradujo en la mayor precisión del panel, RF29 redujo el pico de memoria de entrenamiento registrado de 10.057 a 1.643 GiB mediante la nueva recomputación, RF39 pasó de 5.512 a 4.765 GiB, estas cifras se refieren a memoria asignada por PyTorch y no al consumo total del dispositivo.

Las diecinueve continuaciones sumaron 302.644 minutos de bucle de entrenamiento, la prueba sumó 37.951 minutos y las validaciones iniciales y por época 7.827 minutos, esta última cifra excluye inicializaciones aleatorias y pilotos, la suma parcial no reemplaza el tiempo global de 379.798 minutos que incluye preparación, serialización y cierre.

## Bases y ablaciones, incluidos resultados desfavorables

| Comparación | Base, época/rama | CO, época/rama | mAP O base % | mAP O CO % | Delta O pp | mAP LR base % | mAP LR CO % | Delta LR pp |
|---|---|---|---:|---:|---:|---:|---:|---:|
| BYOL ResNet50 | 1/objetivo | 2/objetivo | 0.903 | 0.605 | -0.298 | 0.892 | 0.502 | -0.389 |
| BYOL RF99 | 2/objetivo | 1/online | 0.552 | 1.324 | +0.771 | 0.353 | 1.140 | +0.787 |
| BYOL BagNet33 | 1/objetivo | 2/objetivo | 2.017 | 2.167 | +0.151 | 1.889 | 2.014 | +0.126 |
| MoCoV3 ResNet50 | 2/objetivo | 2/online | 0.933 | 1.909 | +0.976 | 0.741 | 1.675 | +0.935 |

CO mejora las dos condiciones en RF99, BagNet33 y la pareja MoCoV3 ResNet50 de esta ejecución, pero empeora ambas frente a BYOL sin CO en ResNet50, cada variante usa su selección propia de validación y por eso la comparación práctica incluye diferencias de época y a veces de rama, no debe describirse como un efecto aislado a checkpoint idéntico, las ablationes históricas y sus límites permanecen documentados en fase 1.

## Inicialización aleatoria, una comprobación que faltaba

| Arquitectura | mAP media val aleatoria % | mAP media val elegida % | Delta pp |
|---|---:|---:|---:|
| resnet18 | 5.805 | 9.385 | +3.580 |
| resnet50 | 4.831 | 4.096 | -0.735 |
| rf29_resnet18 | 5.181 | 6.422 | +1.240 |
| rf39_resnet50 | 4.836 | 3.942 | -0.894 |
| rf67_resnet50 | 4.898 | 4.847 | -0.051 |
| rf99_resnet50 | 4.932 | 7.284 | +2.351 |
| rf115_resnet50 | 4.081 | 4.214 | +0.133 |
| rf163_resnet50 | 4.549 | 3.308 | -1.241 |
| rf195_paper193_resnet50 | 4.170 | 2.851 | -1.319 |
| bagnet33 | 8.453 | 10.327 | +1.874 |
| vit_s16 | 3.467 | 4.214 | +0.748 |
| swin_t | 3.626 | 6.357 | +2.731 |
| swin_s | 3.712 | 6.336 | +2.624 |
| swin_b | 3.912 | 4.809 | +0.897 |

ResNet50, RF39, RF67, RF163 y RF195 conservan una puntuación observada inferior a su inicialización aleatoria en validación, RF67 difiere solo 0.051 puntos, no se ha realizado una prueba de significación ni una evaluación de estas inicializaciones en testing, la inicialización aleatoria no es un ranking aleatorio y puede producir representaciones útiles sin aprendizaje, estos datos impiden afirmar una mejora general de SSL en todas las arquitecturas.

BagNet33 pasa de 8.453 a 10.327 por ciento medio de validación frente a su inicialización aleatoria, una mejora observada de 1.874 puntos, en el piloto RF99 el cambio relativo del primer tensor fue aproximadamente 0.088 por ciento con tasa base 0.2 y 0.772 por ciento con tasa 1.6, se midió un tensor y no toda la red, no queda demostrado que BatchNorm sea la causa del rendimiento ni que toda red de fase 1 fuera equivalente a no entrenar.

## Integridad y alcance de esta revisión

Se contrastaron las 152 métricas del CSV seleccionado con sus JSON de prueba, se verificaron los 378 hashes de registros históricos protegidos y las huellas de los trece archivos de código ejecutado, todas las configuraciones conservan batch ocho, 2760 pasos adicionales, dos épocas completas y protocolo de evaluación común, no se recalcularon embeddings, métricas mediante inferencia ni gradientes durante esta revisión.

Se conservaron sesion_independiente.json, verificacion_fase2.json y errores.log del cierre original, el estado completado de esos archivos acredita el panel experimental y convive con el fallo audiovisual que registran, una revisión posterior de documentación no sustituye esos registros ni cambia los pesos o las selecciones.

La selección por validación antecede a la prueba de fase 2, pero la fase se diseñó después de existir pruebas históricas, no constituye un segundo conjunto ciego, no hay evidencia de ajuste con testing ni se propone ahora entrenar para mejorar las filas peores.

## Estado audiovisual y publicación de la entrega

El video de fase 2 se corrigió solo en CPU, el fallo inicial fue ModuleNotFoundError de matplotlib en el entorno de voz, el gráfico se genera ahora con el entorno de análisis que ya lo tiene instalado y la narración y codificación siguen en su entorno CPU, la versión nueva dura 717.56 segundos y pasó decodificación completa y límite por capítulo, se conserva el video histórico de fase 1.

La versión actual y su guion están en [resultados_fase2/audiovisual/VIDEO.mp4](audiovisual/VIDEO.mp4) y [resultados_fase2/audiovisual/GUION_VIDEO.md](audiovisual/GUION_VIDEO.md), el grupo debe revisar la narración sintética y los ejemplos visuales antes de la entrega.

Las figuras incluyen cinco aciertos y cinco fallos por condición, durante esta revisión se inspeccionaron visualmente los veinte casos y se añadieron observaciones concretas sobre prendas, punto de vista, mochila y pérdida de detalle LR en ANALISIS_ERRORES_FASE2.md, esos rasgos visibles apoyan hipótesis de confusión pero no demuestran qué atributo determinó el encoder, la revisión del grupo sigue pendiente.

Persisten una semilla, solo unas 2.908 épocas equivalentes, cien consultas de validación, ausencia de convergencia y de intervalos de incertidumbre, no se reprodujeron los calendarios largos del paper, Places365, la igualación literal de parámetros de BagNet33 ni todos los barridos, los 44 análisis históricos de máscaras, PGD, campos efectivos y lecturas no se repitieron con los pesos nuevos y no deben presentarse como evidencia de fase 2, se mantienen como diferencias declaradas y no como requisitos básicos de ejecución que falten en el panel acordado.

La entrega consolidada de ambas fases se publica en [entrega-v2](https://github.com/juanvasquez-ship-it/Lab-3-deep-learning/releases/tag/entrega-v2), incluye ZIP actualizado, guía PDF, resultados y video de fase 2 y preserva la release anterior, el ZIP añade los encoders seleccionados de fase 2 y declara los pesos omitidos, los registros previos de revisión mantienen su estado histórico de publicación pendiente, revisar PUBLICACION_FASE2.md para las evidencias actuales, siguen pendientes la revisión del grupo, el envío al formulario y la defensa individual.

El contraste de requisitos se conserva en resultados_fase2/REVISION_CIERRE.md, los experimentos básicos del laboratorio tienen evidencia y las actividades humanas y la reproducción experimental íntegra del paper continúan separadas del estado completado de los entrenamientos.
