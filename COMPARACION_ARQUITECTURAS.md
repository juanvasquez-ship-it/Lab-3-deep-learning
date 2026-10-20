# Comparación común de las catorce arquitecturas

Este panel fija CO-BYOL con peso local 0.2, rama online, entrada 224, batch físico y efectivo ocho y 1253 actualizaciones, conserva particiones, semilla, inicialización aleatoria y transformaciones comunes, la resolución original y reducida se leen de los HDF5 suministrados y se evalúan con la misma galería original.

Se reutilizan diez configuraciones existentes y se programan cuatro para completar el panel a resolución común, ResNet18 y RF29 a 64 se conservan como experimentos del panel pequeño del artículo y no se mezclan en la clasificación de esta tabla, el control BYOL sin CO con ResNet50 y las ablaciones con arquitectura fija ya realizadas se conservan y no se repiten.

Los kernels y las dimensiones de las cabezas locales dependen del backbone como en el artículo y producen cantidades de posiciones distintas, CNN y Transformers conservan sus optimizadores y tasas documentados, BagNet33 mantiene su diferencia de parámetros y el bajo número de actualizaciones no permite concluir convergencia ni diferencias estadísticamente concluyentes.

RF39 usa recomputación anidada de activaciones para conservar batch ocho, resolución y todas sus posiciones locales sin duplicar las actualizaciones de estadísticas BatchNorm, el mecanismo tiene una verificación CPU dirigida con salidas, gradientes y buffers coincidentes, aumenta el tiempo y su comparación de coste incorpora ese compromiso del equipo.

Estado actual, 14 de catorce arquitecturas evaluadas bajo este protocolo, las celdas pendientes no representan resultados medidos.

## Métricas de prueba

| Arquitectura | Estado | mAP O % | R1 O % | R5 O % | R10 O % | mAP LR % | R1 LR % | R5 LR % | R10 LR % | Caída pp | Caída relativa % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| resnet18 | evaluado | 1.861 | 4.394 | 9.531 | 13.302 | 1.432 | 2.672 | 7.037 | 10.570 | 0.429 | 23.062 |
| resnet50 | evaluado | 0.324 | 0.445 | 2.227 | 3.593 | 0.209 | 0.267 | 1.306 | 1.989 | 0.115 | 35.422 |
| rf29_resnet18 | evaluado | 1.326 | 3.236 | 7.482 | 11.075 | 0.899 | 1.752 | 4.929 | 6.829 | 0.427 | 32.172 |
| rf39_resnet50 | evaluado | 0.725 | 2.108 | 5.285 | 8.224 | 0.237 | 0.238 | 1.158 | 2.227 | 0.488 | 67.277 |
| rf67_resnet50 | evaluado | 0.649 | 1.485 | 3.504 | 5.938 | 0.408 | 0.534 | 1.841 | 3.118 | 0.240 | 37.047 |
| rf99_resnet50 | evaluado | 0.521 | 0.713 | 2.643 | 4.394 | 0.367 | 0.356 | 1.663 | 2.643 | 0.153 | 29.405 |
| rf115_resnet50 | evaluado | 0.542 | 0.861 | 3.474 | 5.790 | 0.294 | 0.505 | 1.752 | 3.088 | 0.248 | 45.787 |
| rf163_resnet50 | evaluado | 0.369 | 0.505 | 2.346 | 3.711 | 0.238 | 0.208 | 0.950 | 1.871 | 0.131 | 35.426 |
| rf195_paper193_resnet50 | evaluado | 0.305 | 0.356 | 1.336 | 2.257 | 0.212 | 0.148 | 0.772 | 1.306 | 0.093 | 30.577 |
| bagnet33 | evaluado | 1.587 | 4.632 | 10.125 | 13.925 | 1.334 | 3.385 | 7.601 | 10.243 | 0.253 | 15.921 |
| vit_s16 | evaluado | 0.630 | 1.069 | 4.186 | 6.740 | 0.581 | 0.861 | 3.207 | 5.760 | 0.049 | 7.761 |
| swin_t | evaluado | 0.689 | 1.158 | 4.899 | 7.928 | 0.616 | 0.891 | 3.800 | 6.502 | 0.073 | 10.652 |
| swin_s | evaluado | 0.682 | 1.544 | 5.255 | 8.017 | 0.599 | 1.099 | 4.038 | 6.799 | 0.083 | 12.171 |
| swin_b | evaluado | 0.587 | 1.099 | 3.919 | 6.770 | 0.522 | 0.891 | 3.444 | 5.701 | 0.066 | 11.174 |

O corresponde a consultas originales y LR a las versiones reducidas, una caída negativa indica mejora en esa condición y no se recorta a cero, cada fila utiliza online de la configuración predeclarada y las dos ramas completas permanecen disponibles en sus JSON.

## Costes medidos y cobertura

| Arquitectura | Configuración | Parámetros encoder M | Entrenamiento min | Validación s | Prueba s | VRAM entrenamiento GiB | Batch efectivo | Actualizaciones |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| resnet18 | co_byol_resnet18_224 | 11.169 | 3.374 | 9.174 | 117.678 | 2.940 | 8 | 1253 |
| resnet50 | co_byol_r50 | 23.508 | 1.568 | No registrado | No registrado | 1.633 | 8 | 1253 |
| rf29_resnet18 | co_byol_rf29_224 | 7.525 | 10.932 | 26.263 | 307.261 | 10.057 | 8 | 1253 |
| rf39_resnet50 | co_byol_rf39_resnet50 | 23.750 | 41.481 | 71.811 | 1075.515 | 5.512 | 8 | 1253 |
| rf67_resnet50 | co_byol_rf67_resnet50 | 23.750 | 3.912 | No registrado | No registrado | 5.795 | 8 | 1253 |
| rf99_resnet50 | co_byol_rf99 | 23.750 | 2.456 | No registrado | No registrado | 3.549 | 8 | 1253 |
| rf115_resnet50 | co_byol_rf115_resnet50 | 23.750 | 2.801 | 5.299 | 101.769 | 2.978 | 8 | 1253 |
| rf163_resnet50 | co_byol_rf163_resnet50 | 23.750 | 1.456 | No registrado | No registrado | 1.784 | 8 | 1253 |
| rf195_paper193_resnet50 | co_byol_rf195_paper193_resnet50 | 23.750 | 1.486 | No registrado | No registrado | 1.643 | 8 | 1253 |
| bagnet33 | co_byol_bagnet33 | 16.261 | 2.858 | No registrado | No registrado | 4.096 | 8 | 1253 |
| vit_s16 | co_byol_vit_s16 | 21.666 | 1.509 | No registrado | No registrado | 2.516 | 8 | 1253 |
| swin_t | co_byol_swin_t | 27.519 | 2.050 | No registrado | No registrado | 2.594 | 8 | 1253 |
| swin_s | co_byol_swin_s | 48.837 | 3.581 | No registrado | No registrado | 3.581 | 8 | 1253 |
| swin_b | co_byol_swin_b | 86.743 | 3.686 | No registrado | No registrado | 4.975 | 8 | 1253 |

El CSV conserva también exposición, épocas equivalentes, espera y preparación cuando fueron medidas, los tiempos históricos de validación y prueba que no se registraron por separado permanecen pendientes de medición y no se reconstruyen restando intervalos con alcances diferentes, no se repiten experimentos para llenar esas columnas.

El pico de VRAM corresponde a memoria asignada por PyTorch durante entrenamiento, no es el consumo total del sistema ni una medición de evaluación, los tiempos son de pared e incluyen las transferencias y los guardados periódicos dentro del bucle, la memoria libre y las demás aplicaciones del equipo pueden afectar la comparación temporal.

El hardware de la sesión nueva se conserva en resultados/hardware_comparacion.json y su presupuesto independiente en presupuesto_comparacion.json, el presupuesto anterior permanece sin cambios, todos los modelos comparables deben conservar el mismo hash del conjunto de índices fuente visitados y eso no demuestra por sí solo identidad de toda la secuencia de aumentos.

## Interpretación de resultados

La mayor media descriptiva de mAP entre condiciones en prueba corresponde a resnet18 con 1.646 por ciento, su mAP original es 1.861 y la reducida es 1.432, esta ordenación describe resultados y no se utiliza para ajustar hiperparámetros ni sustituir la selección por validación.

La menor caída relativa corresponde a vit_s16 con 7.761 por ciento y mAP reducida 0.581, retener un porcentaje alto de un rendimiento inicial bajo no equivale a obtener la mayor precisión absoluta y la tabla permite distinguir ambos criterios.

El mayor cociente descriptivo entre media de mAP porcentual y minutos de entrenamiento corresponde a bagnet33, el frente no dominado en mAP media y tiempo incluye resnet18, rf163_resnet50, bagnet33, vit_s16, swin_t, ese cociente no representa energía medida ni una ley de escalado y no obliga a escoger un único ganador.

Las diferencias pueden relacionarse con campo receptivo, capacidad, geometría de las personas y coste de memoria, estas explicaciones son hipótesis salvo los efectos aislados por las ablaciones existentes, una sola semilla y menos de una época no permiten presentar diferencias pequeñas como concluyentes.

## Comparaciones complementarias y límites

Los resultados de modelo base y ablaciones se conservan en INFORME.md y metricas.csv, el contraste adicional de RF99 con recorte mínimo 0.9 usa control sin CO y variante CO bajo las mismas actualizaciones y se registra por separado, esos dos experimentos tienen prioridad posterior a completar las arquitecturas y no se incorporan como filas intercambiables del panel con recorte 0.2.

La selección ampliada se registra en seleccion_comparacion.json exclusivamente con validación y sin modificar seleccion_validacion.json del experimento histórico, esta ampliación se ha solicitado después de existir pruebas anteriores y no representa una nueva evaluación completamente ciega, se mantienen fijas las configuraciones y no se eligen por sus puntuaciones de prueba.

La matriz histórica ampliada mantiene combinaciones no ejecutadas fuera de esta cola mínima, eso no equivale a arquitecturas pendientes del panel, los calendarios largos, Places365 y la igualación literal de parámetros de BagNet33 conservan sus límites declarados y no se anuncian como reproducción completa del artículo.

## Control de recorte mínimo 0.9 completado

| Peso local RF99 | mAP original % | mAP reducida % |
|---|---:|---:|
| 0 | 0.522 | 0.447 |
| 0.2 | 0.382 | 0.297 |

Ambos controles completaron entrenamiento y evaluación y se mantienen separados de la tabla con recorte mínimo 0.2, en esta ejecución RF99 con CO obtiene menor mAP que el control sin CO en las dos condiciones, este contraste no respalda una mejora local sin recortes pequeños y no se extrapola a los calendarios largos de clasificación del artículo.
