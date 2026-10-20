# Publicación consolidada de fase 2

La entrega vigente se encuentra en [GitHub](https://github.com/juanvasquez-ship-it/Lab-3-deep-learning/releases/tag/entrega-v2), incluye código y evidencia de ambas fases, la guía PDF actualizada y el ZIP con tres archivos compactos de encoders, conserva la release entrega-v1 y los documentos históricos, la publicación se preparó en CPU sin entrenar, sin inferencia de encoders y sin GPU.

## Resultado que debe evaluar la otra IA

Fase 1 contiene 54 configuraciones cortas, fase 2 continúa 19 checkpoints de ese conjunto con dos épocas adicionales completas, incluye las catorce arquitecturas acordadas y cinco controles, el total no equivale a 73 entrenamientos independientes ni a 73 arquitecturas, cada continuación recibió el mismo número de pasos adicionales y la elección de época y rama utiliza validación.

La elección global fue CO-BYOL BagNet33, rama objetivo, época adicional dos, mAP de prueba original 2.167244 por ciento y LR 2.014327 por ciento, Rank-1 original 5.908551 y LR 4.661520 por ciento, doce arquitecturas mejoran su mAP media observada, ResNet18 conserva el checkpoint inicial y RF29 empeora la media, los resultados son preliminares y cinco referencias siguen por debajo de su inicialización aleatoria en validación.

## Archivos vigentes y archivos históricos

| Elemento | Uso |
|---|---|
| INFORME.md | Informe consolidado, mejoras, resultados desfavorables, límites y pendientes humanos |
| docs/GUIA_EVALUACION_IA.pdf | Guía actualizada para revisión independiente de ambas fases |
| resultados_fase2/comparacion_fase2.csv | Diecinueve configuraciones, ocho métricas, caídas, épocas elegidas, tiempos y memoria |
| resultados_fase2/comparacion_fase1_fase2.csv | Comparación de catorce arquitecturas con la misma política de elección de rama |
| resultados_fase2/comparacion_fase1_fase2_misma_rama.csv | Comparación auxiliar manteniendo fija cada rama |
| resultados_fase2/audiovisual/VIDEO.mp4 | Video vigente de fase 2, 717.56 segundos |
| VIDEO.mp4 y docs/*FASE1_HISTORIC* | Video, informe, guion y guía anteriores conservados |
| resultados_fase2/revision_resultados_20261020.json | Auditoría previa a esta publicación, sus campos de publicación pendiente son históricos |
| resultados_fase2/codigo_ejecucion_SHA256.json | Trece fuentes congeladas antes de ejecutar la cola, no modificadas en esta publicación |
| CONTENIDO_PAQUETE.json | Hashes de los tres archivos compactos y lista explícita de todos los pesos omitidos |

## Alcance del ZIP y reproducibilidad

El ZIP incluye encoders_fase2_seleccionados.pt, copia idéntica por SHA256 de resultados_fase2/co_byol_bagnet33/mejores_encoders.pt, contiene ambos encoders y su configuración, la rama elegida es objetivo, los dos archivos compactos históricos se conservan con sus nombres originales, no se incluyen cabezas de entrenamiento ni optimizadores en estos tres archivos y no permiten reanudar todo el entrenamiento.

Se omiten 92 archivos de pesos locales que ocupan 55.074 GiB, entre ellos los 54 checkpoints históricos completos, los 19 estados completos de fase 2 y los 19 exportados por configuración cuyo seleccionado se copia a la raíz, no se confunde conservar sus métricas con distribuir todos sus pesos, los datos y los entornos son externos, requirements.txt fija las dependencias utilizadas y las rutas se reciben explícitamente.

Para una revisión estática basta clonar el repositorio y consultar la guía, el código y los JSON, para cargar el seleccionado descargar el ZIP y extraerlo en una carpeta distinta, un comando de reevaluación autorizado sería el siguiente, no se ejecutó durante la publicación.

```powershell
python main.py evaluar --checkpoint encoders_fase2_seleccionados.pt --datos RUTA_A_LOS_HDF5 --resultados evaluacion_fase2_autorizada --particion testing --dispositivo cpu
```

Ese comando exige los HDF5 y cómputo adicional, no reproduce las diecinueve continuaciones, los comandos ejecutar, ampliar, completar-arquitecturas, analizar-panel y fase2 pueden iniciar cómputo y no son necesarios para leer la entrega, main.py informe y fase2 consolidar regeneran documentación experimental y pueden reemplazar documentos editados, no ejecutarlos como si fueran una consulta inocua del informe consolidado.

## Límites que no deben ocultarse

Una semilla y aproximadamente 2.908 épocas equivalentes no prueban convergencia ni reproducen los calendarios del paper, cien consultas de validación restringen la selección, no hay intervalos estadísticos ni prueba nueva ciega, fase 2 reinicia optimizador y calendario EMA por lo que la mejora no aísla solo el efecto de más épocas, BagNet33 no reproduce la igualación de parámetros del artículo y compartir cabeza cambia otros componentes a la vez.

Los 44 análisis de máscaras, PGD, campos efectivos y lecturas son de fase 1 y no se aplican como evidencia a los pesos nuevos, no se ejecutó Places365 ni el panel bibliográfico completo, CO no supera a su base en todos los controles, el fallo audiovisual inicial permanece en los registros y su reparación CPU se acredita en una revisión posterior, la revisión del grupo, el envío al formulario y la defensa siguen pendientes.

La validación del ZIP comprueba nombres, lista de pesos permitidos, integridad CRC y SHA256, GitHub se contrasta con el árbol del commit publicado y los hashes de sus assets, los comprobantes de publicación se guardan fuera del ZIP para evitar una referencia circular a su propio hash, no se presenta una carga numérica de todos los checkpoints como si se hubiera realizado en esta actualización.
