# Laboratorio 3, reidentificación de personas mediante CO-SSL

Este proyecto adapta CO-BYOL y CO-MoCoV3 del artículo Seeing the whole in the parts with self-supervised representation learning al problema de recuperación de personas del laboratorio, la implementación del aprendizaje autosupervisado y de RF-ResNet es propia, las bibliotecas se utilizan para operaciones generales y arquitecturas estándar.

## Estado actual, revisión de fase 2 del 20 de octubre

La fase 2 terminó a las 06:52 de Bogotá con catorce arquitecturas y cinco controles evaluados, todas las configuraciones completaron dos épocas adicionales con batch ocho, la selección por validación es CO-BYOL BagNet33 objetivo con mAP de prueba original 2.167 por ciento y LR 2.014 por ciento, consultar [INFORME.md](INFORME.md) y [la tabla completa de fase 2](resultados_fase2/RESULTADOS_FASE2.md).

Doce arquitecturas mejoran su mAP media observada, ResNet18 conserva su checkpoint anterior y RF29 empeora la media, cinco referencias siguen bajo su inicialización aleatoria en validación, estos resultados son preliminares y no prueban convergencia ni superioridad general de CO.

El video de fase 2 está generado y verificado en [resultados_fase2/audiovisual/VIDEO.mp4](resultados_fase2/audiovisual/VIDEO.mp4), se corrigió su generación solo en CPU y el registro de cierre original mantiene visible el fallo inicial.

La publicación vigente incorpora ambas fases en [entrega-v2](https://github.com/juanvasquez-ship-it/Lab-3-deep-learning/releases/tag/entrega-v2), la [guía PDF](docs/GUIA_EVALUACION_IA.pdf), el informe y los manifiestos distinguen código implementado, ejecuciones comprobadas y límites, esta actualización no entrenó ni realizó inferencia de encoders ni utilizó GPU.

## Evaluación independiente y archivos publicados

El ZIP y el PDF actualizados están en [la entrega de fase 2](https://github.com/juanvasquez-ship-it/Lab-3-deep-learning/releases/tag/entrega-v2), el repositorio conserva código, métricas, registros, gráficos y los dos videos, el video vigente es [el de fase 2](resultados_fase2/audiovisual/VIDEO.mp4), el archivo VIDEO.mp4 de la raíz es histórico de fase 1.

El ZIP incluye tres archivos de pesos compactos en la raíz, encoders_fase2_seleccionados.pt contiene ambas ramas de BagNet33 de fase 2 y la rama elegida es objetivo, encoders_seleccionados.pt y encoders_comparacion.pt conservan los modelos seleccionados de fase 1, los pesos no se almacenan en Git, sus hashes, tamaños y las omisiones están en [CONTENIDO_PAQUETE.json](CONTENIDO_PAQUETE.json).

[docs/inventario_ejecuciones.json](docs/inventario_ejecuciones.json) describe las 54 configuraciones de fase 1 y [docs/inventario_fase2.json](docs/inventario_fase2.json) las 19 continuaciones de fase 2, no son 73 arquitecturas ni 73 inicializaciones independientes, los HDF5, entornos y restantes checkpoints se conservan localmente fuera del ZIP por volumen, comprobar código y métricas no requiere esos pesos, reevaluar los modelos omitidos sí requiere recuperarlos.

La [release anterior](https://github.com/juanvasquez-ship-it/Lab-3-deep-learning/releases/tag/entrega-v1) y la [guía anterior](docs/GUIA_EVALUACION_IA_FASE1_HISTORICA.pdf) se conservan como historia, los JSON de revisión anteriores que indican publicación pendiente describen el estado de entonces y no se reescriben, el comando main.py informe regenera el informe histórico y no sustituye INFORME.md consolidado, revisar [PUBLICACION_FASE2.md](PUBLICACION_FASE2.md) antes de usar comandos de ejecución.

## Estado de la entrega

Esta sección conserva las instrucciones y el estado publicados de fase 1, el estado vigente de ambas fases está resumido al inicio y desarrollado en INFORME.md.

La continuación autorizada para completar arquitecturas tiene su plan en resultados/plan_comparacion.json y su progreso en resultados/estado_comparacion.json, COMPARACION_ARQUITECTURAS.md y resultados/comparacion_arquitecturas.csv muestran la referencia común y distinguen métricas existentes de pendientes, esta fase utiliza un presupuesto independiente y no reinicia ni altera la sesión inicial.

```powershell
python main.py tabla-arquitecturas --dispositivo cpu
python main.py completar-arquitecturas --datos RUTA_A_LOS_HDF5 --horas 3
```

El primer comando consulta y actualiza la tabla sin entrenar, el segundo continúa la cola predeclarada únicamente dentro de su presupuesto complementario y reutiliza las fases completas, un checkpoint parcial se conserva sin presentarlo como entrenamiento completo y no se sobrescribe con un reinicio automático, los registros de la ejecución independiente se encuentran en resultados/comparacion.log y resultados/sesion_independiente.json.

La revisión contra las fuentes se encuentra en [REVISION_CUMPLIMIENTO.md](REVISION_CUMPLIMIENTO.md), la entrega cubre los experimentos básicos del laboratorio y mantiene parcial el alcance experimental completo del paper, la aprobación automática de coherencia no certifica reproducción íntegra ni finalización de los pendientes.

Los registros contienen 54 modelos que completaron el calendario local y 54 con prueba de ambas ramas y resoluciones, el panel analítico histórico conserva sus 44 análisis, COMPARACION_ARQUITECTURAS.md presenta las catorce arquitecturas bajo la referencia común y verificacion_comparacion.json distingue cobertura y pendientes.

La sesión inicial de GPU registrada duró 171.58 minutos dentro del máximo de 180, cada entrenamiento completo recibió 1253 actualizaciones con batch ocho y 10024 imágenes fuente distintas, este calendario equivale a menos de una época y no reproduce la convergencia del artículo.

La matriz histórica conserva combinaciones no ejecutadas fuera del panel mínimo solicitado para esta continuación, todas las arquitecturas se comprueban individualmente en la tabla y los calendarios largos y la transferencia siguen fuera de la reproducción local, una configuración disponible no constituye un resultado.

VIDEO.mp4 tiene una duración verificada de 14.09 minutos y utiliza narración sintética como apoyo, actualización de la comparación ampliada en el video, completada, el grupo debe revisar el material, realizar el envío al formulario y preparar la defensa individual, esas actividades no se presentan como realizadas por el programa.

El paquete Laboratorio3_entrega.zip incluye código, documentación, métricas, gráficos, video y ambos encoders del modelo principal, los checkpoints completos de todas las ejecuciones se conservan en la carpeta local y se omiten del ZIP por volumen, los HDF5 originales y el entorno de Python se mantienen fuera del paquete.

## Organización

| Elemento | Contenido |
|---|---|
| `main.py` | Punto de entrada para verificar, entrenar, evaluar y generar informes |
| `METODOLOGIA.md` | Relación con el artículo, fórmulas, decisiones y discrepancias de las fuentes |
| `INFORME.md` | Resultados reales, comparaciones, limitaciones y tabla de requisitos |
| `COMPARACION_ARQUITECTURAS.md` | Catorce arquitecturas evaluadas bajo la referencia común, métricas y costes |
| `ANALISIS_ERRORES.md` | Veinte casos visuales explicados y sus límites de interpretación |
| `REVISION_CUMPLIMIENTO.md` | Revisión contra las fuentes, correcciones y separación entre implementación, ejecución y verificación |
| `GUION_VIDEO.md` | Narración, estructura y créditos del video |
| `VIDEO.mp4` | Material audiovisual con narración sintética local |
| `encoders_seleccionados.pt` | Ambas ramas de BagNet33, selección histórica de fase 1 |
| `encoders_fase2_seleccionados.pt` | Ambas ramas de BagNet33 de fase 2, selección vigente objetivo, disponible en el ZIP |
| `encoders_comparacion.pt` | Ambas ramas de ResNet18 a 224, selección de la comparación ampliada mediante validación |
| `codigo/modelos.py` | Encoders, RF-ResNet, cabezas, BYOL, MoCoV3, extensiones CO y DINO |
| `codigo/datos.py` | Lectura HDF5 sin modificar originales y transformaciones de entrada |
| `codigo/entrenamiento.py` | Optimizadores, semillas, EMA, registros y presupuesto temporal |
| `codigo/evaluacion.py` | Representaciones L2, coseno, mAP, Rank-k y ejemplos |
| `codigo/experimentos.py` | Comparaciones principales y matriz ampliada |
| `codigo/ampliacion.py` | Configuraciones predeclaradas y prioridad por cobertura dentro del mismo presupuesto |
| `codigo/analisis.py` | Máscaras, PGD, lecturas intermedias, similitud local y gradientes |
| `codigo/informe.py` | Informe y gráficos construidos a partir de resultados guardados |
| `codigo/verificaciones.py` | Verificaciones técnicas con resultados conocidos |
| `codigo/auditoria.py` | Coherencia de registros existentes y cobertura de experimentos, sin entrenamiento |
| `codigo/comparacion.py` | Cola complementaria y tabla común de arquitecturas |
| `resultados` | Configuraciones, checkpoints, métricas, gráficos y estados reales |

## Entorno

Se utiliza Python 3.12 y el entrenamiento requiere una GPU NVIDIA compatible con CUDA 12.8, consultar resultados y ejecutar la auditoría se hace sin GPU, el entorno local preparado para esta entrega está en `../../work/entorno`, no se modifica la instalación global de Python del equipo.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py verificar --datos RUTA_A_LOS_HDF5
.venv\Scripts\python.exe main.py matriz
.venv\Scripts\python.exe main.py ejecutar --datos RUTA_A_LOS_HDF5 --horas 3 --batch 8 --trabajadores 2
.venv\Scripts\python.exe main.py informe
.venv\Scripts\python.exe main.py auditar
```

Para revisar esta entrega sin iniciar cómputo GPU se puede utilizar el siguiente comando, la auditoría solo lee resultados y metadatos y actualiza sus archivos de coherencia y cobertura.

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
python main.py auditar --dispositivo cpu --datos RUTA_A_LOS_HDF5
```

La suite principal rechaza repetir una entrega que ya tiene selección y prueba registradas, el panel analítico conserva las tareas completas sin reabrir inferencia ni modificar el cierre del presupuesto, las pruebas CPU de esas correcciones se documentan en resultados/revision_correcciones_cpu.json.

La ejecución local puede utilizar `../../work/entorno/Scripts/python.exe` en lugar del intérprete mostrado, los datos ya extraídos se encuentran en `../../work/data`, para otro equipo se debe indicar la ruta real mediante `--datos`.

```powershell
python main.py preparar-datos --zip RUTA_AL_ZIP --datos RUTA_DESTINO
```

## Particiones y etiquetas

Se conservan las 11043 imágenes de `training.h5`, las 100 consultas y 1793 imágenes de galería de `validation.h5`, y las 3368 consultas y 19732 imágenes de galería de `testing.h5`, no se crea un nuevo train test split y no se incorpora validación al entrenamiento, la interfaz de entrenamiento no devuelve identidades.

El atributo `final_training` propone una fase adicional que el PDF no describe, esa fase no se aplica por defecto y cualquier ejecución futura debe documentarla por separado.

## Presupuesto y comparación

El máximo autorizado es de tres horas para la sesión de GPU, el reloj incluye calibración, entrenamiento, validación y prueba, el inicio y el límite se conservan en `resultados/presupuesto.json` y reiniciar el programa no reinicia ese presupuesto.

El PDF del laboratorio y los HDF5 no fijan un tamaño de batch, el valor 64 del apéndice H del paper corresponde a su análisis de coste, el batch local es una decisión de recursos y queda registrado junto con la cantidad efectiva de actualizaciones y de imágenes procesadas.

La calibración determina una cantidad común de actualizaciones para los modelos seleccionados, cada modelo recibe el mismo batch físico y el mismo origen de imágenes, un límite temporal puede impedir completar una época o una evaluación y en ese caso el registro lo indica expresamente.

La selección utiliza la media de mAP de validación original y reducida, se comparan encoder entrenable y encoder EMA, el conjunto de prueba no se utiliza para elegir modelo, rama o hiperparámetros.

Las configuraciones de ampliación se guardan antes de prueba y solo se ejecutan con el margen del presupuesto original, se registran las reordenaciones para cubrir tipos de componentes antes de multiplicar valores y recortes sin cambiar las configuraciones ni usar métricas de prueba como prioridad, plan_ampliacion.json conserva el orden previo y el historial de prioridad y la selección principal se mantiene.

```powershell
python main.py ampliar --datos RUTA_A_LOS_HDF5 --horas 3 --batch 8 --trabajadores 2
python main.py analizar-panel --datos RUTA_A_LOS_HDF5 --horas 3
python main.py analizar --checkpoint resultados/co_byol_rf99/checkpoint.pt --analisis representaciones
python main.py analizar --checkpoint resultados/co_byol_rf99/checkpoint.pt --analisis mascaras --particion validation
python main.py analizar --checkpoint resultados/co_byol_rf99/checkpoint.pt --analisis pgd --particion validation
python main.py analizar --checkpoint resultados/co_byol_rf99/checkpoint.pt --analisis antes_mlp --particion validation
python main.py analizar --checkpoint resultados/co_byol_rf99/checkpoint.pt --analisis capas --particion validation
python main.py analizar --checkpoint resultados/co_byol_rf99/checkpoint.pt --analisis eficiencia
```

Los análisis de máscaras, PGD y lectura utilizan por defecto validación y no producen cambios de parámetros, las consultas originales de training se utilizan sin etiquetas para el análisis de representaciones y gradientes, la implementación de PGD modifica copias en memoria y mantiene los archivos originales intactos.

El panel de análisis utiliza la rama online de todos sus comparadores y un orden predeclarado, en una llamada individual `--rama online`, `--rama objetivo` o `--rama seleccionada` permite especificar la rama y el JSON conserva esa elección, no se mezclan valores de ramas diferentes al presentar una comparación controlada.

Una vez agotado el presupuesto local el programa impide nuevas operaciones GPU en la misma carpeta de resultados, verificar e informe siguen disponibles sin entrenamiento, para reproducir otra sesión autorizada en otro equipo debe elegirse expresamente una carpeta nueva mediante `--resultados`, esa opción no se utiliza en esta entrega para ampliar las tres horas.

```powershell
python main.py ejecutar --datos RUTA_A_LOS_HDF5 --resultados nueva_ejecucion_autorizada --horas 3 --batch 8
python main.py evaluar --checkpoint resultados/co_byol_rf99/checkpoint.pt --datos RUTA_A_LOS_HDF5 --resultados nueva_evaluacion_autorizada --particion testing
```

El segundo comando evalúa las dos ramas de un checkpoint sin entrenarlo, los nombres de checkpoint deben corresponder a archivos existentes, no es necesario volver a entrenar para leer las métricas o regenerar el informe.

```powershell
python main.py exportar
python main.py evaluar --checkpoint encoders_seleccionados.pt --datos RUTA_A_LOS_HDF5 --resultados evaluacion_encoders_autorizada --particion testing
```

La exportación conserva los dos encoders del modelo seleccionado con todos sus parámetros y estadísticas de normalización y omite las cabezas que solo se utilizan para entrenamiento, la igualdad de tensores se comprueba en CPU y no requiere repetir inferencia, el archivo compacto permite evaluar el modelo seleccionado sin cargar sus estados de optimización.

## Adaptación y protocolo

El panel principal utiliza vistas de 224 por 224 y el panel pequeño vistas de 64 por 64, los recortes se adaptan al aspecto vertical de las imágenes de personas, las diferencias entre paneles no se atribuyen exclusivamente a la pérdida autosupervisada.

Las cabezas locales independientes, el promedio de posiciones, las dos direcciones local global y el intercambio de vistas se conservan, las características para recuperación proceden del encoder y no de las proyecciones de entrenamiento, los pesos se inicializan aleatoriamente y no se descargan modelos CO existentes.

La temperatura contrastiva 0.2 sigue el detalle de CO-MoCoV3 en la prepublicación de los autores, la publicación no desarrolla ese valor al mismo nivel, los calendarios cortos se adaptan a la duración real y no equivalen a sus entrenamientos de ImageNet.

El protocolo `market_documentado` ignora identidad -1, conserva identidad 0 como distractor y excluye coincidencias de igual identidad y cámara, estas reglas se documentan en las preguntas frecuentes y el apartado de evaluación de la fuente oficial del dataset, AP se calcula como la media de precisiones en las posiciones relevantes y los empates conservan el orden original de la galería.

Se evalúa el mismo encoder congelado con consultas originales y reducidas manteniendo la galería original, se informa mAP, Rank-1, Rank-5, Rank-10 y caída de mAP absoluta y relativa, la clasificación Top-1 del artículo no se reproduce entrenando identidades porque infringiría las condiciones del laboratorio.

## Referencias

Aubret, Teulière y Triesch, Seeing the whole in the parts with self-supervised representation learning, Neurocomputing 675, 132973, 2026, https://doi.org/10.1016/j.neucom.2026.132973.

Zheng y colaboradores, Market-1501 y documentación de evaluación, https://zheng-lab-anu.github.io/Project/project_reid.html.

Grill y colaboradores, Bootstrap your own latent, https://arxiv.org/abs/2006.07733.

Chen, Xie y He, An Empirical Study of Training Self-Supervised Vision Transformers, https://arxiv.org/abs/2104.02057.

Caron y colaboradores, Emerging Properties in Self-Supervised Vision Transformers, https://arxiv.org/abs/2104.14294.

## Comparación ampliada de arquitecturas

La continuación autorizada se documenta en COMPARACION_ARQUITECTURAS.md, actualmente hay 14 de catorce arquitecturas evaluadas bajo el protocolo común a 224, resultados/comparacion_arquitecturas.csv contiene todas las métricas y costes disponibles, las combinaciones históricas no esenciales no se anuncian como entrenamientos pendientes de arquitectura.

La nueva sesión autorizada registra 98.96 minutos dentro de su límite separado de 180, el presupuesto inicial se conserva sin cambios y no se afirma que ambas sesiones juntas hayan durado tres horas.

## Fase 2, continuación autorizada del 20 de octubre

La fase 1 y su video conservan sus resultados históricos, la fase 2 tiene estado y tabla actualizados en [resultados_fase2/RESULTADOS_FASE2.md](resultados_fase2/RESULTADOS_FASE2.md), los pesos de fase 2 y las épocas elegidas se documentan por separado, las cifras de la fase 1 no se atribuyen al código o entrenamiento de fase 2, la narración del video debe revisarse antes de presentar los nuevos resultados.

La versión audiovisual de fase 2 se encuentra en [resultados_fase2/audiovisual/VIDEO.mp4](resultados_fase2/audiovisual/VIDEO.mp4), su guion y verificación de duración y decodificación están en la misma carpeta, el video original de fase 1 se conserva sin cambios y la revisión humana de ambos materiales sigue pendiente.
