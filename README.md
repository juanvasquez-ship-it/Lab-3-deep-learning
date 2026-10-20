# Laboratorio 3, reidentificación de personas mediante aprendizaje autosupervisado

El trabajo adapta CO-BYOL y CO-MoCoV3 de Aubret, Teulière y Triesch a recuperación de personas entre cámaras, el mecanismo de aprendizaje se implementa en este repositorio y utiliza torchvision y timm para arquitecturas generales, el entrenamiento no recibe etiquetas de identidad ni pesos preentrenados.

## Resultados y alcance

La fase 1 contiene 54 configuraciones con 1253 actualizaciones cada una, la fase 2 continúa 19 de esos checkpoints con dos épocas adicionales completas para todos, incluye las catorce arquitecturas acordadas y cinco controles, son continuaciones y no diecinueve inicializaciones independientes.

La selección mediante validación fue CO-BYOL BagNet33, rama objetivo, con mAP de prueba de 2.167244% en consultas originales y 2.014327% en consultas reducidas, Rank-1 de 5.908551% y 4.661520%, los resultados son preliminares, una semilla y unas 2.908 épocas equivalentes no prueban convergencia ni reproducen los calendarios del artículo.

| Material | Contenido |
|---|---|
| [INFORME.md](INFORME.md) | Problema, resultados entre fases, bases, ablaciones, costes, límites y cobertura del laboratorio |
| [METODOLOGIA.md](METODOLOGIA.md) | Ecuaciones, arquitecturas, adaptaciones, referencias y uso de herramientas |
| [Resultados de fase 2](resultados_fase2/RESULTADOS_FASE2.md) | Diecinueve configuraciones con ambas resoluciones y tiempos |
| [Comparación de fase 1](COMPARACION_ARQUITECTURAS.md) | Catorce arquitecturas con política histórica de rama online |
| [Análisis de errores](resultados_fase2/ANALISIS_ERRORES_FASE2.md) | Veinte casos, cinco aciertos y cinco fallos por condición, con figuras y explicación |
| [Video final](resultados_fase2/audiovisual/VIDEO.mp4) | 11 min 57.56 s, metodología, implementación y resultados |
| [GUION_VIDEO.md](GUION_VIDEO.md) | Transcripción de la narración y atribuciones |

No se incluye una presentación de diapositivas separada, el PDF del laboratorio exige código, resultados e instrucciones y video, el guion y el video conservan las atribuciones de la narración sintética, la exposición humana, el envío y la defensa no se acreditan mediante estos archivos.

## Organización

El código está en main.py y codigo, resultados conserva la evidencia de las 54 configuraciones iniciales y las ablaciones, resultados_fase2 contiene las 19 continuaciones, los pilotos de validación, las dos ramas, curvas y casos del modelo final, las configuraciones, tiempos y selecciones permanecen junto a sus métricas originales.

Se conservan planes y verificaciones que respaldan la integridad del experimento, las rutas de procedencia que aparecen en registros históricos describen el espacio original y no son rutas de instalación, los hashes de ejecución corresponden al commit a6dc59ab03ac869b5ab100c250492d19c07c5a39, las modificaciones posteriores de auditoría y documentación no cambian modelos, pérdidas, datos ni métricas.

## Instalación y revisión en CPU

Se utilizó Python 3.12, PyTorch 2.8 con CUDA 12.8 y las versiones fijadas en requirements.txt, la GPU se necesita para entrenar, consultar resultados y comprobar la entrega no requiere una GPU.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:CUDA_VISIBLE_DEVICES='-1'
.venv\Scripts\python.exe verificar_entrega.py
```

La verificación contrasta archivos, importaciones, configuraciones, métricas guardadas, selección y video, sin calcular representaciones ni entrenar, con los HDF5 puede comprobar también sus formas, la alineación de metadatos y las pruebas numéricas pequeñas ya incluidas en codigo/verificaciones.py.

```powershell
python verificar_entrega.py --datos RUTA_A_LOS_HDF5
```

Los HDF5 externos son training.h5, validation.h5 y testing.h5, sus particiones se conservan, training contiene 11043 imágenes sin identidades, validación 100 consultas y 1793 imágenes de galería, prueba 3368 consultas y 19732 de galería, no se crea otro split ni se incorpora validación al aprendizaje.

## Pesos y evaluación

Los pesos no se almacenan en Git, la [release entrega-v2](https://github.com/juanvasquez-ship-it/Lab-3-deep-learning/releases/tag/entrega-v2) conserva el ZIP con encoders_fase2_seleccionados.pt y los dos exportados históricos, CONTENIDO_PAQUETE.json indica sus tamaños y SHA256, el archivo vigente contiene ambas ramas de BagNet33 y la elegida es objetivo, no incluye cabezas ni optimizador.

```powershell
python main.py evaluar --checkpoint encoders_fase2_seleccionados.pt --datos RUTA_A_LOS_HDF5 --resultados evaluacion_reproducida --particion testing --dispositivo cpu
```

Ese comando calcula de nuevo ambas ramas y puede ser lento en CPU, leer las tablas no lo requiere, reevaluar el resto de modelos requiere sus checkpoints completos, los 19 checkpoints de fase 1 necesarios para repetir la continuación no están en el ZIP compacto y se conservan en el equipo original, el repositorio tiene acceso restringido y su lectura requiere permisos de GitHub.

## Reproducción del entrenamiento

preparar_reproduccion.py prepara una carpeta nueva, no inicia entrenamiento y no modifica los resultados entregados, el ejemplo de fase 1 deja disponibles el panel principal, las ampliaciones y los análisis mediante el mismo main.py, el número de pasos reproduce el calendario local y no una convergencia del paper.

```powershell
python preparar_reproduccion.py --fase 1 --destino ../reproduccion_fase1
$env:CUDA_VISIBLE_DEVICES='0'
python ../reproduccion_fase1/main.py ejecutar --datos RUTA_A_LOS_HDF5 --pasos 1253 --batch 8 --trabajadores 2 --horas 3
```

Las acciones ampliar, analizar-panel y completar-arquitecturas conservan las configuraciones y las guardas temporales de fase 1, los planes guardados identifican las 54 ejecuciones realmente realizadas, una matriz de configuraciones también puede contener combinaciones no ejecutadas, repetir toda la secuencia exige recursos y tiempo de evaluación suficientes.

Para reproducir fase 2 a partir de los pesos iniciales se indica la carpeta que contiene los diecinueve subdirectorios con checkpoint.pt, la preparación copia esos estados y los registros de fase 1 en otro destino y crea allí un plazo nuevo de diez horas y una captura de código, no cambia el plazo ni los archivos históricos del experimento entregado.

```powershell
python preparar_reproduccion.py --fase 2 --destino ../reproduccion_fase2 --pesos-fase1 RUTA_RESULTADOS_FASE1_COMPLETOS
$env:CUDA_VISIBLE_DEVICES='0'
python ../reproduccion_fase2/main.py fase2 --datos RUTA_A_LOS_HDF5
```

La reproducción mantiene el horizonte original de hasta tres épocas adicionales y la reserva de evaluación, en la ejecución entregada se completaron dos rondas, la cantidad que cabe antes del plazo depende del hardware y no se garantiza igualdad numérica de otra ejecución, los pilotos se descartan y las tasas y checkpoints se seleccionan mediante validación.

## Generación de tablas y verificaciones

main.py informe regenera los gráficos y el informe automático de fase 1, tabla-arquitecturas actualiza su comparación y fase2 --fase2-etapa consolidar reconstruye las tablas y curvas de continuación, estas acciones y main.py auditar escriben archivos y se reservan para una copia de reproducción, INFORME.md es la interpretación académica consolidada y no debe confundirse con un reporte automático de una sola fase.

codigo/verificaciones.py conserva comprobaciones analíticas de stop-gradient, pérdidas y AP/Rank-k, codigo/auditoria.py comprueba la evidencia de fase 1 y utiliza las rutas del entregable final declaradas en el manifiesto, los mtime de un clon no demuestran el orden histórico de selección y prueba, esa limitación se declara en lugar de convertir fechas de copia en evidencia experimental.

Las decisiones y las partes no reproducidas se describen en METODOLOGIA.md e INFORME.md, las referencias y la declaración de uso de herramientas se mantienen en la metodología, las actividades de revisión grupal, envío al formulario y defensa individual siguen pendientes de realización humana.
