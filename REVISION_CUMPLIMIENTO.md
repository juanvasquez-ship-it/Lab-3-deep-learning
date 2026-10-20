# Revisión del proyecto contra el laboratorio y el artículo

Este documento conserva la revisión anterior a la continuación de ejecución autorizada después, el estado actualizado de arquitecturas y resultados se presenta en COMPARACION_ARQUITECTURAS.md y los registros de la sesión complementaria, las cantidades y huellas de esta revisión corresponden a su fecha y no sustituyen esos registros posteriores.

## Dictamen

La entrega contiene implementación propia del núcleo CO-SSL y evidencia de los experimentos básicos exigidos por el laboratorio, el alcance experimental completo solicitado a partir del artículo sigue parcial y no debe presentarse como una reproducción íntegra, esta revisión no entrenó modelos ni ejecutó inferencia y mantuvo CUDA oculta en las comprobaciones de PyTorch.

Se contrastaron los PDF originales de tres y doce páginas, las ecuaciones y figuras relevantes, los once archivos Python del proyecto, las configuraciones y registros de las 48 ejecuciones y los 44 análisis, también se amplió la comprobación estructural de los checkpoints y se corrigieron problemas del verificador y de protección de resultados, las huellas de las fuentes y el alcance de cada comprobación se conservan en resultados/revision_cumplimiento.json.

El estado aprobado de verificacion_final.json acredita coherencia de los archivos comprobados, no acredita convergencia, igualdad con los resultados de ImageNet, ejecución de toda la matriz ni cumplimiento de las actividades humanas de entrega y defensa.

## Requisitos del laboratorio

| Requisito explícito | Página | Implementación y evidencia | Dictamen y límites |
|---|---:|---|---|
| Recuperar la misma persona entre cámaras mediante una representación y un orden de similitud | 1 | Encoder congelado en evaluacion.py y métricas por consulta | Implementado y evaluado, la clasificación del artículo se adapta a recuperación |
| Entrenamiento sin etiquetas de identidad | 1 | Entrenamiento en datos.py solo abre images e image_ids y devuelve vistas e índice, las pérdidas no reciben person_ids | Comprobado en código, los metadatos de identidad se usan en evaluación |
| Utilizar exclusivamente imágenes como entrada, sin texto ni descripciones | 1 | Vistas RGB y tensores de imagen en modelos.py | Comprobado en las interfaces |
| Usar training.h5 con 11043 imágenes | 1 | Origen único de entrenar, longitud 11043 en los 48 resúmenes | No equivale a utilizar todas las imágenes en una ejecución menor de una época |
| Usar validación de 100 consultas y 1793 imágenes de galería para elegir configuración | 1 y 2 | validacion.json por modelo y seleccion_validacion.json | Ambas ramas y resoluciones, criterio local explícito de media de mAP |
| Usar prueba de 3368 consultas y 19732 imágenes de galería | 1 y 2 | prueba.json de las 48 ejecuciones | Todas las consultas evaluadas en las cuatro combinaciones de rama y resolución |
| Leer las consultas reducidas en query/images_low_resolution y conservar su correspondencia por orden | 1 | Evaluacion en datos.py y extracción ordenada sin shuffle | Utiliza las versiones suministradas, no genera una partición nueva |
| Separación de identidades entre entrenamiento y prueba | 1 | Declaración del PDF y de los materiales del dataset | No verificable directamente sin etiquetas de training, no se deduce de ausencia de duplicados de píxeles |
| Estudiar e implementar el método del paper y adaptar entradas, arquitectura, representación y pérdida | 1 y 2 | METODOLOGIA.md y modelos.py | Núcleo CO implementado, las desviaciones y la cobertura parcial del paper se detallan abajo |
| Implementación propia de la propuesta central, sin una implementación existente del mismo método | 2 | Pérdidas, cabezas, ramas, EMA y RF-ResNet en código propio | torchvision y timm aportan arquitecturas estándar, no se importa solo-learn ni una implementación SSL completa |
| Entrenar y seleccionar usando validación | 2 | 48 resúmenes, historiales y checkpoints, selección principal entre las 29 configuraciones iniciales | Selección conservada, las 19 ejecuciones adicionales no reabren la elección después de prueba |
| Comparar con un modelo base | 2 | BYOL y MoCoV3 sin CO y tablas de INFORME.md | Existen controles con arquitectura fija, además hay una base BagNet33 sin CO para el modelo principal |
| Realizar una ablación sobre un componente importante | 2 | Peso local cero frente a CO, variantes sin MLP y cabeza compartida | Ejecutadas, no se confunde quitar MLP al entrenar con leer antes del MLP |
| Conservar la transferencia maestro estudiante y comparar ambos si el método los usa | 2 | Stop gradient, EMA y evaluaciones online y objetivo | Implementado y ambas ramas evaluadas, las estadísticas BatchNorm objetivo proceden de sus propias pasadas |
| Justificar reemplazos de información ausente conservando la propuesta central | 2 | Recuperación congelada en lugar de sonda de clasificación, METODOLOGIA.md | Adaptación explícita, no se entrenan clasificadores de identidad |
| Evaluar originales y consultas reducidas manteniendo galería original y el mismo modelo entrenado | 2 | evaluar_encoder reutiliza encoder y características de galería | Implementado y registrado para 48 modelos y ambas ramas |
| Vectores normalizados L2 y similitud coseno | 2 | F.normalize y producto entre matrices en evaluacion.py | Comprobado en código, esta revisión no regeneró embeddings |
| Reportar mAP y Rank-1, Rank-5 y Rank-10 | 2 | 384 conjuntos de métricas en validación y prueba | Valores válidos, cantidades de consultas y orden de Rank-k contrastados |
| Reportar caída de mAP al reducir resolución | 2 | Caída en puntos porcentuales y relativa en cada JSON | El PDF no define una sola convención, se declaran ambas |
| No entrenar ni ajustar con prueba | 2 | Cargador de training separado, selección principal anterior al primer prueba.json y planes predeclarados | Código y registros consistentes, las fechas y el código actual no constituyen una prueba forense de cualquier actividad externa |
| Código .py o notebook con carga, preprocesamiento, método, entrenamiento, validación, base, ablación y evaluación | 2 | main.py y diez módulos de codigo | Incluido, sin comentarios ni docstrings conforme a la petición del usuario |
| Gráficos y resultados en notebook o Markdown e instrucciones y diferencias importantes | 2 y 3 | README.md, METODOLOGIA.md, INFORME.md y resultados | Incluido, esta revisión añade límites que no estaban suficientemente destacados |
| Al menos cinco aciertos y cinco fallos explicados | 3 | ANALISIS_ERRORES.md, cuatro imágenes e indices_ejemplos.json | Veinte casos, cinco de cada tipo por resolución, la inspección visual específica previa se conserva |
| Video de máximo 18 minutos con metodología, implementación y resultados | 3 | VIDEO.mp4 y video_verificacion.json | 848.8 segundos verificados previamente, capítulos inferiores a seis minutos, huella actual idéntica al archivo decodificado |
| Entrega grupal al formulario el jueves 22 a las 23:59 | 1 a 3 | Paquete y documentación preparados | Envío y revisión del grupo pendientes, no se declara envío automático |
| Defensa individual el sábado 24 y comprensión de método, código, decisiones, resultados y límites | 1 y 3 | Material de explicación y resultados trazables | Actividad humana pendiente |
| Bibliotecas de procesamiento de audio | 2 | Texto literal contradictorio con la entrada exclusiva de imágenes de la página 1 | Posible errata, no hay aclaración oficial ni autorización inferida para introducir audio como entrada del modelo |

## Correspondencia con el paper

| Elemento del artículo | Referencia | Implementación y ejecución observadas | Diferencia o pendiente |
|---|---|---|---|
| CO-BYOL y CO-MoCoV3, pérdida global y dos direcciones local global con intercambio de vistas | pp 2 a 4, algoritmo 1 y ecuaciones 1 a 3 | modelos.py implementa stop gradient, promedio espacial, peso local y EMA, ambas familias entrenadas | Comprobación metodológica del código, no certificación de equivalencia numérica con solo-learn |
| RF-ResNet y MLP residual tras promedio | p 8, figura A.6 y apéndices A y B | RF29, RF67, RF99, RF163 y RF195 entrenadas, RF39 y RF115 construidas y comprobadas previamente en CPU | RF193 del gráfico da 195 al seguir sus operaciones, ResNet50 de torchvision da 427 frente al aproximadamente 425 del texto |
| Control BagNet33 igualado en parámetros | p 5, figura 4 | Versión propia RF33 con y sin CO, el modelo principal usa esta arquitectura | Sus 16.261 millones de parámetros no reproducen la igualación a aproximadamente 23.5 millones del artículo, corregirla requeriría otra arquitectura y nuevos pesos |
| CNN, ViT-S/16 y Swin-T, Swin-S y Swin-B | tablas 1 y 2 y apéndice E | Todas las familias del panel principal se entrenaron | Panel pequeño a 64 y grande a 224, no se atribuyen diferencias entre paneles exclusivamente a la pérdida |
| Multicrop y DINO como comparadores propios del artículo | tablas 1 y 2 | Pérdidas propias y modelos entrenados | DINO usa seis vistas pequeñas y un calendario menor de una época, su última capa permanece congelada durante esa ejecución |
| Retirar el efecto de recortes pequeños mediante mínimo 0.9 y barridos de recorte | pp 4 y 5, figura 3 y figura D.8 | Configuraciones disponibles | Ninguna configuración con crop0.9 se ejecutó, no se ha demostrado localmente la afirmación central sobre aprender sin recortes pequeños |
| Efecto del campo receptivo y recorte | figura 4 | Algunos RF entrenados con recorte 0.2 | El barrido de la figura 4 está incompleto y no equivale a tener una arquitectura construida |
| Cantidad K de representaciones locales | tabla 3 | K 1, 4, 16, 64 y valor por defecto 196 entrenados | Submuestreo determinista en cuadrícula, K 1 toma la esquina superior izquierda, el artículo no fija ese detalle |
| Peso local | tabla 7 | Valores positivos 0.1, 0.2, 0.5 y 1 más control sin CO | Ejecutados con calendario corto y una semilla, no se establece un óptimo general |
| Ablaciones del MLP, lectura antes del MLP y cabeza compartida | tabla 6 | Dos entrenamientos sin MLP, una cabeza compartida y tres análisis de lectura | No se reproduce automáticamente la selección de la mejor lectura antes o después del MLP que describe la tabla, la cabeza compartida también cambia profundidad local y estadísticas BatchNorm |
| Kernel local de Swin | tabla F.10 | Kernels 1, 2 y 3 ejecutados, 4 y 5 configurados | Barrido parcial, padding válido reduce también el número de posiciones |
| Alineamiento en capas intermedias | apéndice I | Capa 2 y control de capa 4 a K 64 ejecutados | Capa 3 pendiente, no se confunde esta intervención de entrenamiento con lectura de checkpoints en capas 3 y 4 |
| Máscaras internas y PGD | tablas 4 y 5 | Diez análisis de máscaras con cinco proporciones y nueve de PGD con cinco ataques cada uno | Métricas de recuperación, PGD utiliza centro de positivos de otras cámaras, acceso evaluativo a etiquetas, inicio aleatorio declarado y parámetros de la tabla cuando difiere el texto |
| Similitud local y campos efectivos | figura 5 y apéndice C | Catorce muestras de representaciones y una lectura adicional a 384, dieciséis imágenes por tarea | No usa el conjunto completo ImageNet ni el submuestreo espacial de paso dos de la figura C.7, tampoco reproduce las dos posiciones de la figura 2, se utiliza la posición central y gradiente absoluto |
| Lectura intermedia y transferencia | tablas G.11 y G.12 | Tres análisis de lectura de capas, Places365 no disponible | La transferencia literal no está implementada ni ejecutada, solo está identificada como pendiente en metadatos |
| Eficiencia | tabla H.13 | Cuatro mediciones locales guardadas | Se cuenta Conv2d y Linear, no todas las operaciones ni el coste completo de entrenamiento y cabezas de la tabla, el hardware y el batch difieren |
| Calendarios y ajuste de tasas | p 4, tabla 1 y apéndice E | Tasas locales declaradas y 1253 actualizaciones por modelo | Barridos de tasas y calendarios de 100, 300, 400 y 1000 épocas no ejecutados como en el artículo, algunas filas largas son solo metadatos |
| AMDIM, SD-BYOL y comparadores bibliográficos | tabla 1 y sección 2 | No se presentan como modelos implementados | La procedencia no está marcada de manera uniforme en la tabla, no se sustituye esa ambigüedad por resultados inventados |

El calentamiento y descenso coseno se comprimen al número local de pasos, AdamW aplica aquí el mismo decaimiento a todos los parámetros entrenables de su grupo sin separar sesgos y normalizaciones, las implementaciones originales citadas pueden utilizar otras agrupaciones y estos detalles locales impiden presentar los hiperparámetros como una reproducción literal completa.

El texto del apéndice E especifica ResNet18 y RF29 para Tiny-ImageNet mientras que el encabezado de la tabla E.9 habla genéricamente de ResNet50, el panel pequeño sigue la descripción explícita y se declara esa elección sin presentar el encabezado y el texto como si fueran idénticos.

## Qué se ejecutó realmente

Los registros contienen 29 configuraciones principales y 19 adicionales con entrenamiento completo dentro del calendario local, todas tienen 1253 actualizaciones y batch físico ocho, cada modelo procesó 10024 imágenes fuente distintas de las 11043 disponibles y los 48 hashes del conjunto de índices coinciden, el hash no acredita por sí solo igualdad de orden de lectura ni de aumentos y la fracción 0.907971 se calcula respecto a 1380 batches completos por época.

Hay 48 validaciones y 48 pruebas con las dos ramas y las dos resoluciones, corresponden a 192 pares de evaluación y 384 conjuntos de métricas, las 144 filas de la matriz incluyen redundancias y condiciones publicadas distintas, 58 filas tienen una configuración equivalente ejecutada y ese valor no significa que se entrenaran 58 modelos adicionales.

El panel analítico contiene catorce tareas de representaciones, diez de máscaras, nueve de PGD, tres de lectura antes del MLP, tres de lectura de capas, cuatro de eficiencia y una de representaciones a 384, se contrastaron las 44 tareas declaradas como completas con sus archivos y condiciones internas, los cien casos de validación de cada condición analítica con métricas están presentes.

La selección conservada es co_byol_bagnet33 online con media de mAP de validación 7.533 por ciento, su mAP de prueba es 1.587 por ciento en consultas originales y 1.334 por ciento en reducidas, la caída es 0.253 puntos porcentuales y 15.921 por ciento relativo, esta revisión no cambió selección, pesos ni valores experimentales.

La sesión GPU previa registra 171.58 minutos y conserva su cierre original, la calibración incluyó actualizaciones descartadas para medir coste y no se presenta como entrenamiento de los checkpoints finales, esta revisión no prolongó esa sesión ni abrió otra.

## Comprobaciones y límites de la evidencia

| Comprobación | Alcance y procedimiento | Resultado y límite |
|---|---|---|
| Identificación de las fuentes | Lectura nueva de los dos PDF originales y SHA256, contraste visual de las tres páginas del laboratorio y de figuras y tablas relevantes del artículo | Tres y doce páginas, hashes conservados, las instrucciones de los documentos se tratan como requisitos de la tarea y no como autorización para acciones externas |
| Cobertura del código | Once archivos Python, lectura del flujo de carga, entrenamiento, modelos, evaluación, análisis, informes, planes y auditoría, AST y tokens | Núcleo propio y sin comentarios ni docstrings, no se vuelve a ejecutar entrenamiento para verificar su funcionamiento |
| Checkpoints | Los 48 archivos, lectura CPU con mmap y FakeTensorMode, comparación de configuración, resumen, claves, forma y tipo de cada tensor contra la construcción actual y cantidad de parámetros | Todos compatibles estructuralmente, no se leyeron los valores de todos los pesos ni se verificó finitud o CRC interno de cada archivo |
| Exportación compacta | Evidencia CPU previa de carga estricta y torch.equal de los 638 tensores de ambos encoders | Comprobación conservada, no se repitió al no haber cambiado pesos ni arquitectura |
| Métricas | Todos los JSON de validación y prueba, cantidades, límites, Rank-k, caída absoluta y criterio medio de validación | Coherentes, no se volvieron a generar embeddings ni similitudes |
| Rankings del seleccionado | 6736 consultas entre ambas resoluciones, top diez, identidades, cámaras, exclusiones y primer positivo | Etiquetas y rangos guardados coherentes, el promedio de AP coincide con mAP, no se reconstruye AP de galería completa a partir del top diez |
| Análisis | 44 tareas, archivos y ramas, cinco proporciones por máscara, cinco ataques y sus cotas, dos lecturas de capas, lecturas antes del MLP y evidencias de representaciones | Cobertura guardada consistente, campos externos de partición históricos discrepantes en quince tareas |
| Selección | Valores de validación de las 29 candidatas, configuración elegida y fechas locales de selección, planes y prueba | Selección consistente con validación, la marca temporal depende del sistema de archivos y no sustituye un historial inmutable |
| Datos y separación de identidades | Código HDF5 abierto en modo lectura, evidencia exploratoria previa y tamaños y fechas antes y después de esta revisión | No se repitió el hash de todos los píxeles ni la inspección visual ya documentada, no se puede certificar ausencia de identidades compartidas con training sin etiquetas |
| Video | SHA256 del archivo actual contra la huella de la decodificación completa previa, duración y capítulos del registro | Archivo sin cambios, no se repite el decodificador ni se presenta narración sintética como exposición humana |
| Correcciones | Seis casos dirigidos en CPU y copias sintéticas separadas de la entrega | Fallos anteriores reproducidos y correcciones comprobadas sin ejecutar redes ni tocar datos originales |

No existe una huella del código tomada exactamente al iniciar los entrenamientos que permita certificar retrospectivamente cada detalle de la versión ejecutada, la consistencia de configuraciones, registros y estructura de los checkpoints aporta evidencia sustancial pero no sustituye esa procedencia, la captura de esta revisión identifica el código recibido y el código corregido sin atribuir resultados nuevos a las correcciones.

## Hallazgos y correcciones

| Hallazgo | Corrección realizada | Comprobación |
|---|---|---|
| La auditoría anterior podía aprobar una rama de prueba ausente | Ahora rechaza ramas ausentes, resúmenes principales faltantes, inconsistencias del estado y criterios medios mal calculados | Una copia con rama objetivo eliminada era aprobada antes y es rechazada ahora |
| La auditoría no contrastaba los 44 análisis con sus archivos | Comprueba tareas declaradas, duplicados, rama, partición, condiciones, consultas y cota PGD | Una tarea completa sin archivo ahora produce error, los 44 análisis reales pasan |
| El comando auditar devolvía éxito de terminal incluso con errores | Devuelve código uno cuando la revisión requiere correcciones | Caso defectuoso comprobado desde un proceso separado |
| Repetir el panel analítico podía sustituir tareas completas por pendientes y cambiar el cierre del presupuesto | Reutiliza tareas completas y no modifica el presupuesto cuando ya no hay trabajo, rechaza trabajo pendiente con presupuesto agotado | Los 44 estados y los bytes del presupuesto se conservan al solicitar nuevamente un panel completado |
| Repetir la suite podía reabrir selección y evaluación ya registradas | Bloqueo antes de consultar CUDA cuando ya existen selección y prueba | Caso dirigido rechaza repetir la suite sin abrir GPU |
| Quince tareas de representaciones tienen particion externa validation e interna training | Se corrige el generador de metadatos y se deja advertencia explícita sobre registros históricos | Caso de metadatos con función sustituida, no se regeneran ni alteran los quince resultados originales |
| Preparar datos aceptaba archivos existentes comprobando únicamente su tamaño | Verifica también CRC contra el miembro del ZIP | Acepta copias idénticas y rechaza una alteración del mismo tamaño en archivos sintéticos |
| Faltaba destacar diferencias de algunas adaptaciones | Se precisan transferencia solo en metadatos, ausencia de crop0.9 ejecutado, selección de lectura compartida, coste parcial y muestreo ERF | Contraste con tablas, figuras, ejecutores y archivos reales |

No se cambió el núcleo de modelos, pérdidas ni optimización utilizado para obtener los pesos, las correcciones operativas no se presentan como si hubieran sido validadas mediante una nueva ejecución experimental, las pruebas dirigidas solo verifican los caminos corregidos y la compatibilidad estructural de los modelos permanece comprobada por separado.

## Pendientes con impacto en la entrega

Permanecen 69 configuraciones de ampliación sin ejecutar y los calendarios largos sin reproducir, no hay evidencia local para concluir que el método sustituya los recortes pequeños ni para elegir un óptimo en los barridos incompletos, la cobertura de entrenamiento inferior a una época y una sola semilla limitan todas las conclusiones comparativas.

La igualación de parámetros de BagNet33 no se puede corregir manteniendo válidos sus pesos actuales, la transferencia literal a Places365 requiere datos y un protocolo externos ausentes, AMDIM y SD-BYOL conservan su cuestión de alcance, estos asuntos no se consideran resueltos por la existencia de configuraciones en JSON.

Se mantienen como asuntos de las fuentes el significado de final_training frente al entrenamiento exclusivo previsto, la errata de audio y los detalles de evaluación ausentes del paper de clasificación, el proyecto declara su protocolo de Market-1501 y ambas convenciones de caída de mAP sin atribuir esas reglas al artículo.

La revisión del material por el grupo, el envío al formulario y la defensa individual siguen pendientes, el ZIP conserva los encoders del seleccionado y todos los resultados pero omite por volumen los 48 checkpoints completos que permanecen en la carpeta local, quien reciba solo el ZIP podrá evaluar el seleccionado y no volver a evaluar los demás sin sus pesos locales.

## Consultar la evidencia sin GPU

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
python main.py auditar --dispositivo cpu --datos RUTA_A_LOS_HDF5
```

El comando inspecciona resultados existentes y actualiza el informe automático de coherencia y cobertura, no entrena ni ejecuta inferencia, los resultados de esta revisión se encuentran en revision_cumplimiento.json, revision_checkpoints_cpu.json y revision_correcciones_cpu.json dentro de resultados, la evidencia anterior de métricas, pesos, análisis, selección y presupuesto se conserva.
