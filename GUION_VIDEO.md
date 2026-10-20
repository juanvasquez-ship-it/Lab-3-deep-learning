# Guion vigente, fase 2

Corresponde a [resultados_fase2/audiovisual/VIDEO.mp4](resultados_fase2/audiovisual/VIDEO.mp4), el guion y el video de fase 1 se conservan como evidencia histórica, la revisión del grupo y su defensa siguen pendientes.

# Guion del video explicativo

El video utiliza narración sintética local como material de apoyo, el grupo debe revisar su contenido y preparar la explicación y defensa individual exigida por el laboratorio.

## 1, Metodología, Aprender a reconocer personas sin etiquetas

Este video presenta la adaptación de CO SSL al laboratorio de reidentificación de personas, la tarea consiste en representar una imagen de consulta y ordenar una galería para encontrar a la misma persona observada por otra cámara, el entrenamiento utiliza exclusivamente imágenes sin etiquetas de identidad, la referencia es el artículo Seeing the whole in the parts with self supervised representation learning de Aubret, Teulière y Triesch, publicado en dos mil veintiséis, separaremos la propuesta del artículo de las adaptaciones exigidas por el laboratorio y de las limitaciones de la ejecución local, esta narración sintética es un material de apoyo y no sustituye la preparación de la defensa individual.

## 2, Metodología, De vistas aumentadas a representaciones

El aprendizaje autosupervisado parte de una imagen y genera dos vistas mediante recortes y cambios de apariencia, el encoder extrae un mapa espacial de características y el promedio de ese mapa produce una representación global, un proyector transforma la representación para calcular la pérdida y un predictor aparece únicamente en la rama online, las dos vistas de una imagen forman la correspondencia positiva sin conocer quién aparece en ella, dos fotografías distintas de la misma persona no se fuerzan como positivas porque sus identidades no están disponibles durante el aprendizaje, esta diferencia entre correspondencia de vistas e identidad de persona es fundamental para entender el alcance del método.

## 3, Metodología, Rama online y objetivo EMA

El modelo mantiene una rama online y una rama objetivo, la primera se actualiza por descenso de gradiente y contiene el predictor, la segunda recibe una media móvil exponencial de los parámetros online y sus salidas se detienen para el cálculo del gradiente, la actualización objetivo combina los parámetros anteriores con una fracción pequeña de los nuevos parámetros online, el coeficiente comienza en cero coma novecientos noventa y seis y aumenta hacia uno durante el calendario local, las estadísticas de normalización por lotes proceden de las propias pasadas de cada rama, después del entrenamiento se evalúan ambos encoders congelados porque el laboratorio pide comparar maestro y estudiante.

## 4, Metodología, Qué añade CO a BYOL y MoCoV3

BYOL alinea una predicción con una proyección objetivo mediante coseno, MoCo versión tres utiliza una pérdida contrastiva con negativos procedentes del batch, la contribución CO conserva esa pérdida global y añade dos alineamientos, cada representación local online se compara con la global objetivo de la otra vista y la predicción global online se compara con las locales objetivo, los términos se promedian por posición para que la escala no crezca con el número de locales y se calculan intercambiando las vistas, el peso local controla su contribución, poner ese peso a cero permite una ablación que conserva el encoder y elimina el componente principal de la propuesta.

## 5, Metodología, Campo receptivo y arquitecturas

El artículo propone limitar el campo receptivo para que las representaciones locales dependan de partes de la imagen, RF ResNet reemplaza muchas convoluciones espaciales por convoluciones de uno por uno y modifica algunos pasos de reducción, RF noventa y nueve conserva un contexto local menor que la imagen de doscientos veinticuatro píxeles y añade un bloque residual después del promedio, también se implementan ResNet estándar, RF veintinueve para el panel pequeño, BagNet y variantes adicionales, ViT y Swin utilizan sus tokens finales y una cabeza local que agrupa ventanas espaciales, estas familias tienen costes y resoluciones diferentes por lo que una comparación entre todas ellas no aísla exclusivamente la pérdida CO.

## 6, Metodología, Adaptación a recuperación de personas

El artículo evalúa representaciones mediante clasificación supervisada con una sonda lineal, el laboratorio exige recuperar personas sin utilizar sus identidades durante el entrenamiento, por eso conservamos el aprendizaje autosupervisado y utilizamos directamente la representación del encoder para ordenar la galería, los vectores se normalizan en norma ele dos y su producto escalar equivale a similitud coseno, esta adaptación evita entrenar un clasificador de identidad y respeta las entradas exclusivamente visuales, sus resultados de mAP y Rank no son numéricamente comparables con la exactitud de clasificación del artículo, reproducir la idea metodológica no significa haber reproducido el rendimiento publicado en otro dataset.

## 7, Implementación, Mismas particiones en ambas fases

La segunda fase conserva exactamente los archivos suministrados, once mil cuarenta y tres imágenes de entrenamiento sin identidades, cien consultas y mil setecientas noventa y tres imágenes de galería de validación, tres mil trescientas sesenta y ocho consultas y diecinueve mil setecientas treinta y dos imágenes de galería de prueba, no se crea una nueva división ni se incorpora validación al aprendizaje, las identidades solo intervienen al medir recuperación, el atributo final training no se aplica como una instrucción adicional ausente del laboratorio.

## 8, Implementación, Entrada común y mejoras pospuestas

Se mantiene la entrada cuadrada de doscientos veinticuatro por doscientos veinticuatro para las catorce arquitecturas de la referencia común, adaptar las personas a ese tamaño deforma su proporción y aumenta el área respecto de la imagen original, esa limitación se declara, pasar a un formato rectangular exigiría adaptar las cuadrículas de los transformers y la geometría local y volver a evaluar la comparación, la degradación de resolución durante aprendizaje está permitida pero no es obligatoria en el PDF, se pospone para no mezclar otra intervención con la continuación de entrenamiento.

## 9, Implementación, Continuación del método propio

El núcleo propio de CO BYOL y CO MoCo versión tres se conserva, las pérdidas y el mecanismo de transferencia no se sustituyen por una biblioteca de aprendizaje autosupervisado, la recomputación de activaciones reduce memoria sin duplicar las actualizaciones de BatchNorm, se verificaron salidas, gradientes y buffers en una prueba dirigida CPU y se ejecutaron pasos reales en las arquitecturas de mayor memoria, la continuación utiliza los pesos válidos de la primera fase y reinicia explícitamente su optimización y calendario de transferencia, no se presenta como una reanudación idéntica de un entrenamiento largo originalmente planificado.

## 10, Implementación, Un plazo global para toda la fase 2

La fase uno conservó cincuenta y cuatro configuraciones con mil doscientas cincuenta y tres actualizaciones cada una, recibió dos autorizaciones separadas y no se afirma que todas esas ejecuciones cupieran en tres horas, la fase dos dispone de un plazo global nuevo de diez horas desde el inicio de la revisión, no por modelo, completó 2 rondas adicionales comunes para diecinueve configuraciones, catorce arquitecturas y cinco controles, se conservó el batch físico ocho, las rondas se planificaron con tiempos medidos y se reservó tiempo para validación, prueba y cierre, las pocas épocas y una sola semilla siguen siendo una limitación.

## 11, Implementación, Selección sin consultar prueba

Un barrido reducido de tasas publicadas se realizó con RF noventa y nueve para las CNN y ViT para los transformers, sus pesos se descartaron y su validación eligió una tasa común por familia, no se pretende reproducir el ajuste individual completo del artículo, después cada configuración seleccionó época y rama mediante la media de mAP de validación original y reducida, toda la selección se cerró antes de la prueba de fase dos, se conservaron ambas ramas del mismo checkpoint elegido, como ya existían pruebas históricas, esta ampliación no constituye un conjunto de prueba completamente ciego.

## 12, Implementación, Red sin entrenar y curvas reales

Ahora se mide la representación de cada arquitectura con inicialización aleatoria en validación y se contrasta con sus checkpoints aprendidos, cinco de las catorce referencias mantienen menor mAP de validación que su inicialización aleatoria, esta diferencia observada no tiene una prueba de significación ni demuestra inestabilidad de BatchNorm, los pilotos registran el cambio relativo del primer tensor y no el de toda la red, se comparan las curvas de validación y las pérdidas sin asumir convergencia, cien consultas y una semilla limitan la interpretación de diferencias pequeñas.

## 13, Resultados, Resultados de las catorce arquitecturas

La tabla completa conserva las catorce arquitecturas y los controles con sus épocas, ramas, configuraciones y costes, la mayor media descriptiva de mAP entre consultas originales y reducidas en el panel de arquitecturas corresponde a bagnet33, esto describe los resultados finales y no se utiliza para ajustar hiperparámetros, deben leerse por separado la precisión absoluta, la caída relativa y el tiempo de entrenamiento, una arquitectura que retiene una fracción alta de una precisión inicial baja no es necesariamente la mejor recuperadora absoluta.

## 14, Resultados, Modelo elegido por validación

La selección global mediante validación corresponde a co_byol_bagnet33, rama objetivo, en prueba obtiene mAP original 2.167 por ciento y mAP reducida 2.014 por ciento, la época adicional elegida es 2, una época adicional cero significa conservar los pesos aprendidos de fase uno y no una red aleatoria, la selección puede conservar un checkpoint anterior si validación no mejora, eso se informa sin cambiar las conclusiones para favorecer la continuación o el componente CO.

## 15, Resultados, Bases y ablación a arquitectura fija

Los controles incluyen BYOL sin CO para ResNet cincuenta, RF noventa y nueve y BagNet treinta y tres, además de MoCo versión tres con y sin CO en ResNet cincuenta, se comparan con presupuesto común y las mismas particiones, quitar el término local conserva el encoder y constituye la ablación principal, no se sustituye una arquitectura diferente y se atribuye toda la diferencia a CO, las tablas conservan mejoras y deterioros en ambas condiciones sin cambiar la conclusión para favorecer la propuesta, el control histórico de recorte mínimo cero coma nueve sigue siendo una evidencia separada y no se atribuye al entrenamiento nuevo.

## 16, Resultados, Maestro, estudiante y costes

Las dos ramas del mismo checkpoint seleccionado por validación se comparan en los JSON de prueba, no se elige la rama mirando prueba, el costo conserva tiempos reales de entrenamiento de ambas fases y los de evaluación cuando están medidos, el máximo de memoria corresponde a memoria asignada por PyTorch y no al consumo total del dispositivo, el cociente entre mAP media y minutos describe un equilibrio local entre calidad y tiempo y no es una medición energética, la actividad de otras aplicaciones y la recomputación también afectan los tiempos, las explicaciones causales requieren comparaciones controladas adicionales.

## 17, Resultados, Aciertos y fallos con índices verificables

Los ejemplos se eligen de forma reproducible por índice y resultado de Rank uno, conservando identidades de consulta distintas, se incluyen cinco aciertos y cinco fallos por condición de resolución, las explicaciones verifican identidad, cámara del primer recuperado y rango del primer positivo válido, un fallo sitúa otra persona antes de cualquier positivo después de las exclusiones del protocolo, la similitud de ropa, el fondo, la oclusión o el encuadre son posibles explicaciones que deben comprobarse visualmente y no se anuncian como causas demostradas por una tabla de etiquetas.

## 18, Resultados, Alcance y pendientes declarados

La fase dos amplía la exposición de entrenamiento y añade mediciones que faltaban, conserva los resultados históricos y deja explícita su relación con el código actualizado, aun así no demuestra convergencia ni reproduce los experimentos largos de clasificación del artículo, permanecen fuera de alcance Places trescientos sesenta y cinco, múltiples semillas, la igualación literal de parámetros de BagNet y todos los barridos redundantes, el grupo debe revisar los documentos y este material sintético, realizar el envío académico y preparar la defensa individual, subir código o generar un video no equivale a haber realizado esas actividades humanas.

## Créditos de narración

Síntesis local con Piper es_ES-sharvard-medium, datos Spanish Harvard sentences de la Universidad de Edimburgo, licencia CC BY 3.0.

Modelo, https://huggingface.co/rhasspy/piper-voices/tree/main/es/es_ES/sharvard/medium

Datos, https://datashare.ed.ac.uk/handle/10283/574

Motor, https://github.com/OHF-Voice/piper1-gpl
