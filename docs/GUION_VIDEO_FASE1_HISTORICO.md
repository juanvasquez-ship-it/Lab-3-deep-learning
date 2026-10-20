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

## 7, Implementación, Datos y particiones proporcionadas

Los datos se leen desde los tres archivos HDF cinco proporcionados, training contiene once mil cuarenta y tres imágenes y no incluye etiquetas de persona, validación contiene cien consultas y mil setecientas noventa y tres imágenes de galería, prueba contiene tres mil trescientas sesenta y ocho consultas y diecinueve mil setecientas treinta y dos imágenes de galería, las versiones reducidas ya están disponibles y conservan el orden de las consultas originales, no se crea una nueva división y no se modifica ningún archivo original, el atributo final training sugiere incorporar validación posteriormente pero esa fase no se aplica porque no está exigida por el PDF y cambiaría la comparación del entrenamiento.

## 8, Implementación, Entradas, aumentos y resolución

Cada imagen de training se transforma únicamente en memoria para generar las vistas de entrenamiento, se usan recorte, inversión horizontal y variaciones de color, escala de grises, desenfoque y solarización, el rango de aspecto de los recortes se adapta a las imágenes verticales de personas y se declara como una diferencia respecto a las imágenes cuadradas de referencia, los modelos principales utilizan entradas de doscientos veinticuatro por doscientos veinticuatro y el panel pequeño utiliza sesenta y cuatro, durante evaluación las consultas originales y reducidas reciben la misma adaptación de tamaño del encoder, ampliar una consulta reducida no recupera detalle ni constituye un modelo de superresolución., La comparación ampliada utiliza además entrada doscientos veinticuatro para las catorce arquitecturas, ResNet dieciocho y RF veintinueve a sesenta y cuatro permanecen como panel pequeño separado.

## 9, Implementación, Código propio y comparadores

El núcleo del método se implementa en modelos punto py y no se importa una implementación completa de CO SSL, las bibliotecas generales aportan operaciones y arquitecturas estándar, el código distingue el proyector de BYOL del de MoCo versión tres y conserva cabezas locales independientes salvo en la ablación compartida, los comparadores multicrop generan vistas pequeñas adicionales y DINO utiliza centrado, temperatura y una cabeza propia con actualización objetivo, también existen configuraciones de peso local, cantidad de posiciones, recorte, MLP, kernel y capa de alineamiento, programar una configuración no demuestra haberla ejecutado y el informe comprueba cada caso mediante sus registros.

## 10, Implementación, Dos sesiones autorizadas y presupuesto comparable

La primera autorización permitió tres horas y la sesión registrada duró 171.6 minutos, después el usuario autorizó continuar para completar las arquitecturas, esa continuación tiene un presupuesto separado y duró 99.0 minutos, no se modifica el reloj anterior ni se presenta la suma como una única sesión de tres horas, los modelos conservan batch físico y efectivo ocho y mil doscientas cincuenta y tres actualizaciones, RF treinta y nueve usa recomputación para reducir memoria y conservar las estadísticas de normalización, esa técnica aumenta el tiempo y no sustituye los datos ni las actualizaciones.

## 11, Implementación, Validación y evaluación sin ajuste de prueba

La configuración y la rama se seleccionan con la media de mAP de validación en las condiciones original y reducida, esa elección se guarda antes de evaluar prueba, para Market mil quinientos uno la documentación oficial indica ignorar los elementos con identidad menos uno y las coincidencias de la misma persona y cámara, los elementos con identidad cero permanecen como distractores, estas reglas no se atribuyen al artículo porque no define el protocolo de reidentificación, AP es la media de precisiones en las posiciones positivas y los empates conservan el orden de galería, se informan mAP, Rank uno, cinco y diez y caída de mAP absoluta y relativa.

## 12, Implementación, Qué se comprueba y qué queda pendiente

Las verificaciones comprueban pérdidas con valores conocidos, detención de gradiente y métricas con distractores y exclusiones, las siete colecciones de imágenes se revisaron en la fase de análisis y las consultas reducidas coincidieron con una reducción bicúbica de sus originales, la ausencia de duplicados exactos no demuestra separación de identidades porque training no tiene sus etiquetas, durante ejecución cada modelo guarda configuración, pasos, imágenes distintas, pérdida, memoria, checkpoint y métricas, los calendarios de cientos de épocas, la transferencia a Places y los experimentos que no caben en tres horas permanecen pendientes, esa distinción evita presentar como resultados números que el equipo no ha calculado., La recomputación se verificó por separado con dos vistas y estadísticas BatchNorm, salidas, gradientes y buffers coinciden en el caso de control CPU, los checkpoints y los tiempos de cada fase quedan registrados para revisar el progreso.

## 13, Resultados, Comparación de las catorce arquitecturas

El proyecto conserva 54 entrenamientos completos contando los resultados anteriores y la continuación, la comparación común contiene catorce arquitecturas con CO BYOL, peso local cero coma dos y entrada doscientos veinticuatro, RF treinta y nueve y RF ciento quince completan las arquitecturas pendientes y se añaden versiones grandes de ResNet dieciocho y RF veintinueve para no mezclar resoluciones en la tabla, se reutilizan diez configuraciones anteriores sin reentrenarlas, el archivo comparativo muestra todas las métricas, tiempo, parámetros y memoria, los tiempos históricos de validación o prueba que no se midieron por separado se declaran ausentes.

## 14, Resultados, Modelo seleccionado mediante validación

La selección principal corresponde a CO-BYOL bagnet33 con rama online, se eligió entre las configuraciones principales utilizando exclusivamente validación y se conservó antes de calcular prueba, en las consultas originales obtiene mAP de 1,59 por ciento y Rank uno de 4,63 por ciento, con consultas reducidas obtiene mAP de 1,33 por ciento y Rank uno de 3,38 por ciento, las métricas Rank cinco y diez también aparecen en pantalla y en los archivos, la diferencia de mAP es 0,25 puntos porcentuales, una diferencia negativa se conserva porque representa una mejora bajo reducción en esta ejecución.

## 15, Resultados, Modelo base y ablación del término local

Para analizar el efecto del componente CO se comparan modelos que conservan la arquitectura, en la rama online de prueba CO-BYOL RF99 frente a BYOL RF99 presenta una diferencia de 0,09 puntos con originales y 0,11 con consultas reducidas, CO-BYOL bagnet33 frente a BYOL bagnet33 presenta una diferencia de 0,04 puntos con originales y -0,14 con consultas reducidas, esas diferencias se calculan directamente de los archivos de métricas y pueden tener signos distintos entre condiciones, eliminar la pérdida local manteniendo RF noventa y nueve constituye la ablación principal del laboratorio, comparar ResNet cincuenta con RF noventa y nueve también cambia la arquitectura y se interpreta como una comparación adicional, las cabezas compartidas, la ausencia de MLP y los barridos de peso y recorte se documentan por separado cuando se ejecutaron, una sola semilla y menos de una época no permiten establecer superioridad general.

## 16, Resultados, Comparación entre online y objetivo EMA

El gráfico compara las dos ramas del entrenamiento seleccionado, con consultas originales la rama online obtiene mAP de 1,59 por ciento y la rama objetivo obtiene 1,37 por ciento, las dos condiciones se evalúan con cada encoder congelado y la misma galería, EMA suaviza la evolución de parámetros pero eso no garantiza una mejor recuperación en un entrenamiento tan corto, la tabla del informe conserva esta comparación para cada modelo, los análisis adicionales de robustez fijan online para evitar mezclar ramas entre comparadores, las curvas de pérdida describen el objetivo de aprendizaje y no sustituyen las métricas de recuperación.

## 17, Resultados, Aciertos y errores comprobables

El informe incluye al menos cinco aciertos y cinco fallos por condición, se seleccionan los primeros índices de identidades distintas con Rank uno correcto o incorrecto y no se eligen manualmente por apariencia, la consulta 18 corresponde a un acierto y recupera la identidad 725 desde otra cámara, en la consulta 0 el primer resultado tiene identidad 294 y el primer positivo relevante aparece en Rank 30, la figura añade un positivo de referencia para comparar el contenido disponible y no afirma que ese ejemplo sea el primer positivo recuperado, el análisis visual distingue observaciones de iluminación, encuadre y apariencia de hipótesis sobre las causas del error, las hipótesis no se presentan como causalidad demostrada.

## 18, Resultados, Conclusiones y límites del trabajo

La mayor media descriptiva de mAP en prueba corresponde a resnet18, la menor caída relativa corresponde a vit_s16 y el mayor cociente entre mAP y minutos corresponde a bagnet33, esos criterios pueden favorecer arquitecturas distintas y no se fuerza un único ganador, conservar un porcentaje alto de un resultado bajo no equivale a obtener mayor precisión absoluta, las configuraciones se fijaron por cobertura y validación y estas puntuaciones de prueba no se utilizaron para ajustar hiperparámetros, las diferencias pequeñas no son concluyentes con una sola semilla y menos de una época, BagNet treinta y tres mantiene su diferencia de parámetros frente al artículo y los calendarios largos y la transferencia permanecen fuera de la ejecución local, el grupo debe revisar el material y preparar su defensa.

## Créditos de narración

Síntesis local con Piper es_ES-sharvard-medium, datos Spanish Harvard sentences de la Universidad de Edimburgo, licencia CC BY 3.0.

Modelo, https://huggingface.co/rhasspy/piper-voices/tree/main/es/es_ES/sharvard/medium

Datos, https://datashare.ed.ac.uk/handle/10283/574

Motor, https://github.com/OHF-Voice/piper1-gpl
