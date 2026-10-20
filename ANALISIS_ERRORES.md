# Aciertos y errores del modelo seleccionado

## Alcance y selección

Este análisis corresponde exclusivamente a CO-BYOL con BagNet33 y encoder online, la configuración se seleccionó mediante validación antes de calcular prueba, las observaciones siguientes no se utilizan para modificar modelos, hiperparámetros ni el orden de los experimentos posteriores.

Se inspeccionaron visualmente veinte filas de resultados correspondientes a quince consultas distintas, para cada condición se tomaron los primeros cinco índices con identidades distintas y Rank-1 correcto y los primeros cinco con identidades distintas y Rank-1 incorrecto, la muestra se eligió por ese criterio reproducible y no por apariencia.

La selección contiene deliberadamente igual cantidad de aciertos y fallos y no estima la frecuencia real de error, en las 3368 consultas de prueba Rank-1 es 4.632 % con originales y 3.385 % con consultas reducidas, el mAP correspondiente es 1.587 % y 1.334 %, los valores sin redondear y los índices se conservan en prueba.json e indices_ejemplos.json.

La identidad correcta se establece mediante las etiquetas suministradas y la cámara de captura, no mediante reconocimiento manual de rostros, las tablas separan lo observado en las imágenes de las hipótesis que podrían explicar la recuperación, esas hipótesis no constituyen una demostración causal.

## Cinco aciertos con consultas originales

| Consulta | Identidad | Primer positivo | AP % | Observación visual | Interpretación limitada |
|---|---:|---:|---:|---|---|
| 18 | 725 | Rank 1 | 8.958 | Camiseta amarilla, pantalón corto oscuro y mochila oscura en consulta y primer resultado, otros candidatos también llevan amarillo | La combinación visible puede ayudar, el amarillo por sí solo no distingue la identidad |
| 65 | 473 | Rank 1 | 5.526 | Prenda superior azul, parte inferior negra y mochila, vista posterior o lateral en ambos | La distribución de colores y el objeto transportado ofrecen correspondencias, quedan otros positivos mal ordenados |
| 77 | 746 | Rank 1 | 9.523 | Parte superior clara y falda oscura, cambian la pose y el fondo | El primer resultado conserva la combinación de prendas, varios negativos comparten zonas claras y oscuras |
| 103 | 345 | Rank 1 | 12.830 | Camiseta azul oscura y pantalón corto azul, los dos primeros resultados tienen la identidad correcta | La apariencia se mantiene entre cámaras en los ejemplos, no demuestra invariancia general |
| 117 | 1270 | Rank 1 | 7.698 | Camiseta granate con dibujo y pantalón corto negro, el resto de candidatos incluye otras prendas rojizas | El detalle de la prenda podría contribuir, la paleta también produce vecinos incorrectos |

![Cinco aciertos originales](resultados/ejemplos/original_aciertos.png)

Un acierto en Rank-1 no implica recuperar pronto todos los positivos, AP utiliza las posiciones de todos los positivos relevantes de la galería y no solo las cinco imágenes mostradas, por eso una consulta puede acertar el primer resultado y conservar un AP reducido.

## Cinco fallos con consultas originales

| Consulta | Identidad real | Primer resultado | Primer positivo | AP % | Observación visual | Hipótesis, no causalidad demostrada |
|---|---:|---:|---:|---:|---|---|
| 0 | 1191 | ID 294 | Rank 30 | 1.013 | La consulta combina chaqueta clara y prenda verde, los primeros vecinos muestran grandes regiones claras y accesorios oscuros, el positivo de referencia es más borroso y cambia la vista | La representación podría priorizar regiones claras y perder la combinación verde con chaqueta, también intervienen encuadre y nitidez |
| 1 | 873 | ID 776 | Rank 81 | 0.203 | Consulta con parte superior azul y pantalón negro, el primer resultado tiene zonas azules y claras con peor nitidez y otro encuadre | La similitud de regiones de color podría superar diferencias de prendas y silueta |
| 2 | 360 | ID 722 | Rank 2645 | 0.038 | Predominan camiseta amarilla y bolso oscuro en consulta y negativos, aparecen bicicletas y distintas prendas inferiores | El color y los objetos comunes pueden crear confusión entre personas con apariencia semejante |
| 3 | 1383 | ID 505 | Rank 527 | 0.221 | Consulta con prenda superior clara, pantalón azul y bolso con dibujo naranja, el primer vecino comparte parte superior clara y pantalón azul pero cambia el bolso | La representación podría no dar suficiente peso al detalle distintivo del accesorio |
| 4 | 1271 | ID 63 | Rank 135 | 0.643 | Consulta con ropa oscura y mochila con dibujo, el primer resultado tiene ropa clara y una bicicleta ocupa gran parte de la zona inferior | El fondo, los objetos y el recorte podrían influir, la diferencia de prendas impide explicar el error únicamente por coincidencia de color |

![Cinco fallos originales](resultados/ejemplos/original_fallos.png)

El primer resultado de la consulta cero pertenece a cámara tres y la consulta también, se conserva porque tiene otra identidad, el protocolo excluye únicamente la combinación de misma identidad y misma cámara y no todas las imágenes de una cámara.

El positivo de referencia mostrado en el material audiovisual es el primer índice de galería con la identidad correcta y otra cámara, se añade para comparar su apariencia y no se afirma que su rango sea el del primer positivo recuperado, su índice y cámara se conservan en referencias_video.json.

## Cinco aciertos con consultas reducidas

| Consulta | Identidad | Primer positivo | AP % | Observación visual | Interpretación limitada |
|---|---:|---:|---:|---|---|
| 20 | 784 | Rank 1 | 7.051 | Persisten zonas claras de camiseta y bolso y una zona inferior gris, los puestos dos y cuatro incluyen distractores con etiqueta cero | La información de color y disposición puede bastar para este primer acierto, los distractores siguen compitiendo |
| 28 | 137 | Rank 1 | 4.266 | La consulta pixelada mantiene la división entre parte superior blanca y pantalón oscuro, varios negativos tienen esa combinación | La reducción no elimina toda la estructura gruesa, el acierto no resuelve la ambigüedad de las prendas comunes |
| 164 | 520 | Rank 1 | 19.123 | Camiseta naranja rojiza y pantalón corto oscuro, varios de los primeros puestos son positivos de otras cámaras | La paleta y la silueta permanecen distinguibles en este caso |
| 171 | 824 | Rank 1 | 20.931 | Camiseta blanca, pantalón claro y objeto verde grande a la altura de la cintura, aparecen varios positivos próximos | El objeto transportado puede servir como señal compartida en estas imágenes, no demuestra reconocer a la persona sin ese objeto |
| 205 | 889 | Rank 1 | 16.534 | Prenda superior rosa, pantalón corto oscuro y bolso claro, los tres primeros resultados son positivos | Las regiones amplias de apariencia sobreviven a la reducción en este ejemplo |

![Cinco aciertos reducidos](resultados/ejemplos/baja_resolucion_aciertos.png)

## Cinco fallos con consultas reducidas

| Consulta | Identidad real | Primer resultado | Primer positivo | AP % | Observación visual | Hipótesis, no causalidad demostrada |
|---|---:|---:|---:|---:|---|---|
| 0 | 1191 | ID 0 | Rank 485 | 0.163 | La consulta pierde detalle de chaqueta y prenda verde, el primer vecino es un recorte parcial con verde, piernas y elementos del entorno | La información parcial del distractor podría dominar sobre la configuración completa de prendas |
| 1 | 873 | ID 731 | Rank 2 | 3.729 | El primer candidato combina zonas azules y claras con pantalón oscuro, el segundo tiene la identidad correcta | La confusión local persiste pero el primer positivo mejora mucho frente al original |
| 2 | 360 | ID 721 | Rank 1667 | 0.060 | Los primeros vecinos comparten amarillo y accesorios oscuros, sus identidades difieren de la consulta | La apariencia gruesa es insuficiente para separar estas identidades aunque el rango mejore |
| 3 | 1383 | ID 1085 | Rank 455 | 0.234 | Se pierde detalle del bolso naranja, los vecinos muestran combinaciones de prendas claras y accesorios oscuros | La pérdida de detalles puede reducir la discriminación entre configuraciones de color parecidas |
| 4 | 1271 | ID 831 | Rank 2048 | 0.071 | El dibujo de la mochila es poco distinguible, aparecen vecinos de prendas claras y un distractor con bicicleta | La reducción dificulta conservar señales finas y puede dejar mayor influencia a contornos y contexto |

![Cinco fallos reducidos](resultados/ejemplos/baja_resolucion_fallos.png)

La etiqueta cero del primer resultado de la consulta cero identifica un distractor según el protocolo, permanece como negativo y cuenta como fallo, no se transforma en una identidad positiva ni se elimina para mejorar las métricas, las imágenes con etiqueta menos uno se excluyen por su papel de junk.

## Cambios de rango con la misma consulta

| Consulta | Primer positivo original | Primer positivo reducido | AP original % | AP reducido % |
|---|---:|---:|---:|---:|
| 0 | 30 | 485 | 1.013 | 0.163 |
| 1 | 81 | 2 | 0.203 | 3.729 |
| 2 | 2645 | 1667 | 0.038 | 0.060 |
| 3 | 527 | 455 | 0.221 | 0.234 |
| 4 | 135 | 2048 | 0.643 | 0.071 |
| 18 | 1 | 28 | 8.958 | 2.369 |
| 20 | 10 | 1 | 2.685 | 7.051 |
| 28 | 174 | 1 | 0.365 | 4.266 |
| 65 | 1 | 90 | 5.526 | 0.272 |
| 77 | 1 | 2992 | 9.523 | 0.080 |
| 103 | 1 | 124 | 12.830 | 0.324 |
| 117 | 1 | 10 | 7.698 | 2.395 |
| 164 | 3 | 1 | 7.607 | 19.123 |
| 171 | 1 | 1 | 5.279 | 20.931 |
| 205 | 1 | 1 | 7.613 | 16.534 |

La reducción no empeora todas las consultas, la consulta setenta y siete pasa de Rank uno a 2992 y la consulta uno mejora de 81 a dos, las consultas 171 y 205 conservan Rank uno mientras mejora su AP porque más positivos aparecen antes, el promedio global reduce mAP en 0.252703 puntos porcentuales y 15.921314 % relativo.

Las consultas originales y reducidas corresponden a los mismos registros y utilizan los mismos pesos y galería, las diferencias anteriores describen el efecto observado al cambiar la información de entrada y no permiten identificar por sí solas qué textura, color o región interna produjo cada cambio.

## Límites de interpretación

En estos ejemplos predominan confusiones de apariencia, recortes parciales, objetos del entorno y pérdida de nitidez por resolución, no se cuantifica una causa específica atribuible a iluminación ni se inspeccionaron visualmente todas las consultas de prueba, para establecer causas harían falta intervenciones controladas adicionales.

La galería de prueba tiene 19732 registros frente a 1793 en validación y contiene más identidades y distractores, por eso la diferencia entre mAP de validación y prueba no se atribuye exclusivamente a sobreajuste, también cambia la dificultad del conjunto de recuperación.

El entrenamiento de 1253 actualizaciones con batch ocho procesa 10024 imágenes fuente y equivale a aproximadamente 0.91 épocas, la baja exposición, una sola semilla y la comparación entre arquitecturas de costes distintos limitan las conclusiones, los errores observados no establecen el rendimiento final del método después de un entrenamiento largo.
