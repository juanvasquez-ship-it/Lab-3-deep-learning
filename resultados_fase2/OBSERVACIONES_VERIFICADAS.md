# Observaciones verificadas y decisiones de fase 2

Las observaciones externas son propuestas que se contrastaron con el código, los resultados guardados, las tres páginas del laboratorio y las ecuaciones, sección 3 y apéndices del paper recibido, el ZIP mejoras_lab3_noche.zip citado en el texto no fue recibido y no se aplicaron archivos ni instrucciones de ese parche ausente.

| Observación | Dictamen y evidencia | Decisión y motivo |
|---|---|---|
| Menos de una época por configuración | Confirmado, cada resumen registra 1253 pasos con batch 8 y 0.907971 épocas equivalentes | Conservar fase 1 como evidencia preliminar válida y continuar un panel mínimo con presupuestos comparables, no invalidar automáticamente resultados correctamente medidos |
| LARS con tasa pequeña y factor de confianza 0.001 | Confirmados los valores del código, tasa inicial 0.2*8/256 igual a 0.00625 antes del calendario, que el encoder apenas cambie o sea equivalente a uno sin entrenar no estaba medido | Medir validación de inicialización aleatoria y cambio relativo del primer tensor en pilotos, probar un subconjunto de tasas publicadas en validación, sin sustituir LARS ni afirmar inestabilidad BatchNorm sin medición |
| Deformación de 128x64 a 224x224 | Confirmada en Vistas y preparar_entrada, la razón de áreas es 6.125 y no demuestra igual factor de tiempo real | Mantener 224 en las catorce arquitecturas para conservar protocolo común, tamaños locales y geometría de ViT/Swin, una variante rectangular requiere adaptación y comparación aparte |
| Ausencia de degradación LR en training | Confirmada, el laboratorio permite generarla pero no la exige | Posponer para no mezclar una intervención nueva con calendario, tasas y arquitecturas, se usan las LR suministradas únicamente al evaluar |
| Tope histórico fijo de tres horas | Confirmado y coherente con la autorización anterior, no es un fallo de aquellas ejecuciones | Añadir una acción fase2 que lee el nuevo plazo absoluto de diez horas, sin cambiar presupuestos ni guardas históricos ni crear diez horas por modelo |
| Cien consultas de validación y una semilla | Confirmado por los HDF5 y los registros | Conservar las particiones, limitar el barrido y las conclusiones, no inventar consultas, nuevas divisiones, significación estadística ni convergencia |

## Experimentos seleccionados y alcance

El panel nuevo continúa los catorce checkpoints CO-BYOL de la referencia a 224 y cinco controles, BYOL sin CO para ResNet50, RF99 y BagNet33, y MoCoV3 con y sin CO para ResNet50, el presupuesto común es de hasta tres épocas completas adicionales por configuración en rondas, se deja de iniciar otra ronda si no queda reserva suficiente para terminarla y evaluar.

Se conserva batch físico ocho, validado mediante pasos reales de RF39, RF29 y Swin-B con recomputación operativa, los pesos históricos se cargan estrictamente y el estado del optimizador y el calendario EMA de continuación se reinician de manera explícita, no se presentan como una reanudación idéntica de un calendario largo originalmente planificado.

Los pilotos usan RF99 para una tasa común CNN entre 0.2, 0.8 y 1.6, y ViT-S/16 para una tasa común Transformer entre 0.0003, 0.0005 y 0.001, se eligen por validación y se descartan sus pesos, este barrido reducido no sustituye el ajuste completo por arquitectura del paper, el cambio relativo del primer tensor tampoco equivale al cambio de todos los pesos ni demuestra la estabilidad de BatchNorm.

La prueba rápida comprobó entrenamiento, checkpoint, recuperación de optimizador y RNG, validación completa de ambas ramas y exportación exacta de encoders, no abrió testing, se descartó la sugerencia de usar prueba durante el diagnóstico para preservar su función final.

Cada configuración elige época y rama con la media de mAP de validación original y reducida, ambas ramas de esa misma época se evalúan tras congelar toda la selección, si se conserva el checkpoint de inicio se reutilizan sus métricas históricas válidas cuando no hacen falta nuevos rankings para los ejemplos, conocer pruebas históricas impide presentar esta fase como un nuevo conjunto ciego.

## Plazo y ejecución independiente

El plazo comenzó el 20 de octubre a las 00:32:19 de Bogotá, las correcciones y el inicio de los entrenamientos deben quedar dentro de las 08:32:19 y el cierre total dentro de las 10:32:19, plazo_global.json conserva los instantes absolutos y no se reinicia al reanudar, se reserva una hora y media para evaluación y el supervisor deja además margen de cierre CPU.

El supervisor mantiene el equipo despierto únicamente mientras vive su proceso y no modifica la política global de energía, registra su PID, fase y salida, guarda checkpoints atómicos y deja tablas, curvas y verificación al finalizar, no configura notificaciones ni seguimiento automático en el chat.

Permanecen fuera de alcance los calendarios largos y datasets de clasificación del paper, Places365, múltiples semillas, la igualación literal de parámetros BagNet33 y todos los cruces redundantes, no se prometerán sus conclusiones con unas pocas épocas de recuperación.
