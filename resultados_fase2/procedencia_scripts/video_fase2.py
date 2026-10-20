import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
root = Path(__file__).resolve().parents[1]
project = root / 'outputs/Laboratorio3'
destination = project / 'resultados_fase2'
mini = destination / 'audiovisual'
assets = mini / 'resultados'
assets.mkdir(parents=True, exist_ok=True)
cache = root / 'work/video_fase2'
cache.mkdir(exist_ok=True)
old_cache = root / 'work/video'
rows = json.loads((destination / 'comparacion_fase2.json').read_text(encoding='utf-8'))
reference = [r for r in rows if r['grupo'] == 'arquitecturas' and r['estado'] == 'evaluado']
assert len(reference) == 14
selection = json.loads((destination / 'seleccion_global_validacion.json').read_text(encoding='utf-8'))['modelo']
picked = next(r for r in rows if r['nombre'] == selection['nombre'])
epochs = min(r['epocas_fase2_completas'] for r in rows)
best = max(reference, key=lambda r: r['original_mAP_porcentaje'] + r['baja_resolucion_mAP_porcentaje'])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
figure, axis = plt.subplots(figsize=(12, 7))
positions = np.arange(14)
axis.barh(positions - 0.18, [r['original_mAP_porcentaje'] for r in reference], 0.35, label='Original')
axis.barh(positions + 0.18, [r['baja_resolucion_mAP_porcentaje'] for r in reference], 0.35, label='Consulta LR')
axis.set_yticks(positions, [r['arquitectura'] for r in reference])
axis.set_xlabel('mAP %')
axis.set_title('Fase 2, época y rama elegidas por validación')
axis.invert_yaxis()
axis.legend()
figure.tight_layout()
figure.savefig(assets / 'comparacion.png', dpi=150)
plt.close(figure)
shutil.copy2(project / 'resultados/arquitectura.png', assets / 'arquitectura.png')
shutil.copy2(destination / 'curvas_fase2.png', assets / 'curvas.png')
shutil.copy2(destination / 'ejemplos/original_fallos.png', assets / 'fallos.png')
slides = json.loads((old_cache / 'contenido.json').read_text(encoding='utf-8'))

def change(index, title, screen, narration, image=None):
    slides[index] = {'seccion': slides[index]['seccion'], 'titulo': title, 'pantalla': screen, 'narracion': narration}
    if image:
        slides[index]['imagen'] = image

change(6, 'Mismas particiones en ambas fases', '11043 imágenes de training sin identidades\n100 consultas de validación y 3368 de prueba\nSin nuevas divisiones ni incorporación de validación a training', 'La segunda fase conserva exactamente los archivos suministrados, once mil cuarenta y tres imágenes de entrenamiento sin identidades, cien consultas y mil setecientas noventa y tres imágenes de galería de validación, tres mil trescientas sesenta y ocho consultas y diecinueve mil setecientas treinta y dos imágenes de galería de prueba, no se crea una nueva división ni se incorpora validación al aprendizaje, las identidades solo intervienen al medir recuperación, el atributo final training no se aplica como una instrucción adicional ausente del laboratorio.')
change(7, 'Entrada común y mejoras pospuestas', '224 por 224 en las catorce arquitecturas\nSin degradación LR nueva durante aprendizaje\nCambiar a rectángulos exigiría otra comparación', 'Se mantiene la entrada cuadrada de doscientos veinticuatro por doscientos veinticuatro para las catorce arquitecturas de la referencia común, adaptar las personas a ese tamaño deforma su proporción y aumenta el área respecto de la imagen original, esa limitación se declara, pasar a un formato rectangular exigiría adaptar las cuadrículas de los transformers y la geometría local y volver a evaluar la comparación, la degradación de resolución durante aprendizaje está permitida pero no es obligatoria en el PDF, se pospone para no mezclar otra intervención con la continuación de entrenamiento.')
change(8, 'Continuación del método propio', 'CO-BYOL y CO-MoCoV3 propios\nRecomputación para mantener batch ocho\nPesos válidos de fase 1 y optimización nueva explícita', 'El núcleo propio de CO BYOL y CO MoCo versión tres se conserva, las pérdidas y el mecanismo de transferencia no se sustituyen por una biblioteca de aprendizaje autosupervisado, la recomputación de activaciones reduce memoria sin duplicar las actualizaciones de BatchNorm, se verificaron salidas, gradientes y buffers en una prueba dirigida CPU y se ejecutaron pasos reales en las arquitecturas de mayor memoria, la continuación utiliza los pesos válidos de la primera fase y reinicia explícitamente su optimización y calendario de transferencia, no se presenta como una reanudación idéntica de un entrenamiento largo originalmente planificado.')
change(9, 'Un plazo global para toda la fase 2', f'19 configuraciones, 14 arquitecturas y 5 controles\n{epochs} rondas completas adicionales, una semilla\nDiez horas máximas desde el inicio, con reserva de evaluación', f'La fase uno conservó cincuenta y cuatro configuraciones con mil doscientas cincuenta y tres actualizaciones cada una, recibió dos autorizaciones separadas y no se afirma que todas esas ejecuciones cupieran en tres horas, la fase dos dispone de un plazo global nuevo de diez horas desde el inicio de la revisión, no por modelo, completó {epochs} rondas adicionales comunes para diecinueve configuraciones, catorce arquitecturas y cinco controles, se conservó el batch físico ocho, las rondas se planificaron con tiempos medidos y se reservó tiempo para validación, prueba y cierre, las pocas épocas y una sola semilla siguen siendo una limitación.')
change(10, 'Selección sin consultar prueba', 'Pilotos de tasa en validación, pesos descartados\nMejor época y rama por media mAP original y LR\nPrueba tras cerrar toda la selección', 'Un barrido reducido de tasas publicadas se realizó con RF noventa y nueve para las CNN y ViT para los transformers, sus pesos se descartaron y su validación eligió una tasa común por familia, no se pretende reproducir el ajuste individual completo del artículo, después cada configuración seleccionó época y rama mediante la media de mAP de validación original y reducida, toda la selección se cerró antes de la prueba de fase dos, se conservaron ambas ramas del mismo checkpoint elegido, como ya existían pruebas históricas, esta ampliación no constituye un conjunto de prueba completamente ciego.')
change(11, 'Red sin entrenar y curvas reales', 'Inicialización aleatoria medida en validación\nCurvas de pérdida y de validación por época\nUna validación de cien consultas limita la precisión de la selección', 'Ahora se mide la representación de cada arquitectura con inicialización aleatoria en validación y se contrasta con sus checkpoints aprendidos, esta medición faltaba y permite examinar la hipótesis de que los resultados iniciales fueran próximos a una red sin aprender, no demuestra por sí sola la causa de una diferencia, los pilotos registran además el cambio relativo del primer tensor del encoder y no se interpreta como el cambio de toda la red, las curvas conservan pérdida y validación por época, cien consultas y una semilla siguen impidiendo afirmar diferencias pequeñas como significativas.', 'resultados/curvas.png')
change(12, 'Resultados de las catorce arquitecturas', 'Prueba con selección previa por validación, ambas resoluciones', f'La tabla completa conserva las catorce arquitecturas y los controles con sus épocas, ramas, configuraciones y costes, la mayor media descriptiva de mAP entre consultas originales y reducidas en el panel de arquitecturas corresponde a {best["arquitectura"]}, esto describe los resultados finales y no se utiliza para ajustar hiperparámetros, deben leerse por separado la precisión absoluta, la caída relativa y el tiempo de entrenamiento, una arquitectura que retiene una fracción alta de una precisión inicial baja no es necesariamente la mejor recuperadora absoluta.', 'resultados/comparacion.png')
change(13, 'Modelo elegido por validación', f'{selection["nombre"]}, rama {selection["rama"]}\nmAP original {picked["original_mAP_porcentaje"]:.3f} %, LR {picked["baja_resolucion_mAP_porcentaje"]:.3f} %\nÉpoca adicional elegida {selection["epoca_adicional_seleccionada"]}', f'La selección global mediante validación corresponde a {selection["nombre"]}, rama {selection["rama"]}, en prueba obtiene mAP original {picked["original_mAP_porcentaje"]:.3f} por ciento y mAP reducida {picked["baja_resolucion_mAP_porcentaje"]:.3f} por ciento, la época adicional elegida es {selection["epoca_adicional_seleccionada"]}, una época adicional cero significa conservar los pesos aprendidos de fase uno y no una red aleatoria, la selección puede conservar un checkpoint anterior si validación no mejora, eso se informa sin cambiar las conclusiones para favorecer la continuación o el componente CO.')
change(14, 'Bases y ablación a arquitectura fija', 'BYOL sin CO para ResNet50, RF99 y BagNet33\nMoCoV3 con y sin CO en ResNet50\nResultados favorables y desfavorables se conservan', 'Los controles incluyen BYOL sin CO para ResNet cincuenta, RF noventa y nueve y BagNet treinta y tres, además de MoCo versión tres con y sin CO en ResNet cincuenta, se comparan con presupuesto común y las mismas particiones, quitar el término local conserva el encoder y constituye la ablación principal, no se sustituye una arquitectura diferente y se atribuye toda la diferencia a CO, las tablas conservan mejoras y deterioros en ambas condiciones sin cambiar la conclusión para favorecer la propuesta, el control histórico de recorte mínimo cero coma nueve sigue siendo una evidencia separada y no se atribuye al entrenamiento nuevo.')
change(15, 'Maestro, estudiante y costes', 'Ambas ramas del mismo checkpoint elegido\nTiempos reales y pico PyTorch de entrenamiento\nCociente mAP por minuto descriptivo, sin medir energía', 'Las dos ramas del mismo checkpoint seleccionado por validación se comparan en los JSON de prueba, no se elige la rama mirando prueba, el costo conserva tiempos reales de entrenamiento de ambas fases y los de evaluación cuando están medidos, el máximo de memoria corresponde a memoria asignada por PyTorch y no al consumo total del dispositivo, el cociente entre mAP media y minutos describe un equilibrio local entre calidad y tiempo y no es una medición energética, la actividad de otras aplicaciones y la recomputación también afectan los tiempos, las explicaciones causales requieren comparaciones controladas adicionales.')
change(16, 'Aciertos y fallos con índices verificables', 'Cinco aciertos y cinco fallos por condición\nExplicaciones factuales de identidad, cámara y rango\nLas causas visuales deben revisarse en las imágenes', 'Los ejemplos se eligen de forma reproducible por índice y resultado de Rank uno, conservando identidades de consulta distintas, se incluyen cinco aciertos y cinco fallos por condición de resolución, las explicaciones verifican identidad, cámara del primer recuperado y rango del primer positivo válido, un fallo sitúa otra persona antes de cualquier positivo después de las exclusiones del protocolo, la similitud de ropa, el fondo, la oclusión o el encuadre son posibles explicaciones que deben comprobarse visualmente y no se anuncian como causas demostradas por una tabla de etiquetas.', 'resultados/fallos.png')
change(17, 'Alcance y pendientes declarados', 'Resultados preliminares, una semilla y pocas épocas', 'La fase dos amplía la exposición de entrenamiento y añade mediciones que faltaban, conserva los resultados históricos y deja explícita su relación con el código actualizado, aun así no demuestra convergencia ni reproduce los experimentos largos de clasificación del artículo, permanecen fuera de alcance Places trescientos sesenta y cinco, múltiples semillas, la igualación literal de parámetros de BagNet y todos los barridos redundantes, el grupo debe revisar los documentos y este material sintético, realizar el envío académico y preparar la defensa individual, subir código o generar un video no equivale a haber realizado esas actividades humanas.')
for path in old_cache.iterdir():
    if path.name.endswith(('.onnx', '.onnx.json')) or path.name.startswith('audio_'):
        target = cache / path.name
        if not target.exists():
            if path.name.startswith('audio_'):
                shutil.copy2(path, target)
            else:
                try:
                    os.link(path, target)
                except OSError:
                    shutil.copy2(path, target)
(cache / 'contenido.json').write_text(json.dumps(slides, ensure_ascii=False, indent=2), encoding='utf-8')
spec = importlib.util.spec_from_file_location('render_fase2', root / 'work/preparar_video.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.PROYECTO = mini
module.CARPETA = cache
sys.argv = ['video_fase2']
module.main()
import imageio_ffmpeg
decode = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-xerror', '-i', str(mini / 'VIDEO.mp4'), '-f', 'null', '-'], capture_output=True, text=True)
assert decode.returncode == 0
match = re.search(r'Duration:\s*(\d+):(\d+):([\d.]+)', decode.stderr)
assert match is not None
duration = int(match[1]) * 3600 + int(match[2]) * 60 + float(match[3])
proof_path = assets / 'video_verificacion.json'
proof = json.loads(proof_path.read_text(encoding='utf-8'))
assert 0 < duration <= 1080 and all(value <= 360 for value in proof['capitulos_segundos'].values())
proof.update(duracion_segundos=duration, decodificacion_completa_sin_errores=True, sha256_video=hashlib.sha256((mini / 'VIDEO.mp4').read_bytes()).hexdigest(), fase=2, fase1_preservada=True)
proof_path.write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding='utf-8')
for name in ['README.md', 'GUION_VIDEO.md']:
    path = project / name
    with path.open('a', encoding='utf-8') as stream:
        stream.write('\nLa versión audiovisual de fase 2 se encuentra en [resultados_fase2/audiovisual/VIDEO.mp4](resultados_fase2/audiovisual/VIDEO.mp4), su guion y verificación de duración y decodificación están en la misma carpeta, el video original de fase 1 se conserva sin cambios y la revisión humana de ambos materiales sigue pendiente.\n')
print(json.dumps({'video_fase2': str(mini / 'VIDEO.mp4'), 'duracion_segundos': duration, 'verificacion': 'aprobado'}, ensure_ascii=False))
