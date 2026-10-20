import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
root = Path(__file__).resolve().parents[1]
project = root / 'outputs/Laboratorio3'
destination = project / 'resultados_fase2'
sys.path.insert(0, str(project))
from codigo.fase2 import consolidar, huella, leer
from codigo.entrenamiento import guardar_json

consolidar(project, destination)
plan = leer(destination / 'plan_fase2.json')
rows = leer(destination / 'comparacion_fase2.json')
selection = leer(destination / 'seleccion_global_validacion.json', {})
errors = []
for relative, digest in plan['historicos_SHA256'].items():
    if huella(project / relative) != digest:
        errors.append('Registro histórico alterado: ' + relative)
for row in rows:
    folder = destination / row['nombre']
    result = leer(folder / 'prueba.json', {})
    if row['estado'] == 'evaluado':
        for branch in ['online', 'objetivo']:
            for condition in ['original', 'baja_resolucion']:
                values = result[branch][condition]
                if values['consultas_evaluadas'] != 3368 or values['protocolo'] != 'market_documentado':
                    errors.append('Cobertura o protocolo incorrecto: ' + row['nombre'])
                if not 0 <= values['Rank-1'] <= values['Rank-5'] <= values['Rank-10'] <= 1:
                    errors.append('Rank-k incoherente: ' + row['nombre'])
                if not 0 <= values['mAP'] <= 1:
                    errors.append('mAP fuera de rango: ' + row['nombre'])
        if not selection or (folder / 'prueba.json').stat().st_mtime < selection['congelada_unix'] - 1:
            errors.append('Falta evidencia temporal de selección antes de prueba: ' + row['nombre'])
examples = leer(destination / 'ejemplos/indices_ejemplos.json', [])
counts = {f'{condition}_{kind}': sum(e['condicion'] == condition and e['tipo'] == kind for e in examples) for condition in ['original', 'baja_resolucion'] for kind in ['aciertos', 'fallos']}
for key, value in counts.items():
    if value < 5:
        errors.append('Ejemplos insuficientes: ' + key)
code = leer(destination / 'codigo_ejecucion_SHA256.json', {}).get('archivos', {})
if not code or any(huella(project / path) != digest for path, digest in code.items()):
    errors.append('Código ejecutado sin huella coincidente al cierre')
requirements = [
    ['Solo imágenes y sin identidades en aprendizaje', '1', 'Entrenamiento sin person_ids, no se modificó el lector de training'],
    ['Particiones suministradas', '1', 'Sin nuevo split ni fase final_training, training como origen único'],
    ['Implementación central propia y EMA', '2', 'Núcleo CO propio conservado, recomputación operativa con prueba CPU y GPU'],
    ['Validación y selección', '2', 'Pilotos y época/rama con validación, cierre global anterior a prueba de fase 2'],
    ['Base y ablación', '2', 'BYOL sin CO en ResNet50, RF99 y BagNet33, pareja MoCoV3 sin/con CO, con estados explícitos en tabla'],
    ['Ambas ramas y resoluciones, L2, coseno y métricas', '2', 'JSON por configuración, conteos y protocolo comprobados cuando evaluado'],
    ['Al menos 5 aciertos y 5 fallos explicados', '2-3', 'Explicaciones factuales y figuras de fase 2, la causalidad visual requiere revisión humana'],
    ['Código, resultados e instrucciones', '2-3', 'Tabla CSV/Markdown, configuración por trabajo, curvas y registros, históricos preservados'],
    ['Video máximo 18 minutos', '3', 'VIDEO.mp4 conserva fase 1, se prepara una versión de fase 2 separada si la consolidación dispone de tiempo y resultados suficientes'],
    ['Entrega al formulario y defensa', '3', 'Pendientes humanos, no se declaran realizados por los scripts']
]
text = ['# Verificación de cierre de fase 2', '', '| Requisito del PDF | Página | Evidencia y límites |', '|---|---:|---|']
text += ['| ' + ' | '.join(row) + ' |' for row in requirements]
text += ['', 'La matriz experimental íntegra del paper continúa parcial, esta revisión distingue requisitos básicos del laboratorio, cobertura de arquitecturas y reproducción experimental publicada, una ejecución incompleta aparece como tal en verificacion_fase2.json.', '', 'No se altera ni republica automáticamente el ZIP o el repositorio histórico, la siguiente revisión humana debe decidir qué evidencia de fase 2 consolidar en la entrega definitiva.', '']
(destination / 'REVISION_CIERRE.md').write_text('\n'.join(text), encoding='utf-8')
proof = leer(destination / 'verificacion_fase2.json')
deadline = leer(destination / 'plazo_global.json')['fin_unix']
if not errors and proof.get('estado') == 'completado' and deadline - time.time() > 300:
    try:
        process = subprocess.Popen([str(root / 'work/video_entorno/Scripts/python.exe'), str(root / 'work/video_fase2.py')], env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'))
        code_video = process.wait(timeout=max(1, deadline - time.time() - 90))
        proof['video_fase2_CPU'] = 'completado' if code_video == 0 else 'fallo_revision_pendiente'
    except subprocess.TimeoutExpired:
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
        proof['video_fase2_CPU'] = 'pendiente_por_plazo'
else:
    proof['video_fase2_CPU'] = 'pendiente_por_resultados_o_plazo'
proof.update(verificacion_CPU_final='aprobado' if not errors else 'requiere_revision', errores_cierre=errors, ejemplos_por_condicion=counts, codigo_fuente_SHA256_conservado=bool(code) and all(huella(project / path) == digest for path, digest in code.items()), CUDA_VISIBLE_DEVICES='-1')
if errors:
    proof['estado'] = 'parcial_o_requiere_revision'
guardar_json(destination / 'verificacion_fase2.json', proof)
print(json.dumps(proof, ensure_ascii=False), flush=True)
if errors:
    raise SystemExit(1)
