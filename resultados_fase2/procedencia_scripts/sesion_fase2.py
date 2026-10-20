import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
project = root / 'outputs/Laboratorio3'
destination = project / 'resultados_fase2'
deadline = json.loads((destination / 'plazo_global.json').read_text(encoding='utf-8'))
record = destination / 'sesion_independiente.json'

def status(phase, **values):
    content = {'pid_supervisor': os.getpid(), 'fase': phase, 'actualizacion_unix': time.time(), 'inicio_plazo_unix': deadline['inicio_unix'], 'limite_global_unix': deadline['fin_unix'], 'sin_seguimiento_automatico_en_chat': True, **values}
    tmp = record.with_suffix('.tmp')
    tmp.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(record)

def launch(command, env, phase, reserve):
    timeout = deadline['fin_unix'] - time.time() - reserve
    if timeout <= 0:
        return 124
    process = subprocess.Popen(command, cwd=str(project), env=env)
    status(phase, pid_trabajador=process.pid)
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if process.poll() is None:
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                pass
        status('limite_supervisor', fase_interrumpida=phase, codigo=124)
        return 124

keep_awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
status('iniciando', inhibicion_suspension_solo_durante_proceso=keep_awake)
gpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1')
cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONUNBUFFERED='1')
gpu_code = None
try:
    for attempt in range(2):
        gpu_code = launch([sys.executable, str(project / 'main.py'), 'fase2', '--datos', str(root / 'work/data')], gpu_env, 'GPU_entrenamiento_validacion_prueba', 900)
        if gpu_code == 0 or deadline['fin_unix'] - time.time() < 7200:
            break
        status('reintento_acotado', intento=attempt + 1, codigo_previo=gpu_code)
        if attempt == 0:
            time.sleep(90)
    cpu_code = launch([sys.executable, str(root / 'work/finalizar_fase2.py')], cpu_env, 'consolidacion_CPU', 30)
    proof = json.loads((destination / 'verificacion_fase2.json').read_text(encoding='utf-8')) if (destination / 'verificacion_fase2.json').exists() else {}
    phase = 'finalizado' if gpu_code == 0 and cpu_code == 0 and proof.get('estado') == 'completado' else 'requiere_revision'
    status(phase, codigo_GPU=gpu_code, codigo_CPU=cpu_code, transcurridos_desde_inicio_segundos=time.time() - deadline['inicio_unix'], resumen=proof)
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
