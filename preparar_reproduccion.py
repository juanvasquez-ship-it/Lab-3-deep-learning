import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path


def leer(ruta):
    return json.loads(ruta.read_text(encoding='utf-8-sig'))


def guardar(ruta, valor):
    ruta.write_text(json.dumps(valor, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def preparar(destino, fase, pesos):
    origen = Path(__file__).resolve().parent
    destino = Path(destino).resolve()
    if destino.exists():
        raise FileExistsError('El destino debe ser una carpeta nueva')
    plan = leer(origen / 'resultados_fase2/plan_fase2.json') if fase == 2 else None
    checkpoints = []
    if plan:
        if pesos is None:
            raise ValueError('Fase 2 requiere --pesos-fase1 con los checkpoints completos originales')
        for trabajo in plan['trabajos']:
            fuente = Path(pesos) / trabajo['nombre'] / 'checkpoint.pt'
            if not fuente.is_file():
                raise FileNotFoundError(str(fuente))
            checkpoints.append((fuente, destino / trabajo['origen']))
    destino.mkdir(parents=True)
    for ruta in [origen / 'main.py', *sorted((origen / 'codigo').glob('*.py')), *[origen / n for n in ['requirements.txt', 'README.md', 'METODOLOGIA.md', 'INFORME.md']]]:
        salida = destino / ruta.relative_to(origen)
        salida.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ruta, salida)
    if plan:
        for nombre, esperado in plan['historicos_SHA256'].items():
            fuente = origen / nombre
            with fuente.open('rb') as archivo:
                if hashlib.file_digest(archivo, 'sha256').hexdigest() != esperado:
                    raise ValueError('Registro de origen distinto al declarado: ' + nombre)
            salida = destino / nombre
            salida.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fuente, salida)
        for fuente, salida in checkpoints:
            salida.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fuente, salida)
        carpeta = destino / 'resultados_fase2'
        carpeta.mkdir()
        inicio = time.time()
        plazo = {'inicio_unix': inicio, 'limite_correcciones_inicio_entrenamientos_unix': inicio + 28800, 'fin_unix': inicio + 36000, 'maximo_segundos': 36000, 'incluye': 'reproducción independiente, pilotos, entrenamiento, validación y prueba', 'origen': 'preparación explícita de una nueva ejecución, no modifica el plazo del experimento entregado'}
        plan['plazo'] = plazo
        plan['declaracion_unix'] = inicio
        plan['reproduccion_de'] = 'entrega-v2, configuraciones históricas conservadas, nuevos tiempos y nueva ejecución'
        guardar(carpeta / 'plazo_global.json', plazo)
        guardar(carpeta / 'plan_fase2.json', plan)
        fuentes = {}
        for ruta in [destino / 'main.py', *sorted((destino / 'codigo').glob('*.py'))]:
            with ruta.open('rb') as archivo:
                fuentes[ruta.relative_to(destino).as_posix()] = hashlib.file_digest(archivo, 'sha256').hexdigest()
        guardar(carpeta / 'codigo_ejecucion_SHA256.json', {'archivos': fuentes, 'alcance': 'fuentes de la reproducción, posteriores a la organización académica, no sustituye las huellas originales'})
    return {'destino': str(destino), 'fase': fase, 'checkpoints_completos_copiados': len(checkpoints), 'entrenamientos_iniciados': 0, 'evaluaciones_iniciadas': 0, 'GPU_utilizada': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preparar una reproducción en una carpeta nueva sin entrenar ni evaluar')
    parser.add_argument('--destino', type=Path, required=True)
    parser.add_argument('--fase', type=int, choices=[1, 2], required=True)
    parser.add_argument('--pesos-fase1', type=Path)
    argumentos = parser.parse_args()
    print(json.dumps(preparar(argumentos.destino, argumentos.fase, argumentos.pesos_fase1), ensure_ascii=False, indent=2))
