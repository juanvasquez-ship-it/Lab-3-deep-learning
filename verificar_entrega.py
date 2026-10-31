import argparse
import ast
import csv
import hashlib
import importlib
import io
import json
import math
import os
import re
import tokenize
from pathlib import Path
from urllib.parse import unquote, urlsplit

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def leer(ruta):
    return json.loads(ruta.read_text(encoding='utf-8-sig'))


def sha(ruta):
    with ruta.open('rb') as archivo:
        return hashlib.file_digest(archivo, 'sha256').hexdigest()


def cerca(a, b):
    assert math.isclose(float(a), float(b), abs_tol=1e-9), (a, b)


def verificar(datos=None):
    raiz = Path(__file__).resolve().parent
    entrega = leer(raiz / 'CONTENIDO_PAQUETE.json')
    fuentes = [raiz / 'main.py', *sorted((raiz / 'codigo').glob('*.py')), raiz / 'preparar_reproduccion.py', Path(__file__)]
    for ruta in fuentes:
        contenido = ruta.read_text(encoding='utf-8')
        arbol = ast.parse(contenido)
        assert not any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(contenido).readline)), ruta.name
        assert not any(ast.get_docstring(n) for n in ast.walk(arbol) if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))), ruta.name
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.ImportFrom) and nodo.module and nodo.module.startswith('codigo.'):
                assert (raiz / (nodo.module.replace('.', '/') + '.py')).is_file(), nodo.module
    for ruta in sorted((raiz / 'codigo').glob('*.py')):
        importlib.import_module('codigo.' + ruta.stem)
    plan = leer(raiz / 'resultados_fase2/plan_fase2.json')
    revisiones_documentales = entrega.get('registros_documentales_actualizados', {})
    for nombre, revision in revisiones_documentales.items():
        actual = leer(raiz / nombre)
        assert sha(raiz / nombre) == revision['sha256_actual'], nombre
        assert all(campo not in actual for campo in revision['campos_retirados']), nombre
        contenido = json.dumps(actual, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
        assert hashlib.sha256(contenido).hexdigest() == revision['sha256_contenido_tecnico'], nombre
        if nombre in plan['historicos_SHA256']:
            assert revision['sha256_original'] == plan['historicos_SHA256'][nombre], nombre
    for nombre, esperado in plan['historicos_SHA256'].items():
        esperado_actual = revisiones_documentales.get(nombre, {}).get('sha256_actual', esperado)
        assert sha(raiz / nombre) == esperado_actual, nombre
    captura = leer(raiz / 'resultados_fase2/codigo_ejecucion_SHA256.json')['archivos']
    cambios = {nombre: sha(raiz / nombre) for nombre, esperado in captura.items() if sha(raiz / nombre) != esperado}
    assert cambios == entrega['fuentes_modificadas_despues_de_ejecucion'], cambios
    metricas = ['mAP', 'Rank-1', 'Rank-5', 'Rank-10']
    condiciones = ['original', 'baja_resolucion']
    fase1 = sorted((raiz / 'resultados').glob('*/resumen.json'))
    assert len(fase1) == 54
    for ruta in fase1:
        resumen = leer(ruta)
        assert resumen['pasos_realizados'] == 1253 and resumen['batch_fisico'] == 8
        assert resumen['imagenes_fuente_procesadas'] == 10024
        assert leer(ruta.parent / 'historial.json')[-1]['paso'] == 1253
        assert leer(ruta.parent / 'configuracion.json')['nombre'] == ruta.parent.name
        for archivo, consultas in [('validacion.json', 100), ('prueba.json', 3368)]:
            resultado = leer(ruta.parent / archivo)
            for rama in ['online', 'objetivo']:
                for condicion in condiciones:
                    registro = resultado[rama][condicion]
                    assert registro['consultas_evaluadas'] == consultas and not registro['consultas_omitidas']
                    assert registro['protocolo'] == 'market_documentado'
                    assert all(math.isfinite(registro[m]) and 0 <= registro[m] <= 1 for m in metricas)
                    assert registro['Rank-1'] <= registro['Rank-5'] <= registro['Rank-10']
    with (raiz / 'resultados_fase2/comparacion_fase2.csv').open(encoding='utf-8-sig', newline='') as archivo:
        filas = list(csv.DictReader(archivo))
    assert len(filas) == 19
    from codigo.comparacion import ARQUITECTURAS
    assert {f['arquitectura'] for f in filas if f['grupo'] == 'arquitecturas'} == set(ARQUITECTURAS)
    assert sum(f['grupo'] == 'controles' for f in filas) == 5
    candidatos = []
    for fila in filas:
        carpeta = raiz / 'resultados_fase2' / fila['nombre']
        config = leer(carpeta / 'configuracion.json')
        seleccion = leer(carpeta / 'seleccion_validacion.json')
        assert fila['arquitectura'] == config['arquitectura']
        assert int(fila['batch']) == 8 and int(fila['pasos_fase2']) == 2760 and int(fila['epocas_fase2_completas']) == 2
        assert seleccion['rama'] == fila['rama'] and seleccion['epoca_adicional_seleccionada'] == int(fila['epoca_fase2_seleccionada'])
        criterio_mejor = -1
        for nombre in ['validacion_inicio_fase2.json', 'validacion_epoca_1.json', 'validacion_epoca_2.json']:
            validacion = leer(carpeta / nombre)
            for rama in ['online', 'objetivo']:
                media = sum(validacion[rama][c]['mAP'] for c in condiciones) / 2
                criterio_mejor = max(criterio_mejor, media)
        cerca(criterio_mejor, seleccion['criterio'])
        candidatos.append((fila['nombre'], seleccion['rama'], seleccion['criterio']))
        prueba = leer(carpeta / 'prueba.json')
        for rama in ['online', 'objetivo']:
            for condicion in condiciones:
                registro = prueba[rama][condicion]
                assert registro['consultas_evaluadas'] == 3368 and not registro['consultas_omitidas']
                assert registro['protocolo'] == 'market_documentado'
                assert all(math.isfinite(registro[m]) and 0 <= registro[m] <= 1 for m in metricas)
                assert registro['Rank-1'] <= registro['Rank-5'] <= registro['Rank-10']
        for condicion in condiciones:
            for metrica in metricas:
                cerca(float(fila[f'{condicion}_{metrica}_porcentaje']), 100 * prueba[fila['rama']][condicion][metrica])
        seleccionado = prueba[fila['rama']]
        caida = 100 * (seleccionado['original']['mAP'] - seleccionado['baja_resolucion']['mAP'])
        cerca(caida, fila['caida_mAP_pp'])
        cerca(100 * (seleccionado['original']['mAP'] - seleccionado['baja_resolucion']['mAP']) / seleccionado['original']['mAP'], fila['caida_mAP_relativa_porcentaje'])
    global_ = leer(raiz / 'resultados_fase2/seleccion_global_validacion.json')['modelo']
    mejor = max(candidatos, key=lambda c: c[2])
    assert mejor[:2] == (global_['nombre'], global_['rama'])
    cerca(mejor[2], global_['criterio'])
    ejemplos = leer(raiz / 'resultados_fase2/ejemplos/indices_ejemplos.json')
    for condicion in condiciones:
        for tipo in ['aciertos', 'fallos']:
            grupo = [e for e in ejemplos if e['condicion'] == condicion and e['tipo'] == tipo]
            assert len(grupo) >= 5
            assert all((e['person_id'] == e['person_ids_galeria'][0]) == (tipo == 'aciertos') for e in grupo)
            assert (raiz / f'resultados_fase2/ejemplos/{condicion}_{tipo}.png').is_file()
    video = leer(raiz / entrega['verificacion_video'])
    assert video['duracion_segundos'] <= 1080 and video['decodificacion_completa_sin_errores']
    assert all(v <= 360 for v in video['capitulos_segundos'].values())
    assert sha(raiz / entrega['video']) == video['sha256_video']
    enlaces = 0
    for nombre in entrega['documentos_evaluacion']:
        ruta = raiz / nombre
        assert ruta.is_file(), nombre
        for enlace in re.findall(r'\]\(([^)]+)\)', ruta.read_text(encoding='utf-8')):
            if urlsplit(enlace).scheme or enlace.startswith('#'):
                continue
            destino = unquote(enlace.split('#', 1)[0]).strip('<>')
            assert (ruta.parent / destino).is_file(), (nombre, enlace)
            enlaces += 1
    numericas = None
    if datos is not None:
        from codigo.verificaciones import verificar as verificar_funciones
        numericas = verificar_funciones(datos)
        assert numericas['estado'] == 'aprobado'
    return {'estado': 'aprobado', 'alcance': 'archivos e importaciones, evidencia y métricas guardadas, selección por validación, video y enlaces, no estima embeddings ni vuelve a evaluar modelos', 'fuentes_python_sin_comentarios_ni_docstrings': len(fuentes), 'modulos_importados_CPU': len(list((raiz / 'codigo').glob('*.py'))), 'configuraciones_fase1': len(fase1), 'continuaciones_fase2': len(filas), 'arquitecturas': len(ARQUITECTURAS), 'controles': 5, 'metricas_CSV_JSON_fase2': len(filas) * 8, 'historicos_SHA256': len(plan['historicos_SHA256']), 'historicos_SHA256_sin_cambios': len(plan['historicos_SHA256']) - sum(n in plan['historicos_SHA256'] for n in revisiones_documentales), 'registros_documentales_actualizados': sorted(revisiones_documentales), 'fuentes_ejecucion_identicas': len(captura) - len(cambios), 'fuentes_cambiadas_solo_auditoria_entregables': cambios, 'seleccion': global_, 'video_SHA256': video['sha256_video'], 'enlaces_internos_comprobados': enlaces, 'datos_HDF5_contrastados': datos is not None, 'verificaciones_numericas_y_datos': numericas, 'entrenamientos_nuevos': 0, 'inferencias_encoders_nuevas': 0, 'GPU_utilizada': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verificar la entrega existente en CPU sin entrenar ni evaluar encoders')
    parser.add_argument('--datos', type=Path)
    parser.add_argument('--salida', type=Path)
    args = parser.parse_args()
    resultado = verificar(args.datos)
    texto = json.dumps(resultado, ensure_ascii=False, indent=2)
    if args.salida:
        args.salida.write_text(texto + '\n', encoding='utf-8')
    print(texto)
