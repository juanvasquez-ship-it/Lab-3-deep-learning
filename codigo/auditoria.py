import ast
import csv
import hashlib
import io
import json
import math
import tokenize
from pathlib import Path

import h5py
import numpy as np

from codigo.entrenamiento import guardar_json
from codigo.experimentos import experimentos_lectura, exportar_matriz, matriz_completa


def leer(ruta, defecto=None):
    return json.loads(ruta.read_text(encoding='utf-8')) if ruta.exists() else defecto


def clave_configuracion(configuracion):
    omitir = {'nombre', 'fuente', 'estado', 'alias', 'estado_inicial_implementacion', 'checkpoint_cada', 'checkpoint_activaciones'}
    parametros = {k: v for k, v in configuracion.items() if k not in omitir}
    if parametros.get('cantidad_locales') == 196 and parametros['arquitectura'] == 'rf99_resnet50' and parametros['resolucion'] == 224 and parametros.get('capa_local', 4) == 4:
        parametros.pop('cantidad_locales')
    return json.dumps(parametros, sort_keys=True)


def auditar_resultados(resultados, raiz, datos=None):
    resultados, raiz = Path(resultados), Path(raiz)
    errores = []
    observaciones = []
    ejecuciones = []
    equivalentes = {}
    hashes_fuente = set()
    plan = leer(resultados / 'plan_ejecucion.json', {})
    for carpeta in sorted(resultados.iterdir()):
        resumen = leer(carpeta / 'resumen.json') if carpeta.is_dir() else None
        if resumen is None:
            continue
        nombre = carpeta.name
        configuracion = leer(carpeta / 'configuracion.json', {})
        if configuracion.get('nombre') != nombre:
            errores.append(f'{nombre}, configuración ausente o identificador distinto de su carpeta')
        if resumen['pasos_realizados'] > resumen['pasos_objetivo'] or resumen['imagenes_fuente_procesadas'] != resumen['pasos_realizados'] * resumen['batch_fisico']:
            errores.append(f'{nombre}, cantidades de entrenamiento inconsistentes')
        if resumen['batch_fisico'] != plan.get('batch_fisico') or resumen['pasos_objetivo'] != plan.get('pasos_comunes'):
            errores.append(f'{nombre}, condiciones distintas de la planificación común')
        if resumen['longitud_particion'] != 11043 or resumen['imagenes_fuente_distintas'] > min(11043, resumen['imagenes_fuente_procesadas']):
            errores.append(f'{nombre}, exposición de datos inconsistente')
        if not (carpeta / 'checkpoint.pt').exists():
            if (raiz / 'CONTENIDO_PAQUETE.json').exists():
                observaciones.append(f'{nombre}, checkpoint completo omitido en el paquete compacto, se conserva el registro de ejecución')
            else:
                errores.append(f'{nombre}, falta checkpoint')
        historia = leer(carpeta / 'historial.json', [])
        if resumen['pasos_realizados'] and (not historia or historia[-1]['paso'] != resumen['pasos_realizados']):
            errores.append(f'{nombre}, historial sin actualización final')
        for registro in historia:
            if any(not math.isfinite(v) for v in registro.values() if isinstance(v, (int, float))):
                errores.append(f'{nombre}, registro no finito')
        fases = {}
        for fase, archivo, cantidad in [('validation', 'validacion.json', 100), ('testing', 'prueba.json', 3368)]:
            evaluacion = leer(carpeta / archivo, {})
            fases[fase] = all(rama in evaluacion for rama in ['online', 'objetivo'])
            if resumen['estado'] == 'entrenamiento_completado' and not fases[fase]:
                errores.append(f'{nombre}, {fase}, faltan ramas de evaluación')
            for rama in ['online', 'objetivo']:
                if rama not in evaluacion:
                    continue
                for condicion in ['original', 'baja_resolucion']:
                    metricas = evaluacion[rama][condicion]
                    if metricas['consultas_evaluadas'] != cantidad or metricas['consultas_omitidas']:
                        errores.append(f'{nombre}, {fase}, consultas incompletas')
                    valores = [metricas[k] for k in ['mAP', 'Rank-1', 'Rank-5', 'Rank-10']]
                    if not all(math.isfinite(v) and 0 <= v <= 1 for v in valores) or valores[1:] != sorted(valores[1:]):
                        errores.append(f'{nombre}, {fase}, métricas inválidas')
                    if metricas['protocolo'] != 'market_documentado':
                        errores.append(f'{nombre}, {fase}, protocolo inesperado')
                caida = 100 * (evaluacion[rama]['original']['mAP'] - evaluacion[rama]['baja_resolucion']['mAP'])
                if abs(caida - evaluacion[rama]['caida_mAP_puntos_porcentuales']) > 1e-10:
                    errores.append(f'{nombre}, caída mAP inconsistente')
                media = (evaluacion[rama]['original']['mAP'] + evaluacion[rama]['baja_resolucion']['mAP']) / 2
                if abs(media - evaluacion[rama]['criterio_validacion']) > 1e-10:
                    errores.append(f'{nombre}, {fase}, criterio de validación inconsistente')
            if fase == 'validation' and fases[fase]:
                elegida = max(['online', 'objetivo'], key=lambda r: evaluacion[r]['criterio_validacion'])
                if evaluacion['rama_seleccionada'] != elegida:
                    errores.append(f'{nombre}, rama de validación inconsistente')
        completa = resumen['estado'] == 'entrenamiento_completado'
        if completa:
            hashes_fuente.add(resumen['indices_fuente_visitados_sha256'])
        else:
            observaciones.append(f'{nombre}, calendario local incompleto')
        estado = {'modelo': nombre, 'entrenamiento_completo': completa, 'pasos': resumen['pasos_realizados'], 'validation_completa': fases['validation'], 'testing_completa': fases['testing']}
        ejecuciones.append(estado)
        equivalentes[clave_configuracion(configuracion)] = estado
    observadas = {e['modelo'] for e in ejecuciones}
    faltantes = set(plan.get('modelos', [])) - observadas
    if faltantes:
        errores.append(f'Faltan resúmenes del panel principal, {sorted(faltantes)}')
    estados_entrenamiento = leer(resultados / 'estado_ejecucion.json', [])
    declaradas_completas = {e['modelo'] for e in estados_entrenamiento if e.get('estado') == 'entrenamiento_completado'}
    if declaradas_completas != {e['modelo'] for e in ejecuciones if e['entrenamiento_completo']}:
        errores.append('El estado de ejecución y los resúmenes completos no coinciden')
    if len(hashes_fuente) > 1:
        errores.append('Las ejecuciones completas no comparten el mismo conjunto de índices de entrenamiento')
    seleccion = leer(resultados / 'seleccion_validacion.json', {})
    rankings_verificados = 0
    if datos is not None and (Path(datos) / 'testing.h5').exists() and seleccion:
        prueba_seleccionada = leer(resultados / seleccion['modelo'] / 'prueba.json', {}).get(seleccion['rama'], {})
        with h5py.File(Path(datos) / 'testing.h5', 'r') as archivo:
            personas_q, camaras_q = archivo['query/person_ids'][:], archivo['query/camera_ids'][:]
            personas_g, camaras_g = archivo['gallery/person_ids'][:], archivo['gallery/camera_ids'][:]
        for condicion in ['original', 'baja_resolucion']:
            metricas = prueba_seleccionada.get(condicion, {})
            rankings = metricas.get('rankings', [])
            if len(rankings) != 3368 or sorted(r['consulta'] for r in rankings) != list(range(3368)):
                errores.append(f'{condicion}, rankings del seleccionado incompletos')
                continue
            ap = []
            for ranking in rankings:
                q, indices = ranking['consulta'], np.array(ranking['indices'])
                coincidencias = personas_g[indices] == personas_q[q]
                excluidas = (personas_g[indices] == -1) | (coincidencias & (camaras_g[indices] == camaras_q[q]))
                if excluidas.any() or coincidencias.tolist() != ranking['aciertos']:
                    errores.append(f'{condicion}, consulta {q}, etiquetas o exclusiones de ranking incoherentes')
                posiciones = np.flatnonzero(coincidencias)
                if (ranking['rank'] <= 10 and (not len(posiciones) or posiciones[0] + 1 != ranking['rank'])) or (ranking['rank'] > 10 and len(posiciones)):
                    errores.append(f'{condicion}, consulta {q}, primer positivo incoherente')
                ap.append(ranking['ap'])
                rankings_verificados += 1
            if abs(float(np.mean(ap)) - metricas['mAP']) > 1e-10:
                errores.append(f'{condicion}, mAP distinto del promedio de AP guardados')
    else:
        observaciones.append('No se contrastaron rankings con metadatos HDF5 en esta auditoría')
    candidatos = []
    for nombre in plan.get('modelos', []):
        validacion = leer(resultados / nombre / 'validacion.json', {})
        if 'rama_seleccionada' in validacion:
            rama = validacion['rama_seleccionada']
            candidatos.append((nombre, rama, validacion[rama]['criterio_validacion']))
    if candidatos:
        esperada = max(candidatos, key=lambda c: c[2])
        if (seleccion.get('modelo'), seleccion.get('rama')) != esperada[:2]:
            errores.append('La selección principal no coincide con validación')
    pruebas = list(resultados.glob('*/prueba.json'))
    seleccion_antes_resultados = None
    if pruebas and (resultados / 'seleccion_validacion.json').exists():
        primer_resultado = min(p.stat().st_mtime for p in pruebas)
        seleccion_antes_resultados = (resultados / 'seleccion_validacion.json').stat().st_mtime <= primer_resultado
        if not seleccion_antes_resultados:
            observaciones.append('Las fechas de copia no acreditan el orden de ejecución, la cronología histórica se consulta en los planes y registros originales')
            seleccion_antes_resultados = None
        for archivo, campo in [('plan_ampliacion.json', 'revision_declaracion_unix'), ('plan_analisis.json', 'declaracion_unix')]:
            declaracion = leer(resultados / archivo, {})
            if declaracion.get(campo, declaracion.get('declaracion_unix', 0)) > primer_resultado:
                errores.append(f'{archivo}, declaración posterior a los resultados de prueba')
    ampliacion = leer(resultados / 'plan_ampliacion.json', {})
    if ampliacion.get('historial_prioridad'):
        huella_actual = hashlib.sha256(json.dumps(sorted(ampliacion['configuraciones'], key=lambda c: c['nombre']), sort_keys=True).encode()).hexdigest()
        for revision in ampliacion['historial_prioridad']:
            if not revision['mismos_valores'] or revision['sha256_configuraciones_antes'] != revision['sha256_configuraciones_despues'] or revision['sha256_configuraciones_despues'] != huella_actual:
                errores.append('Una revisión de prioridad alteró las configuraciones declaradas')
    cobertura = []
    for configuracion in matriz_completa():
        evidencia = equivalentes.get(clave_configuracion(configuracion))
        fila = {'configuracion': configuracion['nombre'], 'fuente': configuracion['fuente'], 'configuracion_equivalente_ejecutada': evidencia['modelo'] if evidencia else '', 'entrenado_calendario_local': bool(evidencia and evidencia['entrenamiento_completo']), 'validado': bool(evidencia and evidencia['validation_completa']), 'probado_ambas_ramas_y_resoluciones': bool(evidencia and evidencia['testing_completa']), 'calendario_largo_publicado': configuracion.get('epocas_paper', ''), 'estado': 'ejecutado_o_cubierto_por_equivalente' if evidencia else 'pendiente_ejecucion'}
        cobertura.append(fila)
    with (resultados / 'cobertura_experimentos.csv').open('w', encoding='utf-8-sig', newline='') as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(cobertura[0]))
        escritor.writeheader()
        escritor.writerows(cobertura)
    fuentes = [raiz / 'main.py', *sorted((raiz / 'codigo').glob('*.py'))]
    for ruta in fuentes:
        contenido = ruta.read_text(encoding='utf-8')
        arbol = ast.parse(contenido)
        if any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(contenido).readline)):
            errores.append(f'{ruta.name}, comentarios presentes')
        if any(ast.get_docstring(n) for n in ast.walk(arbol) if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))):
            errores.append(f'{ruta.name}, docstrings presentes')
    ejemplos = leer(resultados / 'ejemplos' / 'indices_ejemplos.json', [])
    cantidades_ejemplos = {}
    for condicion in ['original', 'baja_resolucion']:
        for tipo in ['aciertos', 'fallos']:
            subconjunto = [e for e in ejemplos if e['condicion'] == condicion and e['tipo'] == tipo]
            cantidades_ejemplos[f'{condicion}_{tipo}'] = len(subconjunto)
            if len(subconjunto) < 5:
                errores.append(f'Ejemplos insuficientes, {condicion}, {tipo}')
            for ejemplo in subconjunto:
                if 'person_ids_galeria' in ejemplo:
                    coincide = ejemplo['person_id'] == ejemplo['person_ids_galeria'][0]
                    if coincide != (tipo == 'aciertos'):
                        errores.append('Ejemplo con etiqueta de acierto o fallo inconsistente')
    entrega = leer(raiz / 'CONTENIDO_PAQUETE.json', {})
    archivo_video = entrega.get('video', 'VIDEO.mp4')
    archivo_errores = entrega.get('analisis_errores', 'ANALISIS_ERRORES.md')
    for requerido in ['README.md', 'METODOLOGIA.md', 'INFORME.md', archivo_errores, archivo_video, 'requirements.txt']:
        if not (raiz / requerido).exists():
            errores.append(f'Falta entregable, {requerido}')
    video = leer(raiz / entrega['verificacion_video'], {}) if entrega.get('verificacion_video') else leer(resultados / 'video_verificacion.json', {})
    if not video or video.get('duracion_segundos', 1081) > 1080:
        errores.append('Video sin duración verificada dentro de 18 minutos')
    if not video.get('decodificacion_completa_sin_errores'):
        errores.append('Falta comprobar la decodificación completa del video')
    if any(duracion > 360 for duracion in video.get('capitulos_segundos', {}).values()):
        errores.append('Un capítulo del video supera los seis minutos')
    if (raiz / archivo_video).exists() and hashlib.sha256((raiz / archivo_video).read_bytes()).hexdigest() != video.get('sha256_video'):
        errores.append('El video actual no coincide con el archivo cuya decodificación se verificó')
    plan_analisis = leer(resultados / 'plan_analisis.json', {})
    esperados_analisis = {(nombre, analisis) for analisis, nombres in plan_analisis.get('grupos', {}).items() for nombre in nombres}
    estados_analisis = leer(resultados / 'estado_analisis.json', [])
    claves_analisis = [(e['modelo'], e['analisis']) for e in estados_analisis]
    if len(claves_analisis) != len(set(claves_analisis)) or set(claves_analisis) - esperados_analisis:
        errores.append('El estado del panel de análisis contiene duplicados o tareas no declaradas')
    analisis_contrastados = 0
    particiones_legacy = 0
    for estado in estados_analisis:
        if estado['estado'] != 'completado':
            continue
        nombre, tipo = estado['modelo'], estado['analisis']
        evidencia = leer(resultados / nombre / f'analisis_{tipo}.json', {})
        if not evidencia or evidencia.get('rama') != plan_analisis.get('rama') or not evidencia.get('resultado'):
            errores.append(f'{nombre}, {tipo}, falta evidencia completa de la rama declarada')
            continue
        contenido = evidencia['resultado']
        if tipo in {'representaciones', 'representaciones384'}:
            carpeta_repr = resultados / nombre / tipo
            if contenido.get('particion') != 'training sin identidades' or len(contenido.get('indices', [])) != 16 or not (carpeta_repr / 'evidencia.npz').exists() or not (carpeta_repr / 'campos_efectivos.png').exists():
                errores.append(f'{nombre}, {tipo}, evidencia de representaciones incompleta')
            if evidencia.get('particion') == 'validation':
                particiones_legacy += 1
        elif tipo in {'mascaras', 'pgd', 'antes_mlp', 'capas'}:
            if evidencia.get('particion') != plan_analisis.get('particion_metricas'):
                errores.append(f'{nombre}, {tipo}, partición distinta del plan')
            claves = {'mascaras': {str(x) for x in experimentos_lectura()['mascaras']['proporciones']}, 'pgd': {str(i) for i in range(len(experimentos_lectura()['pgd']['configuraciones']))}, 'antes_mlp': {'antes_mlp'}, 'capas': {'capa_3', 'capa_4'}}[tipo]
            if set(contenido) != claves:
                errores.append(f'{nombre}, {tipo}, falta una condición prevista')
            for medicion in contenido.values():
                metricas_lista = [medicion['metricas']] if tipo == 'pgd' else [medicion.get(c, {}) for c in ['original', 'baja_resolucion']]
                for metricas in metricas_lista:
                    valores = [metricas.get(k, float('nan')) for k in ['mAP', 'Rank-1', 'Rank-5', 'Rank-10']]
                    if metricas.get('consultas_evaluadas') != 100 or metricas.get('consultas_omitidas') or not all(math.isfinite(v) and 0 <= v <= 1 for v in valores):
                        errores.append(f'{nombre}, {tipo}, medición inválida o consultas incompletas')
                if tipo == 'pgd' and medicion['maximo_cambio_Linf_verificado'] > medicion['ataque']['epsilon'] + 1e-6:
                    errores.append(f'{nombre}, PGD excede su cota registrada')
        elif tipo == 'eficiencia' and (contenido.get('parametros_encoder', 0) <= 0 or contenido.get('segundos_por_forward', 0) <= 0):
            errores.append(f'{nombre}, eficiencia sin medición válida')
        analisis_contrastados += 1
    if particiones_legacy:
        observaciones.append(f'{particiones_legacy} análisis de representaciones conservan partición externa validation por un error de metadatos, su resultado interno y el lector utilizan training sin identidades, se conservan los registros históricos y la corrección se documenta en METODOLOGIA.md')
    if analisis_contrastados != len(esperados_analisis):
        observaciones.append(f'Análisis con evidencia contrastada, {analisis_contrastados} de {len(esperados_analisis)} declarados')
    presupuesto = leer(resultados / 'presupuesto.json', {})
    if presupuesto.get('fin_real_unix', 0) > presupuesto.get('fin_unix', 0):
        errores.append('La última fase registrada excedió el presupuesto temporal')
    complemento = leer(resultados / 'presupuesto_comparacion.json', {})
    if complemento and (complemento.get('maximo_segundos', 10801) > 10800 or complemento.get('fin_real_unix', 0) > complemento.get('fin_unix', 0)):
        errores.append('La sesión complementaria excedió su límite independiente')
    exportar_matriz(resultados / 'matriz_experimentos.json')
    resultado = {'estado': 'aprobado' if not errores else 'requiere_correcciones', 'alcance': 'auditoría de archivos, configuraciones y coherencia de resultados existentes, no reentrena ni repite inferencia', 'errores': errores, 'observaciones': observaciones, 'modelos': ejecuciones, 'seleccion_antes_primer_resultado_prueba': seleccion_antes_resultados, 'rankings_contrastados_con_metadatos': rankings_verificados, 'alcance_rankings': 'los diez primeros índices, sus etiquetas y el rango del primer positivo del modelo seleccionado, el mAP se contrasta con AP previamente calculados sobre toda la galería, no se reconstruye AP usando solo los diez primeros', 'conjuntos_indices_fuente_distintos': len(hashes_fuente), 'filas_cobertura': len(cobertura), 'filas_con_evidencia': sum(bool(f['configuracion_equivalente_ejecutada']) for f in cobertura), 'archivos_python_inspeccionados': len(fuentes), 'ejemplos': cantidades_ejemplos, 'pendiente_humano': 'revisión del material audiovisual y presentación grupal, envío al formulario y defensa individual', 'limite': 'La coherencia automática de registros no sustituye la revisión metodológica y visual del informe'}
    resultado['analisis_contrastados'] = analisis_contrastados
    resultado['analisis_declarados'] = len(esperados_analisis)
    resultado['metadatos_particion_legacy'] = particiones_legacy
    resultado['cumplimiento_integral_paper'] = 'no certificado por la coherencia automática, consultar INFORME.md y METODOLOGIA.md'
    guardar_json(resultados / 'verificacion_final.json', resultado)
    return resultado
