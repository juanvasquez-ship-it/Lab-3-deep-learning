import csv
import hashlib
import json
import os
import platform
import subprocess
import time
import traceback
from pathlib import Path

import torch
from filelock import FileLock

from codigo.entrenamiento import Presupuesto, cargar_checkpoint, entrenar, guardar_json, liberar
from codigo.evaluacion import evaluar_encoder
from codigo.experimentos import configuracion
from codigo.modelos import Encoder


ARQUITECTURAS = ['resnet18', 'resnet50', 'rf29_resnet18', 'rf39_resnet50', 'rf67_resnet50', 'rf99_resnet50', 'rf115_resnet50', 'rf163_resnet50', 'rf195_paper193_resnet50', 'bagnet33', 'vit_s16', 'swin_t', 'swin_s', 'swin_b']
NOMBRES = {'resnet18': 'co_byol_resnet18_224', 'resnet50': 'co_byol_r50', 'rf29_resnet18': 'co_byol_rf29_224', 'rf99_resnet50': 'co_byol_rf99'}
METRICAS = ['mAP', 'Rank-1', 'Rank-5', 'Rank-10']


def leer(ruta, defecto=None):
    return json.loads(Path(ruta).read_text(encoding='utf-8')) if Path(ruta).exists() else defecto


def completa_medicion(contenido, consultas):
    return all(rama in contenido and all(contenido[rama].get(c, {}).get('consultas_evaluadas') == consultas for c in ['original', 'baja_resolucion']) for rama in ['online', 'objetivo'])


def inventario(resultados):
    filas = []
    for ruta in sorted(resultados.glob('*/configuracion.json')):
        config = leer(ruta)
        resumen = leer(ruta.parent / 'resumen.json', {})
        validacion = leer(ruta.parent / 'validacion.json', {})
        prueba = leer(ruta.parent / 'prueba.json', {})
        filas.append({'nombre': config['nombre'], 'arquitectura': config['arquitectura'], 'iniciado': True, 'completo': resumen.get('estado') == 'entrenamiento_completado', 'pasos': resumen.get('pasos_realizados'), 'validado': completa_medicion(validacion, 100), 'evaluado': completa_medicion(prueba, 3368)})
    return filas


def declarar(args):
    ruta = args.resultados / 'plan_comparacion.json'
    if ruta.exists():
        return leer(ruta)
    referencias = []
    for arquitectura in ARQUITECTURAS:
        nombre = NOMBRES.get(arquitectura, f'co_byol_{arquitectura}')
        config = leer(args.resultados / nombre / 'configuracion.json')
        if config is None:
            config = configuracion(nombre, arquitectura=arquitectura, peso_local=0.2, resolucion=224)
        if not (config['metodo'] == 'byol' and config['peso_local'] == 0.2 and config['resolucion'] == 224 and config['recorte'] == 0.2 and not config.get('multicrop') and not config.get('sin_mlp') and not config.get('cabeza_compartida')):
            raise ValueError(f'La referencia no cumple el protocolo común, {nombre}')
        referencias.append(config)
    orden = ['rf115_resnet50', 'rf39_resnet50', 'resnet18', 'rf29_resnet18']
    cola = []
    for arquitectura in orden:
        config = dict(next(c for c in referencias if c['arquitectura'] == arquitectura))
        resumen = leer(args.resultados / config['nombre'] / 'resumen.json', {})
        if resumen.get('estado') == 'entrenamiento_completado':
            continue
        config['checkpoint_cada'] = 250
        if arquitectura == 'rf39_resnet50':
            config['checkpoint_activaciones'] = True
        cola.append({'configuracion': config, 'prioridad': 1, 'motivo': 'arquitectura pendiente' if arquitectura.startswith(('rf39', 'rf115')) else 'misma resolución 224 para comparar todas las arquitecturas con CO-BYOL'})
    for peso in [0.0, 0.2]:
        nombre = f'byol_rf99_resnet50_co{peso}_crop0.9'
        resumen = leer(args.resultados / nombre / 'resumen.json', {})
        if resumen.get('estado') != 'entrenamiento_completado':
            cola.append({'configuracion': configuracion(nombre, arquitectura='rf99_resnet50', peso_local=peso, recorte=0.9, checkpoint_cada=250), 'prioridad': 2, 'motivo': 'contraste mínimo de la afirmación central del artículo con recortes pequeños prácticamente retirados, control sin CO y variante CO'})
    plan_previo = leer(args.resultados / 'plan_ejecucion.json')
    if plan_previo['batch_fisico'] != 8 or plan_previo['pasos_comunes'] != 1253:
        raise ValueError('Las condiciones existentes cambiaron, no se sustituyen por constantes sin revisión')
    protegidos = [p for p in args.resultados.glob('*/*.json')]
    protegidos += [args.resultados / n for n in ['presupuesto.json', 'seleccion_validacion.json', 'plan_ejecucion.json', 'plan_ampliacion.json', 'plan_analisis.json', 'estado_analisis.json']]
    plan = {'declaracion_unix': time.time(), 'solicitud': 'comparación de las catorce arquitecturas y controles complementarios bajo un protocolo común', 'limite_nueva_sesion_segundos': min(10800, args.horas * 3600), 'presupuesto_previo': 'presupuesto.json se conserva y la nueva autorización utiliza presupuesto_comparacion.json', 'batch_fisico': 8, 'batch_efectivo': 8, 'pasos': 1253, 'semilla': 20261019, 'resolucion_referencia': 224, 'rama_tabla_principal': 'online', 'peso_local_referencia': 0.2, 'recorte_referencia': 0.2, 'inicializacion': 'aleatoria, sin pesos preentrenados', 'referencias': referencias, 'cola': cola, 'inventario_inicial': inventario(args.resultados), 'seleccion': 'pool y configuraciones fijados por cobertura y protocolo antes de nuevas pruebas, selección ampliada mediante la media de mAP de validación, la selección histórica no cambia', 'limitacion_prueba_previa': 'La ampliación se define después de existir pruebas históricas, no constituye un segundo conjunto de prueba ciego, no se usa su puntuación para elegir estas ejecuciones', 'estimacion_previa': 'dos a tres horas, extrapolación de RF67 y RF99 a mapas mayores y recomputación, sin benchmark adicional, no es una medición', 'preservacion_sha256': {str(p.relative_to(args.resultados)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protegidos}}
    guardar_json(ruta, plan)
    return plan


def parametros_referencia(args, plan):
    ruta = args.resultados / 'parametros_arquitecturas.json'
    if ruta.exists():
        return leer(ruta)
    torch.set_num_threads(2)
    resultado = {}
    for config in plan['referencias']:
        encoder = Encoder(config)
        resultado[config['arquitectura']] = sum(p.numel() for p in encoder.parameters())
        del encoder
    guardar_json(ruta, resultado)
    return resultado


def generar_tabla(args, raiz):
    plan = declarar(args)
    parametros = parametros_referencia(args, plan)
    hardware = leer(args.resultados / 'hardware_comparacion.json', {})
    filas = []
    for config in plan['referencias']:
        carpeta = args.resultados / config['nombre']
        resumen = leer(carpeta / 'resumen.json', {})
        validacion = leer(carpeta / 'validacion.json', {})
        prueba = leer(carpeta / 'prueba.json', {})
        tiempos = leer(carpeta / 'tiempos_fases.json', {})
        estado = 'evaluado' if completa_medicion(prueba, 3368) and completa_medicion(validacion, 100) and resumen.get('estado') == 'entrenamiento_completado' else 'entrenado_sin_evaluacion_completa' if resumen.get('estado') == 'entrenamiento_completado' else resumen.get('estado', 'pendiente')
        fila = {'arquitectura': config['arquitectura'], 'configuracion': config['nombre'], 'metodo': 'CO-BYOL', 'rama': 'online', 'estado': estado, 'resolucion_encoder': 224, 'batch_fisico': resumen.get('batch_fisico'), 'batch_efectivo': resumen.get('batch_efectivo'), 'actualizaciones': resumen.get('pasos_realizados'), 'imagenes_fuente_procesadas': resumen.get('imagenes_fuente_procesadas'), 'imagenes_fuente_distintas': resumen.get('imagenes_fuente_distintas'), 'epocas_equivalentes_batches_completos': (resumen.get('epocas_completas', 0) + resumen.get('fraccion_epoca', 0)) if resumen else None, 'parametros_encoder': parametros[config['arquitectura']], 'parametros_online_con_cabezas': resumen.get('parametros_online'), 'vram_pico_entrenamiento_GiB': resumen.get('vram_pico_GiB'), 'segundos_entrenamiento': resumen.get('segundos_entrenamiento'), 'segundos_validacion_dos_ramas': sum(tiempos.get('validation', {}).values()) if tiempos.get('validation') else None, 'segundos_prueba_dos_ramas': sum(tiempos.get('testing', {}).values()) if tiempos.get('testing') else None, 'segundos_preparacion_y_guardado': tiempos.get('preparacion_y_guardado'), 'espera_antes_inicio_segundos': tiempos.get('espera_antes_inicio'), 'GPU': hardware.get('GPU', 'equipo local, consultar hardware_comparacion.json'), 'inicializacion': resumen.get('initializacion', 'aleatoria planificada'), 'recomputacion_activaciones': bool(leer(carpeta / 'configuracion.json', {}).get('checkpoint_activaciones', False))}
        for condicion in ['original', 'baja_resolucion']:
            for metrica in METRICAS:
                valor = prueba.get('online', {}).get(condicion, {}).get(metrica)
                fila[f'{condicion}_{metrica}_porcentaje'] = 100 * valor if valor is not None else None
        for campo in ['caida_mAP_puntos_porcentuales', 'caida_mAP_relativa_porcentaje']:
            fila[campo] = prueba.get('online', {}).get(campo)
        fila['mAP_media_validacion_porcentaje'] = 100 * validacion['online']['criterio_validacion'] if 'online' in validacion else None
        filas.append(fila)
    ruta_csv = args.resultados / 'comparacion_arquitecturas.csv'
    with ruta_csv.open('w', encoding='utf-8-sig', newline='') as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)
    guardar_json(args.resultados / 'comparacion_arquitecturas.json', filas)
    formato = lambda x: f'{x:.3f}' if isinstance(x, (int, float)) else 'Pendiente'
    texto = ['# Comparación común de las catorce arquitecturas', '', 'Este panel fija CO-BYOL con peso local 0.2, rama online, entrada 224, batch físico y efectivo ocho y 1253 actualizaciones, conserva particiones, semilla, inicialización aleatoria y transformaciones comunes, la resolución original y reducida se leen de los HDF5 suministrados y se evalúan con la misma galería original.', '', 'Se reutilizan diez configuraciones existentes y se programan cuatro para completar el panel a resolución común, ResNet18 y RF29 a 64 se conservan como experimentos del panel pequeño del artículo y no se mezclan en la clasificación de esta tabla, el control BYOL sin CO con ResNet50 y las ablaciones con arquitectura fija ya realizadas se conservan y no se repiten.', '', 'Los kernels y las dimensiones de las cabezas locales dependen del backbone como en el artículo y producen cantidades de posiciones distintas, CNN y Transformers conservan sus optimizadores y tasas documentados, BagNet33 mantiene su diferencia de parámetros y el bajo número de actualizaciones no permite concluir convergencia ni diferencias estadísticamente concluyentes.', '', 'RF39 usa recomputación anidada de activaciones para conservar batch ocho, resolución y todas sus posiciones locales sin duplicar las actualizaciones de estadísticas BatchNorm, el mecanismo tiene una verificación CPU dirigida con salidas, gradientes y buffers coincidentes, aumenta el tiempo y su comparación de coste incorpora ese compromiso del equipo.', '', f'Estado actual, {sum(f["estado"] == "evaluado" for f in filas)} de catorce arquitecturas evaluadas bajo este protocolo, las celdas pendientes no representan resultados medidos.', '', '## Métricas de prueba', '', '| Arquitectura | Estado | mAP O % | R1 O % | R5 O % | R10 O % | mAP LR % | R1 LR % | R5 LR % | R10 LR % | Caída pp | Caída relativa % |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for f in filas:
        valores = [formato(f[f'{condicion}_{metrica}_porcentaje']) for condicion in ['original', 'baja_resolucion'] for metrica in METRICAS]
        texto.append('| ' + ' | '.join([f['arquitectura'], f['estado'], *valores, formato(f['caida_mAP_puntos_porcentuales']), formato(f['caida_mAP_relativa_porcentaje'])]) + ' |')
    texto += ['', 'O corresponde a consultas originales y LR a las versiones reducidas, una caída negativa indica mejora en esa condición y no se recorta a cero, cada fila utiliza online de la configuración predeclarada y las dos ramas completas permanecen disponibles en sus JSON.', '', '## Costes medidos y cobertura', '', '| Arquitectura | Configuración | Parámetros encoder M | Entrenamiento min | Validación s | Prueba s | VRAM entrenamiento GiB | Batch efectivo | Actualizaciones |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for f in filas:
        tiempo_val = formato(f['segundos_validacion_dos_ramas']) if f['segundos_validacion_dos_ramas'] is not None else 'No registrado'
        tiempo_prueba = formato(f['segundos_prueba_dos_ramas']) if f['segundos_prueba_dos_ramas'] is not None else 'No registrado'
        texto.append('| ' + ' | '.join([f['arquitectura'], f['configuracion'], formato(f['parametros_encoder'] / 1e6), formato(f['segundos_entrenamiento'] / 60 if f['segundos_entrenamiento'] is not None else None), tiempo_val, tiempo_prueba, formato(f['vram_pico_entrenamiento_GiB']), str(f['batch_efectivo'] or 'Pendiente'), str(f['actualizaciones'] or 'Pendiente')]) + ' |')
    texto += ['', 'El CSV conserva también exposición, épocas equivalentes, espera y preparación cuando fueron medidas, los tiempos históricos de validación y prueba que no se registraron por separado permanecen pendientes de medición y no se reconstruyen restando intervalos con alcances diferentes, no se repiten experimentos para llenar esas columnas.', '', 'El pico de VRAM corresponde a memoria asignada por PyTorch durante entrenamiento, no es el consumo total del sistema ni una medición de evaluación, los tiempos son de pared e incluyen las transferencias y los guardados periódicos dentro del bucle, la memoria libre y las demás aplicaciones del equipo pueden afectar la comparación temporal.', '', 'El hardware de la sesión nueva se conserva en resultados/hardware_comparacion.json y su presupuesto independiente en presupuesto_comparacion.json, el presupuesto anterior permanece sin cambios, todos los modelos comparables deben conservar el mismo hash del conjunto de índices fuente visitados y eso no demuestra por sí solo identidad de toda la secuencia de aumentos.', '', '## Interpretación de resultados', '']
    completas = [f for f in filas if f['estado'] == 'evaluado']
    if len(completas) == 14:
        rendimiento = lambda f: (f['original_mAP_porcentaje'] + f['baja_resolucion_mAP_porcentaje']) / 2
        mejor = max(completas, key=rendimiento)
        robusta = min(completas, key=lambda f: f['caida_mAP_relativa_porcentaje'])
        elegibles = [f for f in completas if f['segundos_entrenamiento'] and f['segundos_entrenamiento'] > 0]
        eficiente = max(elegibles, key=lambda f: rendimiento(f) / (f['segundos_entrenamiento'] / 60))
        frente = [f['arquitectura'] for f in elegibles if not any(rendimiento(g) >= rendimiento(f) and g['segundos_entrenamiento'] <= f['segundos_entrenamiento'] and (rendimiento(g) > rendimiento(f) or g['segundos_entrenamiento'] < f['segundos_entrenamiento']) for g in elegibles)]
        texto += [f'La mayor media descriptiva de mAP entre condiciones en prueba corresponde a {mejor["arquitectura"]} con {rendimiento(mejor):.3f} por ciento, su mAP original es {mejor["original_mAP_porcentaje"]:.3f} y la reducida es {mejor["baja_resolucion_mAP_porcentaje"]:.3f}, esta ordenación describe resultados y no se utiliza para ajustar hiperparámetros ni sustituir la selección por validación.', '', f'La menor caída relativa corresponde a {robusta["arquitectura"]} con {robusta["caida_mAP_relativa_porcentaje"]:.3f} por ciento y mAP reducida {robusta["baja_resolucion_mAP_porcentaje"]:.3f}, retener un porcentaje alto de un rendimiento inicial bajo no equivale a obtener la mayor precisión absoluta y la tabla permite distinguir ambos criterios.', '', f'El mayor cociente descriptivo entre media de mAP porcentual y minutos de entrenamiento corresponde a {eficiente["arquitectura"]}, el frente no dominado en mAP media y tiempo incluye {", ".join(frente)}, ese cociente no representa energía medida ni una ley de escalado y no obliga a escoger un único ganador.', '', 'Las diferencias pueden relacionarse con campo receptivo, capacidad, geometría de las personas y coste de memoria, estas explicaciones son hipótesis salvo los efectos aislados por las ablaciones existentes, una sola semilla y menos de una época no permiten presentar diferencias pequeñas como concluyentes.', '']
    else:
        texto += ['La comparación global permanece abierta hasta disponer de las catorce filas completas, no se declara un ganador del conjunto ni se rellenan las métricas faltantes con estimaciones.', '']
    texto += ['## Comparaciones complementarias y límites', '', 'Los resultados de modelo base y ablaciones se conservan en INFORME.md y metricas.csv, el contraste adicional de RF99 con recorte mínimo 0.9 usa control sin CO y variante CO bajo las mismas actualizaciones y se registra por separado, esos dos experimentos tienen prioridad posterior a completar las arquitecturas y no se incorporan como filas intercambiables del panel con recorte 0.2.', '', 'La selección ampliada se registra en seleccion_comparacion.json exclusivamente con validación y sin modificar seleccion_validacion.json del experimento histórico, la ampliación se diseñó después de disponer de pruebas históricas y no representa una nueva evaluación completamente ciega, se mantienen fijas las configuraciones y no se eligen por sus puntuaciones de prueba.', '', 'La matriz histórica ampliada mantiene combinaciones no ejecutadas fuera de esta cola mínima, eso no equivale a arquitecturas pendientes del panel, los calendarios largos, Places365 y la igualación literal de parámetros de BagNet33 conservan sus límites declarados y no se anuncian como reproducción completa del artículo.', '']
    controles_recorte = []
    for peso in [0.0, 0.2]:
        nombre = f'byol_rf99_resnet50_co{peso}_crop0.9'
        medicion = leer(args.resultados / nombre / 'prueba.json', {})
        if completa_medicion(medicion, 3368):
            controles_recorte.append((peso, medicion['online']))
    if len(controles_recorte) == 2:
        texto += ['## Control de recorte mínimo 0.9 completado', '', '| Peso local RF99 | mAP original % | mAP reducida % |', '|---|---:|---:|']
        for peso, medicion in controles_recorte:
            texto.append(f'| {peso:g} | {100 * medicion["original"]["mAP"]:.3f} | {100 * medicion["baja_resolucion"]["mAP"]:.3f} |')
        texto += ['', 'Ambos controles completaron entrenamiento y evaluación y se mantienen separados de la tabla con recorte mínimo 0.2, en esta ejecución RF99 con CO obtiene menor mAP que el control sin CO en las dos condiciones, este contraste no respalda una mejora local sin recortes pequeños y no se extrapola a los calendarios largos de clasificación del artículo.', '']
    (raiz / 'COMPARACION_ARQUITECTURAS.md').write_text('\n'.join(texto), encoding='utf-8')
    return filas


def evaluar_fase(args, modelo, config, fase, presupuesto):
    carpeta = args.resultados / config['nombre']
    ruta = carpeta / ('validacion.json' if fase == 'validation' else 'prueba.json')
    resultado = leer(ruta, {})
    tiempos = leer(carpeta / 'tiempos_fases.json', {})
    batch = 4 if config['arquitectura'] == 'rf39_resnet50' else 32
    for rama in ['online', 'objetivo']:
        if rama in resultado:
            continue
        inicio = time.perf_counter()
        resultado[rama] = evaluar_encoder(getattr(modelo, rama).encoder, args.datos, fase, 224, torch.device(args.dispositivo), batch=batch, protocolo='market_documentado', limite=presupuesto)
        tiempos.setdefault(fase, {})[rama] = time.perf_counter() - inicio
        guardar_json(ruta, resultado)
        guardar_json(carpeta / 'tiempos_fases.json', tiempos)
    if fase == 'validation':
        resultado['rama_seleccionada'] = max(['online', 'objetivo'], key=lambda r: resultado[r]['criterio_validacion'])
        resultado['seleccion_exclusivamente_validacion'] = True
        guardar_json(ruta, resultado)
    return resultado


def actualizar_estado_general(args, config, resumen, validacion, prueba_completa=False):
    ruta = args.resultados / 'estado_ejecucion.json'
    estados = leer(ruta, [])
    nombre = config['nombre']
    if any(e['modelo'] == nombre and e.get('grupo') != 'comparacion_arquitecturas' for e in estados):
        return
    estados = [e for e in estados if e['modelo'] != nombre]
    rama = validacion.get('rama_seleccionada')
    estado = {'modelo': nombre, 'grupo': 'comparacion_arquitecturas', 'estado': resumen['estado'], 'pasos': resumen['pasos_realizados'], 'rama': rama, 'evaluacion_prueba': 'completada' if prueba_completa else 'pendiente'}
    if rama:
        estado['validacion_mAP_media'] = validacion[rama]['criterio_validacion']
    estados.append(estado)
    guardar_json(ruta, estados)


def ejecutar_cola(args, raiz):
    plan = declarar(args)
    ruta_estado = args.resultados / 'estado_comparacion.json'
    with FileLock(str(args.resultados / 'comparacion.lock'), timeout=0):
        anteriores = leer(ruta_estado, {})
        if anteriores.get('estado') == 'completado':
            generar_tabla(args, raiz)
            return anteriores
        presupuesto = Presupuesto(args.resultados / 'presupuesto_comparacion.json', plan['limite_nueva_sesion_segundos'])
        if presupuesto.agotado():
            raise TimeoutError('El presupuesto complementario terminó y no se reinicia automáticamente')
        dispositivo = torch.device(args.dispositivo)
        if dispositivo.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('La cola requiere CUDA, no se sustituye por entrenamiento CPU')
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        propiedades = torch.cuda.get_device_properties(dispositivo)
        hardware = {'GPU': propiedades.name, 'VRAM_total_GiB': propiedades.total_memory / 1024 ** 3, 'CPU': platform.processor(), 'sistema': platform.platform(), 'torch': torch.__version__, 'CUDA_torch': torch.version.cuda, 'medicion_unix': time.time(), 'RAM': '33.4 GB instalados y aproximadamente 4.8 GB libres en la inspección previa, no se supone disponibilidad completa', 'limite_segundos': plan['limite_nueva_sesion_segundos']}
        guardar_json(args.resultados / 'hardware_comparacion.json', hardware)
        estado = {'estado': 'en_curso', 'pid': os.getpid(), 'inicio_unix': presupuesto.datos['inicio_unix'], 'cola': [c['configuracion']['nombre'] for c in plan['cola']], 'tareas': anteriores.get('tareas', {}), 'sesion': 'presupuesto_comparacion.json', 'reanudar': 'se reutilizan entrenamiento completo, validación y ramas de prueba guardadas, no se finge reanudación exacta de un entrenamiento parcial'}
        def registrar(nombre, fase, **extra):
            estado['actual'] = nombre
            estado['fase'] = fase
            estado['actualizacion_unix'] = time.time()
            estado['tareas'].setdefault(nombre, {}).update(fase=fase, **extra)
            guardar_json(ruta_estado, estado)
            print(json.dumps({'evento': 'cola', 'modelo': nombre, 'fase': fase, **extra}), flush=True)
        guardar_json(ruta_estado, estado)
        try:
            for item in plan['cola']:
                config = item['configuracion']
                nombre = config['nombre']
                carpeta = args.resultados / nombre
                resumen = leer(carpeta / 'resumen.json', {})
                validacion = leer(carpeta / 'validacion.json', {})
                if resumen.get('estado') == 'entrenamiento_completado' and completa_medicion(validacion, 100):
                    registrar(nombre, 'validacion_completa_reutilizada')
                    continue
                if presupuesto.agotado(1800):
                    registrar(nombre, 'pendiente_presupuesto')
                    continue
                modelo = None
                try:
                    if resumen and resumen.get('estado') != 'entrenamiento_completado':
                        registrar(nombre, 'parcial_conservado_sin_reinicio_automatico')
                        continue
                    if not resumen and (carpeta / 'checkpoint_periodico.pt').exists():
                        registrar(nombre, 'checkpoint_parcial_conservado_sin_reinicio_automatico')
                        continue
                    inicio = time.perf_counter()
                    if resumen.get('estado') == 'entrenamiento_completado':
                        modelo, config, resumen = cargar_checkpoint(carpeta / 'checkpoint.pt', dispositivo)
                    else:
                        registrar(nombre, 'entrenando', inicio_entrenamiento_unix=time.time())
                        modelo, resumen = entrenar(config, args.datos, dispositivo, 8, 1253, 2, args.resultados, presupuesto, reserva=1800)
                        tiempos = {'espera_antes_inicio': estado['tareas'][nombre]['inicio_entrenamiento_unix'] - presupuesto.datos['inicio_unix'], 'preparacion_y_guardado': max(0.0, time.perf_counter() - inicio - resumen['segundos_entrenamiento'])}
                        guardar_json(carpeta / 'tiempos_fases.json', tiempos)
                        parcial = carpeta / 'checkpoint_periodico.pt'
                        if resumen['estado'] == 'entrenamiento_completado' and parcial.exists():
                            parcial.unlink()
                        guardar_json(carpeta / 'progreso.json', {'pasos': resumen['pasos_realizados'], 'pasos_objetivo': 1253, 'estado': resumen['estado'], 'checkpoint': 'checkpoint.pt', 'actualizacion_unix': time.time()})
                    if resumen['estado'] != 'entrenamiento_completado':
                        actualizar_estado_general(args, config, resumen, {})
                        registrar(nombre, 'entrenamiento_parcial')
                        continue
                    registrar(nombre, 'validando')
                    validacion = evaluar_fase(args, modelo, config, 'validation', presupuesto)
                    actualizar_estado_general(args, config, resumen, validacion)
                    registrar(nombre, 'validacion_completa')
                except Exception as error:
                    registrar(nombre, 'fallo', error=str(error), traza=traceback.format_exc())
                finally:
                    modelo = None
                    liberar()
                generar_tabla(args, raiz)
            seleccion = args.resultados / 'seleccion_comparacion.json'
            if not seleccion.exists():
                candidatas = []
                for config in plan['referencias']:
                    validacion = leer(args.resultados / config['nombre'] / 'validacion.json', {})
                    if completa_medicion(validacion, 100):
                        candidatas.append({'modelo': config['nombre'], 'arquitectura': config['arquitectura'], 'rama': 'online', 'mAP_media_validacion': validacion['online']['criterio_validacion']})
                if candidatas:
                    guardar_json(seleccion, {**max(candidatas, key=lambda c: c['mAP_media_validacion']), 'creacion_unix': time.time(), 'pool_predeclarado': [c['nombre'] for c in plan['referencias']], 'candidatas_con_validacion': len(candidatas), 'criterio': 'media de mAP de validación original y reducida, rama online fijada para comparar, selección histórica preservada'})
            for item in plan['cola']:
                config = item['configuracion']
                nombre = config['nombre']
                carpeta = args.resultados / nombre
                resumen = leer(carpeta / 'resumen.json', {})
                validacion = leer(carpeta / 'validacion.json', {})
                prueba = leer(carpeta / 'prueba.json', {})
                if resumen.get('estado') != 'entrenamiento_completado' or not completa_medicion(validacion, 100):
                    continue
                if completa_medicion(prueba, 3368):
                    registrar(nombre, 'evaluado_reutilizado')
                    continue
                modelo = None
                try:
                    if presupuesto.agotado(60):
                        registrar(nombre, 'prueba_pendiente_presupuesto')
                        continue
                    registrar(nombre, 'evaluando_prueba')
                    modelo, config, resumen = cargar_checkpoint(carpeta / 'checkpoint.pt', dispositivo)
                    evaluar_fase(args, modelo, config, 'testing', presupuesto)
                    actualizar_estado_general(args, config, resumen, validacion, True)
                    registrar(nombre, 'evaluado')
                except Exception as error:
                    registrar(nombre, 'fallo_evaluacion', error=str(error), traza=traceback.format_exc())
                finally:
                    modelo = None
                    liberar()
                generar_tabla(args, raiz)
        finally:
            presupuesto.cerrar()
            filas = generar_tabla(args, raiz)
            estado['arquitecturas_referencia_evaluadas'] = sum(f['estado'] == 'evaluado' for f in filas)
            tareas_completas = all(completa_medicion(leer(args.resultados / item['configuracion']['nombre'] / 'prueba.json', {}), 3368) for item in plan['cola'])
            estado['estado'] = 'completado' if tareas_completas and estado['arquitecturas_referencia_evaluadas'] == 14 else 'parcial'
            estado['fin_unix'] = time.time()
            guardar_json(ruta_estado, estado)
        for relativo, sha in plan['preservacion_sha256'].items():
            if hashlib.sha256((args.resultados / relativo).read_bytes()).hexdigest() != sha:
                raise ValueError(f'Cambió un resultado histórico protegido, {relativo}')
        from codigo.informe import generar_informe
        from codigo.auditoria import auditar_resultados
        generar_informe(args.resultados, raiz)
        auditoria = auditar_resultados(args.resultados, raiz, args.datos)
        verificacion = {'estado': 'aprobado' if auditoria['estado'] == 'aprobado' and estado['arquitecturas_referencia_evaluadas'] == 14 else 'parcial_o_requiere_revision', 'arquitecturas': estado['arquitecturas_referencia_evaluadas'], 'cola_completa': estado['estado'] == 'completado', 'registros_historicos_preservados': len(plan['preservacion_sha256']), 'mismo_conjunto_indices': auditoria['conjuntos_indices_fuente_distintos'] == 1, 'estado_auditoria': auditoria['estado'], 'errores_auditoria': auditoria['errores'], 'seleccion_historica_preservada': True, 'pendientes_humanos': auditoria['pendiente_humano']}
        guardar_json(args.resultados / 'verificacion_comparacion.json', verificacion)
        texto = (raiz / 'README.md').read_text(encoding='utf-8')
        aviso = f'La continuación autorizada se documenta en COMPARACION_ARQUITECTURAS.md, actualmente hay {estado["arquitecturas_referencia_evaluadas"]} de catorce arquitecturas evaluadas bajo el protocolo común a 224, resultados/comparacion_arquitecturas.csv contiene todas las métricas y costes disponibles, las combinaciones históricas no esenciales no se anuncian como entrenamientos pendientes de arquitectura.'
        marcador = '## Comparación ampliada de arquitecturas'
        if marcador in texto:
            texto = texto[:texto.index(marcador)].rstrip() + '\n'
        (raiz / 'README.md').write_text(texto.rstrip() + '\n\n' + marcador + '\n\n' + aviso + '\n', encoding='utf-8')
        print(json.dumps({'evento': 'fin_comparacion', **verificacion}), flush=True)
        return estado
