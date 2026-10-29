import ast
import copy
import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import DataLoader, Subset

from codigo.comparacion import ARQUITECTURAS, METRICAS
from codigo.datos import Entrenamiento
from codigo.entrenamiento import construir_optimizador, fijar_semilla, guardar_json, liberar
from codigo.evaluacion import evaluar_encoder, generar_ejemplos
from codigo.modelos import EncodersRecuperacion, construir_modelo


def leer(ruta, defecto=None):
    ruta = Path(ruta)
    return json.loads(ruta.read_text(encoding='utf-8')) if ruta.exists() else defecto


def huella(ruta):
    with Path(ruta).open('rb') as archivo:
        return hashlib.file_digest(archivo, 'sha256').hexdigest()


def guardar_torch(contenido, ruta):
    ruta = Path(ruta)
    temporal = ruta.with_suffix('.tmp')
    torch.save(contenido, temporal)
    temporal.replace(ruta)


class PlazoGlobal:
    def __init__(self, ruta):
        self.ruta = Path(ruta)
        self.datos = leer(self.ruta)
        if not self.datos or self.datos['maximo_segundos'] != 36000 or self.datos['fin_unix'] - self.datos['inicio_unix'] != 36000:
            raise ValueError('Falta el plazo humano absoluto de diez horas, no se crea ni reinicia automáticamente')

    def restante(self):
        return max(0.0, self.datos['fin_unix'] - time.time())

    def agotado(self, reserva=0):
        return self.restante() <= reserva


def configuracion_nueva(configuracion):
    resultado = dict(configuracion)
    resultado['checkpoint_rama'] = True
    if resultado['arquitectura'].startswith('rf'):
        resultado['checkpoint_activaciones'] = True
    return resultado


def registrar(destino, evento, **datos):
    contenido = {'evento': evento, 'unix': time.time(), **datos}
    with (Path(destino) / 'eventos.jsonl').open('a', encoding='utf-8') as archivo:
        archivo.write(json.dumps(contenido, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(contenido, ensure_ascii=False), flush=True)


def declarar(raiz, destino):
    ruta = destino / 'plan_fase2.json'
    if ruta.exists():
        return leer(ruta)
    anteriores = raiz / 'resultados'
    referencia = leer(anteriores / 'plan_comparacion.json')['referencias']
    trabajos = [{'nombre': config['nombre'], 'grupo': 'arquitecturas', 'configuracion': configuracion_nueva(config), 'origen': f"resultados/{config['nombre']}/checkpoint.pt"} for config in referencia]
    for nombre in ['byol_r50', 'byol_rf99', 'byol_bagnet33', 'mocov3_r50', 'co_mocov3_r50']:
        trabajos.append({'nombre': nombre, 'grupo': 'controles', 'configuracion': configuracion_nueva(leer(anteriores / nombre / 'configuracion.json')), 'origen': f'resultados/{nombre}/checkpoint.pt'})
    protegidos = {p.relative_to(raiz).as_posix(): huella(p) for p in anteriores.rglob('*') if p.is_file() and p.suffix in {'.json', '.csv'}}
    fuentes = {n: huella(Path('C:/Users/msdro/Downloads') / n) for n in ['Laboratorio 3.pdf', 'main (3).pdf', 'comentarios_lab3.txt']}
    codigo = {p.relative_to(raiz).as_posix(): huella(p) for p in [raiz / 'main.py', *sorted((raiz / 'codigo').glob('*.py'))]}
    plan = {'declaracion_unix': time.time(), 'plazo': leer(destino / 'plazo_global.json'), 'trabajos': trabajos, 'arquitecturas': ARQUITECTURAS, 'maximo_epocas_adicionales': 3, 'seleccion': 'por configuración, mejor época y rama según media mAP de validación original y reducida, misma época seleccionada para comparar ambas ramas, selección global por el mismo criterio, prueba final posterior a cierre global', 'origen': 'continuación de los checkpoints históricos válidos, sin pesos externos, cada uno recibió 1253 pasos batch 8 en fase 1', 'calendario': 'optimización y EMA de fase 2 reiniciadas explícitamente, mismo número de épocas adicionales completas por ronda, los pilotos se descartan y no se suman a los modelos finales', 'piloto_CNN': {'modelo': 'co_byol_rf99', 'tasas_base': [0.2, 0.8, 1.6], 'pasos': 400}, 'piloto_transformer': {'modelo': 'co_byol_vit_s16', 'tasas_base': [0.0003, 0.0005, 0.001], 'pasos': 400}, 'pilotos': 'selección de una tasa común CNN y una común Transformer, por validación sin usar testing, barrido reducido de los valores publicados, sin pretender ajuste individual óptimo', 'protocolo': 'market_documentado, L2, coseno, mAP, Rank-1/5/10 y ambas caídas, originales y LR suministradas, no cambiar los HDF5 ni sus particiones', 'imagen': '224x224 para mantener el panel de catorce y la geometría local, no se introduce degradación LR en training ni nuevas pérdidas', 'batch': 'mayor batch común entre 8 y 4 que supere la prueba de memoria de RF39, RF29 y SwinB, sin acumulación para simular BatchNorm', 'reanudacion': 'checkpoint con optimizador, cursor dentro de permutación fija de época, RNG Python/NumPy/Torch/CUDA, trabajadores cero y generador del DataLoader separado, no reinicia el plazo', 'historicos_SHA256': protegidos, 'fuentes_SHA256': fuentes, 'codigo_inicial_fase2_SHA256': codigo, 'semilla_unica': True, 'limite_inferencias_prueba': 'cada configuración se evalúa una vez al final con su época elegida por validación, ambas ramas para cumplir laboratorio, no se prueba durante pilotos ni validación rápida', 'limite_procedencia': 'fase 2 se define después de resultados históricos de testing, no es otro conjunto de prueba ciego, no se usa testing para escoger tasas ni época', 'faltante': 'mejoras_lab3_noche.zip no recibido, no se puede auditar cambios.patch ni largo.py ausentes'}
    guardar_json(ruta, plan)
    return plan


def recuperar_rng(contenido):
    if 'rng_torch' not in contenido:
        return
    random.setstate(contenido['rng_python'])
    np.random.set_state(contenido['rng_numpy'])
    torch.set_rng_state(contenido['rng_torch'].cpu())
    torch.cuda.set_rng_state(contenido['rng_cuda'].cpu())


def rng():
    return {'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(), 'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state()}


def preparar_trabajo(trabajo, raiz, dispositivo, batch, tasa=None):
    config = dict(trabajo['configuracion'])
    if tasa is not None:
        config['tasa_base'] = tasa
    fijar_semilla(config['semilla'])
    modelo = construir_modelo(config)
    contenido = torch.load(raiz / trabajo['origen'], map_location='cpu', weights_only=False, mmap=True)
    modelo.load_state_dict(contenido['modelo'], strict=True)
    del contenido
    modelo.to(dispositivo)
    optimizador, tasa_real = construir_optimizador(modelo, config, batch)
    return modelo, optimizador, tasa_real, config


def cargador_epoca(datos, semilla, epoca, batch, cursor=0):
    generador = torch.Generator().manual_seed(semilla + 104729 * (epoca + 1))
    permutacion = torch.randperm(len(datos), generator=generador).tolist()
    total = len(permutacion) // batch * batch
    subconjunto = Subset(datos, permutacion[cursor * batch:total])
    return DataLoader(subconjunto, batch_size=batch, shuffle=False, num_workers=0, drop_last=True, pin_memory=True, generator=torch.Generator().manual_seed(semilla + epoca)), permutacion[:total]


def paso(modelo, optimizador, vistas, tasa, progreso, calentamiento, momentum=0.996):
    factor = (progreso + 1) / calentamiento if progreso < calentamiento else 1.0
    for grupo in optimizador.param_groups:
        grupo['lr'] = tasa * factor
    optimizador.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        perdida, partes = modelo(vistas)
    if not torch.isfinite(perdida):
        raise ValueError('Pérdida no finita, se conserva el último checkpoint')
    perdida.backward()
    norma = torch.nn.utils.clip_grad_norm_(modelo.online.parameters(), float('inf'))
    if not torch.isfinite(norma):
        raise ValueError('Gradiente no finito, se conserva el último checkpoint')
    optimizador.step()
    modelo.actualizar_objetivo(momentum)
    return float(perdida.detach()), float(norma), factor


def validar(modelo, args, config, plazo, batch):
    resultados = {'tiempos_segundos': {}}
    for rama in ['online', 'objetivo']:
        inicio = time.perf_counter()
        resultados[rama] = evaluar_encoder(getattr(modelo, rama).encoder, args.datos, 'validation', 224, torch.device('cuda'), batch=batch, protocolo='market_documentado', limite=plazo)
        resultados['tiempos_segundos'][rama] = time.perf_counter() - inicio
    return resultados


def criterio(validacion):
    rama = max(['online', 'objetivo'], key=lambda nombre: validacion[nombre]['criterio_validacion'])
    return rama, validacion[rama]['criterio_validacion']


def guardar_mejor(modelo, config, carpeta, validacion, epoca):
    rama, puntuacion = criterio(validacion)
    actual = leer(carpeta / 'seleccion_validacion.json')
    if actual is not None and puntuacion <= actual['criterio']:
        return False
    seleccion = {'rama': rama, 'criterio': puntuacion, 'epoca_adicional_seleccionada': epoca, 'actualizacion_unix': time.time(), 'criterio_descripcion': 'media mAP original/LR de validación, ambas ramas del mismo checkpoint'}
    pesos = {nombre: valor.detach().cpu() for nombre, valor in modelo.state_dict().items() if nombre.startswith(('online.encoder.', 'objetivo.encoder.'))}
    guardar_torch({'tipo': 'encoders_recuperacion', 'configuracion': config, 'modelo': pesos, 'resumen': seleccion}, carpeta / 'mejores_encoders.pt')
    guardar_json(carpeta / 'seleccion_validacion.json', seleccion)
    return True


def calibrar(args, raiz, destino, plazo, plan):
    ruta = destino / 'calibracion_fase2.json'
    if ruta.exists():
        return leer(ruta)
    dispositivo = torch.device('cuda')
    resultados = []
    elegido = None
    for batch in [8, 4, 2]:
        correcto = True
        for nombre in ['co_byol_rf39_resnet50', 'co_byol_rf29_224', 'co_byol_swin_b']:
            trabajo = next(t for t in plan['trabajos'] if t['nombre'] == nombre)
            modelo = optimizador = vistas = datos = cargador = None
            inicio = time.perf_counter()
            try:
                if plazo.agotado(7200):
                    raise TimeoutError('No queda reserva para la fase comparativa')
                modelo, optimizador, tasa, config = preparar_trabajo(trabajo, raiz, dispositivo, batch)
                datos = Entrenamiento(args.datos, config)
                cargador, indices = cargador_epoca(datos, config['semilla'], 0, batch)
                modelo.train()
                torch.cuda.reset_peak_memory_stats()
                duraciones = []
                for indice, (vistas_cpu, _) in enumerate(cargador):
                    vistas = [v.to(dispositivo, non_blocking=True) for v in vistas_cpu]
                    torch.cuda.synchronize()
                    paso_inicio = time.perf_counter()
                    perdida, norma, _ = paso(modelo, optimizador, vistas, tasa, indice, 1)
                    torch.cuda.synchronize()
                    duraciones.append(time.perf_counter() - paso_inicio)
                    if indice == 1:
                        break
                resultado = {'nombre': nombre, 'batch': batch, 'estado': 'aprobado', 'segundos_paso': duraciones[-1], 'vram_GiB': torch.cuda.max_memory_allocated() / 1024 ** 3, 'segundos_total': time.perf_counter() - inicio, 'pasos_descartados': 2}
            except torch.cuda.OutOfMemoryError:
                correcto = False
                resultado = {'nombre': nombre, 'batch': batch, 'estado': 'memoria_insuficiente', 'segundos_total': time.perf_counter() - inicio}
            finally:
                modelo = optimizador = vistas = datos = cargador = None
                liberar()
            resultados.append(resultado)
            registrar(destino, 'calibracion', **resultado)
            if not correcto:
                break
        if correcto:
            elegido = batch
            break
    if elegido is None:
        guardar_json(destino / 'fallo_memoria.json', {'estado': 'requiere_revision', 'mediciones': resultados, 'motivo': 'ni batch 4 conserva las catorce arquitecturas con la GPU disponible, no se reduce arbitrariamente un único modelo'})
        raise RuntimeError('No cabe el batch común mínimo 2, revisar procesos ajenos sin cerrarlos automáticamente')
    resultado = {'batch_comun': elegido, 'mediciones': resultados, 'creacion_unix': time.time(), 'checkpoint_rama': True, 'checkpoint_rf': True, 'modificaciones_pesos_calibracion_descartadas': True}
    guardar_json(ruta, resultado)
    return resultado


def barrer_tasas(args, raiz, destino, plazo, plan, batch):
    ruta = destino / 'seleccion_tasas.json'
    if ruta.exists():
        return leer(ruta)
    grupos = {}
    for familia, detalle in [('CNN', plan['piloto_CNN']), ('Transformer', plan['piloto_transformer'])]:
        filas = []
        trabajo = next(t for t in plan['trabajos'] if t['nombre'] == detalle['modelo'])
        for tasa_base in detalle['tasas_base']:
            ruta_piloto = destino / 'pilotos' / f'{familia}_{tasa_base}.json'
            if ruta_piloto.exists():
                filas.append(leer(ruta_piloto))
                continue
            if plazo.agotado(7200):
                raise TimeoutError('Pilotos detenidos para preservar reserva de evaluación')
            modelo, optimizador, tasa, config = preparar_trabajo(trabajo, raiz, torch.device('cuda'), batch, tasa_base)
            datos = Entrenamiento(args.datos, config)
            cargador, indices = cargador_epoca(datos, config['semilla'], 0, batch)
            modelo.train()
            inicio = time.perf_counter()
            parametros_antes = modelo.online.encoder.red.inicio[0].weight.detach().clone() if modelo.online.encoder.rf else next(modelo.online.encoder.parameters()).detach().clone()
            perdidas = []
            for numero, (vistas_cpu, _) in enumerate(cargador):
                vistas = [v.to('cuda', non_blocking=True) for v in vistas_cpu]
                perdida, norma, _ = paso(modelo, optimizador, vistas, tasa, numero, 40)
                perdidas.append(perdida)
                if numero + 1 == detalle['pasos']:
                    break
            parametros_despues = modelo.online.encoder.red.inicio[0].weight if modelo.online.encoder.rf else next(modelo.online.encoder.parameters())
            cambio = float((parametros_despues.detach() - parametros_antes).norm() / parametros_antes.norm())
            validacion = validar(modelo, args, config, plazo, 2 if config['arquitectura'] == 'rf39_resnet50' else 8)
            rama, puntuacion = criterio(validacion)
            fila = {'familia': familia, 'modelo': trabajo['nombre'], 'tasa_base': tasa_base, 'tasa_efectiva': tasa, 'pasos': len(perdidas), 'batch': batch, 'segundos': time.perf_counter() - inicio, 'validacion': validacion, 'criterio': puntuacion, 'rama': rama, 'cambio_relativo_primer_tensor_encoder': cambio, 'alcance_cambio': 'primer tensor, no norma de todos los parámetros, no prueba causal de estabilidad BatchNorm', 'perdida_inicial': perdidas[0], 'perdida_final': perdidas[-1], 'pesos_descartados': True}
            guardar_json(ruta_piloto, fila)
            filas.append(fila)
            registrar(destino, 'piloto', familia=familia, tasa_base=tasa_base, criterio=puntuacion, cambio_relativo=cambio)
            del modelo, optimizador, datos, cargador, vistas, vistas_cpu, parametros_antes, parametros_despues
            liberar()
        grupos[familia] = {'tasa_base': max(filas, key=lambda fila: fila['criterio'])['tasa_base'], 'pilotos': filas}
    guardar_json(ruta, grupos)
    return grupos


def comprobar_recorrido(args, raiz, destino, plazo, plan, batch):
    ruta = destino / 'verificacion_GPU.json'
    if ruta.exists():
        return leer(ruta)
    trabajo = next(t for t in plan['trabajos'] if t['nombre'] == 'co_byol_rf99')
    carpeta = destino / 'prueba_recorrido'
    carpeta.mkdir(exist_ok=True)
    modelo, optimizador, tasa, config = preparar_trabajo(trabajo, raiz, torch.device('cuda'), batch)
    datos = Entrenamiento(args.datos, config)
    cargador, _ = cargador_epoca(datos, config['semilla'], 0, batch)
    modelo.train()
    inicio = time.perf_counter()
    for indice, (vistas_cpu, _) in enumerate(cargador):
        vistas = [v.to('cuda') for v in vistas_cpu]
        perdida, norma, _ = paso(modelo, optimizador, vistas, tasa, indice, 1)
        if indice == 2:
            break
    guardar_estado_entrenamiento(modelo, optimizador, carpeta, config, {'epocas_completas': 0, 'cursor_pasos': 3, 'pasos_totales_fase2': 3})
    del modelo, optimizador, datos, cargador, vistas, vistas_cpu
    liberar()
    modelo, optimizador, tasa, config = preparar_trabajo(trabajo, raiz, torch.device('cuda'), batch)
    recuperado = torch.load(carpeta / 'ultimo_checkpoint.pt', map_location='cpu', weights_only=False, mmap=True)
    modelo.load_state_dict(recuperado['modelo'], strict=True)
    optimizador.load_state_dict(recuperado['optimizador'])
    recuperar_rng(recuperado)
    assert recuperado['estado_fase2']['cursor_pasos'] == 3
    del recuperado
    validacion = validar(modelo, args, config, plazo, 8)
    guardar_mejor(modelo, config, carpeta, validacion, 0)
    exportado = torch.load(carpeta / 'mejores_encoders.pt', map_location='cpu', weights_only=False, mmap=True)
    actuales = modelo.state_dict()
    assert all(torch.equal(tensor, actuales[nombre].detach().cpu()) for nombre, tensor in exportado['modelo'].items())
    resultado = {'estado': 'aprobado', 'modelo': trabajo['nombre'], 'batch': batch, 'pasos_entrenamiento_descartados': 3, 'checkpoint_modelo_optimizador_rng_recuperado': True, 'validacion_completa': all(validacion[r][c]['consultas_evaluadas'] == 100 for r in ['online', 'objetivo'] for c in ['original', 'baja_resolucion']), 'encoders_exportados_tensores_iguales': True, 'testing_abierto': False, 'segundos': time.perf_counter() - inicio, 'validacion': validacion}
    assert resultado['validacion_completa']
    guardar_json(ruta, resultado)
    del modelo, optimizador, exportado, actuales
    liberar()
    for nombre in ['ultimo_checkpoint.pt', 'mejores_encoders.pt']:
        (carpeta / nombre).unlink()
    registrar(destino, 'prueba_recorrido_aprobada', segundos=resultado['segundos'])
    return resultado


def guardar_estado_entrenamiento(modelo, optimizador, carpeta, config, estado):
    guardar_torch({'configuracion': config, 'modelo': modelo.state_dict(), 'optimizador': optimizador.state_dict(), 'estado_fase2': estado, **rng()}, carpeta / 'ultimo_checkpoint.pt')
    guardar_json(carpeta / 'estado_entrenamiento.json', estado)


def entrenar_epoca(args, raiz, destino, plazo, trabajo, batch, tasas, epoca, reserva, estados):
    carpeta = destino / trabajo['nombre']
    carpeta.mkdir(exist_ok=True)
    familia = 'Transformer' if trabajo['configuracion']['arquitectura'].startswith(('vit', 'swin')) else 'CNN'
    modelo, optimizador, tasa, config = preparar_trabajo(trabajo, raiz, torch.device('cuda'), batch, tasas[familia]['tasa_base'])
    guardar_json(carpeta / 'configuracion.json', config)
    estado = {'epocas_completas': 0, 'cursor_pasos': 0, 'pasos_totales_fase2': 0, 'segundos_entrenamiento': 0.0, 'historia': [], 'vram_pico_GiB': 0.0}
    ruta_checkpoint = carpeta / 'ultimo_checkpoint.pt'
    if ruta_checkpoint.exists():
        anterior = torch.load(ruta_checkpoint, map_location='cpu', weights_only=False, mmap=True)
        if anterior['configuracion'] != config:
            raise ValueError('La configuración de reanudación cambió')
        modelo.load_state_dict(anterior['modelo'], strict=True)
        optimizador.load_state_dict(anterior['optimizador'])
        estado = anterior['estado_fase2']
        recuperar_rng(anterior)
        del anterior
    if estado['epocas_completas'] > epoca:
        estados[trabajo['nombre']] = {'epocas_completas': estado['epocas_completas'], 'pasos_fase2': estado['pasos_totales_fase2'], 'estado': 'epoca_completa', 'segundos_entrenamiento': estado['segundos_entrenamiento'], 'vram_pico_GiB': estado['vram_pico_GiB']}
        guardar_json(destino / 'estado_trabajos.json', estados)
        del modelo, optimizador
        liberar()
        return estado
    if estado['epocas_completas'] != epoca:
        raise ValueError('La ronda solicitada no coincide con el checkpoint')
    inferencia_batch = 2 if config['arquitectura'] in {'rf39_resnet50', 'rf29_resnet18'} else 8
    if not (carpeta / 'validacion_sin_entrenar.json').exists():
        estado_rng = rng()
        fijar_semilla(config['semilla'])
        temporal = construir_modelo(config)
        encoder_inicial = temporal.online.encoder.to('cuda')
        validacion_azar = evaluar_encoder(encoder_inicial, args.datos, 'validation', 224, torch.device('cuda'), batch=inferencia_batch, protocolo='market_documentado', limite=plazo)
        guardar_json(carpeta / 'validacion_sin_entrenar.json', {'epoca': 0, 'inicializacion': 'aleatoria con la semilla de configuración, antes de aprender', 'resultado': validacion_azar})
        del encoder_inicial, temporal
        liberar()
        recuperar_rng(estado_rng)
    if not (carpeta / 'validacion_inicio_fase2.json').exists():
        inicial = validar(modelo, args, config, plazo, inferencia_batch)
        guardar_json(carpeta / 'validacion_inicio_fase2.json', inicial)
        guardar_mejor(modelo, config, carpeta, inicial, 0)
    datos = Entrenamiento(args.datos, config)
    cargador, indices = cargador_epoca(datos, config['semilla'], epoca, batch, estado['cursor_pasos'])
    pasos_epoca = len(datos) // batch
    modelo.train()
    torch.cuda.reset_peak_memory_stats()
    inicio = time.perf_counter()
    tiempo_previo = estado['segundos_entrenamiento']
    cursor_inicial = estado['cursor_pasos']
    completo = True
    for contador, (vistas_cpu, visitados) in enumerate(cargador, cursor_inicial):
        if plazo.agotado(reserva):
            completo = False
            break
        vistas = [v.to('cuda', non_blocking=True) for v in vistas_cpu]
        progreso = estado['pasos_totales_fase2']
        progreso_total = max(1, 3 * pasos_epoca)
        factor_coseno = 0.5 * (1 + math.cos(math.pi * max(0, progreso - pasos_epoca // 10) / max(1, progreso_total - pasos_epoca // 10))) if progreso >= pasos_epoca // 10 else 1.0
        momentum = 1 - (1 - 0.996) * (math.cos(math.pi * progreso / progreso_total) + 1) / 2
        perdida, norma, factor = paso(modelo, optimizador, vistas, tasa * factor_coseno, progreso, max(1, pasos_epoca // 10), momentum)
        estado['pasos_totales_fase2'] += 1
        estado['cursor_pasos'] = contador + 1
        estado['segundos_entrenamiento'] = tiempo_previo + time.perf_counter() - inicio
        estado['vram_pico_GiB'] = max(estado['vram_pico_GiB'], torch.cuda.max_memory_allocated() / 1024 ** 3)
        if estado['pasos_totales_fase2'] % 100 == 0 or contador + 1 == pasos_epoca:
            estado['historia'].append({'epoca_adicional': epoca + 1, 'paso_en_epoca': contador + 1, 'pasos_totales_fase2': estado['pasos_totales_fase2'], 'perdida': perdida, 'norma_gradiente': norma, 'tasa': tasa * factor_coseno * factor, 'segundos_entrenamiento': estado['segundos_entrenamiento']})
            guardar_json(carpeta / 'historial.json', estado['historia'])
            registrar(destino, 'progreso', nombre=trabajo['nombre'], epoca=epoca + 1, paso=contador + 1, pasos_epoca=pasos_epoca, perdida=perdida)
        if estado['pasos_totales_fase2'] % 250 == 0:
            guardar_estado_entrenamiento(modelo, optimizador, carpeta, config, estado)
    if completo:
        estado['epocas_completas'] += 1
        estado['cursor_pasos'] = 0
        validacion = validar(modelo, args, config, plazo, inferencia_batch)
        guardar_json(carpeta / f'validacion_epoca_{epoca + 1}.json', validacion)
        guardar_mejor(modelo, config, carpeta, validacion, epoca + 1)
        estado['indices_ultima_epoca_sha256'] = hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()
        estado['imagenes_fuente_por_epoca'] = len(indices)
    guardar_estado_entrenamiento(modelo, optimizador, carpeta, config, estado)
    estados[trabajo['nombre']] = {'epocas_completas': estado['epocas_completas'], 'pasos_fase2': estado['pasos_totales_fase2'], 'estado': 'epoca_completa' if completo else 'parcial_por_plazo', 'segundos_entrenamiento': estado['segundos_entrenamiento'], 'vram_pico_GiB': estado['vram_pico_GiB']}
    guardar_json(destino / 'estado_trabajos.json', estados)
    registrar(destino, 'fin_epoca', nombre=trabajo['nombre'], **estados[trabajo['nombre']])
    del modelo, optimizador, datos, cargador
    liberar()
    return estado


def cargar_encoders(carpeta):
    contenido = torch.load(carpeta / 'mejores_encoders.pt', map_location='cpu', weights_only=False, mmap=True)
    modelo = EncodersRecuperacion(contenido['configuracion'])
    modelo.load_state_dict(contenido['modelo'], strict=True)
    modelo.to('cuda').eval()
    return modelo, contenido['configuracion']


def evaluar_final(args, raiz, destino, plazo, plan, estados):
    ruta_seleccion = destino / 'seleccion_global_validacion.json'
    candidatos = [{'nombre': t['nombre'], 'grupo': t['grupo'], **leer(destino / t['nombre'] / 'seleccion_validacion.json')} for t in plan['trabajos'] if (destino / t['nombre'] / 'seleccion_validacion.json').exists()]
    if not ruta_seleccion.exists():
        if not candidatos:
            raise ValueError('No existe ninguna selección válida')
        guardar_json(ruta_seleccion, {'modelo': max(candidatos, key=lambda c: c['criterio']), 'candidatos': candidatos, 'congelada_unix': time.time(), 'prueba_no_utilizada_para_seleccion': True, 'alcance': 'se conocen pruebas históricas de fase 1, sin ajuste con prueba de fase 2'})
    elegido = leer(ruta_seleccion)['modelo']['nombre']
    orden = sorted(plan['trabajos'], key=lambda t: (t['nombre'] != elegido, t['grupo'] != 'arquitecturas', t['configuracion']['arquitectura'] != 'rf39_resnet50'))
    for trabajo in orden:
        carpeta = destino / trabajo['nombre']
        ruta_prueba = carpeta / 'prueba.json'
        if not (carpeta / 'mejores_encoders.pt').exists():
            continue
        prueba = leer(ruta_prueba, {})
        seleccion = leer(carpeta / 'seleccion_validacion.json')
        if not prueba and seleccion['epoca_adicional_seleccionada'] == 0:
            historica = leer(raiz / 'resultados' / trabajo['nombre'] / 'prueba.json')
            prueba = {rama: copy.deepcopy(historica[rama]) for rama in ['online', 'objetivo']}
            prueba['reutilizacion_fase1'] = {'motivo': 'época adicional seleccionada cero, mismos parámetros y protocolo, métricas históricas válidas reutilizadas después de congelar selección', 'origen': f"resultados/{trabajo['nombre']}/prueba.json", 'sha256': huella(raiz / 'resultados' / trabajo['nombre'] / 'prueba.json')}
            prueba['tiempos_segundos'] = {'online': 0.0, 'objetivo': 0.0}
            if trabajo['nombre'] == elegido and not all('rankings' in prueba[seleccion['rama']][c] for c in ['original', 'baja_resolucion']):
                prueba.pop(seleccion['rama'])
            guardar_json(ruta_prueba, prueba)
        if all(r in prueba for r in ['online', 'objetivo']):
            if trabajo['nombre'] == elegido:
                generar_ejemplos(prueba[seleccion['rama']], args.datos, destino / 'ejemplos')
            continue
        if plazo.agotado(600):
            registrar(destino, 'evaluacion_pendiente_por_plazo', nombre=trabajo['nombre'])
            continue
        modelo, config = cargar_encoders(carpeta)
        batch = 2 if config['arquitectura'] in {'rf39_resnet50', 'rf29_resnet18'} else 8
        for rama in ['online', 'objetivo']:
            if rama in prueba:
                continue
            inicio = time.perf_counter()
            prueba[rama] = evaluar_encoder(getattr(modelo, rama).encoder, args.datos, 'testing', 224, torch.device('cuda'), batch=batch, protocolo='market_documentado', ejemplos=trabajo['nombre'] == elegido, limite=plazo)
            prueba.setdefault('tiempos_segundos', {})[rama] = time.perf_counter() - inicio
            guardar_json(ruta_prueba, prueba)
            registrar(destino, 'prueba', nombre=trabajo['nombre'], rama=rama, segundos=prueba['tiempos_segundos'][rama])
        if trabajo['nombre'] == elegido:
            rama = leer(carpeta / 'seleccion_validacion.json')['rama']
            generar_ejemplos(prueba[rama], args.datos, destino / 'ejemplos')
        del modelo
        liberar()


def consolidar(raiz, destino):
    plan = leer(destino / 'plan_fase2.json')
    if not plan:
        return
    estados = leer(destino / 'estado_trabajos.json', {})
    columnas = ['nombre', 'grupo', 'arquitectura', 'metodo', 'resolucion', 'peso_local', 'tasa_base', 'estado', 'batch', 'epocas_fase2_completas', 'epoca_fase2_seleccionada', 'rama', 'pasos_fase1', 'pasos_fase2', 'segundos_entrenamiento_fase1', 'segundos_entrenamiento_fase2', 'segundos_entrenamiento_total', 'segundos_validacion_fase2', 'segundos_prueba_fase2', 'parametros_encoder', 'VRAM_pico_GiB', 'mAP_validacion_sin_entrenar_porcentaje', 'mAP_validacion_seleccionada_porcentaje']
    columnas += [f'{condicion}_{metrica}_porcentaje' for condicion in ['original', 'baja_resolucion'] for metrica in METRICAS]
    columnas += ['caida_mAP_pp', 'caida_mAP_relativa_porcentaje']
    filas = []
    filas_ramas = []
    for trabajo in plan['trabajos']:
        carpeta = destino / trabajo['nombre']
        seleccion = leer(carpeta / 'seleccion_validacion.json', {})
        config = leer(carpeta / 'configuracion.json', trabajo['configuracion'])
        estado = estados.get(trabajo['nombre'], {})
        prueba = leer(carpeta / 'prueba.json', {})
        resumen_fase1 = leer(raiz / 'resultados' / trabajo['nombre'] / 'resumen.json')
        rama = seleccion.get('rama')
        fila = {'nombre': trabajo['nombre'], 'grupo': trabajo['grupo'], 'arquitectura': config['arquitectura'], 'estado': 'evaluado' if all(r in prueba for r in ['online', 'objetivo']) else 'en_curso_o_pendiente', 'batch': leer(destino / 'calibracion_fase2.json', {}).get('batch_comun'), 'epocas_fase2_completas': estado.get('epocas_completas'), 'epoca_fase2_seleccionada': seleccion.get('epoca_adicional_seleccionada'), 'rama': rama, 'pasos_fase1': 1253, 'pasos_fase2': estado.get('pasos_fase2'), 'segundos_entrenamiento_fase2': estado.get('segundos_entrenamiento'), 'VRAM_pico_GiB': estado.get('vram_pico_GiB'), 'mAP_validacion_sin_entrenar_porcentaje': 100 * leer(carpeta / 'validacion_sin_entrenar.json', {}).get('resultado', {}).get('criterio_validacion', 0) if (carpeta / 'validacion_sin_entrenar.json').exists() else None, 'mAP_validacion_seleccionada_porcentaje': 100 * seleccion['criterio'] if seleccion else None}
        fila.update(metodo=config['metodo'], resolucion=config['resolucion'], peso_local=config.get('peso_local', 0), tasa_base=config.get('tasa_base'), parametros_encoder=leer(raiz / 'resultados/parametros_arquitecturas.json')[config['arquitectura']], segundos_entrenamiento_fase1=resumen_fase1['segundos_entrenamiento'], segundos_entrenamiento_total=resumen_fase1['segundos_entrenamiento'] + (estado.get('segundos_entrenamiento') or 0), segundos_prueba_fase2=sum(prueba.get('tiempos_segundos', {}).values()) if prueba else None, segundos_validacion_fase2=sum(sum(leer(p).get('tiempos_segundos', {}).values()) for p in [carpeta / 'validacion_inicio_fase2.json', *carpeta.glob('validacion_epoca_*.json')] if p.exists()))
        if rama in prueba:
            for condicion in ['original', 'baja_resolucion']:
                for metrica in METRICAS:
                    fila[f'{condicion}_{metrica}_porcentaje'] = 100 * prueba[rama][condicion][metrica]
            fila['caida_mAP_pp'] = prueba[rama]['caida_mAP_puntos_porcentuales']
            fila['caida_mAP_relativa_porcentaje'] = prueba[rama]['caida_mAP_relativa_porcentaje']
        for rama_guardada in ['online', 'objetivo']:
            if rama_guardada in prueba:
                registro_rama = {'modelo': trabajo['nombre'], 'arquitectura': config['arquitectura'], 'rama': rama_guardada, 'epoca_adicional_seleccionada': seleccion['epoca_adicional_seleccionada']}
                for condicion in ['original', 'baja_resolucion']:
                    for metrica in METRICAS:
                        registro_rama[f'{condicion}_{metrica}_porcentaje'] = 100 * prueba[rama_guardada][condicion][metrica]
                filas_ramas.append(registro_rama)
        filas.append(fila)
    if filas_ramas:
        with (destino / 'comparacion_ramas_fase2.csv').open('w', encoding='utf-8-sig', newline='') as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=list(filas_ramas[0]))
            escritor.writeheader()
            escritor.writerows(filas_ramas)
    with (destino / 'comparacion_fase2.csv').open('w', encoding='utf-8-sig', newline='') as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)
    guardar_json(destino / 'comparacion_fase2.json', filas)
    texto = ['# Fase 2, comparación y alcance real', '', 'La fase 1 conserva sus 54 configuraciones y todas sus evidencias sin cambios, esta fase continúa un subconjunto predeclarado sin confundir los pesos históricos con nuevos resultados, estado por configuración en la tabla y en sesion_independiente.json.', '', '| Arquitectura | Grupo | Estado | Épocas adicionales | Época elegida | Rama | mAP O % | R1 O % | R5 O % | R10 O % | mAP LR % | R1 LR % | R5 LR % | R10 LR % | Min train |', '|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def valor(fila, clave):
        dato = fila.get(clave)
        return 'Pendiente' if dato is None else f'{dato:.3f}' if isinstance(dato, float) else str(dato)
    for fila in filas:
        claves = ['arquitectura', 'grupo', 'estado', 'epocas_fase2_completas', 'epoca_fase2_seleccionada', 'rama'] + [f'{c}_{m}_porcentaje' for c in ['original', 'baja_resolucion'] for m in METRICAS]
        texto.append('| ' + ' | '.join(valor(fila, clave) for clave in claves) + ' | ' + (f"{fila['segundos_entrenamiento_fase2'] / 60:.3f}" if fila.get('segundos_entrenamiento_fase2') is not None else 'Pendiente') + ' |')
    texto += ['', 'Todas las métricas de la tabla principal corresponden a la época y rama elegidas por validación, ambas ramas de esa misma época se guardan en prueba.json, no se mezclan rendimientos de fase 1 con fase 2, una época seleccionada cero significa reutilizar el checkpoint válido al inicio de la continuación, no una red sin entrenar.', '', 'El batch común se fija por pruebas de memoria, los pesos proceden de la fase 1 y la optimización y EMA de continuación se reinician de forma explícita, las tasas se eligen con pilotos de validación descartados, no se usa prueba para elegir configuración, una semilla y pocas épocas adicionales impiden afirmar convergencia o reproducción completa del paper.', '', 'El formato cuadrado 224 se mantiene para preservar la referencia común y la geometría local del paper, el coste por área respecto de 128x64 no demuestra por sí solo una mejora seis veces mayor al cambiar la entrada, la degradación LR en training es opcional en el laboratorio y no se incorpora como requisito inventado.']
    evaluadas = [fila for fila in filas if fila['estado'] == 'evaluado' and fila['grupo'] == 'arquitecturas']
    if evaluadas:
        mejor = max(evaluadas, key=lambda fila: (fila['original_mAP_porcentaje'] + fila['baja_resolucion_mAP_porcentaje']) / 2)
        eficiencia = max(evaluadas, key=lambda fila: (fila['original_mAP_porcentaje'] + fila['baja_resolucion_mAP_porcentaje']) / (fila['segundos_entrenamiento_total'] / 60))
        texto += ['', f"La mayor mAP media descriptiva de prueba entre las arquitecturas evaluadas corresponde a {mejor['arquitectura']}, el mayor cociente descriptivo de mAP media por minuto de entrenamiento de ambas fases corresponde a {eficiencia['arquitectura']}, esos resultados no seleccionan hiperparámetros ni equivalen a medir energía, la exposición y época elegida deben consultarse antes de atribuir causas, el coste compartido de pilotos se registra aparte."]
    texto += ['', '## Pendientes y límites', '', 'Los calendarios largos del paper, múltiples semillas, transferencia Places365, igualación literal de parámetros de BagNet33 y barridos exhaustivos permanecen fuera del alcance, los resultados locales son preliminares y no acreditan convergencia ni una reproducción completa del artículo.']
    (destino / 'RESULTADOS_FASE2.md').write_text('\n'.join(texto) + '\n', encoding='utf-8')
    protegidos = all(huella(raiz / ruta) == digest for ruta, digest in plan['historicos_SHA256'].items())
    cobertura = len(evaluadas)
    rondas = [f.get('epocas_fase2_completas') or 0 for f in filas]
    guardar_json(destino / 'verificacion_fase2.json', {'estado': 'completado' if cobertura == 14 and all(f['estado'] == 'evaluado' for f in filas) and protegidos and min(rondas) >= 1 and min(rondas) == max(rondas) else 'parcial_o_en_curso', 'arquitecturas_evaluadas': cobertura, 'configuraciones_evaluadas': sum(f['estado'] == 'evaluado' for f in filas), 'configuraciones_previstas': len(filas), 'rondas_comunes_completas': min(rondas), 'maximo_epocas_adicionales_observado': max(rondas), 'diferencias_exposicion_fase2': min(rondas) != max(rondas), 'registros_historicos_SHA256_conservados': protegidos, 'plazo': leer(destino / 'plazo_global.json'), 'prueba_no_usada_en_pilotos': True, 'limites_no_resueltos': ['una semilla', 'sin convergencia demostrada', 'video fase 1 requiere revisión humana', 'Places365 no disponible', 'BagNet33 no igualado en parámetros', 'parche externo no recibido']})
    dibujar_curvas(destino, plan)
    for nombre in ['README.md', 'INFORME.md']:
        ruta = raiz / nombre
        previo = ruta.read_text(encoding='utf-8')
        marca = '\n## Fase 2, continuación autorizada del 20 de octubre\n'
        base = previo.split(marca)[0]
        nota = '\nLa fase 1 y su video conservan sus resultados históricos, la fase 2 tiene estado y tabla actualizados en [resultados_fase2/RESULTADOS_FASE2.md](resultados_fase2/RESULTADOS_FASE2.md), los pesos de fase 2 y las épocas elegidas se documentan por separado, las cifras de la fase 1 no se atribuyen al código o entrenamiento de fase 2, la documentación audiovisual distingue la versión histórica de los resultados de continuación.\n'
        ruta.write_text(base + marca + nota, encoding='utf-8')
    ejemplos = leer(destino / 'ejemplos/indices_ejemplos.json', [])
    if ejemplos:
        lineas = ['# Casos de recuperación de fase 2', '', 'Selección reproducible por índice y resultado Rank-1, estas explicaciones se basan en las etiquetas y rangos guardados, las figuras permiten revisar oclusión, encuadre e iluminación sin atribuir causas visuales no verificadas automáticamente.', '']
        for ejemplo in ejemplos:
            correcto = ejemplo['tipo'] == 'aciertos'
            linea = f"Consulta {ejemplo['consulta']} de identidad {ejemplo['person_id']} y cámara {ejemplo['camera_id']}, condición {ejemplo['condicion']}, primer elemento recuperado identidad {ejemplo['person_ids_galeria'][0]} y cámara {ejemplo['camera_ids_galeria'][0]}, primer positivo en rango {ejemplo['rank']}, "
            linea += 'el acierto recupera la misma identidad desde otra cámara bajo las exclusiones del protocolo.' if correcto else 'el fallo sitúa otra identidad antes de cualquier positivo válido, una eventual similitud de ropa, pose o fondo debe comprobarse visualmente y no se afirma como causa demostrada.'
            lineas += [linea, '']
        lineas += ['![Aciertos originales](ejemplos/original_aciertos.png)', '', '![Fallos originales](ejemplos/original_fallos.png)', '', '![Aciertos LR](ejemplos/baja_resolucion_aciertos.png)', '', '![Fallos LR](ejemplos/baja_resolucion_fallos.png)', '']
        (destino / 'ANALISIS_ERRORES_FASE2.md').write_text('\n'.join(lineas), encoding='utf-8')


def dibujar_curvas(destino, plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figura, ejes = plt.subplots(1, 2, figsize=(13, 5))
    for trabajo in plan['trabajos']:
        carpeta = destino / trabajo['nombre']
        historia = leer(carpeta / 'historial.json', [])
        if historia:
            ejes[0].plot([h['pasos_totales_fase2'] for h in historia], [h['perdida'] for h in historia], alpha=0.6, label=trabajo['nombre'])
        valores = []
        for ruta in sorted(carpeta.glob('validacion_epoca_*.json')):
            valores.append((int(ruta.stem.rsplit('_', 1)[-1]), 100 * criterio(leer(ruta))[1]))
        if valores:
            ejes[1].plot([v[0] for v in valores], [v[1] for v in valores], marker='o', label=trabajo['nombre'])
    ejes[0].set(xlabel='Actualizaciones adicionales', ylabel='Pérdida, escalas distintas según método', title='Pérdida de fase 2')
    ejes[1].set(xlabel='Época adicional completa', ylabel='mAP media validación %', title='Validación por época, rama elegida')
    if ejes[1].lines:
        ejes[1].legend(fontsize=5, ncol=2)
    figura.tight_layout()
    figura.savefig(destino / 'curvas_fase2.png', dpi=140)
    plt.close(figura)


def ejecutar_fase2(args, raiz):
    raiz = Path(raiz)
    destino = raiz / 'resultados_fase2'
    destino.mkdir(exist_ok=True)
    if args.fase2_etapa == 'consolidar':
        consolidar(raiz, destino)
        return
    plazo = PlazoGlobal(destino / 'plazo_global.json')
    if plazo.agotado(600):
        raise TimeoutError('El plazo humano absoluto está agotado o sin reserva de cierre')
    if not torch.cuda.is_available():
        raise RuntimeError('La fase 2 requiere la GPU expresamente autorizada')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    plan = declarar(raiz, destino)
    captura = leer(destino / 'codigo_ejecucion_SHA256.json')
    if captura is not None and any(huella(raiz / ruta) != digest for ruta, digest in captura['archivos'].items()):
        raise ValueError('El código cambió después de congelar la ejecución, no se reanuda con una versión distinta sin revisión')
    with FileLock(str(destino / 'ejecucion.lock'), timeout=1):
        medicion = calibrar(args, raiz, destino, plazo, plan)
        if args.fase2_etapa == 'calibrar':
            return
        batch = medicion['batch_comun']
        comprobar_recorrido(args, raiz, destino, plazo, plan, batch)
        if args.fase2_etapa == 'comprobar':
            return
        tasas = barrer_tasas(args, raiz, destino, plazo, plan, batch)
        estados = leer(destino / 'estado_trabajos.json', {})
        trabajos = sorted(plan['trabajos'], key=lambda t: ({'rf39_resnet50': 0, 'rf29_resnet18': 1}.get(t['configuracion']['arquitectura'], 2), t['grupo'] != 'arquitecturas', t['nombre']))
        reserva = 5400
        for epoca in range(plan['maximo_epocas_adicionales']):
            if (destino / 'seleccion_global_validacion.json').exists():
                break
            tiempos = [e['segundos_entrenamiento'] / e['epocas_completas'] for e in estados.values() if e.get('epocas_completas', 0)]
            estimacion = sum(tiempos) * 1.35 if len(tiempos) == len(trabajos) else 91.58 * 60 * (8 / batch) * 1.5 + 1800
            if epoca and plazo.restante() < estimacion + reserva:
                registrar(destino, 'ronda_no_iniciada_por_reserva', epoca=epoca + 1, segundos_restantes=plazo.restante(), estimacion_ronda=estimacion)
                break
            for trabajo in trabajos:
                if estados.get(trabajo['nombre'], {}).get('epocas_completas', 0) > epoca:
                    continue
                if plazo.agotado(reserva):
                    registrar(destino, 'entrenamiento_cerrado_por_reserva', nombre=trabajo['nombre'])
                    break
                try:
                    entrenar_epoca(args, raiz, destino, plazo, trabajo, batch, tasas, epoca, reserva, estados)
                except Exception as error:
                    registrar(destino, 'fallo_trabajo', nombre=trabajo['nombre'], error=str(error), traza=traceback.format_exc())
                    liberar()
            consolidar(raiz, destino)
            if plazo.agotado(reserva):
                break
        evaluar_final(args, raiz, destino, plazo, plan, estados)
        consolidar(raiz, destino)
        registrar(destino, 'cola_finalizada', segundos_restantes=plazo.restante())
