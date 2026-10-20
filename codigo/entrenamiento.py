import gc
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from codigo.datos import Entrenamiento
from codigo.evaluacion import evaluar_encoder
from codigo.modelos import EncodersRecuperacion, construir_modelo


def guardar_json(ruta, contenido):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(ruta.suffix + '.tmp')
    temporal.write_text(json.dumps(contenido, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporal.replace(ruta)


def fijar_semilla(semilla):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def inicializar_trabajador(indice):
    semilla = torch.initial_seed() % (2 ** 32)
    np.random.seed(semilla)
    random.seed(semilla)
    torch.set_num_threads(1)


class Presupuesto:
    def __init__(self, ruta, segundos=10800):
        self.ruta = Path(ruta)
        if self.ruta.exists():
            self.datos = json.loads(self.ruta.read_text(encoding='utf-8'))
        else:
            inicio = time.time()
            self.datos = {'inicio_unix': inicio, 'fin_unix': inicio + min(segundos, 10800), 'maximo_segundos': min(segundos, 10800), 'incluye': 'calibracion, entrenamiento, validacion y evaluacion GPU, con pausas entre esas fases'}
            guardar_json(self.ruta, self.datos)

    def restante(self):
        return max(0.0, self.datos['fin_unix'] - time.time())

    def agotado(self, reserva=0):
        return self.restante() <= reserva

    def cerrar(self):
        self.datos['fin_real_unix'] = time.time()
        self.datos['transcurridos_segundos'] = self.datos['fin_real_unix'] - self.datos['inicio_unix']
        self.datos['restantes_segundos'] = self.restante()
        guardar_json(self.ruta, self.datos)


class LARS(torch.optim.Optimizer):
    def __init__(self, parametros, lr, momentum=0.9, weight_decay=1.5e-6):
        super().__init__(parametros, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for grupo in self.param_groups:
            for parametro in grupo['params']:
                if parametro.grad is None:
                    continue
                gradiente = parametro.grad
                if parametro.ndim > 1:
                    gradiente = gradiente.add(parametro, alpha=grupo['weight_decay'])
                    norma_peso = torch.linalg.vector_norm(parametro)
                    norma_gradiente = torch.linalg.vector_norm(gradiente)
                    razon = torch.where((norma_peso > 0) & (norma_gradiente > 0), 0.001 * norma_peso / norma_gradiente, torch.ones_like(norma_peso))
                    gradiente = gradiente * razon
                estado = self.state[parametro]
                if 'momentum' not in estado:
                    estado['momentum'] = torch.zeros_like(parametro)
                estado['momentum'].mul_(grupo['momentum']).add_(gradiente)
                parametro.add_(estado['momentum'], alpha=-grupo['lr'])


def construir_optimizador(modelo, configuracion, batch):
    parametros = [p for p in modelo.online.parameters() if p.requires_grad]
    if configuracion['arquitectura'].startswith(('vit', 'swin')) or configuracion['metodo'] == 'dino':
        tasa = configuracion.get('tasa_base', 0.0005) * batch / 256
        return torch.optim.AdamW(parametros, lr=tasa, weight_decay=0.04), tasa
    tasa = configuracion.get('tasa_base', 0.2) * batch / 256
    return LARS(parametros, tasa), tasa


def liberar():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def calibrar(configuraciones, carpeta, dispositivo, batch, destino, presupuesto):
    perfiles = []
    ruta_perfiles = Path(destino) / 'calibracion.json'
    anteriores = {p['nombre']: p for p in json.loads(ruta_perfiles.read_text(encoding='utf-8'))} if ruta_perfiles.exists() else {}
    for configuracion in configuraciones:
        if presupuesto.agotado(600):
            raise TimeoutError('Presupuesto insuficiente para calibrar el conjunto solicitado')
        anterior = anteriores.get(configuracion['nombre'])
        if anterior is not None and anterior.get('medicion_inferencia') == 'dos_pases_despues_de_calentamiento' and anterior['batch_fisico'] == batch:
            perfiles.append(anterior)
            continue
        fijar_semilla(configuracion['semilla'])
        datos = Entrenamiento(carpeta, configuracion)
        inicio_preparacion = time.perf_counter()
        cargador = DataLoader(datos, batch_size=batch, shuffle=False, num_workers=0, drop_last=True)
        modelo = construir_modelo(configuracion).to(dispositivo)
        modelo.train()
        optimizador, tasa = construir_optimizador(modelo, configuracion, batch)
        iterador = iter(cargador)
        duraciones = []
        torch.cuda.reset_peak_memory_stats()
        for indice in range(1 if anterior is not None else 4):
            inicio_paso = time.perf_counter()
            vistas, indices = next(iterador)
            vistas = [v.to(dispositivo) for v in vistas]
            optimizador.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                perdida, partes = modelo(vistas)
            if not torch.isfinite(perdida):
                raise ValueError(f'Pérdida no finita al calibrar {configuracion["nombre"]}')
            perdida.backward()
            optimizador.step()
            modelo.actualizar_objetivo(0.996)
            torch.cuda.synchronize()
            duraciones.append(time.perf_counter() - inicio_paso)
        modelo.eval()
        muestra = vistas[0].repeat(math.ceil(64 / batch), 1, 1, 1)[:64]
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            modelo.online.encoder(muestra)
        torch.cuda.synchronize()
        inicio_inferencia = time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for _ in range(2):
                modelo.online.encoder(muestra)
        torch.cuda.synchronize()
        por_imagen = (time.perf_counter() - inicio_inferencia) / (2 * len(muestra))
        perfil = {'nombre': configuracion['nombre'], 'batch_fisico': batch, 'pasos_calibracion_descartados': anterior['pasos_calibracion_descartados'] + 1 if anterior is not None else 4, 'segundos_paso_mediana': anterior['segundos_paso_mediana'] if anterior is not None else float(np.median(duraciones[1:])), 'segundos_inferencia_imagen': por_imagen, 'medicion_inferencia': 'dos_pases_despues_de_calentamiento', 'vram_pico_GiB': torch.cuda.max_memory_allocated() / 1024 ** 3, 'parametros_online': sum(p.numel() for p in modelo.online.parameters()), 'segundos_total_calibracion': (anterior.get('segundos_total_calibracion', 0) if anterior else 0) + time.perf_counter() - inicio_preparacion}
        perfiles.append(perfil)
        guardar_json(Path(destino) / 'calibracion.json', perfiles)
        print(json.dumps({'evento': 'calibracion', **perfil}), flush=True)
        del modelo, optimizador, cargador, iterador, vistas, datos, muestra, perdida, partes
        liberar()
    guardar_json(ruta_perfiles, perfiles)
    return perfiles


def entrenar(configuracion, carpeta, dispositivo, batch, pasos_objetivo, trabajadores, destino, presupuesto, reserva=0):
    destino = Path(destino) / configuracion['nombre']
    destino.mkdir(parents=True, exist_ok=True)
    guardar_json(destino / 'configuracion.json', configuracion)
    fijar_semilla(configuracion['semilla'])
    modelo = construir_modelo(configuracion).to(dispositivo)
    modelo.train()
    optimizador, tasa = construir_optimizador(modelo, configuracion, batch)
    datos = Entrenamiento(carpeta, configuracion)
    generador = torch.Generator().manual_seed(configuracion['semilla'])
    cargador = DataLoader(datos, batch_size=batch, shuffle=True, generator=generador, drop_last=True, num_workers=trabajadores, pin_memory=True, persistent_workers=trabajadores > 0, worker_init_fn=inicializar_trabajador)
    iterador = iter(cargador)
    historia = []
    vistos = set()
    inicio = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    pasos = 0
    epocas_completas = 0
    suma_perdida = 0.0
    for paso in range(pasos_objetivo):
        if presupuesto.agotado(reserva):
            break
        try:
            vistas, indices = next(iterador)
        except StopIteration:
            epocas_completas += 1
            iterador = iter(cargador)
            vistas, indices = next(iterador)
        vistos.update(indices.tolist())
        vistas = [v.to(dispositivo, non_blocking=True) for v in vistas]
        calentamiento = max(1, min(pasos_objetivo // 10, 10 * len(cargador)))
        factor = (paso + 1) / calentamiento if paso < calentamiento else 0.5 * (1 + math.cos(math.pi * (paso - calentamiento) / max(1, pasos_objetivo - calentamiento)))
        for grupo in optimizador.param_groups:
            grupo['lr'] = tasa * factor
            if configuracion['metodo'] == 'dino':
                grupo['weight_decay'] = 0.04 + (0.4 - 0.04) * (1 - math.cos(math.pi * paso / max(1, pasos_objetivo))) / 2
        if configuracion['metodo'] == 'dino':
            modelo.temperatura_objetivo = 0.04 + 0.03 * min(1, paso / (30 * len(cargador)))
        optimizador.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            perdida, componentes = modelo(vistas)
        if not torch.isfinite(perdida):
            raise ValueError(f'Pérdida no finita: {configuracion["nombre"]}, paso {paso}')
        perdida.backward()
        if configuracion['metodo'] == 'dino' and paso < len(cargador):
            modelo.online.cabeza.pesos.grad = None
        norma = torch.nn.utils.clip_grad_norm_(modelo.online.parameters(), 3.0 if configuracion['metodo'] == 'dino' else float('inf'))
        if not torch.isfinite(norma):
            raise ValueError('Gradiente no finito')
        optimizador.step()
        momentum = 1 - (1 - 0.996) * (math.cos(math.pi * paso / max(1, pasos_objetivo)) + 1) / 2
        modelo.actualizar_objetivo(momentum)
        pasos += 1
        suma_perdida += float(perdida.detach())
        if pasos == 1 or pasos % 25 == 0 or pasos == pasos_objetivo:
            registro = {'paso': pasos, 'perdida': float(perdida.detach()), 'global': float(componentes['global']), 'local': float(componentes['local']), 'tasa_aprendizaje': tasa * factor, 'momentum_objetivo': momentum, 'segundos': time.perf_counter() - inicio, 'norma_gradiente': float(norma)}
            historia.append(registro)
            guardar_json(destino / 'historial.json', historia)
            print(json.dumps({'evento': 'entrenamiento', 'modelo': configuracion['nombre'], **registro}), flush=True)
        frecuencia = configuracion.get('checkpoint_cada', 0)
        if frecuencia and pasos % frecuencia == 0:
            parcial = {'configuracion': configuracion, 'modelo': modelo.state_dict(), 'optimizador': optimizador.state_dict(), 'pasos_realizados': pasos, 'pasos_objetivo': pasos_objetivo, 'generador_datos': generador.get_state(), 'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state(), 'rng_numpy': np.random.get_state(), 'rng_python': random.getstate(), 'alcance': 'recuperación de pesos y optimizador, no garantiza reanudación numéricamente exacta de aumentos con trabajadores y prefetch'}
            temporal = destino / 'checkpoint_periodico.tmp'
            torch.save(parcial, temporal)
            temporal.replace(destino / 'checkpoint_periodico.pt')
            guardar_json(destino / 'progreso.json', {'pasos': pasos, 'pasos_objetivo': pasos_objetivo, 'checkpoint_periodico': True, 'actualizacion_unix': time.time()})
    if pasos and historia[-1]['paso'] != pasos:
        registro = {'paso': pasos, 'perdida': float(perdida.detach()), 'global': float(componentes['global']), 'local': float(componentes['local']), 'tasa_aprendizaje': tasa * factor, 'momentum_objetivo': momentum, 'segundos': time.perf_counter() - inicio, 'norma_gradiente': float(norma)}
        historia.append(registro)
        guardar_json(destino / 'historial.json', historia)
    torch.cuda.synchronize()
    resumen = {'estado': 'entrenamiento_completado' if pasos == pasos_objetivo else 'interrumpido_por_presupuesto', 'pasos_objetivo': pasos_objetivo, 'pasos_realizados': pasos, 'batch_fisico': batch, 'acumulacion_gradientes': 1, 'batch_efectivo': batch, 'imagenes_fuente_procesadas': pasos * batch, 'imagenes_fuente_distintas': len(vistos), 'epocas_completas': pasos // len(cargador), 'fraccion_epoca': pasos % len(cargador) / len(cargador), 'longitud_particion': len(datos), 'pasos_por_epoca': len(cargador), 'descartadas_por_drop_last_cada_epoca': len(datos) % batch, 'segundos_entrenamiento': time.perf_counter() - inicio, 'perdida_media': suma_perdida / pasos if pasos else None, 'vram_pico_GiB': torch.cuda.max_memory_allocated() / 1024 ** 3, 'parametros_online': sum(p.numel() for p in modelo.online.parameters()), 'initializacion': 'aleatoria, sin pesos externos', 'particion_entrenamiento': 'training.h5 completo como origen, sin etiquetas de identidad', 'indices_fuente_visitados_sha256': __import__('hashlib').sha256(np.array(sorted(vistos), dtype=np.int64).tobytes()).hexdigest()}
    torch.save({'configuracion': configuracion, 'modelo': modelo.state_dict(), 'optimizador': optimizador.state_dict(), 'resumen': resumen, 'rng_torch': torch.get_rng_state(), 'rng_numpy': np.random.get_state(), 'rng_python': random.getstate(), 'generador_datos': generador.get_state()}, destino / 'checkpoint.pt')
    guardar_json(destino / 'resumen.json', resumen)
    del cargador, iterador, optimizador
    if pasos:
        del vistas, perdida, componentes
    liberar()
    return modelo, resumen


def cargar_checkpoint(ruta, dispositivo):
    contenido = torch.load(ruta, map_location='cpu', weights_only=False)
    modelo = EncodersRecuperacion(contenido['configuracion']) if contenido.get('tipo') == 'encoders_recuperacion' else construir_modelo(contenido['configuracion'])
    modelo.load_state_dict(contenido['modelo'])
    modelo.to(dispositivo).eval()
    return modelo, contenido['configuracion'], contenido['resumen']


def exportar_encoders(origen, destino, ruta_evidencia=None):
    contenido = torch.load(origen, map_location='cpu', weights_only=False)
    pesos = {nombre: valor for nombre, valor in contenido['modelo'].items() if nombre.startswith(('online.encoder.', 'objetivo.encoder.'))}
    modelo = EncodersRecuperacion(contenido['configuracion'])
    modelo.load_state_dict(pesos, strict=True)
    if any(not torch.equal(valor, pesos[nombre]) for nombre, valor in modelo.state_dict().items()):
        raise ValueError('La exportación no conserva exactamente los parámetros y buffers')
    resultado = {'tipo': 'encoders_recuperacion', 'configuracion': contenido['configuracion'], 'modelo': pesos, 'resumen': contenido['resumen'], 'uso': 'evaluación y análisis de encoders, no contiene cabezas de entrenamiento ni estado del optimizador'}
    destino = Path(destino)
    temporal = destino.with_suffix('.tmp')
    torch.save(resultado, temporal)
    temporal.replace(destino)
    evidencia = {'origen': str(origen), 'destino': str(destino), 'tensores_verificados': len(pesos), 'igualdad_exacta_parametros_y_buffers': True, 'bytes_completo': Path(origen).stat().st_size, 'bytes_encoders': destino.stat().st_size, 'procedimiento': 'lectura CPU, carga estricta en los mismos Encoder y comparación torch.equal de todos los tensores, sin repetir inferencia ni entrenamiento'}
    guardar_json(ruta_evidencia or destino.parent / 'resultados' / 'exportacion_encoders.json', evidencia)
    return evidencia


def validar_modelo(modelo, configuracion, carpeta, dispositivo, destino, protocolo, presupuesto):
    resultados = {}
    for rama in ['online', 'objetivo']:
        resultados[rama] = evaluar_encoder(getattr(modelo, rama).encoder, carpeta, 'validation', configuracion['resolucion'], dispositivo, protocolo=protocolo, limite=presupuesto)
    resultados['rama_seleccionada'] = max(['online', 'objetivo'], key=lambda rama: resultados[rama]['criterio_validacion'])
    resultados['seleccion_exclusivamente_validacion'] = True
    guardar_json(Path(destino) / configuracion['nombre'] / 'validacion.json', resultados)
    return resultados
