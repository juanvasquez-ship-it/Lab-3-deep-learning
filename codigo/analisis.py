import json
import time
import traceback
from copy import copy
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from codigo.datos import Evaluacion, preparar_entrada
from codigo.entrenamiento import Presupuesto, cargar_checkpoint, fijar_semilla, guardar_json, liberar
from codigo.evaluacion import calcular_metricas, evaluar_encoder, extraer_caracteristicas, perturbacion_pgd
from codigo.experimentos import experimentos_lectura


def comprobar_tiempo(presupuesto):
    if presupuesto.agotado(30):
        raise TimeoutError('Límite temporal antes del siguiente análisis')


def analizar_mascaras(encoder, args, configuracion, presupuesto):
    resultados = {}
    for proporcion in experimentos_lectura()['mascaras']['proporciones']:
        comprobar_tiempo(presupuesto)
        fijar_semilla(configuracion['semilla'])
        resultados[str(proporcion)] = evaluar_encoder(encoder, args.datos, args.particion, configuracion['resolucion'], torch.device(args.dispositivo), protocolo=args.protocolo, mascara=proporcion, limite=presupuesto)
        guardar_json(args.resultados / configuracion['nombre'] / 'analisis_mascaras.json', resultados)
    return resultados


def analizar_lectura(encoder, args, configuracion, presupuesto):
    resultado = {}
    opciones = [('antes_mlp', {'antes_mlp': True})] if args.analisis == 'antes_mlp' else [(f'capa_{capa}', {'capa_lectura': capa}) for capa in [3, 4]]
    if args.analisis == 'capas' and encoder.transformer:
        return {'estado': 'no_aplicable', 'motivo': 'Las capas CNN de la tabla G.12 no equivalen a los bloques Transformer'}
    for nombre, parametros in opciones:
        comprobar_tiempo(presupuesto)
        resultado[nombre] = evaluar_encoder(encoder, args.datos, args.particion, configuracion['resolucion'], torch.device(args.dispositivo), protocolo=args.protocolo, limite=presupuesto, **parametros)
        guardar_json(args.resultados / configuracion['nombre'] / f'analisis_{args.analisis}.json', resultado)
    return resultado


def analizar_pgd(encoder, args, configuracion, presupuesto):
    dispositivo = torch.device(args.dispositivo)
    galeria = Evaluacion(args.datos, args.particion, 'gallery')
    consultas = Evaluacion(args.datos, args.particion, 'query')
    caracteristicas = extraer_caracteristicas(encoder, galeria, configuracion['resolucion'], dispositivo, limite=presupuesto)
    objetivos = []
    for persona, camara in zip(consultas.personas, consultas.camaras):
        positivos = (galeria.personas == persona) & (galeria.camaras != camara)
        if not positivos.any():
            raise ValueError('Consulta sin positivos para definir el ataque de recuperación')
        objetivos.append(caracteristicas[positivos].mean(0))
    objetivos = torch.tensor(np.stack(objetivos), device=dispositivo)
    parametros_antes = [p.requires_grad for p in encoder.parameters()]
    for parametro in encoder.parameters():
        parametro.requires_grad_(False)
    resultados = {}
    try:
        for indice, ataque in enumerate(experimentos_lectura()['pgd']['configuraciones']):
            fijar_semilla(configuracion['semilla'])
            representaciones = []
            maximo_cambio = 0.0
            for inicio in range(0, len(consultas), 8):
                comprobar_tiempo(presupuesto)
                imagenes = torch.stack([consultas[i][0] for i in range(inicio, min(inicio + 8, len(consultas)))]).to(dispositivo).float() / 255
                perturbadas = perturbacion_pgd(encoder, imagenes, objetivos[inicio:inicio + len(imagenes)], **ataque, resolucion=configuracion['resolucion'])
                maximo_cambio = max(maximo_cambio, float((perturbadas - imagenes).abs().max()))
                media = perturbadas.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
                desviacion = perturbadas.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
                entrada = F.interpolate(perturbadas, (configuracion['resolucion'],) * 2, mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
                with torch.inference_mode(), torch.autocast(dispositivo.type, dtype=torch.bfloat16, enabled=dispositivo.type == 'cuda'):
                    representaciones.append(F.normalize(encoder((entrada - media) / desviacion)[1].float(), dim=1).cpu().numpy())
            similitudes = np.concatenate(representaciones) @ caracteristicas.T
            if maximo_cambio > ataque['epsilon'] + 1e-6:
                raise ValueError('El ataque excedió la cota de perturbación')
            resultados[str(indice)] = {'ataque': ataque, 'inicio_aleatorio': True, 'semilla': configuracion['semilla'], 'maximo_cambio_Linf_verificado': maximo_cambio, 'metricas': calcular_metricas(similitudes, consultas.personas, consultas.camaras, galeria.personas, galeria.camaras, args.protocolo), 'objetivo': 'reducir similitud con centro de positivos de otras cámaras, etiquetas utilizadas solo en análisis de robustez'}
            guardar_json(args.resultados / configuracion['nombre'] / 'analisis_pgd.json', resultados)
    finally:
        for parametro, valor in zip(encoder.parameters(), parametros_antes):
            parametro.requires_grad_(valor)
    return resultados


def analizar_representaciones(encoder, args, configuracion, presupuesto):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dispositivo = torch.device(args.dispositivo)
    destino = args.resultados / configuracion['nombre'] / args.analisis
    destino.mkdir(parents=True, exist_ok=True)
    generador = np.random.default_rng(configuracion['semilla'])
    with h5py.File(Path(args.datos) / 'training.h5', 'r') as archivo:
        indices = np.sort(generador.choice(len(archivo['images']), size=16, replace=False))
        originales = archivo['images'][indices]
    entradas = preparar_entrada(torch.tensor(originales), configuracion['resolucion'], dispositivo)
    similitudes = []
    energias = []
    figura, ejes = plt.subplots(4, 4, figsize=(9, 11))
    for i, entrada in enumerate(entradas):
        comprobar_tiempo(presupuesto)
        imagen = entrada.unsqueeze(0).detach().requires_grad_(True)
        mapa = encoder.mapas(imagen)[-1]
        vectores = F.normalize(mapa.float().flatten(2), dim=1)
        afinidades = vectores.transpose(1, 2) @ vectores
        k = afinidades.shape[-1]
        if k > 1:
            similitudes.append(float(((afinidades.sum() - afinidades.diagonal(dim1=-2, dim2=-1).sum()) / (k * (k - 1))).detach()))
        central = mapa[0, :, mapa.shape[2] // 2, mapa.shape[3] // 2].abs().sum()
        gradiente = torch.autograd.grad(central, imagen)[0].abs().sum(1)[0]
        energia = gradiente.detach().cpu().numpy()
        energia = energia / max(1e-12, float(energia.max()))
        energias.append(energia)
        eje = ejes.flat[i]
        eje.imshow(originales[i].transpose(1, 2, 0), extent=(0, 1, 1, 0))
        eje.imshow(energia, cmap='magma', alpha=0.55, extent=(0, 1, 1, 0))
        eje.set_title(f'Índice {indices[i]}', fontsize=9)
        eje.axis('off')
    figura.suptitle('Gradiente de la activación espacial central sobre la entrada')
    figura.tight_layout()
    figura.savefig(destino / 'campos_efectivos.png', dpi=130)
    plt.close(figura)
    fracciones = {str(umbral): float((np.stack(energias) >= umbral).mean()) for umbral in [0.05, 0.32]}
    np.savez_compressed(destino / 'evidencia.npz', indices=indices, energias=np.stack(energias))
    resultado = {'particion': 'training sin identidades', 'resolucion_entrada': configuracion['resolucion'], 'indices': indices.tolist(), 'seleccion': f'16 índices sin reemplazo, generador NumPy con semilla {configuracion["semilla"]}', 'coseno_medio_posiciones_distintas': float(np.mean(similitudes)) if similitudes else None, 'fraccion_area_gradiente_promedio': fracciones, 'adaptacion_ERF': 'gradiente absoluto sumado en RGB y normalizado por máximo por imagen, activación central absoluta sumada en canales, se calcula fracción por imagen y después su media sobre 16 imágenes, no equivale al muestreo de ImageNet del artículo'}
    guardar_json(destino / 'resumen.json', resultado)
    return resultado


def analizar_eficiencia(encoder, args, configuracion, presupuesto):
    comprobar_tiempo(presupuesto)
    dispositivo = torch.device(args.dispositivo)
    datos = Evaluacion(args.datos, 'validation', 'query')
    entrada = preparar_entrada(torch.stack([datos[i][0] for i in range(64)]), configuracion['resolucion'], dispositivo)
    with torch.inference_mode():
        encoder(entrada)
        torch.cuda.synchronize()
        inicio = time.perf_counter()
        for _ in range(5):
            comprobar_tiempo(presupuesto)
            encoder(entrada)
        torch.cuda.synchronize()
    duracion = (time.perf_counter() - inicio) / 5
    comprobar_tiempo(presupuesto)
    operaciones = contar_operaciones(encoder, lambda: encoder(entrada[:1]))
    return {'batch_tiempo': 64, 'batch_operaciones': 1, 'segundos_por_forward': duracion, 'parametros_encoder': sum(p.numel() for p in encoder.parameters()), 'precision': 'float32', 'operaciones_Conv2d_Linear': operaciones, 'convencion': 'una multiplicación y una suma son dos operaciones, solo Conv2d y Linear, excluye normalización, activaciones, productos de atención y backward', 'limitacion': 'Tiempo medido en el equipo local, no se equipara a la tabla de hardware del artículo, el conteo de Transformer es parcial'}


def contar_operaciones(modulo, funcion):
    resultados = {}
    ganchos = []
    def registrar(nombre):
        def contar(capa, argumentos, salida):
            if isinstance(capa, torch.nn.Conv2d):
                multiplicaciones = salida.numel() * capa.kernel_size[0] * capa.kernel_size[1] * capa.in_channels // capa.groups
            else:
                multiplicaciones = salida.numel() * capa.in_features
            resultados[nombre] = resultados.get(nombre, 0) + 2 * multiplicaciones
        return contar
    for nombre, capa in modulo.named_modules():
        if isinstance(capa, (torch.nn.Conv2d, torch.nn.Linear)):
            ganchos.append(capa.register_forward_hook(registrar(nombre)))
    try:
        with torch.inference_mode():
            funcion()
    finally:
        for gancho in ganchos:
            gancho.remove()
    return {'total': sum(resultados.values()), 'por_capa': resultados}


def ejecutar_analisis(args, modelo, configuracion, presupuesto):
    configuracion = copy(configuracion)
    if args.analisis == 'representaciones384':
        if configuracion['arquitectura'].startswith(('vit', 'swin')):
            raise ValueError('Esta lectura a 384 se aplica a las CNN del apéndice C, no modifica posiciones de Transformers')
        configuracion['resolucion'] = 384
    seleccion = args.resultados / configuracion['nombre'] / 'validacion.json'
    rama = json.loads(seleccion.read_text(encoding='utf-8'))['rama_seleccionada'] if seleccion.exists() else 'objetivo'
    if args.rama != 'seleccionada':
        rama = args.rama
    encoder = getattr(modelo, rama).encoder.eval()
    funciones = {'mascaras': analizar_mascaras, 'pgd': analizar_pgd, 'antes_mlp': analizar_lectura, 'capas': analizar_lectura, 'representaciones': analizar_representaciones, 'representaciones384': analizar_representaciones, 'eficiencia': analizar_eficiencia}
    inicio = time.perf_counter()
    resultado = funciones[args.analisis](encoder, args, configuracion, presupuesto)
    particion = 'training sin identidades' if args.analisis in {'representaciones', 'representaciones384'} else ('validation' if args.analisis == 'eficiencia' else args.particion)
    evidencia = {'rama': rama, 'particion': particion, 'segundos': time.perf_counter() - inicio, 'resultado': resultado}
    guardar_json(args.resultados / configuracion['nombre'] / f'analisis_{args.analisis}.json', evidencia)
    return evidencia


def declarar_panel(destino):
    ruta = destino / 'plan_analisis.json'
    if ruta.exists():
        return json.loads(ruta.read_text(encoding='utf-8'))
    grupos = {
        'representaciones': ['byol_r50', 'co_byol_r50', 'byol_rf99', 'co_byol_rf99', 'byol_swin_t', 'co_byol_swin_t', 'byol_vit_s16', 'co_byol_vit_s16', 'byol_mc_r50', 'mocov3_r50', 'mocov3_mc_r50', 'co_mocov3_r50', 'mocov3_rf99', 'co_mocov3_rf99'],
        'mascaras': ['byol_r50', 'byol_mc_r50', 'co_byol_r50', 'byol_rf99', 'co_byol_rf99', 'byol_swin_t', 'co_byol_swin_t', 'mocov3_r50', 'mocov3_mc_r50', 'co_mocov3_r50'],
        'pgd': ['byol_r50', 'byol_mc_r50', 'co_byol_r50', 'co_byol_rf99', 'byol_swin_t', 'co_byol_swin_t', 'mocov3_r50', 'mocov3_mc_r50', 'co_mocov3_r50'],
        'antes_mlp': ['byol_rf99', 'co_byol_rf99', 'co_byol_rf99_cabeza_compartida'],
        'capas': ['byol_r50', 'co_byol_r50', 'co_byol_rf99'],
        'eficiencia': ['byol_r50', 'byol_mc_r50', 'co_byol_r50', 'co_byol_rf99'],
        'representaciones384': ['co_byol_r50'],
    }
    plan = {'declaracion_unix': time.time(), 'rama': 'online', 'particion_metricas': 'validation', 'particion_representaciones': 'training sin identidades', 'grupos': grupos, 'criterio': 'comparadores de las tablas y figuras del artículo, declarados antes de prueba, mismo encoder online para las comparaciones controladas', 'limite': 'presupuesto original de tres horas, se registran pendientes si no caben'}
    guardar_json(ruta, plan)
    return plan


def ejecutar_panel(args, raiz):
    plan = declarar_panel(args.resultados)
    ruta_estados = args.resultados / 'estado_analisis.json'
    anteriores = json.loads(ruta_estados.read_text(encoding='utf-8')) if ruta_estados.exists() else []
    estados = [e for e in anteriores if e.get('estado') == 'completado' and e['modelo'] in plan['grupos'].get(e['analisis'], []) and (args.resultados / e['modelo'] / f'analisis_{e["analisis"]}.json').exists()]
    completados = {(e['modelo'], e['analisis']) for e in estados}
    tareas = {}
    for analisis, nombres in plan['grupos'].items():
        for nombre in nombres:
            if (nombre, analisis) not in completados:
                tareas.setdefault(nombre, []).append(analisis)
    if not tareas:
        return estados
    presupuesto = Presupuesto(args.resultados / 'presupuesto.json', args.horas * 3600)
    if presupuesto.agotado(60):
        raise TimeoutError('El presupuesto finalizó, se conservan los análisis y el cierre originales sin sobrescribirlos')
    torch.set_num_threads(4)
    for nombre, analisis_modelo in tareas.items():
        ruta = args.resultados / nombre / 'checkpoint.pt'
        if presupuesto.agotado(60) or not ruta.exists():
            estados.extend({'modelo': nombre, 'analisis': analisis, 'estado': 'pendiente_presupuesto_o_checkpoint'} for analisis in analisis_modelo)
            continue
        modelo = None
        try:
            modelo, configuracion, resumen = cargar_checkpoint(ruta, torch.device(args.dispositivo))
            for analisis in analisis_modelo:
                if presupuesto.agotado(60):
                    estados.append({'modelo': nombre, 'analisis': analisis, 'estado': 'pendiente_presupuesto'})
                    continue
                parametros = copy(args)
                parametros.analisis = analisis
                parametros.rama = plan['rama']
                parametros.particion = plan['particion_metricas']
                try:
                    resultado = ejecutar_analisis(parametros, modelo, configuracion, presupuesto)
                    estado = {'modelo': nombre, 'analisis': analisis, 'estado': 'completado', 'rama': resultado['rama'], 'segundos': resultado['segundos']}
                except Exception as error:
                    estado = {'modelo': nombre, 'analisis': analisis, 'estado': 'fallo_o_presupuesto', 'error': str(error), 'traza': traceback.format_exc()}
                estados.append(estado)
                guardar_json(args.resultados / 'estado_analisis.json', estados)
                print(json.dumps({'evento': 'analisis', **estado}), flush=True)
        finally:
            modelo = None
            liberar()
    guardar_json(args.resultados / 'estado_analisis.json', estados)
    presupuesto.cerrar()
    from codigo.informe import generar_informe
    generar_informe(args.resultados, raiz)
    return estados
