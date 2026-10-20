import json
import time
import traceback

import torch

from codigo.entrenamiento import Presupuesto, cargar_checkpoint, entrenar, guardar_json, liberar, validar_modelo
from codigo.evaluacion import evaluar_encoder
from codigo.experimentos import configuraciones_unicas, exportar_matriz, matriz_completa, principales


def prioridad_ampliada():
    nombres = ['byol_bagnet33', 'byol_rf67_resnet50', 'byol_rf163_resnet50', 'byol_rf195_paper193_resnet50', 'byol_rf29_r18', 'mocov3_rf29_r18', 'byol_rf99_sin_mlp_co0.0', 'byol_rf99_sin_mlp_co0.2', 'co_byol_rf99_cabeza_compartida', 'co_byol_rf99_peso0.1', 'co_byol_rf99_peso0.5', 'co_byol_rf99_peso1.0']
    nombres += [f'co_byol_rf99_locales{cantidad}' for cantidad in [1, 4, 16, 64]]
    nombres += [f'co_byol_swin_t_kernel{kernel}' for kernel in [1, 2, 4, 5]]
    nombres += [f'co_byol_rf99_capa{capa}' for capa in [2, 3]]
    for recorte in [0.9, 0.08]:
        for metodo in ['byol', 'mocov3']:
            for arquitectura in ['resnet50', 'rf99_resnet50']:
                for peso in [0.0, 0.2]:
                    nombres.append(f'{metodo}_{arquitectura}_co{peso}_crop{recorte}')
    matriz = matriz_completa()
    indice = {c['nombre']: c for c in matriz}
    seleccionados = [indice[nombre] for nombre in nombres]
    conocidos = {c['nombre'] for c in principales()}
    restantes = [c for c in matriz if 'epocas_paper' not in c and c['arquitectura'] not in {'rf39_resnet50', 'rf115_resnet50'} and not (c.get('cantidad_locales') == 196 and c['arquitectura'] == 'rf99_resnet50' and c.get('capa_local', 4) == 4 and c['resolucion'] == 224)]
    return [c for c in configuraciones_unicas(principales() + seleccionados + restantes) if c['nombre'] not in conocidos]


def declarar_ampliacion(destino):
    ruta = destino / 'plan_ampliacion.json'
    if ruta.exists():
        return json.loads(ruta.read_text(encoding='utf-8'))
    plan = {'declaracion_unix': time.time(), 'configuraciones': prioridad_ampliada(), 'orden': 'controles con arquitectura fija, MLP y cabeza, pesos, cantidades locales, kernels, capas y finalmente barridos de recorte', 'seleccion': 'configuraciones declaradas antes de prueba, prioridad basada en cobertura y presupuesto, sin elegir por resultados de prueba', 'pasos': 'los mismos del panel principal', 'restriccion': 'solo las configuraciones que caben en el mismo presupuesto original, reserva de quince minutos para análisis, sin reiniciar las tres horas', 'seleccion_principal': 'se conserva la configuración seleccionada en el panel principal, estos experimentos no la sustituyen tras observar prueba'}
    guardar_json(ruta, plan)
    return plan


def ejecutar_ampliacion(args, raiz):
    dispositivo = torch.device(args.dispositivo)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    presupuesto = Presupuesto(args.resultados / 'presupuesto.json', args.horas * 3600)
    plan_analisis = json.loads((args.resultados / 'plan_analisis.json').read_text(encoding='utf-8')) if (args.resultados / 'plan_analisis.json').exists() else {}
    estado_analisis = json.loads((args.resultados / 'estado_analisis.json').read_text(encoding='utf-8')) if (args.resultados / 'estado_analisis.json').exists() else []
    analisis_previstos = sum(len(nombres) for nombres in plan_analisis.get('grupos', {}).values())
    analisis_completo = analisis_previstos > 0 and len(estado_analisis) == analisis_previstos and all(e['estado'] == 'completado' for e in estado_analisis)
    reserva = 120 if analisis_completo else 900
    if presupuesto.agotado(reserva):
        raise TimeoutError('No queda margen para ampliar el entrenamiento dentro del presupuesto original')
    plan = declarar_ampliacion(args.resultados)
    exportar_matriz(args.resultados / 'matriz_experimentos.json')
    plan_principal = json.loads((args.resultados / 'plan_ejecucion.json').read_text(encoding='utf-8'))
    if args.batch != plan_principal['batch_fisico']:
        raise ValueError('La ampliación debe conservar el batch del panel principal')
    pasos = plan_principal['pasos_comunes']
    estados = json.loads((args.resultados / 'estado_ejecucion.json').read_text(encoding='utf-8'))
    anteriores = {e['modelo'] for e in estados}
    perfiles = json.loads((args.resultados / 'calibracion.json').read_text(encoding='utf-8'))
    configuraciones = {c['nombre']: c for c in principales()}
    for configuracion in plan['configuraciones']:
        nombre = configuracion['nombre']
        if nombre in anteriores:
            continue
        comparables = [p for p in perfiles if configuraciones[p['nombre']]['arquitectura'] == configuracion['arquitectura']]
        segundo_paso = max([p['segundos_paso_mediana'] for p in comparables], default=0.3)
        estimado = pasos * segundo_paso * 1.3 + 180
        if presupuesto.agotado(reserva + estimado):
            break
        inicio = time.perf_counter()
        modelo = None
        try:
            modelo, resumen = entrenar(configuracion, args.datos, dispositivo, args.batch, pasos, args.trabajadores, args.resultados, presupuesto, reserva)
            validacion = validar_modelo(modelo, configuracion, args.datos, dispositivo, args.resultados, args.protocolo, presupuesto)
            rama = validacion['rama_seleccionada']
            estado = {'modelo': nombre, 'estado': resumen['estado'], 'grupo': 'ampliacion_predeclarada', 'rama': rama, 'validacion_mAP_media': validacion[rama]['criterio_validacion'], 'pasos': resumen['pasos_realizados']}
            prueba = {}
            for rama_prueba in ['online', 'objetivo']:
                prueba[rama_prueba] = evaluar_encoder(getattr(modelo, rama_prueba).encoder, args.datos, 'testing', configuracion['resolucion'], dispositivo, protocolo=args.protocolo, limite=presupuesto)
                guardar_json(args.resultados / nombre / 'prueba.json', prueba)
            estado.update(evaluacion_prueba='completada', segundos_totales=time.perf_counter() - inicio)
            del modelo
            liberar()
        except Exception as error:
            estado = {'modelo': nombre, 'estado': 'fallo', 'grupo': 'ampliacion_predeclarada', 'error': str(error), 'traza': traceback.format_exc()}
        finally:
            modelo = None
            liberar()
        estados.append(estado)
        guardar_json(args.resultados / 'estado_ejecucion.json', estados)
        print(json.dumps({'evento': 'ampliacion', **estado}), flush=True)
    realizados = {e['modelo'] for e in estados}
    guardar_json(args.resultados / 'pendientes_ampliacion.json', [c['nombre'] for c in plan['configuraciones'] if c['nombre'] not in realizados])
    presupuesto.cerrar()
    from codigo.informe import generar_informe
    generar_informe(args.resultados, raiz)
