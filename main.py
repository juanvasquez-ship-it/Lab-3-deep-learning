import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

from codigo.datos import preparar_datos
from codigo.entrenamiento import Presupuesto, calibrar, cargar_checkpoint, entrenar, fijar_semilla, guardar_json, liberar, validar_modelo
from codigo.evaluacion import evaluar_encoder, generar_ejemplos
from codigo.experimentos import configuraciones_unicas, exportar_matriz, matriz_completa, principales
from codigo.verificaciones import verificar


RAIZ = Path(__file__).resolve().parent


def ejecutar_argumentos():
    parser = argparse.ArgumentParser(description='Laboratorio 3, CO-SSL y recuperación de personas')
    parser.add_argument('accion', choices=['verificar', 'auditar', 'matriz', 'preparar-datos', 'ejecutar', 'ampliar', 'evaluar', 'analizar', 'analizar-panel', 'exportar', 'informe', 'tabla-arquitecturas', 'completar-arquitecturas', 'fase2'])
    parser.add_argument('--datos', type=Path, default=RAIZ.parents[1] / 'work' / 'data')
    parser.add_argument('--resultados', type=Path, default=RAIZ / 'resultados')
    parser.add_argument('--zip', type=Path)
    parser.add_argument('--horas', type=float, default=3)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--trabajadores', type=int, default=2)
    parser.add_argument('--modelos', default='')
    parser.add_argument('--grupo', choices=['principales', 'completo'], default='principales')
    parser.add_argument('--pasos', type=int)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--particion', choices=['validation', 'testing'], default='validation')
    parser.add_argument('--protocolo', choices=['market_documentado', 'sin_exclusiones'], default='market_documentado')
    parser.add_argument('--analisis', choices=['mascaras', 'pgd', 'representaciones', 'representaciones384', 'antes_mlp', 'capas', 'eficiencia'], default='representaciones')
    parser.add_argument('--dispositivo', default='cuda')
    parser.add_argument('--rama', choices=['seleccionada', 'online', 'objetivo'], default='seleccionada')
    parser.add_argument('--fase2-etapa', choices=['calibrar', 'comprobar', 'ejecutar', 'consolidar'], default='ejecutar')
    args = parser.parse_args()
    if args.horas <= 0 or args.horas > 3:
        parser.error('El presupuesto autorizado está limitado a tres horas')
    if args.batch < 2:
        parser.error('El batch físico debe ser al menos dos por BatchNorm')
    if args.pasos is not None and args.pasos < 1:
        parser.error('La cantidad de pasos debe ser positiva')
    if args.trabajadores < 0:
        parser.error('La cantidad de trabajadores no puede ser negativa')
    return args


def ejecutar_suite(args):
    if (args.resultados / 'seleccion_validacion.json').exists() and any(args.resultados.glob('*/prueba.json')):
        raise RuntimeError('La selección principal y prueba ya están registradas, se conserva esta entrega sin repetir entrenamiento ni evaluación, utilice auditar o informe para revisar los resultados')
    dispositivo = torch.device(args.dispositivo)
    if dispositivo.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('La ejecución limitada requiere PyTorch con CUDA, no se iniciará entrenamiento CPU')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    presupuesto = Presupuesto(args.resultados / 'presupuesto.json', args.horas * 3600)
    if presupuesto.agotado():
        raise TimeoutError('Este presupuesto ya finalizó, no se crea otro automáticamente')
    guardar_json(args.resultados / 'verificaciones.json', verificar(args.datos))
    exportar_matriz(args.resultados / 'matriz_experimentos.json')
    candidatas = principales() if args.grupo == 'principales' else matriz_completa()
    calendarios_publicados = [c for c in candidatas if 'epocas_paper' in c]
    if calendarios_publicados:
        guardar_json(args.resultados / 'calendarios_largos_pendientes.json', {'configuraciones': calendarios_publicados, 'motivo': 'No se sustituyen 300 o 1000 épocas por pasos locales para fingir la reproducción del calendario publicado, requieren recursos adicionales no autorizados'})
    configuraciones = configuraciones_unicas([c for c in candidatas if 'epocas_paper' not in c])
    if args.modelos:
        solicitados = args.modelos.split(',')
        configuraciones = [c for c in configuraciones if c['nombre'] in solicitados]
        if set(solicitados) != {c['nombre'] for c in configuraciones}:
            raise ValueError('Algún modelo solicitado no existe en la matriz')
    calibracion = args.resultados / 'calibracion.json'
    if calibracion.exists():
        perfiles = json.loads(calibracion.read_text(encoding='utf-8'))
        if not {p['nombre'] for p in perfiles}.issubset({c['nombre'] for c in configuraciones}) or any(p['batch_fisico'] != args.batch for p in perfiles):
            raise ValueError('La calibración existente no corresponde a estos modelos o a este batch')
        if len(perfiles) != len(configuraciones) or any(p.get('medicion_inferencia') != 'dos_pases_despues_de_calentamiento' for p in perfiles):
            perfiles = calibrar(configuraciones, args.datos, dispositivo, args.batch, args.resultados, presupuesto)
    else:
        perfiles = calibrar(configuraciones, args.datos, dispositivo, args.batch, args.resultados, presupuesto)
    tiempos = {p['nombre']: p for p in perfiles}
    inferencia_estimada = sum(2 * (19732 + 2 * 3368 + 1793 + 2 * 100) * p['segundos_inferencia_imagen'] for p in perfiles)
    reserva_evaluacion = inferencia_estimada * 2.0 + 1200
    suma_pasos = sum(p['segundos_paso_mediana'] for p in perfiles)
    tiempo_entrenamiento = max(0.0, presupuesto.restante() - reserva_evaluacion)
    pasos_comunes = args.pasos or int(tiempo_entrenamiento / (suma_pasos * 1.30))
    pasos_comunes = min(pasos_comunes, (11043 // args.batch) * 100)
    plan_anterior = args.resultados / 'plan_ejecucion.json'
    if plan_anterior.exists():
        anterior = json.loads(plan_anterior.read_text(encoding='utf-8'))
        if anterior['batch_fisico'] != args.batch or anterior['modelos'] != [c['nombre'] for c in configuraciones]:
            raise ValueError('La reanudación debe conservar la comparación planificada')
        pasos_comunes = anterior['pasos_comunes']
    if pasos_comunes < 1:
        raise RuntimeError('El presupuesto restante no permite una comparación con evaluación completa')
    plan = {'batch_fisico': args.batch, 'pasos_comunes': pasos_comunes, 'imagenes_por_modelo': pasos_comunes * args.batch, 'modelos': [c['nombre'] for c in configuraciones], 'reserva_evaluacion_segundos': reserva_evaluacion, 'suma_segundos_paso_calibrada': suma_pasos, 'presupuesto_restante_al_planificar': presupuesto.restante(), 'seleccion_pasos': 'misma cantidad de actualizaciones y originales por modelo, estimada con calibración, no se crea otra partición', 'modificaciones_datos_originales': False}
    guardar_json(args.resultados / 'plan_ejecucion.json', plan)
    print(json.dumps({'evento': 'plan', **plan}), flush=True)
    estados = []
    for configuracion in configuraciones:
        nombre = configuracion['nombre']
        if presupuesto.agotado(reserva_evaluacion):
            estados.append({'modelo': nombre, 'estado': 'pendiente_por_limite_temporal'})
            guardar_json(args.resultados / 'estado_ejecucion.json', estados)
            continue
        inicio = time.perf_counter()
        modelo = None
        try:
            carpeta = args.resultados / nombre
            resumen_existente = carpeta / 'resumen.json'
            validacion_existente = carpeta / 'validacion.json'
            if resumen_existente.exists() and (carpeta / 'checkpoint.pt').exists():
                resumen = json.loads(resumen_existente.read_text(encoding='utf-8'))
                configuracion_previa = json.loads((carpeta / 'configuracion.json').read_text(encoding='utf-8'))
                claves_metadatos = {'fuente', 'estado', 'alias'}
                if {k: v for k, v in configuracion.items() if k not in claves_metadatos} != {k: v for k, v in configuracion_previa.items() if k not in claves_metadatos}:
                    raise ValueError('La configuración cambió, no se reutiliza su checkpoint automáticamente')
                if resumen['pasos_realizados'] != pasos_comunes:
                    raise ValueError('Checkpoint parcial, no se reanuda como si fuera un entrenamiento completo')
                if validacion_existente.exists():
                    validacion = json.loads(validacion_existente.read_text(encoding='utf-8'))
                    modelo = None
                else:
                    modelo, configuracion_guardada, resumen = cargar_checkpoint(carpeta / 'checkpoint.pt', dispositivo)
                    validacion = validar_modelo(modelo, configuracion, args.datos, dispositivo, args.resultados, args.protocolo, presupuesto)
                print(json.dumps({'evento': 'reutilizacion_sin_reentrenar', 'modelo': nombre}), flush=True)
            else:
                modelo, resumen = entrenar(configuracion, args.datos, dispositivo, args.batch, pasos_comunes, args.trabajadores, args.resultados, presupuesto, reserva_evaluacion)
                validacion = validar_modelo(modelo, configuracion, args.datos, dispositivo, args.resultados, args.protocolo, presupuesto)
            rama = validacion['rama_seleccionada']
            estado = {'modelo': nombre, 'estado': resumen['estado'], 'validacion_mAP_media': validacion[rama]['criterio_validacion'], 'rama': rama, 'pasos': resumen['pasos_realizados'], 'segundos_entrenamiento_y_validacion': time.perf_counter() - inicio}
            del modelo
            liberar()
        except Exception as error:
            estado = {'modelo': nombre, 'estado': 'fallo', 'error': str(error), 'traza': traceback.format_exc()}
        finally:
            modelo = None
            liberar()
        estados.append(estado)
        guardar_json(args.resultados / 'estado_ejecucion.json', estados)
        print(json.dumps({'evento': 'fin_modelo', **estado}), flush=True)
    candidatos = [e for e in estados if 'validacion_mAP_media' in e and e['estado'] == 'entrenamiento_completado']
    if candidatos:
        seleccionado = max(candidatos, key=lambda e: e['validacion_mAP_media'])
        guardar_json(args.resultados / 'seleccion_validacion.json', {**seleccionado, 'configuracion_congelada_antes_de_prueba': True, 'criterio': 'media de mAP de validación original y reducida, desempate por orden de ejecución'})
        orden_prueba = [seleccionado] + [e for e in candidatos if e['modelo'] != seleccionado['modelo']]
        for estado in orden_prueba:
            if presupuesto.agotado(30):
                estado['evaluacion_prueba'] = 'pendiente_por_limite_temporal'
                continue
            nombre = estado['modelo']
            try:
                modelo, configuracion, resumen = cargar_checkpoint(args.resultados / nombre / 'checkpoint.pt', dispositivo)
                resultado = {}
                for rama in ['online', 'objetivo']:
                    resultado[rama] = evaluar_encoder(getattr(modelo, rama).encoder, args.datos, 'testing', configuracion['resolucion'], dispositivo, protocolo=args.protocolo, ejemplos=nombre == seleccionado['modelo'] and rama == seleccionado['rama'], limite=presupuesto)
                    guardar_json(args.resultados / nombre / 'prueba.json', resultado)
                estado['evaluacion_prueba'] = 'completada'
                if nombre == seleccionado['modelo']:
                    generar_ejemplos(resultado[seleccionado['rama']], args.datos, args.resultados / 'ejemplos')
                del modelo
                liberar()
            except Exception as error:
                estado['evaluacion_prueba'] = 'fallo_o_limite_temporal'
                estado['error_prueba'] = str(error)
            finally:
                modelo = None
                liberar()
            guardar_json(args.resultados / 'estado_ejecucion.json', estados)
            print(json.dumps({'evento': 'prueba', **estado}), flush=True)
    presupuesto.cerrar()
    from codigo.informe import generar_informe
    generar_informe(args.resultados, RAIZ)


def main():
    args = ejecutar_argumentos()
    args.resultados.mkdir(parents=True, exist_ok=True)
    if args.accion == 'fase2':
        from codigo.fase2 import ejecutar_fase2
        ejecutar_fase2(args, RAIZ)
    elif args.accion == 'preparar-datos':
        if args.zip is None:
            raise ValueError('Debe proporcionar --zip')
        print(preparar_datos(args.zip, args.datos))
    elif args.accion == 'verificar':
        resultado = verificar(args.datos)
        guardar_json(args.resultados / 'verificaciones.json', resultado)
        print(json.dumps(resultado, indent=2))
    elif args.accion == 'matriz':
        resultado = exportar_matriz(args.resultados / 'matriz_experimentos.json')
        print(json.dumps({'principales': len(resultado['principales']), 'configuraciones': len(resultado['matriz'])}))
    elif args.accion == 'ejecutar':
        ejecutar_suite(args)
    elif args.accion == 'informe':
        from codigo.informe import generar_informe
        generar_informe(args.resultados, RAIZ)
    elif args.accion in {'tabla-arquitecturas', 'completar-arquitecturas'}:
        from codigo.comparacion import ejecutar_cola, generar_tabla
        resultado = generar_tabla(args, RAIZ) if args.accion == 'tabla-arquitecturas' else ejecutar_cola(args, RAIZ)
        print(json.dumps(resultado, indent=2))
    elif args.accion == 'auditar':
        from codigo.auditoria import auditar_resultados
        resultado = auditar_resultados(args.resultados, RAIZ, args.datos)
        print(json.dumps(resultado, indent=2))
        if resultado['estado'] != 'aprobado':
            raise SystemExit(1)
    elif args.accion == 'exportar':
        from codigo.entrenamiento import exportar_encoders
        seleccion = json.loads((args.resultados / 'seleccion_validacion.json').read_text(encoding='utf-8')) if args.checkpoint is None else None
        origen = args.checkpoint or args.resultados / seleccion['modelo'] / 'checkpoint.pt'
        print(json.dumps(exportar_encoders(origen, RAIZ / 'encoders_seleccionados.pt'), indent=2))
    elif args.accion == 'ampliar':
        from codigo.ampliacion import ejecutar_ampliacion
        ejecutar_ampliacion(args, RAIZ)
    elif args.accion == 'analizar-panel':
        from codigo.analisis import ejecutar_panel
        ejecutar_panel(args, RAIZ)
    else:
        if args.checkpoint is None:
            raise ValueError('Debe proporcionar --checkpoint')
        dispositivo = torch.device(args.dispositivo)
        presupuesto = Presupuesto(args.resultados / 'presupuesto.json', args.horas * 3600)
        if presupuesto.agotado():
            raise TimeoutError('El presupuesto autorizado ya finalizó')
        modelo, configuracion, resumen = cargar_checkpoint(args.checkpoint, dispositivo)
        if args.accion == 'evaluar':
            resultado = {rama: evaluar_encoder(getattr(modelo, rama).encoder, args.datos, args.particion, configuracion['resolucion'], dispositivo, protocolo=args.protocolo, limite=presupuesto) for rama in ['online', 'objetivo']}
            guardar_json(args.resultados / configuracion['nombre'] / f'{args.particion}_adicional.json', resultado)
        else:
            from codigo.analisis import ejecutar_analisis
            ejecutar_analisis(args, modelo, configuracion, presupuesto)
        presupuesto.cerrar()


if __name__ == '__main__':
    main()
