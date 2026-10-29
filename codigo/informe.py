import csv
import json
import time
from pathlib import Path

import numpy as np


def leer_json(ruta, defecto=None):
    return json.loads(ruta.read_text(encoding='utf-8')) if ruta.exists() else defecto


def porcentaje(valor):
    return f'{100 * valor:.3f}' if valor is not None else 'Pendiente'


def notas_comparacion(resultados):
    seleccion = leer_json(resultados / 'seleccion_comparacion.json', {})
    if not seleccion:
        return []
    estado = leer_json(resultados / 'estado_comparacion.json', {})
    inicial = leer_json(resultados / 'presupuesto.json', {})
    adicional = leer_json(resultados / 'presupuesto_comparacion.json', {})
    return [f'La comparación ampliada cuenta con {estado.get("arquitecturas_referencia_evaluadas", 0)} arquitecturas evaluadas bajo CO-BYOL online a resolución 224, la selección de ese panel mediante validación es {seleccion["modelo"]} con mAP medio de validación {100 * seleccion["mAP_media_validacion"]:.3f} por ciento, COMPARACION_ARQUITECTURAS.md contiene la tabla común y mantiene separada la selección histórica del panel inicial.', '', f'La sesión inicial registró {inicial.get("transcurridos_segundos", 0) / 60:.2f} minutos y la continuación autorizada registró {adicional.get("transcurridos_segundos", 0) / 60:.2f} minutos, cada autorización tiene su presupuesto independiente y no se presenta la suma como una única sesión de tres horas.', '']


def graficar(resultados, filas):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    disponibles = [f for f in filas if f['particion'] == 'testing']
    seleccionados = [f for f in disponibles if f['rama'] == 'online']
    if seleccionados:
        figura, ejes = plt.subplots(figsize=(12, max(6, len(seleccionados) * 0.35)))
        posiciones = np.arange(len(seleccionados))
        ejes.barh(posiciones - 0.18, [100 * f['original_mAP'] for f in seleccionados], height=0.35, label='Original')
        ejes.barh(posiciones + 0.18, [100 * f['baja_resolucion_mAP'] for f in seleccionados], height=0.35, label='Consulta reducida')
        ejes.set_yticks(posiciones, [f['modelo'] for f in seleccionados], fontsize=8)
        ejes.set_xlabel('mAP, porcentaje')
        ejes.legend()
        ejes.invert_yaxis()
        figura.tight_layout()
        figura.savefig(resultados / 'comparacion_map.png', dpi=150)
        plt.close(figura)
    ejemplos = ['byol_r50', 'co_byol_r50', 'byol_rf99', 'co_byol_rf99', 'mocov3_r50', 'co_mocov3_rf99']
    figura, ejes = plt.subplots(figsize=(10, 5))
    cantidad = 0
    for nombre in ejemplos:
        historia = leer_json(resultados / nombre / 'historial.json', [])
        if historia:
            ejes.plot([h['paso'] for h in historia], [h['perdida'] for h in historia], label=nombre)
            cantidad += 1
    if cantidad:
        ejes.set_xlabel('Actualizaciones')
        ejes.set_ylabel('Pérdida registrada')
        ejes.set_title('Las escalas de pérdidas distintas no miden calidad de recuperación')
        ejes.legend(fontsize=8)
        figura.tight_layout()
        figura.savefig(resultados / 'curvas_entrenamiento.png', dpi=150)
    plt.close(figura)
    if disponibles:
        parejas = {}
        for fila in disponibles:
            parejas.setdefault(fila['modelo'], {})[fila['rama']] = fila['original_mAP']
        parejas = {k: v for k, v in parejas.items() if len(v) == 2}
        if parejas:
            figura, ejes = plt.subplots(figsize=(7, 6))
            ejes.scatter([100 * v['online'] for v in parejas.values()], [100 * v['objetivo'] for v in parejas.values()])
            maximo = max(100 * max(v.values()) for v in parejas.values()) * 1.05
            ejes.plot([0, maximo], [0, maximo], color='gray', linestyle='--')
            ejes.set_xlabel('Encoder online, mAP original %')
            ejes.set_ylabel('Encoder objetivo EMA, mAP original %')
            figura.tight_layout()
            figura.savefig(resultados / 'comparacion_maestro_estudiante.png', dpi=150)
            plt.close(figura)


def resumir_analisis(resultados):
    estados = leer_json(resultados / 'estado_analisis.json', [])
    if not estados:
        return []
    texto = ['## Análisis adicionales sin reentrenamiento', '', 'Las comparaciones de este panel utilizan la rama online y métricas de validación, los gradientes y las similitudes locales se calculan sobre dieciséis imágenes de training sin identidades, los resultados no se utilizan para modificar configuraciones después de prueba.', '', '| Modelo | Análisis | Estado | Segundos |', '|---|---|---|---:|']
    for estado in estados:
        texto.append(f'| {estado["modelo"]} | {estado["analisis"]} | {estado["estado"]} | {estado.get("segundos", 0):.2f} |')
    mascaras = {}
    ataques = {}
    representaciones = {}
    eficiencia = {}
    lecturas = {}
    for estado in estados:
        if estado['estado'] != 'completado':
            continue
        nombre, analisis = estado['modelo'], estado['analisis']
        contenido = leer_json(resultados / nombre / f'analisis_{analisis}.json', {})
        resultado = contenido.get('resultado', {})
        if analisis == 'mascaras':
            mascaras[nombre] = resultado
        elif analisis == 'pgd':
            ataques[nombre] = resultado
        elif analisis.startswith('representaciones'):
            representaciones[nombre if analisis == 'representaciones' else f'{nombre} a 384'] = resultado
        elif analisis == 'eficiencia':
            eficiencia[nombre] = resultado
        else:
            lecturas[(nombre, analisis)] = resultado
    if mascaras:
        texto += ['', '### Máscaras internas', '', 'Se informa mAP original de validación en porcentaje, las proporciones descartan posiciones antes del promedio y las máscaras de consultas correspondientes se acoplan entre resoluciones, los JSON conservan también las métricas con consultas reducidas.', '', '| Modelo | 50 % | 60 % | 70 % | 80 % | 90 % |', '|---|---:|---:|---:|---:|---:|']
        for nombre, datos in mascaras.items():
            texto.append('| ' + ' | '.join([nombre, *[porcentaje(datos.get(str(p), {}).get('original', {}).get('mAP')) for p in [0.5, 0.6, 0.7, 0.8, 0.9]]]) + ' |')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figura, eje = plt.subplots(figsize=(11, 6))
        for nombre, datos in mascaras.items():
            base = leer_json(resultados / nombre / 'validacion.json', {}).get('online', {}).get('original', {}).get('mAP')
            if base:
                eje.plot([50, 60, 70, 80, 90], [100 * datos[str(p)]['original']['mAP'] / base for p in [0.5, 0.6, 0.7, 0.8, 0.9]], marker='o', label=nombre)
        eje.set_xlabel('Posiciones descartadas, porcentaje')
        eje.set_ylabel('mAP retenido respecto a validación sin máscara, porcentaje')
        eje.set_ylim(bottom=0)
        eje.legend(fontsize=7, ncol=2)
        figura.tight_layout()
        figura.savefig(resultados / 'robustez_mascaras.png', dpi=150)
        plt.close(figura)
        texto += ['', '![Robustez a máscaras](resultados/robustez_mascaras.png)', '', 'Una retención superior al cien por ciento significa que la máscara obtuvo un mAP mayor que la evaluación sin máscara de ese mismo modelo, las cifras se mantienen con su valor observado y corresponden a una realización reproducible de máscaras sobre cien consultas de validación.', '']
        if all(n in mascaras for n in ['byol_rf99', 'co_byol_rf99']):
            bases = [leer_json(resultados / n / 'validacion.json', {})['online']['original']['mAP'] for n in ['byol_rf99', 'co_byol_rf99']]
            valores = [mascaras[n]['0.5']['original']['mAP'] for n in ['byol_rf99', 'co_byol_rf99']]
            if all(bases):
                texto += [f'Con el cincuenta por ciento de posiciones descartadas, BYOL RF99 alcanza mAP {porcentaje(valores[0])} por ciento y CO-BYOL RF99 {porcentaje(valores[1])} por ciento, las retenciones respecto a sus propios valores sin máscara son {100 * valores[0] / bases[0]:.2f} y {100 * valores[1] / bases[1]:.2f} por ciento respectivamente.', '', 'CO conserva mayor mAP absoluto pero pierde una fracción mayor de su rendimiento inicial en esta comparación, las dos medidas responden a preguntas diferentes y no sustentan una mejora uniforme de robustez.', '']
    if ataques:
        texto += ['### Perturbaciones PGD', '', 'Las cinco columnas siguen el orden de la tabla cinco del artículo y utilizan un objetivo adaptado a recuperación, el ataque conoce positivos de galería para reducir su similitud con la consulta y no entrena parámetros, las cotas operan en escala de píxel de cero a uno con inicio aleatorio.', '', '| Modelo | ε 0.01, 1 paso | ε 0.03, 1 paso | ε 0.1, 1 paso | ε 0.003, 5 pasos | ε 0.003, 1 paso de ε/10 |', '|---|---:|---:|---:|---:|---:|']
        for nombre, datos in ataques.items():
            texto.append('| ' + ' | '.join([nombre, *[porcentaje(datos.get(str(i), {}).get('metricas', {}).get('mAP')) for i in range(5)]]) + ' |')
        texto += ['', 'Los tres primeros ataques y el cuarto utilizan paso de epsilon dividido entre cuarenta, la última columna utiliza epsilon dividido entre diez, una cota pequeña o pocas iteraciones no certifican robustez y estos resultados no equivalen a un ataque al clasificador del artículo.', '']
    if representaciones:
        texto += ['### Similitud local y campos efectivos', '', '| Modelo | Coseno local medio | Área gradiente ≥ 5 %, porcentaje | Área gradiente ≥ 32 %, porcentaje |', '|---|---:|---:|---:|']
        for nombre, datos in representaciones.items():
            areas = datos['fraccion_area_gradiente_promedio']
            texto.append(f'| {nombre} | {datos["coseno_medio_posiciones_distintas"]:.4f} | {porcentaje(areas["0.05"])} | {porcentaje(areas["0.32"])} |')
        texto += ['', 'Las fracciones se calculan por imagen y después se promedian, los mapas y las matrices numéricas se conservan en la carpeta representaciones de cada modelo, una muestra de dieciséis imágenes no reproduce la estimación de ImageNet de los autores.', '']
        if all(n in representaciones for n in ['byol_rf99', 'co_byol_rf99']):
            a, b = [representaciones[n] for n in ['byol_rf99', 'co_byol_rf99']]
            texto += [f'En la muestra fija de dieciséis imágenes, BYOL RF99 y CO-BYOL RF99 presentan áreas medias por encima del cinco por ciento del máximo de gradiente de {porcentaje(a["fraccion_area_gradiente_promedio"]["0.05"])} y {porcentaje(b["fraccion_area_gradiente_promedio"]["0.05"])} por ciento, sus cosenos locales medios son {a["coseno_medio_posiciones_distintas"]:.4f} y {b["coseno_medio_posiciones_distintas"]:.4f}, estas cifras próximas no reproducen por sí solas una separación fuerte entre métodos y describen sensibilidad de una activación central bajo el entrenamiento local.', '']
        nombres = ['byol_r50', 'co_byol_r50', 'byol_rf99', 'co_byol_rf99']
        rutas = [resultados / n / 'representaciones' / 'evidencia.npz' for n in nombres]
        if all(r.exists() for r in rutas):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            figura, ejes = plt.subplots(2, 4, figsize=(12, 6), layout='constrained')
            for columna, (nombre, ruta) in enumerate(zip(nombres, rutas)):
                with np.load(ruta) as evidencia:
                    for fila in range(2):
                        eje = ejes[fila, columna]
                        mapa = eje.imshow(evidencia['energias'][fila], cmap='magma', vmin=0, vmax=1)
                        eje.set_title(f'{nombre}, índice {evidencia["indices"][fila]}', fontsize=9)
                        eje.axis('off')
            figura.colorbar(mapa, ax=ejes, label='Gradiente normalizado por el máximo de cada imagen', shrink=0.8)
            figura.savefig(resultados / 'comparacion_campos_efectivos.png', dpi=150)
            plt.close(figura)
            texto += ['![Comparación de gradientes centrales](resultados/comparacion_campos_efectivos.png)', '', 'La figura utiliza los dos primeros índices de la muestra fija y las matrices ya calculadas sin nueva inferencia, cada mapa se normaliza por su máximo individual y la escala representa sensibilidad relativa, la diferencia de extensión entre ResNet50 y RF99 también incorpora la arquitectura y se interpreta por separado del efecto de CO.', '']
    if eficiencia:
        texto += ['### Coste de inferencia medido', '', '| Modelo | Parámetros encoder, millones | Milisegundos con batch 64 | Operaciones Conv2d y Linear con batch 1, miles de millones |', '|---|---:|---:|---:|']
        for nombre, datos in eficiencia.items():
            texto.append(f'| {nombre} | {datos["parametros_encoder"] / 1e6:.3f} | {1000 * datos["segundos_por_forward"]:.3f} | {datos["operaciones_Conv2d_Linear"]["total"] / 1e9:.3f} |')
        texto += ['', 'El conteo utiliza dos operaciones por multiplicación acumulada y excluye normalización, activaciones, productos funcionales de atención y backward, las unidades y el hardware se declaran para no equiparar estas cifras con otra convención de FLOPs o con los tiempos distribuidos del artículo.', '']
    if lecturas:
        texto += ['### Lecturas alternativas del mismo checkpoint', '', '| Modelo | Intervención de lectura | mAP original % | mAP reducida % |', '|---|---|---:|---:|']
        for (nombre, analisis), datos in lecturas.items():
            for variante, resultado in datos.items():
                if isinstance(resultado, dict) and 'original' in resultado:
                    texto.append(f'| {nombre} | {variante} | {porcentaje(resultado["original"]["mAP"])} | {porcentaje(resultado["baja_resolucion"]["mAP"])} |')
        texto += ['', 'Leer antes del MLP conserva los pesos aprendidos y no equivale a entrenar una arquitectura sin MLP, la lectura de etapas intermedias utiliza promedio espacial y es una adaptación a recuperación que no reproduce las sondas de clasificación de la tabla G.12.', '']
    return texto


def resumir_variantes(resultados, indice):
    controles = [('MLP retirado en BYOL', 'byol_rf99_sin_mlp_co0.0', 'byol_rf99'), ('MLP retirado con CO', 'byol_rf99_sin_mlp_co0.2', 'co_byol_rf99'), ('Cabeza compartida', 'co_byol_rf99_cabeza_compartida', 'co_byol_rf99')]
    controles += [('Peso local', f'co_byol_rf99_peso{p}', 'co_byol_rf99') for p in [0.1, 0.5, 1.0]]
    controles += [('Cantidad local', f'co_byol_rf99_locales{k}', 'co_byol_rf99') for k in [1, 4, 16, 64]]
    controles += [('Kernel Swin', f'co_byol_swin_t_kernel{k}', 'co_byol_swin_t') for k in [1, 2, 4, 5]]
    controles += [('Capa de alineamiento con K64', f'co_byol_rf99_capa{k}', 'co_byol_rf99_locales64') for k in [2, 3]]
    filas = []
    for componente, variante, control in controles:
        for particion in ['validation', 'testing']:
            a, b = indice.get((control, 'online', particion)), indice.get((variante, 'online', particion))
            if a is not None and b is not None:
                filas.append(f'| {componente} | {variante} | {control} | {particion} | {porcentaje(b["original_mAP"])} | {porcentaje(b["baja_resolucion_mAP"])} | {100 * (b["original_mAP"] - a["original_mAP"]):.3f} | {100 * (b["baja_resolucion_mAP"] - a["baja_resolucion_mAP"]):.3f} |')
    if not filas:
        return []
    texto = ['## Ablaciones de componentes y sensibilidad', '', 'Se utiliza la rama online en todos los comparadores de esta sección, las diferencias son variante menos control y los mAP se muestran en porcentaje, cada comparación conserva las particiones y el calendario local y no modifica la selección principal.', '', '| Componente | Variante | Control | Partición | mAP original % | mAP reducida % | Diferencia original pp | Diferencia reducida pp |', '|---|---|---|---|---:|---:|---:|---:|', *filas, '', 'Retirar el MLP durante entrenamiento cambia el modelo aprendido y se distingue de leer antes del MLP un checkpoint existente, compartir la cabeza BYOL reutiliza también la profundidad y BatchNorm del proyector global, las capas de alineamiento se contrastan con capa cuatro y K64 únicamente cuando existe ese control entrenado.', '']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grupos = [('Peso del término local', [(p, 'co_byol_rf99' if p == 0.2 else f'co_byol_rf99_peso{p}') for p in [0.1, 0.2, 0.5, 1.0]]), ('Posiciones locales RF99', [(k, 'co_byol_rf99' if k == 196 else f'co_byol_rf99_locales{k}') for k in [1, 4, 16, 64, 196]]), ('Kernel de la cabeza Swin-T', [(k, 'co_byol_swin_t' if k == 3 else f'co_byol_swin_t_kernel{k}') for k in [1, 2, 3, 4, 5]]), ('Capa de alineamiento RF99, K64', [(k, 'co_byol_rf99_locales64' if k == 4 else f'co_byol_rf99_capa{k}') for k in [2, 3, 4]])]
    figura, ejes = plt.subplots(2, 2, figsize=(12, 9))
    for eje, (titulo, valores) in zip(ejes.flat, grupos):
        puntos = 0
        for particion, estilo, marcador in [('validation', '--', 's'), ('testing', '-', 'o')]:
            for condicion, color, etiqueta in [('original', '#245b86', 'original'), ('baja_resolucion', '#159782', 'reducida')]:
                datos = [(x, 100 * indice[(nombre, 'online', particion)][f'{condicion}_mAP']) for x, nombre in valores if (nombre, 'online', particion) in indice]
                if datos:
                    eje.plot([d[0] for d in datos], [d[1] for d in datos], linestyle=estilo, marker=marcador, color=color, label=f'{particion}, {etiqueta}')
                    puntos = max(puntos, len(datos))
        eje.set_title(f'{titulo}, {puntos}/{len(valores)} valores', fontsize=10)
        eje.set_ylabel('mAP, porcentaje')
        if 'Posiciones' in titulo:
            eje.set_xscale('log')
        eje.set_xticks([x for x, _ in valores], labels=[str(x) for x, _ in valores])
        if puntos:
            eje.legend(fontsize=7)
        if puntos < 2:
            eje.text(0.5, 0.5, 'Barrido incompleto, menos de dos valores ejecutados', transform=eje.transAxes, ha='center', fontsize=8)
        eje.grid(alpha=0.15)
    figura.tight_layout()
    figura.savefig(resultados / 'sensibilidad_componentes.png', dpi=150)
    plt.close(figura)
    texto += ['![Sensibilidad de componentes](resultados/sensibilidad_componentes.png)', '', 'Los puntos del gráfico corresponden a configuraciones con métricas disponibles y conservan los ejes de los valores planificados, un eje con menos de dos valores ejecutados se identifica como barrido incompleto, los cambios de mAP bajo este presupuesto no establecen una tendencia general ni un óptimo de hiperparámetros.', '']
    return texto


def generar_informe(resultados, raiz):
    resultados = Path(resultados)
    raiz = Path(raiz)
    filas = []
    entrenamientos = []
    for carpeta in sorted(resultados.iterdir()):
        if not carpeta.is_dir():
            continue
        resumen = leer_json(carpeta / 'resumen.json')
        if resumen:
            entrenamientos.append((carpeta.name, resumen))
        for particion, archivo in [('validation', 'validacion.json'), ('testing', 'prueba.json')]:
            evaluacion = leer_json(carpeta / archivo, {})
            for rama in ['online', 'objetivo']:
                datos = evaluacion.get(rama)
                if datos is None:
                    continue
                fila = {'modelo': carpeta.name, 'rama': rama, 'particion': particion}
                for condicion in ['original', 'baja_resolucion']:
                    for metrica in ['mAP', 'Rank-1', 'Rank-5', 'Rank-10']:
                        fila[f'{condicion}_{metrica}'] = datos[condicion][metrica]
                fila.update(caida_pp=datos['caida_mAP_puntos_porcentuales'], caida_relativa=datos['caida_mAP_relativa_porcentaje'], consultas=datos['original']['consultas_evaluadas'])
                filas.append(fila)
    if filas:
        with (resultados / 'metricas.csv').open('w', newline='', encoding='utf-8-sig') as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
            escritor.writeheader()
            escritor.writerows(filas)
        graficar(resultados, filas)
    plan = leer_json(resultados / 'plan_ejecucion.json', {})
    seleccion = leer_json(resultados / 'seleccion_validacion.json', {})
    estados = leer_json(resultados / 'estado_ejecucion.json', [])
    matriz = leer_json(resultados / 'matriz_experimentos.json', {})
    presupuesto = leer_json(resultados / 'presupuesto.json', {})
    transcurridos = presupuesto.get('transcurridos_segundos')
    if transcurridos is None and 'inicio_unix' in presupuesto:
        transcurridos = max(0, time.time() - presupuesto['inicio_unix'])
    tiempo_registrado = f'{transcurridos / 60:.2f} minutos' if transcurridos is not None else 'pendiente de inicio'
    verificaciones = leer_json(resultados / 'verificaciones.json', {})
    comprobadas = sum(e.get('evaluacion_prueba') == 'completada' for e in estados)
    completo_entrenamiento = sum(r['estado'] == 'entrenamiento_completado' for _, r in entrenamientos)
    texto = [
        '# Informe de implementación y ejecución',
        '',
        'El proyecto implementa CO-BYOL y CO-MoCoV3 para recuperación de personas sin etiquetas de identidad durante el entrenamiento, la comparación con ImageNet del artículo se sustituye por el protocolo de recuperación exigido por el laboratorio, las conclusiones se limitan a las ejecuciones registradas en este equipo.',
        '',
        '## Estado comprobable',
        '',
        'La revisión contra las fuentes se encuentra en REVISION_CUMPLIMIENTO.md, los requisitos experimentales básicos del laboratorio tienen evidencia y el alcance completo del artículo sigue parcial, la auditoría automática de coherencia no certifica una reproducción íntegra.',
        '',
        f'Se registran {len(entrenamientos)} entrenamientos con resumen, {completo_entrenamiento} completaron el número local de actualizaciones planificadas y {comprobadas} tienen prueba completa en ambas ramas y resoluciones, el panel principal contiene {len(plan.get("modelos", []))} configuraciones y la matriz ampliada contiene {len(matriz.get("matriz", []))} filas de configuración, una fila programada no constituye un experimento ejecutado.',
        '',
        f'El estado de las verificaciones técnicas es {verificaciones.get("estado", "pendiente")}, el límite de la sesión inicial es de {presupuesto.get("maximo_segundos", 10800) / 3600:g} horas y su registro conserva un inicio único para calibración, entrenamiento, validación y prueba, el tiempo transcurrido de esa sesión es {tiempo_registrado}.',
        '',
        '## Condiciones del experimento',
        '',
        'Se conserva training con 11043 imágenes, validation con 100 consultas y 1793 imágenes de galería, testing con 3368 consultas y 19732 imágenes de galería, las consultas reducidas se leen de su dataset HDF5 original y mantienen la galería original, no se crea una división nueva ni se utiliza final_training para incorporar validación al aprendizaje.',
        '',
        f'El batch físico elegido es {plan.get("batch_fisico", "pendiente")} y la planificación común es de {plan.get("pasos_comunes", "pendiente")} actualizaciones, estos valores proceden de la restricción de recursos y de la calibración local, no se atribuyen al PDF ni a los HDF5, la columna de época equivalente indica cuánta exposición efectiva recibió cada modelo.',
        '',
        'Los modelos parten de inicialización aleatoria, la lectura de entrenamiento devuelve exclusivamente imágenes y sus índices, las identidades y cámaras de validación y prueba intervienen en la medición de recuperación, los índices de cámara no se utilizan para construir objetivos de entrenamiento.',
        '',
        'Las vistas de entrenamiento se construyen con recortes y aumentos fotométricos autosupervisados, el panel principal emplea 224 por 224 y los modelos pequeños 64 por 64, el intervalo de aspecto del recorte se adapta a personas verticales, los resultados entre paneles no aíslan el efecto de la arquitectura o de la pérdida.',
        '',
        'Los encoders se comparan mediante vectores L2 y producto escalar coseno, se ignora person_ids igual a -1 y la coincidencia de identidad y cámara, person_ids igual a 0 permanece como distractor, AP se define como la media de la precisión en las posiciones relevantes, los empates se resuelven conservando el orden original de galería.',
        '',
        'La documentación oficial de Market-1501 fundamenta las exclusiones de evaluación, el artículo seleccionado no define este protocolo de reidentificación, la adaptación se declara de forma separada para no atribuirle reglas ausentes, fuente https://zheng-lab-anu.github.io/Project/project_reid.html.',
        '',
        'La caída absoluta de mAP es 100 por mAP original menos mAP reducida en puntos porcentuales, la caída relativa es 100 por esa diferencia dividida por mAP original, una caída negativa representa una mejora y no se recorta a cero.',
        '',
        f'La selección principal entre las configuraciones del panel inicial es {seleccion.get("modelo", "pendiente")} con rama {seleccion.get("rama", "pendiente")}, el criterio es la media de mAP de validación en las dos resoluciones, esta selección se guarda antes de abrir la fase de prueba y la ampliación de sensibilidad no la sustituye después de conocer prueba.',
        '',
        '## Exposición real de entrenamiento',
        '',
        '| Modelo | Estado | Pasos | Imágenes procesadas | Imágenes distintas | Épocas equivalentes | Minutos |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for nombre, resumen in entrenamientos:
        texto.append(f'| {nombre} | {resumen["estado"]} | {resumen["pasos_realizados"]} | {resumen["imagenes_fuente_procesadas"]} | {resumen["imagenes_fuente_distintas"]} | {resumen["epocas_completas"] + resumen["fraccion_epoca"]:.4f} | {resumen["segundos_entrenamiento"] / 60:.2f} |')
    texto += ['', '## Métricas por modelo y rama', '', 'Las tablas muestran mAP y Rank como porcentajes, metricas.csv y los JSON conservan proporciones de cero a uno sin redondear, caida_pp utiliza puntos porcentuales y caida_relativa utiliza porcentaje.', '', '| Modelo | Rama | Partición | mAP original | R1 | R5 | R10 | mAP reducida | R1 LR | R5 LR | R10 LR | Caída pp |', '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for fila in filas:
        valores = [porcentaje(fila[f'{condicion}_{metrica}']) for condicion in ['original', 'baja_resolucion'] for metrica in ['mAP', 'Rank-1', 'Rank-5', 'Rank-10']]
        texto.append('| ' + ' | '.join([fila['modelo'], fila['rama'], fila['particion'], *valores, f'{fila["caida_pp"]:.3f}']) + ' |')
    texto += ['', '## Modelo base y ablación', '', 'BYOL ResNet50 y MoCoV3 ResNet50 proporcionan modelos base, la comparación de BYOL RF99 frente a CO-BYOL RF99 conserva la arquitectura y elimina únicamente el término local mediante peso cero, la comparación entre ResNet50 estándar y RF99 cambia también la arquitectura y no se presenta como ablación aislada.', '', '| Comparación | Rama | Partición | Diferencia mAP original pp | Diferencia mAP reducida pp |', '|---|---|---|---:|---:|']
    indice = {(f['modelo'], f['rama'], f['particion']): f for f in filas}
    diferencias = []
    comparaciones_controladas = [('byol_r50', 'co_byol_r50'), ('byol_rf99', 'co_byol_rf99'), ('mocov3_r50', 'co_mocov3_r50'), ('mocov3_rf99', 'co_mocov3_rf99'), ('byol_bagnet33', 'co_byol_bagnet33'), ('byol_rf67_resnet50', 'co_byol_rf67_resnet50'), ('byol_rf163_resnet50', 'co_byol_rf163_resnet50'), ('byol_rf195_paper193_resnet50', 'co_byol_rf195_paper193_resnet50'), ('byol_rf29_r18', 'co_byol_rf29_r18'), ('mocov3_rf29_r18', 'co_mocov3_rf29_r18')]
    for base, propuesta in comparaciones_controladas:
        for rama in ['online', 'objetivo']:
            for particion in ['validation', 'testing']:
                a, b = indice.get((base, rama, particion)), indice.get((propuesta, rama, particion))
                if a is not None and b is not None:
                    original = 100 * (b['original_mAP'] - a['original_mAP'])
                    reducida = 100 * (b['baja_resolucion_mAP'] - a['baja_resolucion_mAP'])
                    texto.append(f'| {propuesta} frente a {base} | {rama} | {particion} | {original:.3f} | {reducida:.3f} |')
                    if particion == 'testing':
                        diferencias.append((propuesta, base, rama, original, reducida))
    texto += ['', '## Gráficos y ejemplos', '']
    for nombre, titulo in [('comparacion_map.png', 'Comparación de mAP por condición, rama online'), ('curvas_entrenamiento.png', 'Pérdidas de entrenamiento'), ('comparacion_maestro_estudiante.png', 'Maestro EMA frente a encoder online')]:
        if (resultados / nombre).exists():
            texto += [f'![{titulo}](resultados/{nombre})', '']
    evidencia = leer_json(resultados / 'ejemplos' / 'indices_ejemplos.json', [])
    for condicion in ['original', 'baja_resolucion']:
        for tipo in ['aciertos', 'fallos']:
            muestras = [e for e in evidencia if e['condicion'] == condicion and e['tipo'] == tipo]
            if muestras:
                texto += [f'### {tipo.capitalize()} con consultas {condicion.replace("_", " ")}', '', f'Se muestran {len(muestras)} consultas seleccionadas por el primer índice disponible de cada identidad distinta con Rank-1 {"correcto" if tipo == "aciertos" else "incorrecto"}, la selección es reproducible y los índices de galería y consulta se conservan en indices_ejemplos.json.', '', f'![{tipo} {condicion}](resultados/ejemplos/{condicion}_{tipo}.png)', '']
    if (raiz / 'ANALISIS_ERRORES.md').exists():
        texto += ['El análisis visual de los casos se encuentra en ANALISIS_ERRORES.md, los errores de recuperación se distinguen de las hipótesis sobre sus causas visuales.', '']
    texto += ['## Limitaciones y conclusiones', '', 'La restricción de tres horas obliga a calendarios muy inferiores a los del artículo, una fracción de época con inicialización aleatoria no permite afirmar convergencia ni evaluar el potencial final de CO-SSL, las diferencias medidas describen únicamente este régimen de cómputo.', '', 'El batch reducido cambia el número de negativos de MoCoV3 y la estimación de BatchNorm, multicrop procesa más vistas por actualización y las arquitecturas tienen costes diferentes, igualar imágenes fuente y actualizaciones no equivale a igualar FLOPs ni tiempo.', '', 'No se entrenan clasificadores de identidad, por ello las exactitudes Top-1 de clasificación del artículo no son directamente comparables con mAP y Rank de reidentificación, la adaptación conserva pérdidas, alineamiento local global y actualización EMA.', '', 'La ausencia de etiquetas de identidad en training impide verificar de manera directa la separación de personas frente a validación y prueba, la ausencia de duplicados exactos de imágenes no demuestra ausencia de identidades compartidas, la separación de personas de entrenamiento y prueba es una declaración del PDF.', '']
    contrastes_online = [d for d in diferencias if d[2] == 'online']
    if contrastes_online:
        favorables = sum(d[3] > 0 and d[4] > 0 for d in contrastes_online)
        desfavorables = sum(d[3] < 0 and d[4] < 0 for d in contrastes_online)
        mixtos = len(contrastes_online) - favorables - desfavorables
        texto += [f'En las {len(contrastes_online)} comparaciones emparejadas de la rama online en prueba, CO aumenta el mAP en ambas condiciones en {favorables} casos y lo reduce en ambas en {desfavorables}, las {mixtos} comparaciones restantes presentan signos distintos o igualdad, el componente local no produce una mejora uniforme en este entrenamiento y la tabla de ablaciones conserva cada diferencia y ambas ramas.', '']
    contraste_seleccionado = next((d for d in diferencias if d[0] == seleccion.get('modelo') and d[2] == seleccion.get('rama')), None)
    if contraste_seleccionado:
        propuesta, base, rama, original, reducida = contraste_seleccionado
        texto += [f'El modelo seleccionado {propuesta} frente a su control {base} con la misma arquitectura presenta diferencias de {original:.3f} puntos de mAP original y {reducida:.3f} puntos con consultas reducidas en la rama {rama}, esta comparación aísla el término local dentro de la arquitectura implementada y no demuestra superioridad general con una sola semilla.', '']
    texto += ['La selección principal se cierra entre las veintinueve configuraciones iniciales utilizando validación, los controles y barridos adicionales conservan las configuraciones declaradas y se estudian como ampliación sin reemplazar esa selección después de observar prueba.', '', 'Validación contiene 100 consultas y una galería de 1793 imágenes mientras que prueba contiene 3368 consultas y una galería de 19732 imágenes, sus métricas responden a problemas de recuperación de distinta dificultad y una diferencia entre ellas no permite atribuir por sí sola el resultado a sobreajuste.', '']
    texto += ['## Requisitos y evidencia de cobertura', '', '| Requisito | Fuente | Implementado | Ejecutado y evidencia |', '|---|---|---|---|', '| Entrenamiento sin identidades y entradas de imagen | PDF p 1 | datos.py y entrenamiento.py | Interfaz de training sin person_ids, resúmenes con imágenes vistas |', '| Particiones proporcionadas | PDF p 1 | datos.py | verificaciones.json con formas reales |', '| Implementación propia del núcleo | PDF p 2 | modelos.py, RF-ResNet, pérdidas y EMA propias | checkpoints y registros, arquitecturas estándar de bibliotecas generales |', f'| Entrenar y seleccionar con validación | PDF p 2 | entrenamiento.py y main.py | {len(entrenamientos)} checkpoints, seleccion_validacion.json |', '| Modelo base y ablación | PDF p 2 | peso_local cero frente a CO con misma arquitectura | Tabla de comparaciones y JSON por modelo, solo filas con métricas reales |', '| Comparar maestro y estudiante | PDF p 2 | Dos ramas y selección en validación | Filas online y objetivo, gráfico de comparación |', f'| Prueba en ambas resoluciones con mismo modelo | PDF p 2 | evaluacion.py | {comprobadas} modelos con ambas ramas completas, prueba.json |', '| L2, coseno, mAP y Rank 1, 5 y 10, caída mAP | PDF p 2 | evaluacion.py | verificaciones.json y metricas.csv |', '| No entrenar ni ajustar con prueba | PDF p 2 | Orden de main.py y selección congelada | seleccion_validacion.json anterior a prueba |', f'| Cinco aciertos y cinco fallos explicados | PDF p 3 | evaluacion.py | {len(evidencia)} ejemplos registrados, imágenes e informe visual |', '| Instrucciones y diferencias respecto al paper | PDF pp 2 y 3 | README.md y METODOLOGIA.md | Documentación y configuraciones guardadas |', '| Video explicativo de hasta 18 minutos | PDF p 3 | Video explicativo y registro de duración | ' + ('VIDEO.mp4' if (raiz / 'VIDEO.mp4').exists() else 'Pendiente, el informe no sustituye el video') + ' |', '| Defensa individual y entrega grupal | PDF p 3 | Guía de explicación en documentación | Pendiente de presentación humana y envío al formulario |', '', '## Experimentos ampliados y asuntos pendientes', '', 'Los barridos de recorte, campos receptivos, peso y cantidad de locales, MLP, cabeza compartida, capas de alineamiento y kernel se configuran en experimentos.py, su ejecución se comprueba exclusivamente por presencia de resumen y métricas, los calendarios de 300 y 1000 épocas permanecen pendientes si no existe evidencia de finalización.', '', 'Los análisis de máscaras, PGD, representaciones y lecturas intermedias disponen de un ejecutor limitado por el mismo presupuesto, únicamente los archivos analisis correspondientes demuestran ejecución, PGD utiliza un objetivo de recuperación documentado que difiere de la clasificación del artículo.', '', 'La transferencia literal a Places365 requiere otro dataset y etiquetas de clasificación ausentes, los comparadores AMDIM y SD-BYOL no se presentan como implementaciones propias verificadas, el campo denominado RF193 en la figura del artículo produce RF195 al seguir literalmente sus kernels y strides y se identifica con ambos nombres para conservar la discrepancia.', '', 'La indicación de bibliotecas para audio en el PDF resulta incompatible con sus entradas exclusivamente de imagen, se interpreta como una posible errata y no se incorporan señales de audio, no se presenta esa interpretación como una aclaración oficial.', '', 'Las conclusiones no utilizan el conjunto de prueba para modificar modelos o hiperparámetros, las ejecuciones pendientes y las restricciones de presentación se mantienen visibles para que la entrega no se confunda con una reproducción íntegra del artículo.', '']
    clave_seleccionada = (seleccion.get('modelo'), seleccion.get('rama'), 'testing')
    fila_seleccionada = indice.get(clave_seleccionada)
    if fila_seleccionada:
        posicion = texto.index('## Condiciones del experimento')
        tabla = ['## Resultado del modelo seleccionado en validación', '', f'La selección previa a prueba corresponde a {seleccion["modelo"]} con rama {seleccion["rama"]}, no se escoge esta fila por su resultado de prueba.', '', '| Condición | mAP % | Rank 1 % | Rank 5 % | Rank 10 % |', '|---|---:|---:|---:|---:|']
        for condicion in ['original', 'baja_resolucion']:
            tabla.append('| ' + ' | '.join([condicion.replace('_', ' '), *[porcentaje(fila_seleccionada[f'{condicion}_{m}']) for m in ['mAP', 'Rank-1', 'Rank-5', 'Rank-10']]]) + ' |')
        relativa = fila_seleccionada['caida_relativa']
        tabla += ['', f'La caída absoluta es {fila_seleccionada["caida_pp"]:.3f} puntos porcentuales y la caída relativa es {relativa:.3f} por ciento.' if relativa is not None else 'La caída relativa no puede definirse con mAP original cero.', '']
        texto[posicion:posicion] = tabla
    posicion_variantes = texto.index('## Gráficos y ejemplos')
    texto[posicion_variantes:posicion_variantes] = resumir_variantes(resultados, indice)
    posicion_analisis = texto.index('## Limitaciones y conclusiones')
    texto[posicion_analisis:posicion_analisis] = resumir_analisis(resultados)
    posicion_requisitos = texto.index('## Requisitos y evidencia de cobertura')
    texto[posicion_requisitos:posicion_requisitos] = ['El encoder BagNet33 implementado tiene 16.261 millones de parámetros frente a 23.508 de ResNet50 y 23.750 de RF99, el artículo describe su control BagNet33 como igualado en parámetros pero no detalla esa configuración, esta versión conserva campo 33 y no reproduce aquella igualación, las diferencias entre arquitecturas no se atribuyen exclusivamente al campo receptivo y el control sin CO de la misma BagNet33 permite aislar la pérdida local en esta implementación.', '']
    texto += ['## Presentación y evaluación del curso', '', 'La página uno fija la entrega de video y código para el jueves veintidós de octubre a las 23:59 y la revisión para el sábado veinticuatro en horario de clase, el envío al formulario corresponde al grupo y no se realiza automáticamente desde este proyecto.', '', 'La página tres distingue la evaluación grupal del video y el código de la defensa individual, las preguntas abarcan comprensión del método, funcionamiento del código, justificación de decisiones, interpretación de resultados e identificación de errores y limitaciones, el PDF no proporciona ponderaciones numéricas ni un umbral mínimo de rendimiento.', '', 'El material audiovisual utiliza narración sintética local, sus atribuciones constan en METODOLOGIA.md, esta generación no acredita una exposición humana ni la defensa individual.', '']
    notas = notas_comparacion(resultados)
    if notas:
        posicion = texto.index('## Estado comprobable') + 2
        texto[posicion:posicion] = notas
        if '## Resultado del modelo seleccionado en validación' in texto:
            texto[texto.index('## Resultado del modelo seleccionado en validación')] = '## Resultado de la selección histórica del panel inicial'
    (raiz / 'INFORME.md').write_text('\n'.join(texto), encoding='utf-8')
    return {'modelos_con_checkpoint': len(entrenamientos), 'pruebas_completas': comprobadas, 'filas_metricas': len(filas)}
