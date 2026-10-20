import copy
import json


def configuracion(nombre, metodo='byol', arquitectura='resnet50', peso_local=0.0, **opciones):
    return {'nombre': nombre, 'metodo': metodo, 'arquitectura': arquitectura, 'peso_local': peso_local, 'resolucion': 64 if arquitectura in {'resnet18', 'rf29_resnet18'} else 224, 'recorte': 0.2, 'semilla': 20261019, 'temperatura': 0.2, 'kernel_local': 6 if arquitectura == 'vit_s16' and peso_local else 3 if arquitectura.startswith('swin') and peso_local else 1, **opciones}


def principales():
    lista = []
    for metodo in ['byol', 'mocov3']:
        lista.extend([
            configuracion(f'{metodo}_r50', metodo),
            configuracion(f'{metodo}_mc_r50', metodo, multicrop=True),
            configuracion(f'co_{metodo}_r50', metodo, peso_local=0.2),
            configuracion(f'co_{metodo}_rf99', metodo, 'rf99_resnet50', 0.2),
            configuracion(f'{metodo}_rf99', metodo, 'rf99_resnet50'),
            configuracion(f'{metodo}_r18', metodo, 'resnet18'),
            configuracion(f'{metodo}_mc_r18', metodo, 'resnet18', multicrop=True),
            configuracion(f'co_{metodo}_rf29_r18', metodo, 'rf29_resnet18', 0.2),
        ])
    for arquitectura in ['vit_s16', 'swin_t', 'swin_s', 'swin_b']:
        lista.extend([configuracion(f'byol_{arquitectura}', arquitectura=arquitectura), configuracion(f'co_byol_{arquitectura}', arquitectura=arquitectura, peso_local=0.2)])
    lista.append(configuracion('dino_r50', metodo='dino', multicrop=True, numero_locales=6))
    for arquitectura in ['bagnet33', 'rf67_resnet50', 'rf163_resnet50', 'rf195_paper193_resnet50']:
        lista.append(configuracion(f'co_byol_{arquitectura}', arquitectura=arquitectura, peso_local=0.2))
    return lista


def matriz_completa():
    matriz = principales()
    for metodo in ['byol', 'mocov3']:
        matriz.append(configuracion(f'{metodo}_rf29_r18', metodo, 'rf29_resnet18', fuente='Control con arquitectura fija para panel pequeño'))
    for arquitectura in ['bagnet33', 'rf67_resnet50', 'rf163_resnet50', 'rf195_paper193_resnet50']:
        matriz.append(configuracion(f'byol_{arquitectura}', arquitectura=arquitectura, fuente='Laboratorio PDF p 2, control sin CO con la misma arquitectura'))
    for arquitectura in ['rf39_resnet50', 'rf115_resnet50']:
        matriz.append(configuracion(f'co_byol_{arquitectura}', arquitectura=arquitectura, peso_local=0.2, fuente='Apéndice A, construcción arquitectónica'))
    for metodo in ['byol', 'mocov3']:
        for arquitectura in ['resnet50', 'rf99_resnet50']:
            for peso in [0.0, 0.2]:
                for recorte in [0.08, 0.4, 0.6, 0.8, 0.9]:
                    matriz.append(configuracion(f'{metodo}_{arquitectura}_co{peso}_crop{recorte}', metodo, arquitectura, peso, recorte=recorte, fuente='Figura D.8'))
    for arquitectura in ['bagnet33', 'rf67_resnet50', 'rf99_resnet50', 'rf163_resnet50', 'rf195_paper193_resnet50', 'resnet50']:
        for recorte in [0.1, 0.2, 0.3, 0.4, 0.7]:
            matriz.append(configuracion(f'co_byol_{arquitectura}_crop{recorte}', arquitectura=arquitectura, peso_local=0.2, recorte=recorte, fuente='Figura 4'))
    for cantidad in [1, 4, 16, 64, 196]:
        matriz.append(configuracion(f'co_byol_rf99_locales{cantidad}', arquitectura='rf99_resnet50', peso_local=0.2, cantidad_locales=cantidad, fuente='Tabla 3'))
    for peso in [0.0, 0.1, 0.2, 0.5, 1.0]:
        matriz.append(configuracion(f'co_byol_rf99_peso{peso}', arquitectura='rf99_resnet50', peso_local=peso, fuente='Tabla 7'))
    for peso in [0.0, 0.2]:
        matriz.append(configuracion(f'byol_rf99_sin_mlp_co{peso}', arquitectura='rf99_resnet50', peso_local=peso, sin_mlp=True, fuente='Tabla 6'))
    matriz.append(configuracion('co_byol_rf99_cabeza_compartida', arquitectura='rf99_resnet50', peso_local=0.2, cabeza_compartida=True, fuente='Tabla 6'))
    for kernel in range(1, 6):
        matriz.append(configuracion(f'co_byol_swin_t_kernel{kernel}', arquitectura='swin_t', peso_local=0.2, kernel_local=kernel, fuente='Tabla F.10'))
    for capa in [2, 3]:
        matriz.append(configuracion(f'co_byol_rf99_capa{capa}', arquitectura='rf99_resnet50', peso_local=0.2, capa_local=capa, cantidad_locales=64, fuente='Apéndice I'))
    for arquitectura in ['vit_s16', 'swin_s']:
        for peso in [0.0, 0.2]:
            matriz.append(configuracion(f'byol_{arquitectura}_co{peso}_crop09', arquitectura=arquitectura, peso_local=peso, recorte=0.9, fuente='Tabla D.8'))
    for nombre in ['byol_r50', 'co_byol_r50', 'co_byol_rf99', 'byol_vit_s16', 'co_byol_vit_s16', 'byol_swin_t', 'co_byol_swin_t', 'byol_swin_b', 'co_byol_swin_b']:
        base = next(c for c in principales() if c['nombre'] == nombre)
        for epocas in ([300, 1000] if nombre in ['byol_r50', 'co_byol_r50', 'byol_vit_s16', 'co_byol_vit_s16'] else [300]):
            variante = copy.deepcopy(base)
            variante.update(nombre=f'{nombre}_epocas{epocas}', epocas_paper=epocas, fuente='Tabla 1')
            matriz.append(variante)
    for configuracion_actual in matriz:
        configuracion_actual.setdefault('estado', 'implementado_no_ejecutado')
        configuracion_actual.setdefault('fuente', 'Tablas 1, 2, E.9; apéndice D')
    return matriz


def experimentos_lectura():
    return {
        'antes_mlp': {'modelos': ['byol_rf99', 'co_byol_rf99'], 'reentrenamiento': False, 'fuente': 'Tabla 6'},
        'mascaras': {'proporciones': [0.5, 0.6, 0.7, 0.8, 0.9], 'reentrenamiento': False, 'fuente': 'Tabla 4'},
        'pgd': {'configuraciones': [{'epsilon': 0.01, 'paso': 0.01 / 40, 'iteraciones': 1}, {'epsilon': 0.03, 'paso': 0.03 / 40, 'iteraciones': 1}, {'epsilon': 0.1, 'paso': 0.1 / 40, 'iteraciones': 1}, {'epsilon': 0.003, 'paso': 0.003 / 40, 'iteraciones': 5}, {'epsilon': 0.003, 'paso': 0.003 / 10, 'iteraciones': 1}], 'reentrenamiento': False, 'adaptacion': 'ataque a consulta para reducir similitud con representación media de positivos, pesos congelados', 'fuente': 'Tabla 5, diferencia respecto al texto documentada'},
        'capas_lectura': {'capas': [3, 4], 'reentrenamiento': False, 'fuente': 'Tabla G.12'},
        'similitud_local': {'reentrenamiento': False, 'fuente': 'Figura 5'},
        'campos_receptivos_efectivos': {'umbrales': [0.05, 0.32], 'reentrenamiento': False, 'fuente': 'Figuras 2 y C.7'},
        'representaciones384': {'modelo': 'co_byol_r50', 'resolucion': 384, 'reentrenamiento': False, 'fuente': 'Apéndice C', 'interpretacion': 'El texto no exige reentrenar a 384, se analiza el mismo encoder con otra resolución de entrada'},
        'transferencia_places365': {'estado': 'no_reproducida_en_esta_adaptacion_por_datos_no_disponibles', 'fuente': 'Tabla G.11'},
        'comparadores_ambiguos': {'modelos': ['AMDIM', 'SD-BYOL'], 'estado': 'pendiente_aclaracion_de_alcance', 'fuente': 'Tabla 1'},
    }


def exportar_matriz(ruta):
    contenido = {'principales': principales(), 'matriz': matriz_completa(), 'lecturas': experimentos_lectura(), 'calendarios_publicados': {'ImageNet1k': [100, 300, 1000], 'ImageNet100': [400], 'TinyImageNet': [400]}, 'nota_calendarios': 'Condiciones de los datasets originales del artículo, no son épocas realizadas en el dataset del laboratorio'}
    for fila in contenido['matriz']:
        fila['estado_inicial_implementacion'] = fila.pop('estado')
    ruta_estados = ruta.parent / 'estado_ejecucion.json'
    contenido['ejecuciones_observadas'] = json.loads(ruta_estados.read_text(encoding='utf-8')) if ruta_estados.exists() else []
    contenido['alcance_estados'] = 'Los estados iniciales describen parametrización, la ejecución real se consulta en ejecuciones_observadas y cobertura_experimentos.csv'
    contenido['hiperparametros_fuente'] = {'tasa_CNN': [0.2, 0.4, 0.8, 1.6], 'tasa_Transformer': [0.0003, 0.0005, 0.001], 'peso_local_principal': [0.2, 0.5], 'nota': 'El código admite tasa_base, no se ejecuta el barrido completo de tasas bajo el presupuesto local'}
    ruta.write_text(json.dumps(contenido, indent=2, ensure_ascii=False), encoding='utf-8')
    return contenido


def configuraciones_unicas(configuraciones):
    resultado = []
    posiciones = {}
    for actual in configuraciones:
        clave = json.dumps({k: v for k, v in actual.items() if k not in {'nombre', 'fuente', 'estado'}}, sort_keys=True)
        if clave in posiciones:
            resultado[posiciones[clave]].setdefault('alias', []).append(actual['nombre'])
        else:
            posiciones[clave] = len(resultado)
            resultado.append(copy.deepcopy(actual))
    return resultado
