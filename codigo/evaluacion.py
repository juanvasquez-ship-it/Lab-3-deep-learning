import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from codigo.datos import Evaluacion, preparar_entrada


def calcular_metricas(similitudes, personas_consulta, camaras_consulta, personas_galeria, camaras_galeria, protocolo='market_documentado', guardar_rankings=False):
    if protocolo not in {'market_documentado', 'sin_exclusiones'}:
        raise ValueError('Debe especificarse un protocolo de evaluación conocido')
    precisiones = []
    rangos = []
    rankings = []
    omitidas = []
    personas_galeria = np.asarray(personas_galeria)
    camaras_galeria = np.asarray(camaras_galeria)
    for indice, (persona, camara) in enumerate(zip(personas_consulta, camaras_consulta)):
        orden = np.argsort(-similitudes[indice], kind='stable')
        if protocolo == 'market_documentado':
            excluidas = (personas_galeria[orden] == -1) | ((personas_galeria[orden] == persona) & (camaras_galeria[orden] == camara))
            orden = orden[~excluidas]
        aciertos = personas_galeria[orden] == persona
        posiciones = np.flatnonzero(aciertos)
        if not len(posiciones):
            omitidas.append(indice)
            continue
        precisiones.append(float(np.mean(np.arange(1, len(posiciones) + 1) / (posiciones + 1))))
        rangos.append(int(posiciones[0]) + 1)
        if guardar_rankings:
            rankings.append({'consulta': indice, 'rank': int(posiciones[0]) + 1, 'ap': precisiones[-1], 'indices': orden[:10].tolist(), 'aciertos': aciertos[:10].tolist()})
    if not precisiones:
        raise ValueError('Ninguna consulta tiene positivos bajo el protocolo seleccionado')
    resultado = {'mAP': float(np.mean(precisiones)), 'Rank-1': float(np.mean(np.array(rangos) <= 1)), 'Rank-5': float(np.mean(np.array(rangos) <= 5)), 'Rank-10': float(np.mean(np.array(rangos) <= 10)), 'consultas_evaluadas': len(precisiones), 'consultas_omitidas': omitidas, 'protocolo': protocolo, 'AP': 'media de la precisión en cada posición relevante', 'desempate': 'orden estable de los índices originales de galería'}
    if guardar_rankings:
        resultado['rankings'] = rankings
    return resultado


@torch.inference_mode()
def extraer_caracteristicas(encoder, dataset, resolucion, dispositivo, batch=64, trabajadores=0, antes_mlp=False, mascara=0, capa_lectura=None, limite=None):
    cargador = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=trabajadores, pin_memory=dispositivo.type == 'cuda')
    caracteristicas = []
    encoder.eval()
    for imagenes, indices in cargador:
        if limite is not None and limite.agotado(30):
            raise TimeoutError('Se agotó el presupuesto global durante la extracción')
        entrada = preparar_entrada(imagenes, resolucion, dispositivo)
        with torch.autocast(dispositivo.type, dtype=torch.bfloat16, enabled=dispositivo.type == 'cuda'):
            mapas = encoder.mapas(entrada)
            if capa_lectura is not None:
                globales = mapas[capa_lectura - 1].mean((2, 3))
            else:
                globales = encoder.agregar(mapas[-1], antes_mlp, mascara)
        if not torch.isfinite(globales).all():
            raise ValueError('Representación no finita durante evaluación')
        caracteristicas.append(F.normalize(globales.float(), dim=1).cpu())
    return torch.cat(caracteristicas).numpy()


def evaluar_encoder(encoder, carpeta, particion, resolucion, dispositivo, batch=64, trabajadores=0, protocolo='market_documentado', ejemplos=False, antes_mlp=False, mascara=0, capa_lectura=None, limite=None):
    galeria = Evaluacion(carpeta, particion, 'gallery')
    consulta = Evaluacion(carpeta, particion, 'query')
    reducida = Evaluacion(carpeta, particion, 'query', True)
    parametros = dict(resolucion=resolucion, dispositivo=dispositivo, batch=batch, trabajadores=trabajadores, antes_mlp=antes_mlp, mascara=mascara, capa_lectura=capa_lectura, limite=limite)
    caracteristicas_galeria = extraer_caracteristicas(encoder, galeria, **parametros)
    estado_cpu = torch.get_rng_state() if mascara else None
    estado_cuda = torch.cuda.get_rng_state(dispositivo) if mascara and dispositivo.type == 'cuda' else None
    resultado = {}
    for nombre, datos in [('original', consulta), ('baja_resolucion', reducida)]:
        if mascara and nombre == 'baja_resolucion':
            torch.set_rng_state(estado_cpu)
            if estado_cuda is not None:
                torch.cuda.set_rng_state(estado_cuda, dispositivo)
        caracteristicas_consulta = extraer_caracteristicas(encoder, datos, **parametros)
        similitudes = caracteristicas_consulta @ caracteristicas_galeria.T
        resultado[nombre] = calcular_metricas(similitudes, datos.personas, datos.camaras, galeria.personas, galeria.camaras, protocolo, ejemplos)
    original = resultado['original']['mAP']
    reducida_map = resultado['baja_resolucion']['mAP']
    resultado['caida_mAP_puntos_porcentuales'] = 100 * (original - reducida_map)
    resultado['caida_mAP_relativa_porcentaje'] = 100 * (original - reducida_map) / original if original else None
    resultado['criterio_validacion'] = (original + reducida_map) / 2
    if mascara:
        resultado['mascaras_acopladas_consultas'] = True
    return resultado


def generar_ejemplos(resultado, carpeta_datos, destino, particion='testing', cantidad=5):
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    evidencia = []
    with h5py.File(Path(carpeta_datos) / f'{particion}.h5', 'r') as archivo:
        for condicion in ['original', 'baja_resolucion']:
            rankings = resultado[condicion].get('rankings', [])
            for nombre, acierto in [('aciertos', True), ('fallos', False)]:
                seleccion = []
                identidades = set()
                for registro in rankings:
                    persona = int(archivo['query/person_ids'][registro['consulta']])
                    if (registro['rank'] == 1) == acierto and persona not in identidades:
                        seleccion.append(registro)
                        identidades.add(persona)
                    if len(seleccion) == cantidad:
                        break
                if not seleccion:
                    continue
                figura, ejes = plt.subplots(len(seleccion), 6, figsize=(11, max(3, len(seleccion) * 2.8)), squeeze=False)
                for fila, muestra in enumerate(seleccion):
                    q = muestra['consulta']
                    campo = 'images' if condicion == 'original' else 'images_low_resolution'
                    imagen = archivo['query'][campo][q].transpose(1, 2, 0)
                    ejes[fila, 0].imshow(imagen)
                    ejes[fila, 0].set_title(f"Consulta {q}\nID {archivo['query/person_ids'][q]}", fontsize=9)
                    for columna, indice in enumerate(muestra['indices'][:5], 1):
                        ejes[fila, columna].imshow(archivo['gallery/images'][indice].transpose(1, 2, 0))
                        coincide = muestra['aciertos'][columna - 1]
                        ejes[fila, columna].set_title(f"Rank {columna}, ID {archivo['gallery/person_ids'][indice]}", color='green' if coincide else 'red', fontsize=9)
                    evidencia.append({'condicion': condicion, 'tipo': nombre, 'person_id': int(archivo['query/person_ids'][q]), 'camera_id': int(archivo['query/camera_ids'][q]), 'person_ids_galeria': [int(archivo['gallery/person_ids'][i]) for i in muestra['indices']], 'camera_ids_galeria': [int(archivo['gallery/camera_ids'][i]) for i in muestra['indices']], 'criterio_seleccion': 'primeros índices con identidades de consulta distintas y Rank-1 correcto o incorrecto, sin selección por apariencia', **muestra})
                for eje in ejes.flat:
                    eje.axis('off')
                figura.suptitle(f'{nombre.capitalize()} en {condicion.replace("_", " ")}')
                figura.tight_layout()
                figura.savefig(destino / f'{condicion}_{nombre}.png', dpi=140)
                plt.close(figura)
    (destino / 'indices_ejemplos.json').write_text(json.dumps(evidencia, indent=2), encoding='utf-8')
    return {'ejemplos': len(evidencia), 'criterio': 'primeros índices originales con identidades distintas y Rank-1 correcto o incorrecto, sin selección manual por apariencia'}


def perturbacion_pgd(encoder, imagenes, positivos_galeria, epsilon, paso, iteraciones, resolucion, inicio_aleatorio=True):
    original = imagenes.detach().float().clamp(0, 1)
    perturbadas = (original + torch.empty_like(original).uniform_(-epsilon, epsilon)).clamp(0, 1) if inicio_aleatorio else original.clone()
    media = original.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    desviacion = original.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    for _ in range(iteraciones):
        perturbadas.requires_grad_(True)
        entrada = F.interpolate(perturbadas, (resolucion, resolucion), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
        with torch.autocast(original.device.type, dtype=torch.bfloat16, enabled=original.device.type == 'cuda'):
            globales = encoder((entrada - media) / desviacion)[1]
        representacion = F.normalize(globales.float(), dim=1)
        objetivo = -(representacion * F.normalize(positivos_galeria, dim=1)).sum(1).mean()
        gradiente = torch.autograd.grad(objetivo, perturbadas)[0]
        if not torch.isfinite(gradiente).all():
            raise ValueError('Gradiente de ataque no finito')
        perturbadas = perturbadas.detach() + paso * gradiente.sign()
        perturbadas = torch.maximum(torch.minimum(perturbadas, original + epsilon), original - epsilon).clamp(0, 1)
    return perturbadas.detach()
