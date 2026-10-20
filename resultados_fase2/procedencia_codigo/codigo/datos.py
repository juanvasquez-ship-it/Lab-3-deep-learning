import math
import zlib
from pathlib import Path
from zipfile import ZipFile

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


MEDIA = (0.485, 0.456, 0.406)
DESVIACION = (0.229, 0.224, 0.225)


class Vistas:
    def __init__(self, resolucion, recorte=0.2, multicrop=False, numero_locales=4):
        escala_global = (0.14, 1.0) if multicrop else (recorte, 1.0)
        self.transformaciones = [self.construir(resolucion, escala_global, 1.0, 0.0), self.construir(resolucion, escala_global, 0.1, 0.2)]
        if multicrop:
            pequeno = round(resolucion * 96 / 224)
            self.transformaciones.extend([self.construir(pequeno, (0.05, 0.14), 0.5, 0.0) for _ in range(numero_locales)])

    def construir(self, resolucion, escala, desenfoque, solarizacion):
        return transforms.Compose([
            transforms.RandomResizedCrop(resolucion, scale=escala, ratio=(0.375, 2 / 3), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(max(3, (resolucion // 10) | 1), sigma=(0.1, 2.0))], p=desenfoque),
            transforms.RandomSolarize(128, p=solarizacion),
            transforms.ToTensor(),
            transforms.Normalize(MEDIA, DESVIACION),
        ])

    def __call__(self, imagen):
        return [transformacion(imagen) for transformacion in self.transformaciones]


class Entrenamiento(Dataset):
    def __init__(self, carpeta, configuracion):
        self.ruta = str(Path(carpeta) / 'training.h5')
        self.archivo = None
        with h5py.File(self.ruta, 'r') as archivo:
            self.cantidad = len(archivo['images'])
            self.identificadores = archivo['image_ids'][:]
        self.vistas = Vistas(configuracion.get('resolucion', 224), configuracion.get('recorte', 0.2), configuracion.get('multicrop', False), configuracion.get('numero_locales', 4))

    def __len__(self):
        return self.cantidad

    def __getitem__(self, indice):
        if self.archivo is None:
            self.archivo = h5py.File(self.ruta, 'r')
        imagen = Image.fromarray(self.archivo['images'][indice].transpose(1, 2, 0))
        return self.vistas(imagen), indice


class Evaluacion(Dataset):
    def __init__(self, carpeta, particion, grupo, reducida=False):
        self.ruta = str(Path(carpeta) / (particion + '.h5'))
        self.grupo = grupo
        self.campo = 'images_low_resolution' if reducida else 'images'
        self.archivo = None
        with h5py.File(self.ruta, 'r') as archivo:
            self.cantidad = len(archivo[grupo][self.campo])
            self.personas = archivo[grupo]['person_ids'][:]
            self.camaras = archivo[grupo]['camera_ids'][:]
            self.identificadores = archivo[grupo]['image_ids'][:]

    def __len__(self):
        return self.cantidad

    def __getitem__(self, indice):
        if self.archivo is None:
            self.archivo = h5py.File(self.ruta, 'r')
        return torch.from_numpy(self.archivo[self.grupo][self.campo][indice]), indice


def preparar_entrada(imagenes, resolucion, dispositivo):
    imagenes = imagenes.to(dispositivo, non_blocking=True).float() / 255
    imagenes = torch.nn.functional.interpolate(imagenes, (resolucion, resolucion), mode='bicubic', align_corners=False, antialias=True).clamp_(0, 1)
    media = imagenes.new_tensor(MEDIA).view(1, 3, 1, 1)
    desviacion = imagenes.new_tensor(DESVIACION).view(1, 3, 1, 1)
    return (imagenes - media) / desviacion


def preparar_datos(ruta_zip, destino):
    destino = Path(destino).resolve()
    destino.mkdir(parents=True, exist_ok=True)
    with ZipFile(ruta_zip) as archivo:
        for nombre in ['training.h5', 'validation.h5', 'testing.h5']:
            miembro = archivo.getinfo(nombre)
            ruta = destino / nombre
            if ruta.exists():
                if ruta.stat().st_size != miembro.file_size:
                    raise ValueError(f'El archivo existente tiene otro tamaño: {ruta}')
                crc = 0
                with ruta.open('rb') as existente:
                    for bloque in iter(lambda: existente.read(8 * 1024 ** 2), b''):
                        crc = zlib.crc32(bloque, crc)
                if crc != miembro.CRC:
                    raise ValueError(f'El archivo existente no coincide con el CRC del ZIP: {ruta}')
            else:
                archivo.extract(nombre, destino)
    return str(destino)


def verificar_particiones(carpeta):
    esperado = {'training': {'images': 11043}, 'validation': {'query/images': 100, 'query/images_low_resolution': 100, 'gallery/images': 1793}, 'testing': {'query/images': 3368, 'query/images_low_resolution': 3368, 'gallery/images': 19732}}
    resultado = {}
    for particion, campos in esperado.items():
        with h5py.File(Path(carpeta) / f'{particion}.h5', 'r') as archivo:
            resultado[particion] = {}
            for campo, cantidad in campos.items():
                datos = archivo[campo]
                espacial = (32, 16) if campo.endswith('images_low_resolution') else (128, 64)
                if datos.shape != (cantidad, 3, *espacial) or datos.dtype != np.uint8:
                    raise ValueError(f'Estructura inesperada: {particion}/{campo}')
                resultado[particion][campo] = list(datos.shape)
                grupo = archivo[campo.rsplit('/', 1)[0]] if '/' in campo else archivo
                for metadato in ['image_ids', 'camera_ids'] + ([] if particion == 'training' else ['person_ids']):
                    if grupo[metadato].shape != (cantidad,):
                        raise ValueError(f'Metadatos desalineados: {particion}/{campo}/{metadato}')
            if particion == 'training' and 'person_ids' in archivo:
                raise ValueError('La interfaz de entrenamiento no admite etiquetas de identidad')
    return resultado
