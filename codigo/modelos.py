import copy
import math
from contextlib import contextmanager, nullcontext

import timm
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch.utils.checkpoint import checkpoint


@contextmanager
def buffers_recomputacion(modulo):
    originales = []
    for capa in modulo.modules():
        if isinstance(capa, nn.modules.batchnorm._BatchNorm):
            for nombre in ['running_mean', 'running_var', 'num_batches_tracked']:
                valor = getattr(capa, nombre)
                if valor is not None:
                    originales.append((capa, nombre, valor))
                    setattr(capa, nombre, valor.clone())
    try:
        yield
    finally:
        for capa, nombre, valor in originales:
            setattr(capa, nombre, valor)


def ejecutar_memoria(modulo, entrada, activado, funcion=None):
    funcion = funcion or modulo
    if not activado or not torch.is_grad_enabled():
        return funcion(entrada)
    return checkpoint(funcion, entrada, use_reentrant=False, context_fn=lambda: (nullcontext(), buffers_recomputacion(modulo)))


class BloqueEspacial(nn.Module):
    def __init__(self, entrada, ancho, kernel, paso, basico=False, sin_padding=False):
        super().__init__()
        salida = ancho if basico else ancho * 4
        padding = 0 if sin_padding else kernel // 2
        if basico:
            capas = [nn.Conv2d(entrada, ancho, kernel, paso, padding, bias=False), nn.BatchNorm2d(ancho), nn.ReLU(), nn.Conv2d(ancho, salida, 1, bias=False), nn.BatchNorm2d(salida)]
        else:
            capas = [nn.Conv2d(entrada, ancho, 1, bias=False), nn.BatchNorm2d(ancho), nn.ReLU(), nn.Conv2d(ancho, ancho, kernel, paso, padding, bias=False), nn.BatchNorm2d(ancho), nn.ReLU(), nn.Conv2d(ancho, salida, 1, bias=False), nn.BatchNorm2d(salida)]
        self.capas = nn.Sequential(*capas)
        self.atajo = nn.Sequential(nn.Conv2d(entrada, salida, 1, paso, bias=False), nn.BatchNorm2d(salida)) if entrada != salida or paso != 1 else nn.Identity()

    def forward(self, imagenes):
        resultado = self.capas(imagenes)
        atajo = self.atajo(imagenes)
        diferencia_alto = atajo.shape[2] - resultado.shape[2]
        diferencia_ancho = atajo.shape[3] - resultado.shape[3]
        atajo = atajo[:, :, diferencia_alto // 2:diferencia_alto // 2 + resultado.shape[2], diferencia_ancho // 2:diferencia_ancho // 2 + resultado.shape[3]]
        return F.relu(resultado + atajo)


class MLPResidual(nn.Module):
    def __init__(self, dimension, basico=False):
        super().__init__()
        interno = dimension if basico else dimension // 4
        capas = [nn.Linear(dimension, interno, bias=False), nn.BatchNorm1d(interno), nn.ReLU()]
        if not basico:
            capas.extend([nn.Linear(interno, interno, bias=False), nn.BatchNorm1d(interno), nn.ReLU()])
        capas.extend([nn.Linear(interno, dimension, bias=False), nn.BatchNorm1d(dimension)])
        self.capas = nn.Sequential(*capas)

    def forward(self, representacion):
        return F.relu(representacion + self.capas(representacion))


class RFResNet(nn.Module):
    def __init__(self, nombre, quitar_mlp=False, memoria_reducida=False):
        super().__init__()
        self.memoria_reducida = memoria_reducida
        perfiles = {
            'rf39_resnet50': (False, (1, 1, 1), False),
            'rf67_resnet50': (False, (2, 1, 2), False),
            'rf99_resnet50': (False, (2, 2, 2), False),
            'rf115_resnet50': (True, (2, 1, 1), False),
            'rf163_resnet50': (True, (2, 2, 1), False),
            'rf195_paper193_resnet50': (True, (2, 2, 2), False),
            'rf29_resnet18': (False, (1, 2, 2), True),
            'bagnet33': (False, (2, 2, 1), False),
        }
        piscina, pasos, basico = perfiles[nombre]
        bolsa = nombre == 'bagnet33'
        if bolsa:
            self.inicio = nn.Sequential(nn.Conv2d(3, 64, 1, bias=False), nn.Conv2d(64, 64, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        else:
            kernel_inicial, paso_inicial = (3, 1) if basico else (7, 2)
            self.inicio = nn.Sequential(nn.Conv2d(3, 64, kernel_inicial, paso_inicial, kernel_inicial // 2, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, 2, 1) if piscina else nn.Identity())
        cantidades = (4, 4, 4, 4) if basico else (3, 4, 6, 3 if bolsa else 4)
        entrada = 64
        etapas = []
        for indice_etapa, (ancho, cantidad, paso) in enumerate(zip((64, 128, 256, 512), cantidades, (2, 2, 2, 1) if bolsa else (1, *pasos))):
            bloques = []
            for indice_bloque in range(cantidad):
                kernel = 3 if indice_bloque in ((0,) if bolsa else (0, cantidad // 2)) else 1
                bloques.append(BloqueEspacial(entrada, ancho, kernel, paso if indice_bloque == 0 else 1, basico, bolsa))
                entrada = ancho if basico else ancho * 4
            etapas.append(nn.Sequential(*bloques))
        self.etapas = nn.ModuleList(etapas)
        self.dimension = entrada
        self.canales_etapas = [64, 128, 256, 512] if basico else [256, 512, 1024, 2048]
        self.mlp = nn.Identity() if quitar_mlp or bolsa else MLPResidual(entrada, basico)

    def calcular_etapa(self, etapa, entrada):
        for bloque in etapa:
            entrada = ejecutar_memoria(bloque, entrada, self.memoria_reducida)
        return entrada

    def mapas(self, imagenes):
        actual = self.inicio(imagenes)
        mapas = []
        for etapa in self.etapas:
            actual = ejecutar_memoria(etapa, actual, self.memoria_reducida, lambda x, etapa=etapa: self.calcular_etapa(etapa, x))
            mapas.append(actual)
        return mapas


class Encoder(nn.Module):
    def __init__(self, configuracion):
        super().__init__()
        self.nombre = configuracion['arquitectura']
        self.capa_local = configuracion.get('capa_local', 4)
        self.rf = self.nombre.startswith('rf') or self.nombre == 'bagnet33'
        self.transformer = self.nombre.startswith(('vit', 'swin'))
        if self.rf:
            self.red = RFResNet(self.nombre, configuracion.get('sin_mlp', False), configuracion.get('checkpoint_activaciones', False))
            self.dimension = self.red.dimension
            self.dimension_local = self.red.canales_etapas[self.capa_local - 1]
        elif self.transformer:
            nombres = {'vit_s16': 'vit_small_patch16_224', 'swin_t': 'swin_tiny_patch4_window7_224', 'swin_s': 'swin_small_patch4_window7_224', 'swin_b': 'swin_base_patch4_window7_224'}
            self.red = timm.create_model(nombres[self.nombre], pretrained=False, num_classes=0, drop_path_rate=0.1 if self.nombre == 'swin_b' else 0.0)
            self.dimension = self.red.num_features
            self.dimension_local = self.dimension
        else:
            self.red = models.resnet18(weights=None) if self.nombre == 'resnet18' else models.resnet50(weights=None)
            if self.nombre == 'resnet18':
                self.red.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
                self.red.maxpool = nn.Identity()
            self.dimension = self.red.fc.in_features
            self.dimension_local = ([64, 128, 256, 512] if self.nombre == 'resnet18' else [256, 512, 1024, 2048])[self.capa_local - 1]
            self.red.fc = nn.Identity()

    def mapas(self, imagenes):
        if self.rf:
            return self.red.mapas(imagenes)
        if self.transformer:
            salida = self.red.forward_features(imagenes)
            if salida.ndim == 3:
                tokens = salida[:, self.red.num_prefix_tokens:]
                lado = math.isqrt(tokens.shape[1])
                return [tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], lado, lado)]
            return [salida.permute(0, 3, 1, 2).contiguous()]
        actual = self.red.maxpool(self.red.relu(self.red.bn1(self.red.conv1(imagenes))))
        mapas = []
        for etapa in [self.red.layer1, self.red.layer2, self.red.layer3, self.red.layer4]:
            actual = etapa(actual)
            mapas.append(actual)
        return mapas

    def agregar(self, mapa, antes_mlp=False, mascara=0.0):
        if mascara:
            posiciones = mapa.shape[2] * mapa.shape[3]
            cantidad = max(1, round(posiciones * (1 - mascara)))
            elegidas = torch.rand(mapa.shape[0], posiciones, device=mapa.device).topk(cantidad, dim=1, sorted=False).indices
            conservar = torch.zeros(mapa.shape[0], posiciones, dtype=torch.bool, device=mapa.device).scatter_(1, elegidas, True).view(mapa.shape[0], 1, *mapa.shape[2:])
            globales = (mapa * conservar).sum((2, 3)) / cantidad
        else:
            globales = mapa.mean((2, 3))
        if self.rf and not antes_mlp:
            globales = self.red.mlp(globales)
        return globales

    def forward(self, imagenes):
        mapas = self.mapas(imagenes)
        locales = mapas[self.capa_local - 1] if len(mapas) == 4 else mapas[-1]
        return locales, self.agregar(mapas[-1])


def proyector_global(entrada, ocultas=2, dimension=256, normalizacion_final=False, sesgo_final=True):
    capas = []
    for _ in range(ocultas):
        capas.extend([nn.Linear(entrada, 4096, bias=False), nn.BatchNorm1d(4096), nn.ReLU()])
        entrada = 4096
    capas.append(nn.Linear(entrada, dimension, bias=sesgo_final))
    if normalizacion_final:
        capas.append(nn.BatchNorm1d(dimension, affine=False))
    return nn.Sequential(*capas)


class Rama(nn.Module):
    def __init__(self, configuracion):
        super().__init__()
        self.encoder = Encoder(configuracion)
        self.memoria_reducida = configuracion.get('checkpoint_activaciones', False)
        moco = configuracion['metodo'] == 'mocov3'
        self.globales = proyector_global(self.encoder.dimension, 1 if moco and not self.encoder.transformer else 2, normalizacion_final=moco, sesgo_final=not moco)
        self.prediccion_global = proyector_global(256, 1, normalizacion_final=moco and self.encoder.transformer, sesgo_final=not moco)
        self.co = configuracion.get('peso_local', 0) > 0
        self.compartida = configuracion.get('cabeza_compartida', False)
        self.cantidad_locales = configuracion.get('cantidad_locales')
        if self.co and not self.compartida:
            kernel = configuracion.get('kernel_local', 1)
            self.locales = nn.Sequential(nn.Conv2d(self.encoder.dimension_local, 4096, kernel, bias=False), nn.BatchNorm2d(4096), nn.ReLU(), nn.Conv2d(4096, 256, 1))
            self.prediccion_local = nn.Sequential(nn.Conv2d(256, 4096, 1, bias=False), nn.BatchNorm2d(4096), nn.ReLU(), nn.Conv2d(4096, 256, 1))

    def proyectar_locales(self, mapa, predecir):
        if self.cantidad_locales:
            lado = math.isqrt(self.cantidad_locales)
            filas = torch.linspace(0, mapa.shape[2] - 1, min(lado, mapa.shape[2]), device=mapa.device).round().long()
            columnas = torch.linspace(0, mapa.shape[3] - 1, min(lado, mapa.shape[3]), device=mapa.device).round().long()
            mapa = mapa.index_select(2, filas).index_select(3, columnas)
        if self.compartida:
            batch, canales, alto, ancho = mapa.shape
            proyectadas = self.globales(mapa.permute(0, 2, 3, 1).reshape(-1, canales))
            predichas = self.prediccion_global(proyectadas) if predecir else None
            proyectadas = proyectadas.reshape(batch, alto * ancho, -1).transpose(1, 2)
            predichas = predichas.reshape(batch, alto * ancho, -1).transpose(1, 2) if predecir else None
        else:
            proyectadas_mapa = ejecutar_memoria(self.locales, mapa, self.memoria_reducida)
            proyectadas = proyectadas_mapa.flatten(2)
            predichas = ejecutar_memoria(self.prediccion_local, proyectadas_mapa, self.memoria_reducida).flatten(2) if predecir else None
        return proyectadas, predichas

    def forward(self, imagenes, predecir=True):
        locales, globales = self.encoder(imagenes)
        proyeccion = self.globales(globales)
        prediccion = self.prediccion_global(proyeccion) if predecir else None
        proyeccion_local, prediccion_local = self.proyectar_locales(locales, predecir) if self.co else (None, None)
        return proyeccion, prediccion, proyeccion_local, prediccion_local


def perdida_coseno(prediccion, objetivo):
    return -2 * F.cosine_similarity(prediccion.float(), objetivo.detach().float(), dim=1).mean()


def perdida_contrastiva(prediccion, objetivo, temperatura):
    prediccion = F.normalize(prediccion.float(), dim=1)
    objetivo = F.normalize(objetivo.detach().float(), dim=1)
    if prediccion.ndim == 2 and objetivo.ndim == 2:
        logits = prediccion @ objetivo.T
        etiquetas = torch.arange(logits.shape[0], device=logits.device)
    else:
        logits = torch.einsum('bdk,cd->kbc', prediccion, objetivo) if prediccion.ndim == 3 else torch.einsum('bd,cdk->kbc', prediccion, objetivo)
        etiquetas = torch.arange(logits.shape[1], device=logits.device).repeat(logits.shape[0])
        logits = logits.reshape(-1, logits.shape[-1])
    return 2 * temperatura * F.cross_entropy(logits / temperatura, etiquetas)


class SSL(nn.Module):
    def __init__(self, configuracion):
        super().__init__()
        self.configuracion = configuracion
        self.online = Rama(configuracion)
        self.objetivo = copy.deepcopy(self.online)
        self.objetivo.prediccion_global = nn.Identity()
        if hasattr(self.objetivo, 'prediccion_local'):
            self.objetivo.prediccion_local = nn.Identity()
        for parametro in self.objetivo.parameters():
            parametro.requires_grad = False
        self.temperatura = configuracion.get('temperatura', 0.2)

    def comparar(self, prediccion, objetivo):
        if self.configuracion['metodo'] == 'mocov3':
            return perdida_contrastiva(prediccion, objetivo, self.temperatura)
        if prediccion.ndim != objetivo.ndim:
            prediccion = prediccion.unsqueeze(-1) if prediccion.ndim == 2 else prediccion
            objetivo = objetivo.unsqueeze(-1) if objetivo.ndim == 2 else objetivo
        return perdida_coseno(prediccion, objetivo)

    def forward(self, vistas):
        online = [ejecutar_memoria(self.online, vista, self.configuracion.get('checkpoint_rama', False)) for vista in vistas]
        with torch.no_grad():
            objetivo = [self.objetivo(vista, False) for vista in vistas[:2]]
        globales = [self.comparar(salida[1], objetivo[j][0]) for i, salida in enumerate(online) for j in range(2) if i != j]
        perdida_global = torch.stack(globales).mean() * 2
        perdida_local = perdida_global.new_zeros(())
        if self.configuracion.get('peso_local', 0):
            terminos = [self.comparar(online[i][3], objetivo[1 - i][0]) + self.comparar(online[i][1], objetivo[1 - i][2]) for i in range(2)]
            perdida_local = torch.stack(terminos).sum()
        return perdida_global + self.configuracion.get('peso_local', 0) * perdida_local, {'global': perdida_global.detach(), 'local': perdida_local.detach()}

    @torch.no_grad()
    def actualizar_objetivo(self, momentum):
        parametros_online = dict(self.online.named_parameters())
        for nombre, destino in self.objetivo.named_parameters():
            destino.lerp_(parametros_online[nombre], 1 - momentum)


class CabezaDINO(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dimension, 2048), nn.GELU(), nn.Linear(2048, 2048), nn.GELU(), nn.Linear(2048, 256))
        self.pesos = nn.Parameter(torch.randn(65536, 256) * 0.02)
        self.escala = nn.Parameter(torch.ones(65536, 1), requires_grad=False)

    def forward(self, representacion):
        return F.linear(F.normalize(self.mlp(representacion), dim=-1), F.normalize(self.pesos, dim=1) * self.escala)


class RamaDINO(nn.Module):
    def __init__(self, configuracion):
        super().__init__()
        self.encoder = Encoder(configuracion)
        self.cabeza = CabezaDINO(self.encoder.dimension)

    def forward(self, imagenes):
        return self.cabeza(self.encoder(imagenes)[1])


class DINO(nn.Module):
    def __init__(self, configuracion):
        super().__init__()
        self.configuracion = configuracion
        self.online = RamaDINO(configuracion)
        self.objetivo = copy.deepcopy(self.online)
        for parametro in self.objetivo.parameters():
            parametro.requires_grad = False
        self.register_buffer('centro', torch.zeros(1, 65536))
        self.temperatura_objetivo = 0.04

    def forward(self, vistas):
        estudiante = [F.log_softmax(self.online(v).float() / 0.1, dim=-1) for v in vistas]
        with torch.no_grad():
            logits = [self.objetivo(v).float() for v in vistas[:2]]
            profesor = [F.softmax((x - self.centro) / self.temperatura_objetivo, dim=-1) for x in logits]
        terminos = [-(profesor[j] * salida).sum(-1).mean() for i, salida in enumerate(estudiante) for j in range(2) if i != j]
        perdida = torch.stack(terminos).mean()
        with torch.no_grad():
            self.centro.lerp_(torch.cat(logits).mean(0, keepdim=True), 0.1)
        return perdida, {'global': perdida.detach(), 'local': perdida.new_zeros(())}

    @torch.no_grad()
    def actualizar_objetivo(self, momentum):
        for origen, destino in zip(self.online.parameters(), self.objetivo.parameters()):
            destino.lerp_(origen, 1 - momentum)


def construir_modelo(configuracion):
    return DINO(configuracion) if configuracion['metodo'] == 'dino' else SSL(configuracion)


class EncodersRecuperacion(nn.Module):
    def __init__(self, configuracion):
        super().__init__()
        self.online = nn.Module()
        self.online.encoder = Encoder(configuracion)
        self.objetivo = nn.Module()
        self.objetivo.encoder = Encoder(configuracion)
