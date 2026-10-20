import math
import time

import numpy as np
import torch

from codigo.datos import verificar_particiones
from codigo.evaluacion import calcular_metricas
from codigo.modelos import perdida_contrastiva, perdida_coseno


def verificar(carpeta):
    inicio = time.perf_counter()
    estructura = verificar_particiones(carpeta)
    prediccion = torch.eye(3, requires_grad=True)
    objetivo = torch.eye(3, requires_grad=True)
    coseno = perdida_coseno(prediccion, objetivo)
    assert torch.allclose(coseno, torch.tensor(-2.0))
    coseno.backward()
    assert objetivo.grad is None
    temperatura = 0.2
    esperado = 2 * temperatura * (math.log(math.exp(1 / temperatura) + 2) - 1 / temperatura)
    contrastiva = perdida_contrastiva(torch.eye(3), torch.eye(3), temperatura)
    assert math.isclose(float(contrastiva), esperado, abs_tol=1e-6)
    locales = torch.eye(3).unsqueeze(-1).repeat(1, 1, 4)
    assert torch.allclose(perdida_contrastiva(locales, torch.eye(3), temperatura), contrastiva)
    assert torch.allclose(perdida_contrastiva(torch.eye(3), locales, temperatura), contrastiva)
    invertida = perdida_contrastiva(torch.eye(3), torch.eye(3).roll(1, 0), temperatura)
    assert invertida > contrastiva
    metricas = calcular_metricas(np.array([[0.99, 0.98, 0.97, 0.96, 0.95, 0.94]]), np.array([1]), np.array([1]), np.array([-1, 1, 0, 1, 2, 1]), np.array([2, 1, 3, 2, 4, 3]))
    assert metricas['mAP'] == 0.5
    assert metricas['Rank-1'] == 0 and metricas['Rank-5'] == 1 and metricas['Rank-10'] == 1
    return {'estado': 'aprobado', 'estructura': estructura, 'comprobaciones': ['particiones originales y tipos', 'coseno con valor conocido y stop-gradient', 'InfoNCE con valor analítico', 'promedio espacial de InfoNCE en ambas direcciones', 'positivo contrastivo mal asignado aumenta pérdida', 'AP y Rank-k con junk, distractor y misma cámara'], 'segundos': time.perf_counter() - inicio, 'entrenamientos': 0}
