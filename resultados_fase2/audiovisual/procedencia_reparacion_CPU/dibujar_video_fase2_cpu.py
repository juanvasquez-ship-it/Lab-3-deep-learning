import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
phase2 = root / 'outputs/Laboratorio3/resultados_fase2'
rows = json.loads((phase2 / 'comparacion_fase2.json').read_text(encoding='utf-8'))
rows = [row for row in rows if row['grupo'] == 'arquitecturas' and row['estado'] == 'evaluado']
assert len(rows) == 14
assets = phase2 / 'audiovisual/resultados'
assets.mkdir(parents=True, exist_ok=True)
figure, axis = plt.subplots(figsize=(12, 7))
positions = np.arange(14)
axis.barh(positions - 0.18, [r['original_mAP_porcentaje'] for r in rows], 0.35, label='Original')
axis.barh(positions + 0.18, [r['baja_resolucion_mAP_porcentaje'] for r in rows], 0.35, label='Consulta LR')
axis.set_yticks(positions, [r['arquitectura'] for r in rows])
axis.set_xlabel('mAP %')
axis.set_title('Fase 2, época y rama elegidas por validación')
axis.invert_yaxis()
axis.legend()
figure.tight_layout()
figure.savefig(assets / 'comparacion.png', dpi=150)
plt.close(figure)
