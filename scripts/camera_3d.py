"""Converte o pixel que o modelo marcou em um ponto 3D no mundo.

A textura de posição do ManiSkill já traz x, y, z da câmera em
milímetros. Multiplicar pela matriz cam2world, como o exemplo oficial
de nuvem de pontos, entrega o ponto em metros.
"""

import numpy as np
import torch


def nuvem_mundo(position, cam2world):
    pos = position.detach().to(dtype=torch.float32).cpu()
    matriz = cam2world.detach().to(dtype=torch.float32).cpu()
    if pos.ndim == 4:
        pos = pos[0]
    if matriz.ndim == 3:
        matriz = matriz[0]
    pos = pos.numpy()
    matriz = matriz.numpy()
    # No OpenGL a câmera olha para -z. Profundidade útil fica entre 5 cm e 2 m.
    profundidade = -pos[..., 2]
    valido = (profundidade > 50) & (profundidade < 2000)
    metros = pos / 1000.0
    homo = np.concatenate([metros, valido[..., None].astype(np.float32)], axis=-1)
    mundo = homo @ matriz.T
    mundo = mundo[..., :3]
    mundo[~valido] = np.nan
    return mundo


def ponto_da_caixa(nuvem, caixa):
    altura, largura = nuvem.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in caixa]
    x1, x2 = sorted((max(0, x1), min(largura, x2)))
    y1, y2 = sorted((max(0, y1), min(altura, y2)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise RuntimeError(f"Caixa pequena demais: {(x1, y1, x2, y2)}")
    recorte = nuvem[y1:y2, x1:x2]
    validos = recorte[np.isfinite(recorte).all(axis=-1)]
    if len(validos) < 10:
        raise RuntimeError("A caixa não cobriu pixels com profundidade.")
    # Pixels da mesa ficam quase em z=0. O objeto é o que está acima dela.
    acima = validos[validos[:, 2] > 0.012]
    if len(acima) >= 10:
        validos = acima
    xy = np.median(validos[:, :2], axis=0)
    z_alto = float(np.percentile(validos[:, 2], 90))
    z_baixo = float(np.percentile(validos[:, 2], 10))
    # A profundidade vê a casca. Descer metade da altura visível
    # coloca a garra no miolo, não só no topo.
    folga = max((z_alto - z_baixo) * 0.5, 0.012)
    ponto = np.array([xy[0], xy[1], max(z_alto - folga, 0.01)])
    return ponto, _eixo_curto(validos)


def _eixo_curto(pontos):
    """Direção em que os dedos devem fechar: perpendicular ao lado longo do objeto."""
    xy = pontos[:, :2] - np.median(pontos[:, :2], axis=0)
    cov = xy.T @ xy
    _, vetores = np.linalg.eigh(cov)
    longo = vetores[:, 1]
    curto = np.array([-longo[1], longo[0], 0.0])
    norma = np.linalg.norm(curto)
    if norma < 1e-6:
        return np.array([1.0, 0.0, 0.0])
    return curto / norma
