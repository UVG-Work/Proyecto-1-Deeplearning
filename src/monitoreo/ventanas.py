"""Ventanas deslizantes representadas como INDICES, no como tensores.

Guardar indices en vez de floats hace que la prueba de permutacion preserve
el contenido por construccion: barajar es permutar enteros dentro de una
fila, y es imposible que altere QUE eventos hay en la ventana.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def construir(df: pd.DataFrame, K: int = cfg.K) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (win_idx, mask).

    `df` debe venir ordenado por (card_id, ts). Los indices son POSICIONALES
    respecto de `df`, de 0 a len(df)-1.
    """
    n = len(df)
    pos_global = np.arange(n, dtype=np.int32)
    pos_en_tarjeta = df.reset_index(drop=True).groupby("card_id").cumcount().to_numpy()

    desplazamientos = np.arange(K - 1, -1, -1, dtype=np.int32)      # K-1 ... 0
    candidatos = pos_global[:, None] - desplazamientos[None, :]
    disponible = pos_en_tarjeta[:, None] >= desplazamientos[None, :]

    win = np.where(disponible, candidatos, pos_global[:, None]).astype(np.int32)
    return win, disponible
