"""Semillas y captura de versiones para el README."""
from __future__ import annotations

import os
import platform
import random
import sys


def fijar_semillas(seed: int) -> None:
    """Fija numpy, random, y (si esta importado) TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    try:
        import tensorflow as tf
    except ImportError:
        return
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def versiones() -> dict[str, str]:
    """Versiones exactas para el README y artefactos/config.json."""
    import importlib.metadata as md

    v = {
        "python": sys.version.split()[0],
        "sistema": f"{platform.system()} {platform.release()}",
    }
    for paquete in ("numpy", "pandas", "scikit-learn", "tensorflow", "keras", "lightgbm"):
        try:
            v[paquete] = md.version(paquete)
        except md.PackageNotFoundError:
            v[paquete] = "no instalado"
    return v
