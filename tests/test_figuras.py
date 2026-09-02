from pathlib import Path
import numpy as np
from monitoreo import figuras


def test_curvas_pr_guarda_png(tmp_path):
    ruta = tmp_path / "pr.png"
    y = np.array([0, 1] * 50)
    figuras.curvas_pr({"A": (y, np.random.rand(100)), "B": (y, np.random.rand(100))}, ruta)
    assert ruta.exists() and ruta.stat().st_size > 1000


def test_auc_vs_k_guarda_png(tmp_path):
    ruta = tmp_path / "k.png"
    figuras.auc_vs_k([1, 3, 5, 10, 20], [0.2, 0.3, 0.4, 0.45, 0.46], 0.25, ruta)
    assert ruta.exists()


def test_curva_costo_marca_los_umbrales(tmp_path):
    ruta = tmp_path / "costo.png"
    u = np.linspace(0, 1, 100)
    figuras.curva_costo({"A": (u, u * 1000 + 500), "B": (u, u * 800 + 400)},
                        {"A": 0.04, "B": 0.05}, ruta)
    assert ruta.exists()
