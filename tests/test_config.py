import os
import pytest
from monitoreo import config as cfg
from monitoreo import reproducibilidad as rep


def test_parametros_congelados():
    assert cfg.SEED_DATOS == 20260831
    assert cfg.SEEDS_MODELO == (7, 13, 29)
    assert cfg.K == 20
    assert cfg.PCT_TRAIN == 0.70
    assert cfg.PCT_VAL == 0.85
    assert cfg.COSTO_FN == 4200.0
    assert cfg.COSTO_FP == 180.0
    assert cfg.PAD == 0 and cfg.UNK == 1


def test_umbral_teorico_coincide_con_los_costos():
    assert cfg.UMBRAL_TEORICO == pytest.approx(cfg.COSTO_FP / cfg.COSTO_FN)
    assert cfg.UMBRAL_TEORICO == pytest.approx(0.042857, abs=1e-5)


def test_dev_mode_reduce_las_tarjetas(monkeypatch):
    monkeypatch.delenv("MONITOREO_DEV", raising=False)
    assert cfg.n_tarjetas() == 4000
    monkeypatch.setenv("MONITOREO_DEV", "1")
    assert cfg.n_tarjetas() == 400


def test_fijar_semillas_hace_reproducible_a_numpy():
    import numpy as np
    rep.fijar_semillas(123)
    a = np.random.rand(5)
    rep.fijar_semillas(123)
    b = np.random.rand(5)
    assert (a == b).all()


def test_versiones_reporta_las_librerias_del_informe():
    v = rep.versiones()
    for clave in ("python", "numpy", "pandas", "scikit-learn", "tensorflow", "keras", "lightgbm"):
        assert clave in v and v[clave]
