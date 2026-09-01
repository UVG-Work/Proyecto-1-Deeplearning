import numpy as np
import pytest
from monitoreo import config as cfg
from monitoreo import features_agregadas as fa
from monitoreo import generador as gen
from monitoreo import metricas as met
from monitoreo import modelos_a as ma
from monitoreo import particion as part


@pytest.fixture(scope="module")
def datos():
    df = gen.generar(cfg.SEED_DATOS, n_tarjetas=300)
    X = fa.construir(df).to_numpy()
    y = df["is_fraud"].to_numpy()
    s = part.asignar_split(df).to_numpy()
    return X, y, s


def test_logistica_supera_el_azar(datos):
    X, y, s = datos
    m = ma.entrenar_logistica(X[s == "train"], y[s == "train"], seed=7)
    p = ma.predecir(m, X[s == "val"])
    assert met.auc_pr(y[s == "val"], p) > y[s == "val"].mean() * 2


def test_gbm_supera_a_la_logistica(datos):
    X, y, s = datos
    log = ma.entrenar_logistica(X[s == "train"], y[s == "train"], seed=7)
    gbm = ma.entrenar_gbm(X[s == "train"], y[s == "train"], X[s == "val"], y[s == "val"], seed=7)
    yv = y[s == "val"]
    assert met.auc_pr(yv, ma.predecir(gbm, X[s == "val"])) >= \
           met.auc_pr(yv, ma.predecir(log, X[s == "val"]))


def test_predecir_devuelve_probabilidades(datos):
    X, y, s = datos
    m = ma.entrenar_logistica(X[s == "train"], y[s == "train"], seed=7)
    p = ma.predecir(m, X[s == "val"])
    assert p.shape == (int((s == "val").sum()),)
    assert (p >= 0).all() and (p <= 1).all()


def test_reproducible_por_semilla(datos):
    X, y, s = datos
    a = ma.entrenar_gbm(X[s == "train"], y[s == "train"], X[s == "val"], y[s == "val"], seed=13)
    b = ma.entrenar_gbm(X[s == "train"], y[s == "train"], X[s == "val"], y[s == "val"], seed=13)
    assert np.allclose(ma.predecir(a, X[s == "val"]), ma.predecir(b, X[s == "val"]))


def test_no_ve_el_conjunto_de_test(datos):
    """El entrenamiento no recibe test por firma: no hay parametro para el."""
    import inspect
    for fn in (ma.entrenar_logistica, ma.entrenar_gbm):
        assert "test" not in inspect.signature(fn).parameters
