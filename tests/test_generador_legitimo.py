import numpy as np
import pandas as pd
import pytest
from monitoreo import config as cfg
from monitoreo import generador as gen

COLUMNAS = [
    "card_id", "ts", "amount", "merchant_id", "mcc",
    "channel", "country", "is_fraud", "fraud_type", "fraud_subtype",
]


@pytest.fixture(scope="module")
def flujo():
    rng = np.random.default_rng(7)
    perf = gen.perfiles(rng, 60)
    return gen.flujo_legitimo(rng, perf)


def test_esquema_completo(flujo):
    assert list(flujo.columns) == COLUMNAS
    assert pd.api.types.is_datetime64_any_dtype(flujo["ts"])
    assert (flujo["amount"] > 0).all()


def test_todo_es_legitimo(flujo):
    assert (flujo["is_fraud"] == 0).all()
    assert (flujo["fraud_type"] == "none").all()
    assert (flujo["fraud_subtype"] == "none").all()


def test_conteo_por_tarjeta_dentro_del_rango(flujo):
    n = flujo.groupby("card_id").size()
    assert n.min() >= cfg.TX_MIN
    assert n.max() <= cfg.TX_MAX
    # la lognormal truncada debe centrarse cerca de TX_MEDIA, no en 130
    assert abs(n.mean() - cfg.TX_MEDIA) < 15


def test_ts_estrictamente_creciente_dentro_de_cada_tarjeta(flujo):
    for _, g in flujo.groupby("card_id"):
        assert g["ts"].is_monotonic_increasing
        assert g["ts"].is_unique


def test_ordenado_por_tarjeta_y_tiempo(flujo):
    esperado = flujo.sort_values(["card_id", "ts"], kind="mergesort")
    pd.testing.assert_frame_equal(flujo, esperado)


def test_hora_del_dia_es_diurna(flujo):
    horas = flujo["ts"].dt.hour
    # mas actividad entre 8 y 22 que en la madrugada
    assert (horas.between(8, 22)).mean() > 0.75


def test_reproducible_por_semilla():
    a = gen.flujo_legitimo(np.random.default_rng(3), gen.perfiles(np.random.default_rng(3), 20))
    b = gen.flujo_legitimo(np.random.default_rng(3), gen.perfiles(np.random.default_rng(3), 20))
    pd.testing.assert_frame_equal(a, b)
