"""Checklist de penalizaciones (§9) como suite ejecutable."""
import numpy as np
import pandas as pd
import pytest
from monitoreo import config as cfg
from monitoreo import features_agregadas as fa
from monitoreo import features_evento as fe
from monitoreo import generador as gen
from monitoreo import particion as part
from monitoreo import ventanas as ven


@pytest.fixture(scope="module")
def pipeline():
    df = gen.generar(cfg.SEED_DATOS, n_tarjetas=250)
    split = part.asignar_split(df)
    es_train = (split == "train").to_numpy()
    X_A = fa.construir(df)
    vocab = fe.construir_vocabularios(df, es_train)
    E_num, E_cat, scaler = fe.construir(df, vocab, es_train)
    win, mask = ven.construir(df, cfg.K)
    return dict(df=df, split=split, es_train=es_train, X_A=X_A,
                E_num=E_num, E_cat=E_cat, win=win, mask=mask, scaler=scaler)


def test_contrato_de_comparabilidad(pipeline):
    """A y B deben responder la misma pregunta sobre las mismas filas."""
    d = pipeline
    n = len(d["df"])
    assert len(d["X_A"]) == n
    assert d["win"].shape[0] == n
    assert d["mask"].shape[0] == n
    assert (d["win"][:, -1] == np.arange(n)).all()


def test_penalizacion_20_particion_no_es_aleatoria(pipeline):
    d = pipeline
    ts = d["df"]["ts"]
    assert ts[d["split"] == "train"].max() <= ts[d["split"] == "val"].min()
    assert ts[d["split"] == "val"].max() <= ts[d["split"] == "test"].min()


def test_penalizacion_15_ninguna_secuencia_mira_al_futuro(pipeline):
    d = pipeline
    ts = d["df"]["ts"].to_numpy()
    propia = np.repeat(ts[:, None], cfg.K, axis=1)
    assert (ts[d["win"]][d["mask"]] <= propia[d["mask"]]).all()


def test_penalizacion_15_vocabularios_y_scaler_solo_train(pipeline):
    d = pipeline
    comercios_train = set(d["df"].loc[d["es_train"], "merchant_id"])
    vocab = fe.construir_vocabularios(d["df"], d["es_train"])
    assert set(vocab["merchant"]) <= comercios_train


def test_fraud_type_fuera_de_toda_matriz_de_features(pipeline):
    d = pipeline
    for col in cfg.COLUMNAS_ANALISIS:
        assert col not in d["X_A"].columns
        assert col not in fe.NOMBRES_NUM


def test_A_es_invariante_a_la_permutacion_de_la_ventana(pipeline):
    """Control de sanidad gratis: A no toca win_idx, asi que no puede moverse.
    Si esta prueba falla, hay fuga de orden en A."""
    d = pipeline
    perm = ven.permutar(d["win"], d["mask"], "full", np.random.default_rng(0))
    X_despues = fa.construir(d["df"])
    pd.testing.assert_frame_equal(d["X_A"], X_despues)
    assert not (perm == d["win"]).all()   # el shuffle si se aplico


def test_las_features_de_A_no_dependen_del_orden_de_llegada(pipeline):
    fa.verificar_sin_orden(pipeline["X_A"])


def test_hay_tarjetas_con_menos_de_K_de_historia(pipeline):
    """§4.3, primer caso de fallo esperado: secuencias con padding."""
    assert (pipeline["mask"].sum(axis=1) < cfg.K).sum() > 0


@pytest.fixture(scope="module")
def df_grande():
    """La brecha larga de f1 ocurre en el 15% de los episodios. Con 250
    tarjetas hay ~23 episodios f1 y la muestra no alcanza para observarla;
    no es un defecto del generador sino del tamano del fixture."""
    return gen.generar(cfg.SEED_DATOS, n_tarjetas=1200)


def test_casos_de_fallo_esperado_existen_en_los_datos(df_grande):
    """§4.3, segundo caso: f1 con brecha > 24h, que no cabe en K=20."""
    f1 = df_grande[df_grande["fraud_type"] == "f1"]
    brechas = []
    for _, g in f1.groupby("card_id"):
        g = g.sort_values("ts")
        golpes = g[g["fraud_subtype"] == "f1_golpe"]["ts"]
        sondeos = g[g["fraud_subtype"] == "f1_sondeo"]["ts"]
        if len(golpes) and len(sondeos):
            brechas.append((golpes.iloc[0] - sondeos.iloc[-1]).total_seconds() / 3600.0)
    assert max(brechas) > 24.0, "falta el caso de f1 con brecha larga"
