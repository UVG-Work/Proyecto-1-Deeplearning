import numpy as np
import pytest
from monitoreo import config as cfg
from monitoreo import features_evento as fe
from monitoreo import generador as gen
from monitoreo import metricas as met
from monitoreo import modelos_b as mb
from monitoreo import particion as part
from monitoreo import ventanas as ven


@pytest.fixture(scope="module")
def datos():
    df = gen.generar(cfg.SEED_DATOS, n_tarjetas=250)
    s = part.asignar_split(df).to_numpy()
    es_train = s == "train"
    vocab = fe.construir_vocabularios(df, es_train)
    E_num, E_cat, _ = fe.construir(df, vocab, es_train)
    win, mask = ven.construir(df, cfg.K)
    y = df["is_fraud"].to_numpy()
    return dict(df=df, s=s, vocab=vocab, E_num=E_num, E_cat=E_cat,
                win=win, mask=mask, y=y)


def test_lotes_hacen_gather_correcto(datos):
    d = datos
    lotes = mb.Lotes(d["win"][:64], d["mask"][:64], d["E_num"], d["E_cat"],
                     y=d["y"][:64], batch_size=32, barajar=False)
    x, y = lotes[0]
    assert x["num"].shape == (32, cfg.K, d["E_num"].shape[1])
    assert x["mcc"].shape == (32, cfg.K)
    assert x["mask"].shape == (32, cfg.K)
    assert y.shape == (32,)
    # la ultima posicion de la ventana es el propio evento
    assert np.allclose(x["num"][:, -1, :], d["E_num"][d["win"][:32, -1]])


def test_lotes_cubren_todas_las_muestras_sin_barajar(datos):
    d = datos
    n = 100
    lotes = mb.Lotes(d["win"][:n], d["mask"][:n], d["E_num"], d["E_cat"],
                     y=d["y"][:n], batch_size=32, barajar=False)
    total = sum(len(lotes[i][1]) for i in range(len(lotes)))
    assert total == n


def _nombres_de_entrada(m):
    return {t.name.split(":")[0] for t in m.inputs}


def test_modelo_tiene_las_entradas_esperadas(datos):
    d = datos
    m = mb.construir_modelo(cfg.K, d["E_num"].shape[1], fe.cardinalidades(d["vocab"]))
    assert _nombres_de_entrada(m) == {"num", "mcc", "channel", "merchant", "mask"}
    assert m.output_shape == (None, 1)


def test_hibrido_agrega_la_entrada_de_agregados(datos):
    d = datos
    m = mb.construir_modelo(cfg.K, d["E_num"].shape[1], fe.cardinalidades(d["vocab"]), d_agg=11)
    assert "agg" in _nombres_de_entrada(m)
    agg = [t for t in m.inputs if t.name.split(":")[0] == "agg"][0]
    assert tuple(agg.shape) == (None, 11)


def test_entrena_y_supera_el_azar(datos):
    d = datos
    tr, va = d["s"] == "train", d["s"] == "val"
    lotes_tr = mb.Lotes(d["win"][tr], d["mask"][tr], d["E_num"], d["E_cat"],
                        y=d["y"][tr], batch_size=256, barajar=True)
    lotes_va = mb.Lotes(d["win"][va], d["mask"][va], d["E_num"], d["E_cat"],
                        y=d["y"][va], batch_size=256)
    m = mb.construir_modelo(cfg.K, d["E_num"].shape[1], fe.cardinalidades(d["vocab"]))
    mb.entrenar(m, lotes_tr, lotes_va, mb.pesos_de_clase(d["y"][tr]), epocas=3)
    p = mb.predecir(m, lotes_va)
    assert p.shape == (int(va.sum()),)
    assert met.auc_pr(d["y"][va], p) > d["y"][va].mean()


def test_pesos_de_clase_favorecen_la_minoritaria(datos):
    y = np.array([0] * 990 + [1] * 10)
    w = mb.pesos_de_clase(y)
    assert w[1] > w[0]
    assert w[1] / w[0] == pytest.approx(99.0, rel=0.01)


def test_el_modelo_expone_auc_pr_como_metrica(datos):
    """EarlyStopping vigila val_auc_pr, nunca val_loss ni nada de test.

    En Keras 3 `model.metrics` solo lista `compile_metrics` hasta que el
    modelo se construye, asi que se verifica el contrato de verdad: que
    `fit` emita la clave que el callback monitorea.
    """
    d = datos
    tr, va = d["s"] == "train", d["s"] == "val"
    lotes_tr = mb.Lotes(d["win"][tr][:512], d["mask"][tr][:512], d["E_num"],
                        d["E_cat"], y=d["y"][tr][:512], batch_size=256)
    lotes_va = mb.Lotes(d["win"][va][:512], d["mask"][va][:512], d["E_num"],
                        d["E_cat"], y=d["y"][va][:512], batch_size=256)
    m = mb.construir_modelo(cfg.K, d["E_num"].shape[1], fe.cardinalidades(d["vocab"]))
    h = mb.entrenar(m, lotes_tr, lotes_va, mb.pesos_de_clase(d["y"][tr]), epocas=1)
    assert "val_auc_pr" in h.history and "auc_pr" in h.history


def test_early_stopping_monitorea_validacion_no_perdida():
    """El monitor es AUC-PR de validacion, no val_loss (spec 4.2)."""
    import inspect
    fuente = inspect.getsource(mb.entrenar)
    assert 'monitor="val_auc_pr"' in fuente
    assert "val_loss" not in fuente.replace("Nunca val_loss", "")


def test_la_permutacion_solo_cambia_el_orden_de_los_lotes(datos):
    """Contrato con la prueba de falsificacion: mismos pesos, misma
    ventana, contenido identico, orden distinto."""
    d = datos
    perm = ven.permutar(d["win"][:64], d["mask"][:64], "history", np.random.default_rng(0))
    a = mb.Lotes(d["win"][:64], d["mask"][:64], d["E_num"], d["E_cat"], batch_size=64)[0]
    b = mb.Lotes(perm, d["mask"][:64], d["E_num"], d["E_cat"], batch_size=64)[0]
    for i in range(64):
        va = np.sort(a["num"][i][a["mask"][i]].sum(axis=1))
        vb = np.sort(b["num"][i][b["mask"][i]].sum(axis=1))
        assert np.allclose(va, vb)
