"""La orquestacion, a escala pequena. Las cifras del informe salen del
notebook a escala completa; aqui solo se verifica que las piezas encajen."""
import numpy as np
import pandas as pd
import pytest

from monitoreo import config as cfg
from monitoreo import experimentos as exp


@pytest.fixture(scope="module")
def d():
    return exp.preparar(n_tarjetas=250)


def test_preparar_respeta_el_contrato(d):
    exp.verificar_contrato(d)


def test_los_agregados_del_hibrido_se_escalan_solo_con_train(d):
    """Si el scaler viera val/test, serian -15 pts."""
    # en float32 el cero exacto no existe; 1e-4 ya es ruido de redondeo
    media_train = np.abs(d["X_A_esc"][d["tr"]].mean(axis=0)).max()
    media_global = np.abs(d["X_A_esc"].mean(axis=0)).max()
    assert media_train < 1e-4
    # la media global NO se centra: prueba de que val y test no entraron al fit
    assert media_global > 100 * media_train


def test_preparar_puede_quitar_delta_t(d):
    sin = exp.preparar(n_tarjetas=250, usar_delta_t=False)
    assert sin["E_num"].shape[1] == d["E_num"].shape[1] - 1


def test_preparar_acepta_otro_K():
    corto = exp.preparar(n_tarjetas=250, K=5)
    assert corto["win"].shape[1] == 5
    assert (corto["win"][:, -1] == np.arange(len(corto["df"]))).all()


def test_correr_a_devuelve_una_auc_por_semilla(d):
    r = exp.correr_a(d, seeds=(7,))
    assert len(r["gbm"]) == 1 and len(r["logistica"]) == 1
    assert r["gbm"][0] > d["y"][d["va"]].mean()


def test_correr_b_entrena_y_puntua_validacion(d):
    r = exp.correr_b(d, seed=7, epocas=2)
    assert r["p_val"].shape == (int(d["va"].sum()),)
    assert 0.0 <= r["auc_pr"] <= 1.0
    assert r["epocas"] <= 2


def test_el_hibrido_recibe_la_entrada_de_agregados(d):
    r = exp.correr_b(d, seed=7, hibrido=True, epocas=1)
    nombres = {t.name.split(":")[0] for t in r["modelo"].inputs}
    assert "agg" in nombres


def test_tabla_permutacion_deja_A_quieto_y_reporta_las_dos_variantes(d):
    r = exp.correr_b(d, seed=7, epocas=1)
    p_a = np.random.default_rng(0).random(int(d["va"].sum()))
    t = exp.tabla_permutacion(r["modelo"], d, p_a)
    assert list(t["variante"]) == ["original", "full", "history"]
    assert t["auc_pr_A"].nunique() == 1          # A es invariante por construccion
    assert t.loc[0, "caida_B"] == pytest.approx(0.0)


def test_predecir_split_cubre_test_sin_reentrenar(d):
    r = exp.correr_b(d, seed=7, epocas=1)
    p = exp.predecir_split(r["modelo"], d, d["te"])
    assert p.shape == (int(d["te"].sum()),)
    assert (p >= 0).all() and (p <= 1).all()


def test_en_cache_no_recalcula(tmp_path):
    llamadas = []

    def caro():
        llamadas.append(1)
        return {"v": 42}

    ruta = tmp_path / "r.json"
    assert exp.en_cache(ruta, caro)["v"] == 42
    assert exp.en_cache(ruta, caro)["v"] == 42
    assert len(llamadas) == 1
    assert exp.en_cache(ruta, caro, forzar=True)["v"] == 42
    assert len(llamadas) == 2


def test_correr_b_cacheado_reusa_los_pesos_del_disco(d, tmp_path):
    """Segunda llamada: mismos puntajes, sin reentrenar."""
    ruta = tmp_path / "b.keras"
    a = exp.correr_b_cacheado(d, seed=7, ruta=ruta, epocas=1)
    assert a["desde_cache"] is False and ruta.exists()
    b = exp.correr_b_cacheado(d, seed=7, ruta=ruta, epocas=1)
    assert b["desde_cache"] is True
    assert np.allclose(a["p_val"], b["p_val"], atol=1e-6)


def test_el_cache_no_reusa_un_modelo_de_otro_dataset(d, tmp_path):
    """Un modelo entrenado en dev no debe cargarse en una corrida completa
    aunque las formas coincidan: reportaria cifras de otro experimento."""
    ruta = tmp_path / "b.keras"
    exp.correr_b_cacheado(d, seed=7, ruta=ruta, epocas=1)
    assert ruta.with_suffix(".huella.json").exists()

    otro = exp.preparar(n_tarjetas=200)
    r = exp.correr_b_cacheado(otro, seed=7, ruta=ruta, epocas=1)
    assert r["desde_cache"] is False, "reuso un modelo entrenado con otros datos"


def test_la_huella_distingue_hibrido_de_puro(d):
    a = exp._huella(d, 7, hibrido=False)
    b = exp._huella(d, 7, hibrido=True)
    assert a != b and a["d_agg"] == 0 and b["d_agg"] > 0
