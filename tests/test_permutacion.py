# tests/test_permutacion.py
import numpy as np
import pytest
from monitoreo import config as cfg
from monitoreo import generador as gen
from monitoreo import ventanas as ven


@pytest.fixture(scope="module")
def datos():
    df = gen.generar(cfg.SEED_DATOS, n_tarjetas=120)
    win, mask = ven.construir(df, K=cfg.K)
    return df, win, mask


@pytest.mark.parametrize("modo", ["full", "history"])
def test_preserva_el_contenido_de_cada_ventana(datos, modo):
    """La propiedad central: barajar cambia el ORDEN, nunca el CONTENIDO."""
    _, win, mask = datos
    perm = ven.permutar(win, mask, modo, np.random.default_rng(0))
    for i in range(0, len(win), 37):
        assert sorted(win[i][mask[i]]) == sorted(perm[i][mask[i]])


@pytest.mark.parametrize("modo", ["full", "history"])
def test_no_muta_la_entrada(datos, modo):
    _, win, mask = datos
    copia = win.copy()
    ven.permutar(win, mask, modo, np.random.default_rng(0))
    assert (win == copia).all()


@pytest.mark.parametrize("modo", ["full", "history"])
def test_el_padding_no_se_mezcla_al_centro(datos, modo):
    """Si el padding entra al medio, la mascara miente y la prueba de
    falsificacion produce basura."""
    _, win, mask = datos
    perm = ven.permutar(win, mask, modo, np.random.default_rng(1))
    # las posiciones enmascaradas siguen siendo exactamente las mismas
    assert (perm[~mask] == win[~mask]).all()


def test_history_deja_fijo_el_evento_objetivo(datos):
    _, win, mask = datos
    perm = ven.permutar(win, mask, "history", np.random.default_rng(2))
    assert (perm[:, -1] == win[:, -1]).all()


def test_full_si_mueve_el_evento_objetivo(datos):
    _, win, mask = datos
    perm = ven.permutar(win, mask, "full", np.random.default_rng(3))
    largas = mask.sum(axis=1) >= 5
    assert (perm[largas, -1] != win[largas, -1]).mean() > 0.5


def test_realmente_baraja(datos):
    """Riesgo de seccion 11 del spec: verificar que el shuffle se aplique."""
    _, win, mask = datos
    perm = ven.permutar(win, mask, "history", np.random.default_rng(4))
    largas = mask.sum(axis=1) >= 5
    cambiaron = (perm[largas] != win[largas]).any(axis=1)
    assert cambiaron.mean() > 0.9


def test_ventanas_de_un_solo_evento_no_cambian(datos):
    _, win, mask = datos
    perm = ven.permutar(win, mask, "history", np.random.default_rng(5))
    cortas = mask.sum(axis=1) == 1
    assert (perm[cortas] == win[cortas]).all()


def test_reproducible_por_semilla(datos):
    _, win, mask = datos
    a = ven.permutar(win, mask, "full", np.random.default_rng(7))
    b = ven.permutar(win, mask, "full", np.random.default_rng(7))
    assert (a == b).all()


def test_modo_invalido_falla(datos):
    _, win, mask = datos
    with pytest.raises(ValueError):
        ven.permutar(win, mask, "aleatorio", np.random.default_rng(0))
