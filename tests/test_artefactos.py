"""Los artefactos que exige §8 del enunciado.

Se saltan mientras el notebook no se haya ejecutado: son el producto de la
corrida completa, no de la suite.
"""
import json

import pytest

from monitoreo import config as cfg

ART = cfg.DIR_ARTEFACTOS
sin_correr = pytest.mark.skipif(
    not (ART / "config.json").exists(), reason="notebook aun no ejecutado")


@sin_correr
def test_config_json_tiene_lo_que_pide_el_spec():
    c = json.loads((ART / "config.json").read_text(encoding="utf-8"))
    for clave in ("K", "seed_datos", "seeds_modelo", "umbral_u_estrella",
                  "fecha_ejecucion_test", "versiones"):
        assert clave in c
    assert c["K"] == cfg.K
    assert c["seed_datos"] == cfg.SEED_DATOS


@sin_correr
def test_todos_los_artefactos_presentes():
    for nombre in ("modelo_candidato.keras", "scaler.pkl", "vocab_embeddings.json",
                   "config.json", "generador_datos.py"):
        assert (ART / nombre).exists(), f"falta {nombre}"


@sin_correr
def test_generador_entregado_es_el_mismo_que_se_uso():
    fuente = (cfg.RAIZ / "src" / "monitoreo" / "generador.py").read_text(encoding="utf-8")
    copia = (ART / "generador_datos.py").read_text(encoding="utf-8")
    assert fuente == copia


@sin_correr
def test_el_umbral_congelado_es_el_de_validacion():
    """u* se elige en validacion y se aplica tal cual a test."""
    c = json.loads((ART / "config.json").read_text(encoding="utf-8"))
    u = c["umbral_u_estrella"]
    assert u["candidato"] in (u["B"], u["C"])
    assert 0.0 < u["candidato"] < 0.5, "un umbral cerca de 0.5 ignoraria los costos"


@sin_correr
def test_la_fecha_de_test_es_posterior_a_la_hipotesis_de_C():
    """La apuesta se registra antes de entrenar C, y test va al final."""
    c = json.loads((ART / "config.json").read_text(encoding="utf-8"))
    assert c["fecha_hipotesis_C"] < c["fecha_ejecucion_test"]


@sin_correr
def test_el_candidato_no_se_eligio_mirando_test():
    c = json.loads((ART / "config.json").read_text(encoding="utf-8"))
    assert "validacion" in c["criterio_de_seleccion"]


@sin_correr
def test_los_vocabularios_reservan_pad_y_unk():
    v = json.loads((ART / "vocab_embeddings.json").read_text(encoding="utf-8"))
    for nombre, tabla in v.items():
        assert min(tabla.values()) >= 2, f"{nombre} pisa PAD=0 o UNK=1"
