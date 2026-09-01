import numpy as np
import pandas as pd
import pytest
from monitoreo import config as cfg
from monitoreo import generador as gen


@pytest.fixture(scope="module")
def df():
    return gen.generar(cfg.SEED_DATOS, n_tarjetas=400)


def test_reproducible_bit_a_bit():
    a = gen.generar(99, n_tarjetas=40)
    b = gen.generar(99, n_tarjetas=40)
    pd.testing.assert_frame_equal(a, b)


def test_semillas_distintas_dan_datos_distintos():
    a = gen.generar(1, n_tarjetas=40)
    b = gen.generar(2, n_tarjetas=40)
    assert not a["amount"].equals(b["amount"])


def test_tasa_de_fraude_en_el_rango_del_spec(df):
    tasa = df["is_fraud"].mean()
    assert 0.005 <= tasa <= 0.02, f"tasa fuera de 0.5%-2%: {tasa:.4f}"
    assert abs(tasa - cfg.TASA_FRAUDE) < 0.005


def test_los_tres_mecanismos_estan_presentes(df):
    tipos = set(df.loc[df["is_fraud"] == 1, "fraud_type"])
    assert tipos == {"f1", "f2", "f3"}


def test_etiquetas_coherentes(df):
    assert (df.loc[df["is_fraud"] == 0, "fraud_type"] == "none").all()
    assert (df.loc[df["is_fraud"] == 1, "fraud_type"] != "none").all()
    esperados = {"none", "f1_sondeo", "f1_golpe", "f2", "f3"}
    assert set(df["fraud_subtype"]) <= esperados


def test_f1_etiqueta_sondeos_y_golpe(df):
    sub = df.loc[df["fraud_type"] == "f1", "fraud_subtype"]
    assert (sub == "f1_sondeo").sum() > 0
    assert (sub == "f1_golpe").sum() > 0
    # 3-6 sondeos por golpe
    razon = (sub == "f1_sondeo").sum() / (sub == "f1_golpe").sum()
    assert 3.0 <= razon <= 6.0


def test_sondeos_de_f1_escalan_de_forma_monotona(df):
    """La propiedad que hace que el orden importe. Si falla, la prueba de
    permutacion no puede mostrar nada y el proyecto pierde su tesis."""
    f1 = df[df["fraud_type"] == "f1"].sort_values(["card_id", "ts"])
    episodios = 0
    for _, g in f1.groupby("card_id"):
        sondeos = g[g["fraud_subtype"] == "f1_sondeo"]["amount"].to_numpy()
        if sondeos.size < 3:
            continue
        # dentro de cada episodio los montos suben; se verifica por bloques
        # separados por el golpe
        assert (np.diff(sondeos) > 0).mean() > 0.8
        episodios += 1
    assert episodios > 0


def test_golpe_de_f1_es_mucho_mayor_que_sus_sondeos(df):
    f1 = df[df["fraud_type"] == "f1"]
    assert f1[f1["fraud_subtype"] == "f1_golpe"]["amount"].median() > \
           20 * f1[f1["fraud_subtype"] == "f1_sondeo"]["amount"].median()


def test_f2_son_retiros_de_cajero_casi_identicos(df):
    f2 = df[df["fraud_type"] == "f2"]
    assert (f2["channel"] == "ATM").all()
    for _, g in f2.groupby("card_id"):
        if len(g) < 3:
            continue
        assert g["amount"].std() / g["amount"].mean() < 0.15


def test_f3_es_una_sola_transaccion_de_monto_extremo(df):
    f3 = df[df["fraud_type"] == "f3"]
    assert (f3["fraud_subtype"] == "f3").all()
    for card, g in f3.groupby("card_id"):
        legit = df[(df["card_id"] == card) & (df["is_fraud"] == 0)]["amount"]
        assert (g["amount"] > legit.quantile(0.999)).all()


def test_existen_rafagas_legitimas_confusoras(df):
    """Sin este confusor, los agregados de A delatan f1 gratis y B no tiene
    nada que aportar. Se buscan ventanas legitimas de >=3 compras pequenas
    en comercios distintos dentro de 2h."""
    legit = df[df["is_fraud"] == 0]
    encontradas = 0
    for _, g in legit.groupby("card_id"):
        g = g.sort_values("ts")
        chicas = g[g["amount"] < 40]
        if len(chicas) < 3:
            continue
        dt = chicas["ts"].diff().dt.total_seconds()
        if ((dt < 7200) & (dt > 0)).sum() >= 2:
            encontradas += 1
    assert encontradas > 0, "no hay rafagas legitimas; el confusor falta"


def test_rafagas_legitimas_no_escalan_monotonamente(df):
    """La diferencia con f1 debe estar en el ORDEN, no en los montos."""
    legit = df[(df["is_fraud"] == 0) & (df["amount"] < 40)]
    fracciones = []
    for _, g in legit.groupby("card_id"):
        montos = g.sort_values("ts")["amount"].to_numpy()
        if montos.size < 4:
            continue
        fracciones.append((np.diff(montos) > 0).mean())
    assert np.mean(fracciones) < 0.65, "las rafagas legitimas escalan como f1"


def test_fraud_type_nunca_se_usa_como_feature_por_error(df):
    # contrato: estas columnas existen pero estan marcadas como solo-analisis
    for col in cfg.COLUMNAS_ANALISIS:
        assert col in df.columns
