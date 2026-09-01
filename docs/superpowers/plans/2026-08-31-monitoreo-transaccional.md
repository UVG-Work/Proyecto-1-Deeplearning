# Monitoreo transaccional — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Producir evidencia falsable sobre si el orden de las transacciones aporta información que los agregados no capturan, y cuánto vale en quetzales.

**Architecture:** Un índice canónico único (`eventos`, `muestras`, `win_idx`, `mask`) que consumen por igual el Modelo A (agregados causales) y el Modelo B (GRU sobre ventana de `K=20`). Las ventanas se guardan como matrices de índices enteros, no como tensores materializados, lo que hace que la prueba de permutación preserve el contenido por construcción y que la invariancia de A sea estructural. Lógica penalizable en `src/monitoreo/` con tests; narrativa y figuras en el notebook.

**Tech Stack:** Python 3.12.3 · numpy 2.4.6 · pandas 2.2.3 · scikit-learn 1.8.0 · TensorFlow 2.21.0 / Keras 3.15.1 · LightGBM 4.7.0 · pytest · matplotlib. CPU únicamente, Windows 10.

**Spec:** `docs/superpowers/specs/2026-08-31-monitoreo-transaccional-design.md`

## Global Constraints

- **Semillas congeladas:** `SEED_DATOS = 20260831`, `SEEDS_MODELO = (7, 13, 29)`. Un solo dataset; las tres semillas varían solo la inicialización del modelo.
- **`K = 20`.** Padding al inicio de la ventana.
- **Partición temporal por percentil de `ts` global:** train 0–70, val 70–85, test 85–100. Nunca aleatoria. Penalización −20 pts.
- **`fit()` solo sobre train.** Scalers, vocabularios e hiperparámetros. Penalización −15 pts.
- **`fraud_type` y `fraud_subtype` jamás entran a una matriz de features.** Solo análisis.
- **AUC-PR es la métrica primaria.** La exactitud no se reporta ni como nota al pie. Penalización −15 pts.
- **El test se ejecuta una sola vez**, al final, con todas las decisiones tomadas y la celda imprimiendo fecha y hora. Penalización −10 pts.
- **Costos:** FN = Q4,200, FP = Q180. Umbral teórico `p* = 0.0429`.
- **Idioma del código:** nombres de módulos, funciones y tests en español, consistentes con el spec.
- **Todo test corre con** `python -m pytest` desde la raíz del repo.

---

## Estructura de archivos

| Archivo | Responsabilidad |
|---|---|
| `pyproject.toml` | Configuración de pytest (`pythonpath = ["src"]`) |
| `requirements.txt` | Versiones exactas verificadas |
| `src/monitoreo/config.py` | Parámetros congelados, rutas, `DEV_MODE` |
| `src/monitoreo/reproducibilidad.py` | `fijar_semillas`, captura de versiones |
| `src/monitoreo/generador.py` | `generar(seed)` — flujo legítimo, ráfagas confusoras, f1/f2/f3 |
| `src/monitoreo/particion.py` | Corte temporal y tabla de particiones |
| `src/monitoreo/features_agregadas.py` | `X_A` con agregados causales `closed='left'` |
| `src/monitoreo/ventanas.py` | `win_idx`, `mask`, y las dos permutaciones |
| `src/monitoreo/features_evento.py` | Vocabularios, `E_num`, `E_cat`, escalado |
| `src/monitoreo/metricas.py` | AUC-PR, métricas en umbral, desglose por tipo |
| `src/monitoreo/modelos_a.py` | Logística + LightGBM |
| `src/monitoreo/modelos_b.py` | GRU, híbrido C, `PyDataset` de lotes |
| `src/monitoreo/calibracion.py` | Isotónica |
| `src/monitoreo/economia.py` | Curva de costo, `u*`, ahorro mensual |
| `notebooks/proyecto1_mazariegos_herrera.ipynb` | Narrativa y las seis evidencias |
| `tests/` | Espeja `src/`; concentra los tests anti-fuga |

---

### Task 1: Andamiaje, configuración y reproducibilidad

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `src/monitoreo/__init__.py`, `src/monitoreo/config.py`, `src/monitoreo/reproducibilidad.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nada.
- Produces: módulo `config` con las constantes del bloque *Global Constraints*; `config.n_tarjetas() -> int` (respeta `DEV_MODE`); `reproducibilidad.fijar_semillas(seed: int) -> None`; `reproducibilidad.versiones() -> dict[str, str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'monitoreo'`

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
filterwarnings = ["ignore::DeprecationWarning"]
```

```
# requirements.txt
numpy==2.4.6
pandas==2.2.3
scikit-learn==1.8.0
tensorflow==2.21.0
keras==3.15.1
lightgbm==4.7.0
matplotlib>=3.8
pytest>=8.0
jupyter
```

```python
# src/monitoreo/config.py
"""Parametros congelados del Proyecto 1. Nada aqui se decide mirando test."""
from __future__ import annotations

import os
from pathlib import Path

SEED_DATOS = 20260831
SEEDS_MODELO = (7, 13, 29)

K = 20

N_TARJETAS = 4000
N_TARJETAS_DEV = 400
TX_MIN, TX_MAX = 60, 200
TX_MEDIA = 100

TASA_FRAUDE = 0.012
MEZCLA_FRAUDE = {"f1": 0.40, "f2": 0.35, "f3": 0.25}
RAFAGAS_LEGITIMAS_POR_F1 = 3.0
PROB_F1_BRECHA_LARGA = 0.15

N_COMERCIOS = 300
MCCS = (
    "supermercado", "restaurante", "combustible", "farmacia", "ropa",
    "electronica", "hogar", "viajes", "entretenimiento", "salud",
    "educacion", "telecom", "transporte", "belleza", "ferreteria",
)
CANALES = ("POS", "online", "ATM", "recurrente")
PAISES = ("GT", "US", "MX", "ES", "CR")

PCT_TRAIN, PCT_VAL = 0.70, 0.85

COSTO_FN = 4200.0
COSTO_FP = 180.0
UMBRAL_TEORICO = COSTO_FP / COSTO_FN

DIM_EMB = {"mcc": 8, "channel": 4, "merchant": 16}
PAD, UNK = 0, 1

BATCH_SIZE = 512
UNIDADES_GRU = 64
DROPOUT = 0.3
LR = 1e-3
PACIENCIA = 5
EPOCAS_MAX = 30

RAIZ = Path(__file__).resolve().parents[2]
DIR_ARTEFACTOS = RAIZ / "artefactos"
DIR_DATOS = RAIZ / "datos"
DIR_FIGURAS = RAIZ / "informe" / "figuras"

COLUMNAS_ANALISIS = ("fraud_type", "fraud_subtype")
PATRONES_PROHIBIDOS_EN_A = ("prev", "lag", "delta", "diff", "anterior", "orden", "seq")


def dev_mode() -> bool:
    return bool(os.environ.get("MONITOREO_DEV"))


def n_tarjetas() -> int:
    return N_TARJETAS_DEV if dev_mode() else N_TARJETAS
```

```python
# src/monitoreo/reproducibilidad.py
"""Semillas y captura de versiones para el README."""
from __future__ import annotations

import os
import platform
import random
import sys


def fijar_semillas(seed: int) -> None:
    """Fija numpy, random, y (si esta importado) TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    try:
        import tensorflow as tf
    except ImportError:
        return
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def versiones() -> dict[str, str]:
    """Versiones exactas para el README y artefactos/config.json."""
    import importlib.metadata as md

    v = {
        "python": sys.version.split()[0],
        "sistema": f"{platform.system()} {platform.release()}",
    }
    for paquete in ("numpy", "pandas", "scikit-learn", "tensorflow", "keras", "lightgbm"):
        try:
            v[paquete] = md.version(paquete)
        except md.PackageNotFoundError:
            v[paquete] = "no instalado"
    return v
```

```python
# src/monitoreo/__init__.py
"""Proyecto 1 - Monitoreo transaccional. Mazariegos / Herrera."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml requirements.txt src/monitoreo/__init__.py src/monitoreo/config.py src/monitoreo/reproducibilidad.py tests/test_config.py
git commit -m "feat: andamiaje, parametros congelados y reproducibilidad"
```

---

### Task 2: Generador — perfiles de tarjeta y flujo legítimo

**Files:**
- Create: `src/monitoreo/generador.py`
- Test: `tests/test_generador_legitimo.py`

**Interfaces:**
- Consumes: `config`.
- Produces:
  - `generador.FECHA_INICIO: pd.Timestamp`
  - `generador.perfiles(rng, n_tarjetas) -> pd.DataFrame` con columnas `monto_base, tasa_dia, comercios_pref (object: np.ndarray), mcc_pref (object), pais_base`
  - `generador.flujo_legitimo(rng, perf) -> pd.DataFrame` con columnas `card_id, ts, amount, merchant_id, mcc, channel, country, is_fraud, fraud_type, fraud_subtype`, ordenada por `(card_id, ts)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generador_legitimo.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_generador_legitimo.py -v`
Expected: FAIL con `AttributeError: module 'monitoreo.generador' has no attribute 'perfiles'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/monitoreo/generador.py
"""Generador sintetico del Proyecto 1.

Entregable en si mismo: generar(seed) devuelve siempre el mismo DataFrame.
El diseno de f1 esta explicado en la seccion 4.1.1 del spec: los sondeos
escalan de forma monotona para que la senal viva en el ORDEN y no en los
agregados, y se inyectan rafagas legitimas con la misma firma agregada
para que el Modelo A no pueda ganar sin leer secuencia.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

FECHA_INICIO = pd.Timestamp("2026-01-01")

COLUMNAS = [
    "card_id", "ts", "amount", "merchant_id", "mcc",
    "channel", "country", "is_fraud", "fraud_type", "fraud_subtype",
]

# Distribucion diurna de la hora de compra (suma 1, indices 0..23).
_PESOS_HORA = np.array(
    [0.4, 0.3, 0.2, 0.2, 0.2, 0.4, 1.0, 2.0, 4.0, 5.0, 5.5, 6.0,
     6.5, 6.0, 5.5, 5.5, 6.0, 6.5, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
)
_PESOS_HORA = _PESOS_HORA / _PESOS_HORA.sum()


def _n_tx_por_tarjeta(rng: np.random.Generator, n: int) -> np.ndarray:
    """Lognormal truncada a [TX_MIN, TX_MAX] con media cercana a TX_MEDIA.

    Una uniforme sobre 60-200 daria media 130 y ~520k eventos, fuera del
    objetivo de 400k del spec.
    """
    out = np.empty(n, dtype=np.int32)
    pendientes = np.arange(n)
    while pendientes.size:
        x = rng.lognormal(np.log(95.0), 0.35, size=pendientes.size)
        ok = (x >= cfg.TX_MIN) & (x <= cfg.TX_MAX)
        out[pendientes[ok]] = x[ok].astype(np.int32)
        pendientes = pendientes[~ok]
    return out


def perfiles(rng: np.random.Generator, n_tarjetas: int) -> pd.DataFrame:
    """Un perfil de gasto estable por tarjeta."""
    return pd.DataFrame(
        {
            "card_id": np.arange(n_tarjetas, dtype=np.int32),
            "n_tx": _n_tx_por_tarjeta(rng, n_tarjetas),
            "monto_base": rng.lognormal(np.log(120.0), 0.6, size=n_tarjetas),
            "tasa_dia": rng.uniform(0.5, 4.0, size=n_tarjetas),
            "comercios_pref": [
                rng.choice(cfg.N_COMERCIOS, size=rng.integers(5, 16), replace=False)
                for _ in range(n_tarjetas)
            ],
            "mcc_pref": [
                rng.choice(len(cfg.MCCS), size=rng.integers(3, 8), replace=False)
                for _ in range(n_tarjetas)
            ],
            "pais_base": np.where(rng.random(n_tarjetas) < 0.97, "GT", "US"),
        }
    )


def _timestamps(rng: np.random.Generator, m: int, tasa_dia: float) -> np.ndarray:
    """Fechas de un proceso de llegadas, con la hora del dia remuestreada
    de una distribucion diurna. Se reordena para preservar monotonia."""
    horas = np.cumsum(rng.exponential(24.0 / tasa_dia, size=m))
    ts = FECHA_INICIO + pd.to_timedelta(horas, unit="h")
    dia = ts.normalize()
    h = rng.choice(24, size=m, p=_PESOS_HORA)
    minuto = rng.integers(0, 60, size=m)
    segundo = rng.integers(0, 60, size=m)
    ts = dia + pd.to_timedelta(h, "h") + pd.to_timedelta(minuto, "m") + pd.to_timedelta(segundo, "s")
    return np.sort(ts.values)


def flujo_legitimo(rng: np.random.Generator, perf: pd.DataFrame) -> pd.DataFrame:
    """Transacciones normales de todas las tarjetas."""
    trozos = []
    for fila in perf.itertuples(index=False):
        m = int(fila.n_tx)
        ts = _timestamps(rng, m, float(fila.tasa_dia))
        pref = np.asarray(fila.comercios_pref)
        usa_pref = rng.random(m) < 0.8
        comercio = np.where(
            usa_pref,
            rng.choice(pref, size=m),
            rng.integers(0, cfg.N_COMERCIOS, size=m),
        )
        mcc_idx = np.where(
            rng.random(m) < 0.85,
            rng.choice(np.asarray(fila.mcc_pref), size=m),
            rng.integers(0, len(cfg.MCCS), size=m),
        )
        monto = fila.monto_base * rng.lognormal(0.0, 0.55, size=m)
        canal = rng.choice(cfg.CANALES, size=m, p=[0.55, 0.28, 0.10, 0.07])
        pais = np.where(rng.random(m) < 0.98, fila.pais_base, rng.choice(cfg.PAISES, size=m))
        trozos.append(
            pd.DataFrame(
                {
                    "card_id": np.full(m, fila.card_id, dtype=np.int32),
                    "ts": ts,
                    "amount": np.round(monto, 2),
                    "merchant_id": comercio.astype(np.int32),
                    "mcc": np.asarray(cfg.MCCS, dtype=object)[mcc_idx],
                    "channel": canal,
                    "country": pais,
                    "is_fraud": np.zeros(m, dtype=np.int8),
                    "fraud_type": np.full(m, "none", dtype=object),
                    "fraud_subtype": np.full(m, "none", dtype=object),
                }
            )
        )
    df = pd.concat(trozos, ignore_index=True)
    df = df.drop_duplicates(subset=["card_id", "ts"], keep="first")
    df = df.sort_values(["card_id", "ts"], kind="mergesort").reset_index(drop=True)
    return df[COLUMNAS]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_generador_legitimo.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/monitoreo/generador.py tests/test_generador_legitimo.py
git commit -m "feat: generador - perfiles de tarjeta y flujo legitimo"
```

---

### Task 3: Generador — ráfagas legítimas y los tres mecanismos de fraude

**Files:**
- Modify: `src/monitoreo/generador.py`
- Test: `tests/test_generador_fraude.py`

**Interfaces:**
- Consumes: `generador.perfiles`, `generador.flujo_legitimo`, `generador.COLUMNAS`.
- Produces: `generador.generar(seed: int, n_tarjetas: int | None = None) -> pd.DataFrame` — flujo completo con fraude inyectado, ordenado por `(card_id, ts)`, índice reiniciado. `fraud_type ∈ {none, f1, f2, f3}`; `fraud_subtype ∈ {none, f1_sondeo, f1_golpe, f2, f3}`.

**Nota de diseño (§4.1.1 del spec):** el corazón de este task son dos propiedades que los tests verifican explícitamente. Los sondeos de f1 son **estrictamente crecientes** (invisible a agregados, destruido por permutación). Las ráfagas legítimas tienen la **misma firma agregada** que f1 pero montos desordenados. Si alguna de las dos falla, el experimento entero pierde sentido.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generador_fraude.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_generador_fraude.py -v`
Expected: FAIL con `AttributeError: module 'monitoreo.generador' has no attribute 'generar'`

- [ ] **Step 3: Write minimal implementation**

Añadir a `src/monitoreo/generador.py`:

```python
def _fila(card_id, ts, amount, merchant, mcc, channel, country, tipo, subtipo):
    return {
        "card_id": np.int32(card_id),
        "ts": pd.Timestamp(ts),
        "amount": round(float(amount), 2),
        "merchant_id": np.int32(merchant),
        "mcc": mcc,
        "channel": channel,
        "country": country,
        "is_fraud": np.int8(0 if tipo == "none" else 1),
        "fraud_type": tipo,
        "fraud_subtype": subtipo,
    }


def _anclas(rng, base, n_episodios):
    """Elige (tarjeta, instante) donde insertar episodios."""
    rangos = base.groupby("card_id")["ts"].agg(["min", "max"])
    cards = rng.choice(rangos.index.to_numpy(), size=n_episodios, replace=True)
    u = rng.uniform(0.1, 0.9, size=n_episodios)
    t0 = rangos.loc[cards, "min"].to_numpy() + (
        (rangos.loc[cards, "max"].to_numpy() - rangos.loc[cards, "min"].to_numpy()) * u
    )
    return cards, pd.to_datetime(t0)


def _episodio_pequeno(rng, monotono):
    """3-6 montos chicos en comercios distintos. Si monotono, escalan."""
    n = int(rng.integers(3, 7))
    montos = np.sort(rng.uniform(5.0, 40.0, size=n))
    if not monotono:
        rng.shuffle(montos)
    comercios = rng.choice(cfg.N_COMERCIOS, size=n, replace=False)
    minutos = np.cumsum(rng.uniform(3.0, 20.0, size=n))
    return montos, comercios, minutos


def _inyectar_f1(rng, base, perf, n_episodios):
    cards, t0s = _anclas(rng, base, n_episodios)
    filas = []
    for card, t0 in zip(cards, t0s):
        montos, comercios, minutos = _episodio_pequeno(rng, monotono=True)
        for monto, com, mi in zip(montos, comercios, minutos):
            filas.append(_fila(
                card, t0 + pd.Timedelta(minutes=float(mi)), monto, com,
                rng.choice(cfg.MCCS), "online", "GT", "f1", "f1_sondeo",
            ))
        if rng.random() < cfg.PROB_F1_BRECHA_LARGA:
            brecha = pd.Timedelta(hours=float(rng.uniform(26.0, 72.0)))  # caso de fallo esperado
        else:
            brecha = pd.Timedelta(minutes=float(rng.uniform(5.0, 55.0)))
        monto_base = float(perf.loc[perf["card_id"] == card, "monto_base"].iloc[0])
        filas.append(_fila(
            card, t0 + pd.Timedelta(minutes=float(minutos[-1])) + brecha,
            monto_base * rng.uniform(8.0, 30.0), rng.integers(0, cfg.N_COMERCIOS),
            rng.choice(cfg.MCCS), "online", rng.choice(("GT", "US")), "f1", "f1_golpe",
        ))
    return filas


def _inyectar_rafagas_legitimas(rng, base, n_episodios):
    """Mismo perfil agregado que f1, montos DESORDENADOS, sin golpe fraudulento.
    A veces termina en una compra grande legitima."""
    cards, t0s = _anclas(rng, base, n_episodios)
    filas = []
    for card, t0 in zip(cards, t0s):
        montos, comercios, minutos = _episodio_pequeno(rng, monotono=False)
        for monto, com, mi in zip(montos, comercios, minutos):
            filas.append(_fila(
                card, t0 + pd.Timedelta(minutes=float(mi)), monto, com,
                rng.choice(cfg.MCCS), rng.choice(("POS", "online")), "GT", "none", "none",
            ))
        if rng.random() < 0.35:
            filas.append(_fila(
                card, t0 + pd.Timedelta(minutes=float(minutos[-1] + rng.uniform(5, 55))),
                rng.uniform(600.0, 3000.0), rng.integers(0, cfg.N_COMERCIOS),
                rng.choice(cfg.MCCS), "POS", "GT", "none", "none",
            ))
    return filas


def _inyectar_f2(rng, base, n_episodios):
    cards, t0s = _anclas(rng, base, n_episodios)
    filas = []
    for card, t0 in zip(cards, t0s):
        n = int(rng.integers(3, 6))
        monto = float(rng.choice((200.0, 500.0, 1000.0, 2000.0)))
        minutos = np.cumsum(rng.uniform(2.0, 9.0, size=n))
        comercio = int(rng.integers(0, cfg.N_COMERCIOS))
        for mi in minutos:
            filas.append(_fila(
                card, t0 + pd.Timedelta(minutes=float(mi)),
                monto * rng.uniform(0.97, 1.03), comercio,
                "transporte", "ATM", "GT", "f2", "f2",
            ))
    return filas


def _inyectar_f3(rng, base, perf, n_episodios):
    """Una sola transaccion extrema y AISLADA: sin actividad de la tarjeta en
    las 6h previas, para que sea estructuralmente distinta del golpe de f1."""
    filas = []
    por_tarjeta = {c: g["ts"].to_numpy() for c, g in base.groupby("card_id")}
    cards, t0s = _anclas(rng, base, n_episodios * 3)
    puestos = 0
    for card, t0 in zip(cards, t0s):
        if puestos >= n_episodios:
            break
        ts = por_tarjeta[card]
        ventana = (ts > np.datetime64(t0 - pd.Timedelta(hours=6))) & (ts <= np.datetime64(t0))
        if ventana.any():
            continue
        p999 = float(np.quantile(base.loc[base["card_id"] == card, "amount"], 0.999))
        filas.append(_fila(
            card, t0, p999 * rng.uniform(1.5, 4.0), rng.integers(0, cfg.N_COMERCIOS),
            rng.choice(cfg.MCCS), rng.choice(("online", "POS")),
            rng.choice(("US", "ES", "CR")), "f3", "f3",
        ))
        puestos += 1
    return filas


def generar(seed: int, n_tarjetas: int | None = None) -> pd.DataFrame:
    """Genera el dataset completo. Identico para la misma semilla."""
    n_tarjetas = cfg.n_tarjetas() if n_tarjetas is None else n_tarjetas
    rng = np.random.default_rng(seed)
    perf = perfiles(rng, n_tarjetas)
    base = flujo_legitimo(rng, perf)

    n_obj = int(len(base) * cfg.TASA_FRAUDE / (1.0 - cfg.TASA_FRAUDE))
    n_f1 = max(1, int(n_obj * cfg.MEZCLA_FRAUDE["f1"] / 5.5))   # ~5.5 tx por episodio
    n_f2 = max(1, int(n_obj * cfg.MEZCLA_FRAUDE["f2"] / 4.0))   # ~4 tx por episodio
    n_f3 = max(1, int(n_obj * cfg.MEZCLA_FRAUDE["f3"]))         # 1 tx por episodio

    filas = []
    filas += _inyectar_f1(rng, base, perf, n_f1)
    filas += _inyectar_rafagas_legitimas(rng, base, int(n_f1 * cfg.RAFAGAS_LEGITIMAS_POR_F1))
    filas += _inyectar_f2(rng, base, n_f2)
    filas += _inyectar_f3(rng, base, perf, n_f3)

    df = pd.concat([base, pd.DataFrame(filas, columns=COLUMNAS)], ignore_index=True)
    df = df.drop_duplicates(subset=["card_id", "ts"], keep="first")
    df = df.sort_values(["card_id", "ts"], kind="mergesort").reset_index(drop=True)
    df["is_fraud"] = df["is_fraud"].astype(np.int8)
    df["card_id"] = df["card_id"].astype(np.int32)
    df["merchant_id"] = df["merchant_id"].astype(np.int32)
    return df[COLUMNAS]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_generador_fraude.py -v`
Expected: 13 passed. Si `test_tasa_de_fraude_en_el_rango_del_spec` falla, ajustar los divisores `5.5` / `4.0` de `generar` a la media real de tx por episodio; no tocar `TASA_FRAUDE`.

- [ ] **Step 5: Commit**

```bash
git add src/monitoreo/generador.py tests/test_generador_fraude.py
git commit -m "feat: generador - f1 monotono, rafagas confusoras, f2 y f3"
```

---

### Task 4: Partición temporal

**Files:**
- Create: `src/monitoreo/particion.py`
- Test: `tests/test_particion.py`

**Interfaces:**
- Consumes: `config.PCT_TRAIN`, `config.PCT_VAL`.
- Produces:
  - `particion.asignar_split(df) -> pd.Series[str]` con valores `train|val|test`, alineada al índice de `df`
  - `particion.tabla(df, split) -> pd.DataFrame` con columnas `split, n, fecha_min, fecha_max, n_fraude, tasa_fraude`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_particion.py
import numpy as np
import pandas as pd
import pytest
from monitoreo import config as cfg
from monitoreo import generador as gen
from monitoreo import particion as part


@pytest.fixture(scope="module")
def datos():
    df = gen.generar(cfg.SEED_DATOS, n_tarjetas=200)
    return df, part.asignar_split(df)


def test_cubre_todas_las_filas_una_sola_vez(datos):
    df, s = datos
    assert len(s) == len(df)
    assert set(s.unique()) == {"train", "val", "test"}


def test_proporciones_cercanas_a_70_15_15(datos):
    _, s = datos
    p = s.value_counts(normalize=True)
    assert abs(p["train"] - 0.70) < 0.02
    assert abs(p["val"] - 0.15) < 0.02
    assert abs(p["test"] - 0.15) < 0.02


def test_sin_solape_temporal(datos):
    df, s = datos
    ts = df["ts"]
    assert ts[s == "train"].max() <= ts[s == "val"].min()
    assert ts[s == "val"].max() <= ts[s == "test"].min()


def test_el_corte_es_global_no_por_tarjeta(datos):
    """Una misma tarjeta puede aparecer en train y en test: es realista."""
    df, s = datos
    compartidas = set(df.loc[s == "train", "card_id"]) & set(df.loc[s == "test", "card_id"])
    assert len(compartidas) > 0


def test_no_es_particion_aleatoria(datos):
    """Penalizacion de -20 pts. Una particion aleatoria mezclaria las fechas."""
    df, s = datos
    orden = df["ts"].rank(method="first")
    corte_train = orden[s == "train"].max()
    assert (orden[s != "train"] > corte_train).all()


def test_tabla_reporta_fechas_y_tasa_de_fraude(datos):
    df, s = datos
    t = part.tabla(df, s)
    assert list(t.columns) == ["split", "n", "fecha_min", "fecha_max", "n_fraude", "tasa_fraude"]
    assert list(t["split"]) == ["train", "val", "test"]
    assert (t["tasa_fraude"] > 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_particion.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'monitoreo.particion'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/monitoreo/particion.py
"""Corte temporal global. Penalizacion de -20 pts por particion aleatoria."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def asignar_split(df: pd.DataFrame) -> pd.Series:
    """Percentiles de `ts` GLOBAL, no por tarjeta.

    Se usa el rango ordinal y no `quantile` sobre las fechas para que los
    empates de timestamp no desbalanceen los cortes.
    """
    orden = df["ts"].rank(method="first")
    n = len(df)
    corte_tr = cfg.PCT_TRAIN * n
    corte_val = cfg.PCT_VAL * n
    s = pd.Series("test", index=df.index, dtype=object)
    s[orden <= corte_val] = "val"
    s[orden <= corte_tr] = "train"
    return s


def tabla(df: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    """Tabla 1 del informe: tamano, fechas de corte y tasa de fraude."""
    filas = []
    for nombre in ("train", "val", "test"):
        g = df[split == nombre]
        filas.append(
            {
                "split": nombre,
                "n": len(g),
                "fecha_min": g["ts"].min(),
                "fecha_max": g["ts"].max(),
                "n_fraude": int(g["is_fraud"].sum()),
                "tasa_fraude": float(g["is_fraud"].mean()),
            }
        )
    return pd.DataFrame(filas)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_particion.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/monitoreo/particion.py tests/test_particion.py
git commit -m "feat: particion temporal por percentil de ts global"
```

---

### Task 5: Features agregadas del Modelo A

**Files:**
- Create: `src/monitoreo/features_agregadas.py`
- Test: `tests/test_features_agregadas.py`

**Interfaces:**
- Consumes: `config.PATRONES_PROHIBIDOS_EN_A`.
- Produces:
  - `features_agregadas.construir(df) -> pd.DataFrame` — `X_A`, mismo número de filas y mismo orden que `df`, solo columnas numéricas
  - `features_agregadas.verificar_sin_orden(X_A) -> None` — lanza `AssertionError` si alguna columna codifica orden

**Nota:** `n_merchants_24h` es un conteo de distintos sobre ventana temporal y pandas no ofrece `rolling.nunique`. Se implementa con un barrido de dos punteros por tarjeta: exacto y O(n).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features_agregadas.py
import numpy as np
import pandas as pd
import pytest
from monitoreo import config as cfg
from monitoreo import features_agregadas as fa
from monitoreo import generador as gen


def _mini(montos, horas, comercios=None):
    n = len(montos)
    comercios = list(range(n)) if comercios is None else comercios
    return pd.DataFrame(
        {
            "card_id": np.zeros(n, dtype=np.int32),
            "ts": [pd.Timestamp("2026-03-01") + pd.Timedelta(hours=h) for h in horas],
            "amount": np.asarray(montos, dtype=float),
            "merchant_id": np.asarray(comercios, dtype=np.int32),
            "mcc": ["ropa"] * n,
            "channel": ["POS"] * n,
            "country": ["GT"] * n,
            "is_fraud": np.zeros(n, dtype=np.int8),
            "fraud_type": ["none"] * n,
            "fraud_subtype": ["none"] * n,
        }
    )


def test_closed_left_excluye_la_transaccion_actual():
    df = _mini([10, 20, 30], [0, 1, 2])
    X = fa.construir(df)
    # en la fila 2, el contexto son 10 y 20 -> media 15, no 20
    assert X.loc[2, "amt_mean_24h"] == pytest.approx(15.0)
    assert X.loc[2, "amt_max_24h"] == pytest.approx(20.0)
    assert X.loc[2, "amt"] == pytest.approx(30.0)


def test_primera_transaccion_sin_contexto():
    df = _mini([10, 20], [0, 1])
    X = fa.construir(df)
    assert X.loc[0, "n_tx_24h"] == 0
    assert X.loc[0, "amt_mean_24h"] == 0.0 or np.isnan(X.loc[0, "amt_mean_24h"]) is False


def test_envenenamiento_del_futuro_no_altera_el_pasado():
    """El test que atrapa cualquier groupby().mean() sobre el historico
    completo. Penalizacion de -15 pts."""
    df = _mini([10, 20, 30], [0, 1, 2])
    X_antes = fa.construir(df)
    df2 = _mini([10, 20, 30, 999999], [0, 1, 2, 3])
    X_despues = fa.construir(df2)
    pd.testing.assert_frame_equal(X_antes, X_despues.iloc[:3])


def test_conteos_respetan_la_ventana_temporal():
    df = _mini([10, 10, 10, 10], [0, 0.5, 0.75, 30])
    X = fa.construir(df)
    assert X.loc[2, "n_tx_1h"] == 2      # las dos previas dentro de 1h
    assert X.loc[3, "n_tx_24h"] == 0     # 30h despues, nada en la ventana


def test_n_merchants_cuenta_distintos_no_totales():
    df = _mini([10, 10, 10], [0, 1, 2], comercios=[5, 5, 7])
    X = fa.construir(df)
    assert X.loc[2, "n_merchants_24h"] == 1   # solo el comercio 5 esta antes
    df2 = _mini([10, 10, 10], [0, 1, 2], comercios=[5, 6, 7])
    assert fa.construir(df2).loc[2, "n_merchants_24h"] == 2


def test_ninguna_feature_codifica_orden():
    df = gen.generar(11, n_tarjetas=30)
    X = fa.construir(df)
    fa.verificar_sin_orden(X)   # no debe lanzar
    for col in X.columns:
        for patron in cfg.PATRONES_PROHIBIDOS_EN_A:
            assert patron not in col.lower()


def test_los_agregados_son_invariantes_a_permutar_los_montos_del_conjunto():
    """Media, maximo y conteo de un conjunto no dependen del orden.
    Es la razon por la que A no puede moverse en la prueba de permutacion."""
    df = _mini([10, 20, 30, 40], [0, 1, 2, 3])
    X = fa.construir(df)
    df_perm = _mini([30, 10, 20, 40], [0, 1, 2, 3])
    X_perm = fa.construir(df_perm)
    assert X.loc[3, "amt_mean_24h"] == pytest.approx(X_perm.loc[3, "amt_mean_24h"])
    assert X.loc[3, "amt_max_24h"] == pytest.approx(X_perm.loc[3, "amt_max_24h"])
    assert X.loc[3, "n_tx_24h"] == X_perm.loc[3, "n_tx_24h"]


def test_fraud_type_no_aparece_en_las_features():
    df = gen.generar(11, n_tarjetas=30)
    X = fa.construir(df)
    for col in cfg.COLUMNAS_ANALISIS:
        assert col not in X.columns
    assert X.select_dtypes(exclude="number").empty


def test_alineado_con_el_dataframe_de_entrada():
    df = gen.generar(11, n_tarjetas=30)
    X = fa.construir(df)
    assert len(X) == len(df)
    assert (X.index == df.index).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_features_agregadas.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'monitoreo.features_agregadas'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/monitoreo/features_agregadas.py
"""Agregados causales del Modelo A.

Todas las features son invariantes al orden por construccion (medias,
conteos, cardinalidades de conjunto). Todos los agregados de contexto usan
`closed='left'`: `amt` es la unica feature que describe la transaccion que
se esta puntuando.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

_VENTANAS = {"1h": pd.Timedelta("1h"), "24h": pd.Timedelta("24h"), "7d": pd.Timedelta("7d")}


def _rolling_causal(g: pd.DataFrame, ventana: str, columna: str, op: str) -> pd.Series:
    s = g.set_index("ts")[columna]
    return s.rolling(ventana, closed="left").agg(op)


def _n_distintos_causal(ts: np.ndarray, comercio: np.ndarray, ventana: pd.Timedelta) -> np.ndarray:
    """Conteo de comercios distintos en (t-ventana, t), excluyendo t.

    Dos punteros con un multiconjunto incremental: exacto y O(n).
    pandas no ofrece rolling.nunique.
    """
    n = len(ts)
    out = np.zeros(n, dtype=np.int32)
    conteo: dict[int, int] = {}
    izq = 0
    for der in range(n):
        limite = ts[der] - ventana.to_timedelta64()
        while izq < der and ts[izq] <= limite:
            c = int(comercio[izq])
            conteo[c] -= 1
            if conteo[c] == 0:
                del conteo[c]
            izq += 1
        out[der] = len(conteo)
        c = int(comercio[der])
        conteo[c] = conteo.get(c, 0) + 1
    return out


def construir(df: pd.DataFrame) -> pd.DataFrame:
    """X_A. Mismo numero de filas y mismo orden que `df`."""
    partes = []
    for _, g in df.groupby("card_id", sort=False):
        g = g.sort_values("ts", kind="mergesort")
        x = pd.DataFrame(index=g.index)
        x["amt"] = g["amount"].to_numpy()
        x["amt_mean_24h"] = _rolling_causal(g, "24h", "amount", "mean").to_numpy()
        x["amt_std_24h"] = _rolling_causal(g, "24h", "amount", "std").to_numpy()
        x["amt_max_24h"] = _rolling_causal(g, "24h", "amount", "max").to_numpy()
        x["n_tx_1h"] = _rolling_causal(g, "1h", "amount", "count").to_numpy()
        x["n_tx_24h"] = _rolling_causal(g, "24h", "amount", "count").to_numpy()
        media_7d = _rolling_causal(g, "7d", "amount", "mean").to_numpy()
        x["n_merchants_24h"] = _n_distintos_causal(
            g["ts"].to_numpy(), g["merchant_id"].to_numpy(), _VENTANAS["24h"]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            x["amt_ratio_to_mean_7d"] = np.where(
                np.isfinite(media_7d) & (media_7d > 0), g["amount"].to_numpy() / media_7d, 1.0
            )
        partes.append(x)

    X = pd.concat(partes).reindex(df.index)
    X = X.fillna(0.0)

    hora = df["ts"].dt.hour.to_numpy() + df["ts"].dt.minute.to_numpy() / 60.0
    X["hour_sin"] = np.sin(2 * np.pi * hora / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * hora / 24.0)
    X["is_weekend"] = (df["ts"].dt.dayofweek >= 5).astype(float).to_numpy()

    for canal in cfg.CANALES:
        X[f"channel_{canal}"] = (df["channel"] == canal).astype(float).to_numpy()
    for mcc in cfg.MCCS:
        X[f"mcc_{mcc}"] = (df["mcc"] == mcc).astype(float).to_numpy()

    verificar_sin_orden(X)
    return X.astype(np.float32)


def verificar_sin_orden(X: pd.DataFrame) -> None:
    """Una feature de orden en A contaminaria la comparacion con B."""
    for col in X.columns:
        bajo = col.lower()
        for patron in cfg.PATRONES_PROHIBIDOS_EN_A:
            assert patron not in bajo, f"'{col}' parece codificar orden; A debe ser ciego al orden"
    for col in cfg.COLUMNAS_ANALISIS:
        assert col not in X.columns, f"'{col}' es solo para analisis, nunca feature"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_features_agregadas.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/monitoreo/features_agregadas.py tests/test_features_agregadas.py
git commit -m "feat: agregados causales de A con closed=left y test de envenenamiento"
```

---

### Task 6: Ventanas como índices

**Files:**
- Create: `src/monitoreo/ventanas.py`
- Test: `tests/test_ventanas.py`

**Interfaces:**
- Consumes: `config.K`.
- Produces: `ventanas.construir(df, K=cfg.K) -> tuple[np.ndarray, np.ndarray]` — `win_idx (N,K) int32` con posiciones enteras dentro de `df` (posicional, no etiqueta) y `mask (N,K) bool`. `win_idx[:, -1] == np.arange(N)`. Las posiciones de padding se rellenan con el propio índice de la fila y `mask=False`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ventanas.py
import numpy as np
import pandas as pd
import pytest
from monitoreo import config as cfg
from monitoreo import generador as gen
from monitoreo import ventanas as ven


@pytest.fixture(scope="module")
def datos():
    df = gen.generar(cfg.SEED_DATOS, n_tarjetas=120)
    win, mask = ven.construir(df, K=cfg.K)
    return df, win, mask


def test_formas_y_tipos(datos):
    df, win, mask = datos
    assert win.shape == (len(df), cfg.K)
    assert mask.shape == (len(df), cfg.K)
    assert win.dtype == np.int32
    assert mask.dtype == bool


def test_la_ultima_posicion_es_la_propia_transaccion(datos):
    df, win, mask = datos
    assert (win[:, -1] == np.arange(len(df))).all()
    assert mask[:, -1].all()


def test_ninguna_ventana_mira_al_futuro(datos):
    df, win, mask = datos
    ts = df["ts"].to_numpy()
    assert (ts[win][mask] <= np.repeat(ts[:, None], cfg.K, axis=1)[mask]).all()


def test_ninguna_ventana_cruza_de_tarjeta(datos):
    df, win, mask = datos
    card = df["card_id"].to_numpy()
    propia = np.repeat(card[:, None], cfg.K, axis=1)
    assert (card[win][mask] == propia[mask]).all()


def test_la_ventana_es_contigua_y_ordenada(datos):
    df, win, mask = datos
    ts = df["ts"].to_numpy()
    for i in np.random.default_rng(0).choice(len(df), size=200, replace=False):
        validos = ts[win[i]][mask[i]]
        assert (np.diff(validos) >= np.timedelta64(0)).all()


def test_padding_al_inicio_para_historia_corta():
    df = gen.generar(5, n_tarjetas=10)
    win, mask = ven.construir(df, K=cfg.K)
    primera_de_cada_tarjeta = df.reset_index(drop=True).groupby("card_id").head(1).index
    for i in primera_de_cada_tarjeta:
        assert mask[i].sum() == 1
        assert mask[i, -1]
        assert not mask[i, :-1].any()


def test_mascara_cuenta_exactamente_la_historia_disponible(datos):
    df, win, mask = datos
    pos = df.reset_index(drop=True).groupby("card_id").cumcount().to_numpy()
    esperado = np.minimum(pos + 1, cfg.K)
    assert (mask.sum(axis=1) == esperado).all()


def test_memoria_razonable(datos):
    _, win, mask = datos
    assert win.nbytes / len(win) == cfg.K * 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ventanas.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'monitoreo.ventanas'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/monitoreo/ventanas.py
"""Ventanas deslizantes representadas como INDICES, no como tensores.

Guardar indices en vez de floats hace que la prueba de permutacion preserve
el contenido por construccion: barajar es permutar enteros dentro de una
fila, y es imposible que altere QUE eventos hay en la ventana.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def construir(df: pd.DataFrame, K: int = cfg.K) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (win_idx, mask).

    `df` debe venir ordenado por (card_id, ts). Los indices son POSICIONALES
    respecto de `df`, de 0 a len(df)-1.
    """
    n = len(df)
    pos_global = np.arange(n, dtype=np.int32)
    pos_en_tarjeta = df.reset_index(drop=True).groupby("card_id").cumcount().to_numpy()

    desplazamientos = np.arange(K - 1, -1, -1, dtype=np.int32)      # K-1 ... 0
    candidatos = pos_global[:, None] - desplazamientos[None, :]
    disponible = pos_en_tarjeta[:, None] >= desplazamientos[None, :]

    win = np.where(disponible, candidatos, pos_global[:, None]).astype(np.int32)
    return win, disponible
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ventanas.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/monitoreo/ventanas.py tests/test_ventanas.py
git commit -m "feat: ventanas como matriz de indices con mascara de padding"
```

---

### Task 7: Permutaciones controladas

**Files:**
- Modify: `src/monitoreo/ventanas.py`
- Test: `tests/test_permutacion.py`

**Interfaces:**
- Consumes: `ventanas.construir`.
- Produces: `ventanas.permutar(win_idx, mask, modo: str, rng) -> np.ndarray` con `modo ∈ {"full", "history"}`. Devuelve una copia; no muta la entrada. Baraja **solo las posiciones válidas**; el padding se queda donde está.

- [ ] **Step 1: Write the failing test**

```python
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
    """Riesgo de §11 del spec: verificar que el shuffle se aplique."""
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_permutacion.py -v`
Expected: FAIL con `AttributeError: module 'monitoreo.ventanas' has no attribute 'permutar'`

- [ ] **Step 3: Write minimal implementation**

Añadir a `src/monitoreo/ventanas.py`:

```python
MODOS = ("full", "history")


def permutar(
    win_idx: np.ndarray, mask: np.ndarray, modo: str, rng: np.random.Generator
) -> np.ndarray:
    """Baraja el orden de la ventana sin alterar su contenido.

    - "full":    baraja las K posiciones validas, evento objetivo incluido.
    - "history": baraja las K-1 previas y deja el objetivo en la ultima
                 posicion. Aisla el aporte del orden de la HISTORIA.

    Solo se permutan posiciones validas: si el padding se mezclara al
    centro, la mascara dejaria de describir la secuencia.
    """
    if modo not in MODOS:
        raise ValueError(f"modo debe ser uno de {MODOS}, no {modo!r}")

    perm = win_idx.copy()
    ultima = win_idx.shape[1] - 1

    for i in range(win_idx.shape[0]):
        posiciones = np.flatnonzero(mask[i])
        if modo == "history":
            posiciones = posiciones[posiciones != ultima]
        if posiciones.size < 2:
            continue
        perm[i, posiciones] = win_idx[i, rng.permutation(posiciones)]

    return perm
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_permutacion.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add src/monitoreo/ventanas.py tests/test_permutacion.py
git commit -m "feat: permutaciones full e history que preservan contenido"
```

---

### Task 8: Features de evento, vocabularios y escalado

**Files:**
- Create: `src/monitoreo/features_evento.py`
- Test: `tests/test_features_evento.py`

**Interfaces:**
- Consumes: `config.PAD`, `config.UNK`, `config.DIM_EMB`.
- Produces:
  - `features_evento.construir_vocabularios(df, es_train) -> dict[str, dict]` — claves `mcc`, `channel`, `merchant`; cada valor mapea categoría → índice ≥ 2
  - `features_evento.construir(df, vocab, es_train, scaler=None, usar_delta_t=True) -> tuple[np.ndarray, np.ndarray, StandardScaler]` — `(E_num float32 [N_ev, d_num], E_cat int32 [N_ev, 3], scaler)`. El orden de columnas de `E_cat` es `(mcc, channel, merchant)`.
  - `features_evento.NOMBRES_NUM: tuple[str, ...]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features_evento.py
import numpy as np
import pandas as pd
import pytest
from monitoreo import config as cfg
from monitoreo import features_evento as fe
from monitoreo import generador as gen
from monitoreo import particion as part


@pytest.fixture(scope="module")
def datos():
    df = gen.generar(cfg.SEED_DATOS, n_tarjetas=150)
    split = part.asignar_split(df)
    es_train = (split == "train").to_numpy()
    vocab = fe.construir_vocabularios(df, es_train)
    E_num, E_cat, scaler = fe.construir(df, vocab, es_train)
    return df, es_train, vocab, E_num, E_cat, scaler


def test_formas_y_tipos(datos):
    df, _, _, E_num, E_cat, _ = datos
    assert E_num.shape == (len(df), len(fe.NOMBRES_NUM))
    assert E_cat.shape == (len(df), 3)
    assert E_num.dtype == np.float32
    assert E_cat.dtype == np.int32


def test_indices_reservados_para_pad_y_unk(datos):
    _, _, vocab, _, E_cat, _ = datos
    for tabla in vocab.values():
        assert min(tabla.values()) >= 2
    assert (E_cat >= cfg.UNK).all()


def test_vocabularios_solo_con_train(datos):
    """Penalizacion de -15 pts. Una categoria exclusiva de test debe caer
    a <UNK>, no ganarse un indice propio."""
    df, es_train, vocab, _, _, _ = datos
    comercios_train = set(df.loc[es_train, "merchant_id"])
    for m in vocab["merchant"]:
        assert m in comercios_train


def test_categoria_nueva_en_test_mapea_a_unk():
    df = gen.generar(21, n_tarjetas=60)
    es_train = np.zeros(len(df), dtype=bool)
    es_train[: len(df) // 2] = True
    vocab = fe.construir_vocabularios(df, es_train)
    df2 = df.copy()
    df2.loc[df2.index[-1], "merchant_id"] = 99999
    _, E_cat, _ = fe.construir(df2, vocab, es_train)
    assert E_cat[-1, 2] == cfg.UNK


def test_scaler_ajustado_solo_en_train(datos):
    df, es_train, vocab, E_num, _, scaler = datos
    # la media de las filas de train debe ser ~0 tras escalar; la global no
    assert abs(E_num[es_train].mean()) < 0.05
    E_crudo, _, _ = fe.construir(df, vocab, es_train, scaler=None)
    assert np.allclose(E_num, E_crudo, atol=1e-5)


def test_scaler_reutilizado_no_reajusta(datos):
    df, es_train, vocab, E_num, _, scaler = datos
    E_otra, _, scaler2 = fe.construir(df, vocab, es_train, scaler=scaler)
    assert scaler2 is scaler
    assert np.allclose(E_num, E_otra, atol=1e-6)


def test_delta_t_presente_y_no_negativo(datos):
    _, _, _, E_num, _, _ = datos
    assert "log_delta_t" in fe.NOMBRES_NUM
    assert "es_primera" in fe.NOMBRES_NUM


def test_ablacion_de_delta_t_elimina_la_columna():
    df = gen.generar(21, n_tarjetas=40)
    es_train = np.ones(len(df), dtype=bool)
    vocab = fe.construir_vocabularios(df, es_train)
    E_con, _, _ = fe.construir(df, vocab, es_train, usar_delta_t=True)
    E_sin, _, _ = fe.construir(df, vocab, es_train, usar_delta_t=False)
    assert E_sin.shape[1] == E_con.shape[1] - 1


def test_delta_t_no_cruza_de_tarjeta():
    df = gen.generar(21, n_tarjetas=40).reset_index(drop=True)
    es_train = np.ones(len(df), dtype=bool)
    vocab = fe.construir_vocabularios(df, es_train)
    E_num, _, _ = fe.construir(df, vocab, es_train)
    i = fe.NOMBRES_NUM.index("es_primera")
    primeras = df.groupby("card_id").head(1).index.to_numpy()
    # tras escalar no vale 1, pero debe ser el valor maximo de la columna
    assert np.isclose(E_num[primeras, i], E_num[:, i].max()).all()


def test_fraud_type_fuera_de_las_features(datos):
    _, _, _, E_num, _, _ = datos
    for col in cfg.COLUMNAS_ANALISIS:
        assert col not in fe.NOMBRES_NUM
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_features_evento.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'monitoreo.features_evento'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/monitoreo/features_evento.py
"""Matriz de eventos para el Modelo B.

`delta_t`, `same_merchant_as_prev` y `amount_ratio_to_prev` se calculan
sobre el flujo ORIGINAL, antes de ventanear: al barajar la ventana cada
evento se lleva consigo su delta_t. Ver seccion 8.2 del spec — por eso
existe la ablacion de delta_t como tercera comprobacion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config as cfg

NOMBRES_NUM = (
    "log_amount",
    "log_delta_t",
    "es_primera",
    "hour_sin",
    "hour_cos",
    "same_merchant_as_prev",
    "amount_ratio_to_prev",
)


def construir_vocabularios(df: pd.DataFrame, es_train: np.ndarray) -> dict[str, dict]:
    """Indices >= 2. El 0 es PAD y el 1 es UNK; deben ser distintos."""
    tr = df[es_train]
    vocab = {}
    for nombre, columna in (("mcc", "mcc"), ("channel", "channel"), ("merchant", "merchant_id")):
        categorias = sorted(tr[columna].unique().tolist())
        vocab[nombre] = {c: i + 2 for i, c in enumerate(categorias)}
    return vocab


def _codificar(serie: pd.Series, tabla: dict) -> np.ndarray:
    return serie.map(tabla).fillna(cfg.UNK).to_numpy(dtype=np.int32)


def construir(
    df: pd.DataFrame,
    vocab: dict[str, dict],
    es_train: np.ndarray,
    scaler: StandardScaler | None = None,
    usar_delta_t: bool = True,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Devuelve (E_num, E_cat, scaler). El scaler se ajusta SOLO con train."""
    g = df.groupby("card_id", sort=False)
    delta = g["ts"].diff().dt.total_seconds()
    es_primera = delta.isna().to_numpy().astype(np.float32)
    delta = delta.fillna(0.0).clip(lower=0.0).to_numpy()

    prev_merchant = g["merchant_id"].shift()
    prev_amount = g["amount"].shift()
    hora = df["ts"].dt.hour.to_numpy() + df["ts"].dt.minute.to_numpy() / 60.0

    columnas = {
        "log_amount": np.log1p(df["amount"].to_numpy()),
        "log_delta_t": np.log1p(delta),
        "es_primera": es_primera,
        "hour_sin": np.sin(2 * np.pi * hora / 24.0),
        "hour_cos": np.cos(2 * np.pi * hora / 24.0),
        "same_merchant_as_prev": (df["merchant_id"] == prev_merchant).astype(float).to_numpy(),
        "amount_ratio_to_prev": (
            df["amount"] / prev_amount.replace(0.0, np.nan)
        ).fillna(1.0).clip(upper=100.0).to_numpy(),
    }
    nombres = [n for n in NOMBRES_NUM if usar_delta_t or n != "log_delta_t"]
    E_num = np.column_stack([columnas[n] for n in nombres]).astype(np.float32)

    if scaler is None:
        scaler = StandardScaler().fit(E_num[es_train])
    E_num = scaler.transform(E_num).astype(np.float32)

    E_cat = np.column_stack(
        [
            _codificar(df["mcc"], vocab["mcc"]),
            _codificar(df["channel"], vocab["channel"]),
            _codificar(df["merchant_id"], vocab["merchant"]),
        ]
    ).astype(np.int32)

    return E_num, E_cat, scaler


def cardinalidades(vocab: dict[str, dict]) -> dict[str, int]:
    """Tamano de cada tabla de embedding, contando PAD y UNK."""
    return {k: max(v.values()) + 1 for k, v in vocab.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_features_evento.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/monitoreo/features_evento.py tests/test_features_evento.py
git commit -m "feat: features de evento, vocabularios solo-train y escalado"
```

---

### Task 9: Métricas

**Files:**
- Create: `src/monitoreo/metricas.py`
- Test: `tests/test_metricas.py`

**Interfaces:**
- Consumes: nada del proyecto.
- Produces:
  - `metricas.auc_pr(y, p) -> float`
  - `metricas.en_umbral(y, p, u) -> dict` con `precision, recall, f1, tp, fp, fn, tn`
  - `metricas.desglose_por_tipo(y, p, subtipo, u) -> pd.DataFrame` con columnas `grupo, n, n_fraude, auc_pr, recall`
  - `metricas.resumen(valores) -> tuple[float, float]` — media y desviación muestral

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metricas.py
import numpy as np
import pandas as pd
import pytest
from monitoreo import metricas as met


def test_auc_pr_perfecto():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert met.auc_pr(y, p) == pytest.approx(1.0)


def test_auc_pr_aleatorio_cerca_de_la_prevalencia():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.012).astype(int)
    p = rng.random(20000)
    assert abs(met.auc_pr(y, p) - 0.012) < 0.01


def test_en_umbral_cuenta_bien():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2])
    m = met.en_umbral(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)


def test_en_umbral_sin_positivos_no_divide_por_cero():
    y = np.array([1, 0])
    p = np.array([0.1, 0.1])
    m = met.en_umbral(y, p, 0.9)
    assert m["precision"] == 0.0 and m["f1"] == 0.0


def test_desglose_separa_los_subtipos():
    y = np.array([0, 1, 1, 1, 0, 0])
    p = np.array([0.1, 0.9, 0.8, 0.2, 0.1, 0.05])
    sub = np.array(["none", "f1_sondeo", "f1_golpe", "f2", "none", "none"])
    t = met.desglose_por_tipo(y, p, sub, 0.5)
    assert set(t["grupo"]) == {"f1_sondeo", "f1_golpe", "f2"}
    assert t.loc[t["grupo"] == "f1_golpe", "recall"].iloc[0] == pytest.approx(1.0)
    assert t.loc[t["grupo"] == "f2", "recall"].iloc[0] == pytest.approx(0.0)


def test_desglose_compara_cada_tipo_contra_los_legitimos():
    """Cada fila mide ese mecanismo vs todo lo legitimo, no vs otros fraudes."""
    y = np.array([0] * 100 + [1] * 5)
    p = np.concatenate([np.full(100, 0.1), np.full(5, 0.9)])
    sub = np.array(["none"] * 100 + ["f3"] * 5)
    t = met.desglose_por_tipo(y, p, sub, 0.5)
    assert t.loc[t["grupo"] == "f3", "n"].iloc[0] == 105


def test_resumen_media_y_sigma():
    m, s = met.resumen([0.50, 0.52, 0.54])
    assert m == pytest.approx(0.52)
    assert s == pytest.approx(np.std([0.50, 0.52, 0.54], ddof=1))


def test_no_expone_exactitud():
    """-15 pts si la exactitud aparece como metrica. No debe existir."""
    assert not hasattr(met, "exactitud")
    assert "accuracy" not in met.en_umbral(np.array([0, 1]), np.array([0.1, 0.9]), 0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metricas.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'monitoreo.metricas'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/monitoreo/metricas.py
"""AUC-PR primaria. La exactitud no se calcula ni se expone (-15 pts)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def auc_pr(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p))


def en_umbral(y: np.ndarray, p: np.ndarray, u: float) -> dict:
    pred = (p >= u).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def desglose_por_tipo(y: np.ndarray, p: np.ndarray, subtipo: np.ndarray, u: float) -> pd.DataFrame:
    """Un mecanismo a la vez, siempre contra el total de legitimos.

    Comparar un tipo de fraude contra otros fraudes no responde ninguna
    pregunta de negocio; contra los legitimos, si.
    """
    legitimos = subtipo == "none"
    filas = []
    for grupo in sorted(set(subtipo) - {"none"}):
        sel = legitimos | (subtipo == grupo)
        yg, pg = y[sel], p[sel]
        filas.append(
            {
                "grupo": grupo,
                "n": int(sel.sum()),
                "n_fraude": int(yg.sum()),
                "auc_pr": auc_pr(yg, pg) if yg.sum() else float("nan"),
                "recall": en_umbral(yg, pg, u)["recall"],
            }
        )
    return pd.DataFrame(filas)


def resumen(valores) -> tuple[float, float]:
    """Media y desviacion muestral sobre las semillas."""
    a = np.asarray(list(valores), dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_metricas.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/monitoreo/metricas.py tests/test_metricas.py
git commit -m "feat: metricas con AUC-PR primaria y desglose por mecanismo"
```

---

### Task 10: Modelo A

**Files:**
- Create: `src/monitoreo/modelos_a.py`
- Test: `tests/test_modelos_a.py`

**Interfaces:**
- Consumes: `X_A` de `features_agregadas.construir`.
- Produces:
  - `modelos_a.entrenar_logistica(X_tr, y_tr, seed) -> Pipeline`
  - `modelos_a.entrenar_gbm(X_tr, y_tr, X_val, y_val, seed) -> modelo`
  - `modelos_a.predecir(modelo, X) -> np.ndarray[float]` en `[0,1]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_modelos_a.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_modelos_a.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'monitoreo.modelos_a'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/monitoreo/modelos_a.py
"""Linea base sin orden. Una linea base debil invalida toda la comparacion."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAY_LIGHTGBM = True
except ImportError:  # fallback declarado en el spec
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAY_LIGHTGBM = False


def entrenar_logistica(X_tr: np.ndarray, y_tr: np.ndarray, seed: int) -> Pipeline:
    """Piso de referencia obligatorio."""
    return Pipeline(
        [
            ("escala", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=seed, n_jobs=-1)),
        ]
    ).fit(X_tr, y_tr)


def entrenar_gbm(X_tr, y_tr, X_val, y_val, seed: int):
    """Primario. Early stopping y todo ajuste ocurren sobre VALIDACION."""
    if HAY_LIGHTGBM:
        m = lgb.LGBMClassifier(
            n_estimators=2000, learning_rate=0.05, num_leaves=31,
            min_child_samples=50, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, class_weight="balanced",
            random_state=seed, n_jobs=-1, verbose=-1,
        )
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)], eval_metric="average_precision",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        return m
    m = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, early_stopping=True,
        validation_fraction=0.15, class_weight="balanced", random_state=seed,
    )
    return m.fit(X_tr, y_tr)


def predecir(modelo, X: np.ndarray) -> np.ndarray:
    return modelo.predict_proba(X)[:, 1].astype(float)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_modelos_a.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/monitoreo/modelos_a.py tests/test_modelos_a.py
git commit -m "feat: Modelo A - logistica y LightGBM sobre agregados causales"
```

---

### Task 11: Modelo B — GRU y lotes por índice

**Files:**
- Create: `src/monitoreo/modelos_b.py`
- Test: `tests/test_modelos_b.py`

**Interfaces:**
- Consumes: `win_idx`, `mask`, `E_num`, `E_cat`, `features_evento.cardinalidades`.
- Produces:
  - `modelos_b.Lotes(win_idx, mask, E_num, E_cat, y=None, batch_size=cfg.BATCH_SIZE, X_agg=None, barajar=False)` — subclase de `keras.utils.PyDataset`
  - `modelos_b.construir_modelo(K, d_num, cardinalidades, d_agg=0, unidades=cfg.UNIDADES_GRU) -> keras.Model`
  - `modelos_b.entrenar(modelo, lotes_tr, lotes_val, class_weight, epocas=cfg.EPOCAS_MAX) -> keras.callbacks.History`
  - `modelos_b.predecir(modelo, lotes) -> np.ndarray`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_modelos_b.py
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


def test_modelo_tiene_las_entradas_esperadas(datos):
    d = datos
    m = mb.construir_modelo(cfg.K, d["E_num"].shape[1], fe.cardinalidades(d["vocab"]))
    assert set(m.input_shape) == {"num", "mcc", "channel", "merchant", "mask"}
    assert m.output_shape == (None, 1)


def test_hibrido_agrega_la_entrada_de_agregados(datos):
    d = datos
    m = mb.construir_modelo(cfg.K, d["E_num"].shape[1], fe.cardinalidades(d["vocab"]), d_agg=11)
    assert "agg" in m.input_shape
    assert m.input_shape["agg"] == (None, 11)


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
    """EarlyStopping vigila val_auc_pr, nunca val_loss ni nada de test."""
    d = datos
    m = mb.construir_modelo(cfg.K, d["E_num"].shape[1], fe.cardinalidades(d["vocab"]))
    assert any(me.name == "auc_pr" for me in m.metrics)


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_modelos_b.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'monitoreo.modelos_b'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/monitoreo/modelos_b.py
"""Modelo B (GRU) y la variante hibrida de la apuesta C.

Los lotes hacen `gather` sobre la matriz de eventos usando win_idx: nunca
se materializa el tensor [N, K, d], que serian ~1.2 GB.
"""
from __future__ import annotations

import keras
import numpy as np
from keras import layers

from . import config as cfg


class Lotes(keras.utils.PyDataset):
    def __init__(self, win_idx, mask, E_num, E_cat, y=None,
                 batch_size=cfg.BATCH_SIZE, X_agg=None, barajar=False, **kw):
        super().__init__(**kw)
        self.win, self.mask = win_idx, mask
        self.E_num, self.E_cat = E_num, E_cat
        self.y, self.X_agg = y, X_agg
        self.batch_size, self.barajar = batch_size, barajar
        self.orden = np.arange(len(win_idx))

    def __len__(self):
        return int(np.ceil(len(self.win) / self.batch_size))

    def on_epoch_end(self):
        if self.barajar:
            np.random.shuffle(self.orden)

    def __getitem__(self, i):
        sel = self.orden[i * self.batch_size : (i + 1) * self.batch_size]
        w = self.win[sel]
        x = {
            "num": self.E_num[w],
            "mcc": self.E_cat[w, 0],
            "channel": self.E_cat[w, 1],
            "merchant": self.E_cat[w, 2],
            "mask": self.mask[sel],
        }
        if self.X_agg is not None:
            x["agg"] = self.X_agg[sel]
        if self.y is None:
            return x
        return x, self.y[sel].astype("float32")


def construir_modelo(K, d_num, cardinalidades, d_agg=0, unidades=cfg.UNIDADES_GRU):
    """GRU sobre LSTM: menos parametros y con K=20 la memoria larga no aporta."""
    ent = {
        "num": keras.Input((K, d_num), name="num"),
        "mcc": keras.Input((K,), dtype="int32", name="mcc"),
        "channel": keras.Input((K,), dtype="int32", name="channel"),
        "merchant": keras.Input((K,), dtype="int32", name="merchant"),
        "mask": keras.Input((K,), dtype="bool", name="mask"),
    }
    embs = [
        layers.Embedding(cardinalidades[k], cfg.DIM_EMB[k], name=f"emb_{k}")(ent[k])
        for k in ("mcc", "channel", "merchant")
    ]
    x = layers.Concatenate(name="secuencia")([ent["num"], *embs])
    x = layers.GRU(unidades, name="gru")(x, mask=ent["mask"])
    x = layers.Dropout(cfg.DROPOUT)(x)

    if d_agg:
        ent["agg"] = keras.Input((d_agg,), name="agg")
        x = layers.Concatenate(name="hibrido")([x, ent["agg"]])

    x = layers.Dense(32, activation="relu")(x)
    salida = layers.Dense(1, activation="sigmoid", name="p")(x)

    modelo = keras.Model(ent, salida)
    modelo.compile(
        optimizer=keras.optimizers.Adam(cfg.LR),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="auc_pr")],
    )
    return modelo


def entrenar(modelo, lotes_tr, lotes_val, class_weight, epocas=cfg.EPOCAS_MAX, verbose=0):
    """EarlyStopping sobre AUC-PR de VALIDACION. Nunca val_loss, jamas test."""
    parada = keras.callbacks.EarlyStopping(
        monitor="val_auc_pr", mode="max", patience=cfg.PACIENCIA,
        restore_best_weights=True, verbose=verbose,
    )
    return modelo.fit(
        lotes_tr, validation_data=lotes_val, epochs=epocas,
        class_weight=class_weight, callbacks=[parada], verbose=verbose,
    )


def predecir(modelo, lotes) -> np.ndarray:
    return modelo.predict(lotes, verbose=0).ravel().astype(float)


def pesos_de_clase(y) -> dict[int, float]:
    n = len(y)
    pos = float(y.sum())
    return {0: n / (2 * (n - pos)), 1: n / (2 * pos)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_modelos_b.py -v`
Expected: 8 passed. `test_entrena_y_supera_el_azar` tarda ~1-2 min en CPU.

- [ ] **Step 5: Commit**

```bash
git add src/monitoreo/modelos_b.py tests/test_modelos_b.py
git commit -m "feat: Modelo B - GRU con lotes por indice y mascara de padding"
```

---

### Task 12: Calibración y decisión económica

**Files:**
- Create: `src/monitoreo/calibracion.py`, `src/monitoreo/economia.py`
- Test: `tests/test_calibracion.py`, `tests/test_economia.py`

**Interfaces:**
- Consumes: `config.COSTO_FN`, `config.COSTO_FP`, `config.UMBRAL_TEORICO`.
- Produces:
  - `calibracion.ajustar(p_val, y_val) -> IsotonicRegression`
  - `calibracion.aplicar(cal, p) -> np.ndarray`
  - `economia.costo(y, p, u) -> float`
  - `economia.curva(y, p, n_pasos=1000) -> tuple[np.ndarray, np.ndarray]`
  - `economia.umbral_optimo(y, p, n_pasos=1000) -> tuple[float, float]`
  - `economia.ahorro_mensual(costo_a, costo_b, dias_test) -> float`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calibracion.py
import numpy as np
import pytest
from monitoreo import calibracion as cal


def test_calibra_puntajes_inflados():
    """Una red con class_weight sobreestima la probabilidad de fraude."""
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.02).astype(int)
    p_crudo = np.clip(np.where(y == 1, rng.beta(6, 3, 20000), rng.beta(2, 4, 20000)), 0, 1)
    c = cal.ajustar(p_crudo, y)
    p_cal = cal.aplicar(c, p_crudo)
    assert abs(p_cal.mean() - y.mean()) < abs(p_crudo.mean() - y.mean())


def test_preserva_el_ranking():
    """La calibracion isotonica es monotona: el AUC-PR no debe cambiar."""
    from monitoreo import metricas as met
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.05).astype(int)
    p = np.clip(y * 0.4 + rng.random(5000) * 0.6, 0, 1)
    c = cal.ajustar(p, y)
    assert met.auc_pr(y, cal.aplicar(c, p)) == pytest.approx(met.auc_pr(y, p), abs=1e-6)


def test_salida_en_cero_uno():
    rng = np.random.default_rng(2)
    y = (rng.random(2000) < 0.1).astype(int)
    p = rng.random(2000)
    q = cal.aplicar(cal.ajustar(p, y), rng.random(500))
    assert (q >= 0).all() and (q <= 1).all()
```

```python
# tests/test_economia.py
import numpy as np
import pytest
from monitoreo import config as cfg
from monitoreo import economia as eco


def test_costo_suma_fn_y_fp():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.01, 0.9, 0.01])
    # u=0.5 -> 1 TP, 1 FN, 1 FP, 1 TN
    assert eco.costo(y, p, 0.5) == pytest.approx(cfg.COSTO_FN + cfg.COSTO_FP)


def test_umbral_optimo_de_un_puntaje_calibrado_cae_cerca_del_teorico():
    """p* = 180/4200 = 0.0429. Es la cifra citable de la presentacion."""
    rng = np.random.default_rng(0)
    n = 200000
    p = rng.beta(0.6, 12.0, n)          # puntajes calibrados y desbalanceados
    y = (rng.random(n) < p).astype(int)  # por construccion P(y=1|p)=p
    u, _ = eco.umbral_optimo(y, p)
    assert abs(u - cfg.UMBRAL_TEORICO) < 0.02


def test_el_umbral_optimo_esta_lejos_de_0_5():
    rng = np.random.default_rng(0)
    n = 100000
    p = rng.beta(0.6, 12.0, n)
    y = (rng.random(n) < p).astype(int)
    u, _ = eco.umbral_optimo(y, p)
    assert u < 0.2


def test_curva_devuelve_umbrales_y_costos_alineados():
    y = np.array([1, 0, 1, 0] * 50)
    p = np.linspace(0, 1, 200)
    us, cs = eco.curva(y, p, n_pasos=100)
    assert us.shape == cs.shape == (100,)
    assert (cs >= 0).all()


def test_el_optimo_de_la_curva_es_el_minimo():
    y = np.array([1, 0, 1, 0] * 50)
    p = np.linspace(0, 1, 200)
    us, cs = eco.curva(y, p, n_pasos=1000)
    u, c = eco.umbral_optimo(y, p, n_pasos=1000)
    assert c == pytest.approx(cs.min())
    assert u == pytest.approx(us[cs.argmin()])


def test_ahorro_mensual_escala_por_dias():
    ahorro = eco.ahorro_mensual(costo_a=10000.0, costo_b=7000.0, dias_test=15.0)
    assert ahorro == pytest.approx(3000.0 * 30.0 / 15.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_calibracion.py tests/test_economia.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'monitoreo.calibracion'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/monitoreo/calibracion.py
"""Calibracion isotonica ajustada en VALIDACION.

Sin esto, el analisis economico compara peras con manzanas entre A y B:
los puntajes de una red con class_weight no son probabilidades.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def ajustar(p_val: np.ndarray, y_val: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_val, y_val)


def aplicar(cal: IsotonicRegression, p: np.ndarray) -> np.ndarray:
    return np.clip(cal.predict(p), 0.0, 1.0)
```

```python
# src/monitoreo/economia.py
"""Umbral por costo, no por F1 ni por percentil.

Dejar pasar un fraude cuesta 23 veces mas que molestar a un cliente
legitimo, asi que el umbral optimo esta lejisimos de 0.5.
"""
from __future__ import annotations

import numpy as np

from . import config as cfg

DIAS_DEL_MES = 30.0


def costo(y: np.ndarray, p: np.ndarray, u: float) -> float:
    pred = p >= u
    fn = int((~pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    return fn * cfg.COSTO_FN + fp * cfg.COSTO_FP


def curva(y: np.ndarray, p: np.ndarray, n_pasos: int = 1000):
    """Barrido de u en [0,1]. Vectorizado sobre el orden de los puntajes."""
    umbrales = np.linspace(0.0, 1.0, n_pasos)
    orden = np.argsort(p)
    y_ord = y[orden]
    p_ord = p[orden]
    # nro de positivos y negativos por debajo de cada umbral
    corte = np.searchsorted(p_ord, umbrales, side="left")
    pos_acum = np.concatenate([[0], np.cumsum(y_ord == 1)])
    neg_acum = np.concatenate([[0], np.cumsum(y_ord == 0)])
    fn = pos_acum[corte]                       # positivos que quedan por debajo
    fp = neg_acum[-1] - neg_acum[corte]        # negativos que quedan por encima
    return umbrales, fn * cfg.COSTO_FN + fp * cfg.COSTO_FP


def umbral_optimo(y: np.ndarray, p: np.ndarray, n_pasos: int = 1000) -> tuple[float, float]:
    """u* que minimiza el costo. Se elige SOBRE VALIDACION y se congela."""
    umbrales, costos = curva(y, p, n_pasos)
    i = int(np.argmin(costos))
    return float(umbrales[i]), float(costos[i])


def ahorro_mensual(costo_a: float, costo_b: float, dias_test: float) -> float:
    """Extrapolacion explicita del periodo de test a un mes de 30 dias."""
    return (costo_a - costo_b) * DIAS_DEL_MES / dias_test
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_calibracion.py tests/test_economia.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/monitoreo/calibracion.py src/monitoreo/economia.py tests/test_calibracion.py tests/test_economia.py
git commit -m "feat: calibracion isotonica y umbral optimo por costo"
```

---

### Task 13: Suite de integridad end-to-end

**Files:**
- Test: `tests/test_integridad.py`

**Interfaces:**
- Consumes: todos los módulos anteriores.
- Produces: nada. Es el checklist de §9 del spec del curso, ejecutable.

Este task no añade código de producción: verifica que las piezas encajen y que las cinco penalizaciones estén cerradas. Un reviewer puede aprobarlo o rechazarlo independientemente de cualquier módulo.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_integridad.py
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


def test_casos_de_fallo_esperado_existen_en_los_datos(pipeline):
    """§4.3: tarjetas con historia < K, y f1 con brecha > 24h."""
    d = pipeline
    cortas = (d["mask"].sum(axis=1) < cfg.K).sum()
    assert cortas > 0

    f1 = d["df"][d["df"]["fraud_type"] == "f1"]
    brechas = []
    for _, g in f1.groupby("card_id"):
        g = g.sort_values("ts")
        golpes = g[g["fraud_subtype"] == "f1_golpe"]["ts"]
        sondeos = g[g["fraud_subtype"] == "f1_sondeo"]["ts"]
        if len(golpes) and len(sondeos):
            brechas.append((golpes.iloc[0] - sondeos.iloc[-1]).total_seconds() / 3600.0)
    assert max(brechas) > 24.0, "falta el caso de f1 con brecha larga"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_integridad.py -v`
Expected: PASS si los tasks 1-12 están bien. Si algo falla aquí, el bug está en un módulo previo — corregirlo ahí, no relajar este test.

- [ ] **Step 3: Correr la suite completa**

Run: `python -m pytest -v`
Expected: todos los tests de tasks 1-13 en verde.

- [ ] **Step 4: Verificar el tiempo de la suite**

Run: `python -m pytest --durations=10`
Expected: la suite completa bajo 5 minutos. Si algún test tarda más, reducir `n_tarjetas` en su fixture.

- [ ] **Step 5: Commit**

```bash
git add tests/test_integridad.py
git commit -m "test: checklist de penalizaciones como suite end-to-end"
```

---

### Task 14: Notebook — datos, Modelo A y Modelo B (evidencias 1 y 2)

**Files:**
- Create: `notebooks/proyecto1_mazariegos_herrera.ipynb`
- Create: `src/monitoreo/figuras.py`

**Interfaces:**
- Consumes: todos los módulos.
- Produces: `figuras.curvas_pr(resultados, ruta) -> None`; `figuras.auc_vs_k(ks, aucs, ruta) -> None`; `figuras.curva_costo(curvas, umbrales, ruta) -> None`. Cada una guarda un PNG a 150 dpi y devuelve `None`.

Secciones del notebook (todas con `MONITOREO_DEV` desactivado para la corrida final):

1. **Portada y contrato.** Versiones (`reproducibilidad.versiones()`), semillas, y el checklist anti-fuga de §2.6 en markdown, apuntando a `tests/`.
2. **Generación y EDA.** `generar(SEED_DATOS)`; tasa de fraude global y por tipo; histograma de montos legítimo vs f1-golpe vs f3; distribución de tx por tarjeta.
3. **Partición temporal.** `particion.tabla(...)` → **Tabla 1** con fechas de corte, tamaños y tasas.
4. **Asserts del contrato de comparabilidad** (los de `test_integridad.py`, ejecutados a la vista del comité).
5. **Modelo A.** Logística y LightGBM sobre las 3 semillas; AUC-PR val media ± σ.
6. **Modelo B.** GRU sobre las 3 semillas; AUC-PR val media ± σ; comparación con A.
7. **Fig. 2** — curvas PR de A y B superpuestas sobre validación.

- [ ] **Step 1: Escribir el módulo de figuras y su test**

```python
# tests/test_figuras.py
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
```

```python
# src/monitoreo/figuras.py
"""Figuras del informe. Cada una guarda un PNG a 150 dpi."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from . import config as cfg


def _guardar(fig, ruta):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def curvas_pr(resultados: dict[str, tuple], ruta) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for nombre, (y, p) in resultados.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=nombre, lw=2)
    ax.set_xlabel("Exhaustividad (recall)")
    ax.set_ylabel("Precisión")
    ax.set_title("Curvas precisión–exhaustividad (validación)")
    ax.legend()
    ax.grid(alpha=0.3)
    _guardar(fig, ruta)


def auc_vs_k(ks, aucs, auc_modelo_a: float, ruta) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, aucs, "o-", lw=2, label="Modelo B (GRU)")
    ax.axhline(auc_modelo_a, ls="--", color="gray", label="Modelo A (agregados)")
    ax.set_xlabel("K — eventos de historia")
    ax.set_ylabel("AUC-PR (validación)")
    ax.set_title("¿Cuánta historia hace falta?")
    ax.legend()
    ax.grid(alpha=0.3)
    _guardar(fig, ruta)


def curva_costo(curvas: dict[str, tuple], umbrales: dict[str, float], ruta) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for nombre, (u, c) in curvas.items():
        linea, = ax.plot(u, c, lw=2, label=nombre)
        ux = umbrales[nombre]
        ax.axvline(ux, ls=":", color=linea.get_color())
        ax.annotate(f"u*={ux:.3f}", (ux, np.interp(ux, u, c)),
                    textcoords="offset points", xytext=(6, 8), color=linea.get_color())
    ax.axvline(cfg.UMBRAL_TEORICO, ls="--", color="black", alpha=0.5,
               label=f"teórico {cfg.UMBRAL_TEORICO:.4f}")
    ax.set_xlim(0, 0.3)
    ax.set_xlabel("Umbral de bloqueo")
    ax.set_ylabel("Costo esperado (Q)")
    ax.set_title("Costo vs umbral — bloquear a partir de ~4.3 %")
    ax.legend()
    ax.grid(alpha=0.3)
    _guardar(fig, ruta)
```

- [ ] **Step 2: Run test to verify it fails, then passes**

Run: `python -m pytest tests/test_figuras.py -v`
Expected: primero FAIL con `ModuleNotFoundError`, luego 3 passed.

- [ ] **Step 3: Escribir el notebook con las secciones 1-7**

Cada sección es una celda markdown de contexto seguida de una celda de código que importa de `monitoreo` y muestra el resultado. Nada de lógica nueva en el notebook: si hace falta una función, va a `src/` con su test.

- [ ] **Step 4: Ejecutar el notebook completo desde kernel limpio**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/proyecto1_mazariegos_herrera.ipynb`
Expected: sin excepciones; Tabla 1 y Fig. 2 generadas.

- [ ] **Step 5: Commit**

```bash
git add src/monitoreo/figuras.py tests/test_figuras.py notebooks/proyecto1_mazariegos_herrera.ipynb informe/figuras
git commit -m "feat: notebook - EDA, particion, Modelo A y Modelo B"
```

---

### Task 15: Notebook — apuesta C y las dos pruebas de falsificación (evidencias 3 y 4)

**Files:**
- Modify: `notebooks/proyecto1_mazariegos_herrera.ipynb`

**Interfaces:**
- Consumes: `ventanas.permutar`, `modelos_b.construir_modelo(d_agg=...)`, `features_evento.construir(usar_delta_t=False)`, `metricas.desglose_por_tipo`.
- Produces: **Tabla 4** (permutación), **Fig. 3** (AUC-PR vs K), **Tabla 5** (apuesta C), tabla de desglose por mecanismo.

**Orden obligatorio:** la celda de la hipótesis de C se escribe y se ejecuta con `datetime.now()` **antes** de la primera celda que entrena C. Si se ejecuta después, la evidencia 4 no puntúa.

- [ ] **Step 1: Celda de hipótesis fechada**

```python
from datetime import datetime
HIPOTESIS_C = (
    "Creemos que concatenar el estado oculto del GRU con el vector de agregados "
    "de A antes de la capa de salida mejorará la AUC-PR en validación porque f2 "
    "y f3 son mayormente capturables por agregados mientras que f1 requiere "
    "orden, y ningún modelo puro cubre ambos regímenes. Lo consideraremos útil "
    "si la AUC-PR en validación supera a la del mejor de A y B por al menos "
    "0.02 absoluto, promediado sobre 3 semillas."
)
FECHA_HIPOTESIS = datetime.now().isoformat(timespec="seconds")
UMBRAL_EXITO_C = 0.02
print(FECHA_HIPOTESIS, "\n", HIPOTESIS_C)
```

- [ ] **Step 2: Prueba de permutación sobre validación**

```python
import numpy as np
from monitoreo import metricas as met, modelos_b as mb, ventanas as ven

filas = []
for modo in ("original", "full", "history"):
    win_eval = win[va] if modo == "original" else ven.permutar(
        win[va], mask[va], modo, np.random.default_rng(0))
    lotes = mb.Lotes(win_eval, mask[va], E_num, E_cat, batch_size=1024)
    filas.append({
        "variante": modo,
        "auc_pr_B": met.auc_pr(y[va], mb.predecir(modelo_b, lotes)),
        "auc_pr_A": met.auc_pr(y[va], p_a_val),   # A no depende de win: constante
    })
tabla_permutacion = pd.DataFrame(filas)
assert tabla_permutacion["auc_pr_A"].nunique() == 1, "BUG: A se movió al permutar"
tabla_permutacion
```

- [ ] **Step 3: Ablación de Δt y curva de recorte de historia**

La ablación reentrena B con `usar_delta_t=False` sobre la misma partición y semillas. La curva evalúa `K ∈ {1,3,5,10,20}` reentrenando una vez por `K` con la semilla 7, y se grafica con `figuras.auc_vs_k` contra la línea de A.

- [ ] **Step 4: Apuesta C y desglose por mecanismo**

Entrenar el híbrido con `d_agg=X_A.shape[1]` sobre las 3 semillas, comparar contra `max(AUC-PR_A, AUC-PR_B)`, y escribir el veredicto explícito (se sostuvo / no se sostuvo) con la cifra. Después, `metricas.desglose_por_tipo` para A, B y C.

- [ ] **Step 5: Commit**

```bash
git add notebooks/proyecto1_mazariegos_herrera.ipynb informe/figuras
git commit -m "feat: notebook - apuesta C, permutacion, ablacion de delta_t y curva de K"
```

---

### Task 16: Corrida única de test, artefactos, README e informe

**Files:**
- Modify: `notebooks/proyecto1_mazariegos_herrera.ipynb`
- Create: `README.md`, `artefactos/config.json`, `artefactos/generador_datos.py`, `informe/informe.md`
- Test: `tests/test_artefactos.py`

**Interfaces:**
- Consumes: todo lo anterior.
- Produces: `artefactos/modelo_candidato.keras`, `artefactos/scaler.pkl`, `artefactos/vocab_embeddings.json`, `artefactos/config.json`, `artefactos/generador_datos.py`.

**Punto de no retorno:** esta es la única vez que se toca el test. Antes de ejecutar la celda, `u*` debe estar congelado desde validación y la arquitectura decidida.

- [ ] **Step 1: Congelar el umbral y ejecutar test una sola vez**

```python
from datetime import datetime
from monitoreo import calibracion as cal, economia as eco

# --- congelado desde VALIDACION, no se reoptimiza ---
cal_a = cal.ajustar(p_a_val, y[va]); cal_b = cal.ajustar(p_b_val, y[va])
u_a, _ = eco.umbral_optimo(y[va], cal.aplicar(cal_a, p_a_val))
u_b, _ = eco.umbral_optimo(y[va], cal.aplicar(cal_b, p_b_val))
print(f"u*_A={u_a:.4f}  u*_B={u_b:.4f}  teórico={cfg.UMBRAL_TEORICO:.4f}")

FECHA_EJECUCION_TEST = datetime.now().isoformat(timespec="seconds")
print("TEST EJECUTADO UNA SOLA VEZ:", FECHA_EJECUCION_TEST)

p_a_test = cal.aplicar(cal_a, ma.predecir(gbm_a, X_A.to_numpy()[te]))
p_b_test = cal.aplicar(cal_b, mb.predecir(modelo_b, lotes_test))
costo_a = eco.costo(y[te], p_a_test, u_a)
costo_b = eco.costo(y[te], p_b_test, u_b)
dias_test = (df.loc[te, "ts"].max() - df.loc[te, "ts"].min()).total_seconds() / 86400
ahorro = eco.ahorro_mensual(costo_a, costo_b, dias_test)
print(f"Ahorro mensual estimado: Q{ahorro:,.0f} (extrapolado de {dias_test:.1f} días)")
```

- [ ] **Step 2: Guardar artefactos y escribir su test**

```python
# tests/test_artefactos.py
import json
from pathlib import Path
import pytest
from monitoreo import config as cfg

ART = cfg.DIR_ARTEFACTOS


@pytest.mark.skipif(not (ART / "config.json").exists(), reason="notebook aun no ejecutado")
def test_config_json_tiene_lo_que_pide_el_spec():
    c = json.loads((ART / "config.json").read_text(encoding="utf-8"))
    for clave in ("K", "seed_datos", "seeds_modelo", "umbral_u_estrella",
                  "fecha_ejecucion_test", "versiones"):
        assert clave in c
    assert c["K"] == cfg.K


@pytest.mark.skipif(not (ART / "config.json").exists(), reason="notebook aun no ejecutado")
def test_todos_los_artefactos_presentes():
    for nombre in ("modelo_candidato.keras", "scaler.pkl", "vocab_embeddings.json",
                   "config.json", "generador_datos.py"):
        assert (ART / nombre).exists(), f"falta {nombre}"


@pytest.mark.skipif(not (ART / "generador_datos.py").exists(), reason="notebook aun no ejecutado")
def test_generador_entregado_es_el_mismo_que_se_uso():
    fuente = (cfg.RAIZ / "src" / "monitoreo" / "generador.py").read_text(encoding="utf-8")
    copia = (ART / "generador_datos.py").read_text(encoding="utf-8")
    assert fuente == copia
```

Celda del notebook:

```python
import json, pickle, shutil
from monitoreo import reproducibilidad as rep

cfg.DIR_ARTEFACTOS.mkdir(parents=True, exist_ok=True)
modelo_candidato.save(cfg.DIR_ARTEFACTOS / "modelo_candidato.keras")
with open(cfg.DIR_ARTEFACTOS / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
(cfg.DIR_ARTEFACTOS / "vocab_embeddings.json").write_text(
    json.dumps({k: {str(a): b for a, b in v.items()} for k, v in vocab.items()}, indent=2),
    encoding="utf-8")
(cfg.DIR_ARTEFACTOS / "config.json").write_text(json.dumps({
    "K": cfg.K, "seed_datos": cfg.SEED_DATOS, "seeds_modelo": list(cfg.SEEDS_MODELO),
    "umbral_u_estrella": {"A": u_a, "B": u_b}, "umbral_teorico": cfg.UMBRAL_TEORICO,
    "costos": {"FN": cfg.COSTO_FN, "FP": cfg.COSTO_FP},
    "fecha_hipotesis_C": FECHA_HIPOTESIS,
    "fecha_ejecucion_test": FECHA_EJECUCION_TEST,
    "versiones": rep.versiones(),
}, indent=2), encoding="utf-8")
shutil.copy(cfg.RAIZ / "src" / "monitoreo" / "generador.py",
            cfg.DIR_ARTEFACTOS / "generador_datos.py")
```

- [ ] **Step 3: Escribir el README con las secciones obligatorias**

`README.md` debe contener, en este orden: **Reproducción** (comandos exactos, semillas, tiempo aproximado); **Versiones** (salida de `rep.versiones()`); **Declaración de uso de IA** (para qué se usó y qué verificó cada quien — Mazariegos y Herrera lo completan a mano); **Tres decisiones técnicas** (GRU vs alternativas / `K=20` justificado por Fig. 3 / Ruta A vs B), cada una con alternativas consideradas y la evidencia que inclinó la decisión; **Candidato al Proyecto Final** con el contrato de entrada/salida:

```
Entrada:  {card_id, últimas 20 transacciones [{ts, amount, mcc, channel, merchant_id}]}
Salida:   {risk_score: float ∈ [0,1], decision: "block"|"review"|"allow", model_version: str}
Latencia objetivo: < 100 ms p95
```

- [ ] **Step 4: Escribir el informe (máx. 7 páginas, sin código)**

`informe/informe.md` → exportar a PDF. Las seis evidencias localizables en <15 s, escritas para el comité de riesgos ("detecta 8 de cada 10 fraudes bloqueando 1 de cada 400 compras legítimas"). Última página: matriz de evidencias con una limitación honesta por fila, incluyendo la de §11.3 del spec (el calibrador y `u*` se ajustan ambos en validación).

- [ ] **Step 5: Verificación final y commit**

```bash
python -m pytest -v
jupyter nbconvert --to notebook --execute --inplace notebooks/proyecto1_mazariegos_herrera.ipynb
git add -A
git commit -m "feat: corrida de test, artefactos, README e informe"
```

Antes de entregar, recorrer §9 del spec del curso punto por punto contra la salida del notebook.

---

## Self-Review

**Cobertura del spec:**

| Sección del spec | Task |
|---|---|
| §1 pregunta / §2 decisiones | Documentadas en README (Task 16) |
| §3 contrato de comparabilidad | Tasks 6, 13 |
| §4 generador, f1 monótono, ráfagas confusoras | Tasks 2, 3 |
| §4.3 casos de fallo esperado | Task 3 (brecha larga), Task 13 (verificación) |
| §5 partición temporal | Task 4 |
| §6 anti-fuga como tests | Tasks 5, 6, 8, 13 |
| §7 Modelo A, `closed='left'`, dos punteros | Tasks 5, 10 |
| §8 Modelo B, permutación, padding | Tasks 7, 8, 11 |
| §9 apuesta C híbrida | Tasks 11 (`d_agg`), 15 |
| §10 métricas y falsificación | Tasks 9, 15 |
| §11 calibración y economía | Tasks 12, 15, 16 |
| §12 estructura y parámetros | Task 1 |
| §13 presupuesto de cómputo | Task 1 (`DEV_MODE`) |
| §14 entregables y seis evidencias | Tasks 14, 15, 16 |
| §16 definición de hecho | Tasks 13, 16 |

**Consistencia de tipos verificada:** `construir` devuelve `pd.DataFrame` en `features_agregadas` y `tuple[np.ndarray, np.ndarray, StandardScaler]` en `features_evento` — nombres iguales, módulos distintos, sin colisión porque siempre se importa el módulo, nunca la función suelta. `ventanas.construir` y `ventanas.permutar` comparten la firma `(win_idx, mask, ...)`. `metricas.auc_pr(y, p)` mantiene el orden `(y, p)` en los nueve puntos donde se llama.

**Riesgo conocido:** los divisores `5.5` y `4.0` de `generar` son estimaciones de tx por episodio. Si `test_tasa_de_fraude_en_el_rango_del_spec` falla, se ajustan esos divisores — nunca `TASA_FRAUDE`, que es el parámetro que el spec congela.
