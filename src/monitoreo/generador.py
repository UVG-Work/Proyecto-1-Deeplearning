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
    ts = np.sort(ts.values)
    # Remuestrear la hora puede producir colisiones. Se desempatan con
    # microsegundos crecientes: sin esto, drop_duplicates bajaria el conteo
    # de la tarjeta por debajo de TX_MIN y romperia el contrato del generador.
    iguales = np.concatenate([[False], ts[1:] == ts[:-1]])
    if iguales.any():
        ts = ts + np.cumsum(iguales) * np.timedelta64(1, "us")
        ts = np.sort(ts)
    return ts


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
