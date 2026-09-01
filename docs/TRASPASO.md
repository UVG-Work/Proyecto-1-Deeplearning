# Traspaso — Proyecto 1, Monitoreo transaccional

**De:** Andres Mazariegos · **Para:** June Herrera
**Fecha:** 1 de septiembre de 2026
**Rama:** `feat/monitoreo-transaccional`

---

## 1. Qué está hecho

La **capa de datos y representación completa** — tasks 1 a 8 del plan. Todo con tests, todo revisado.

| Módulo | Qué hace |
|---|---|
| `src/monitoreo/config.py` | Parámetros congelados: semillas, `K=20`, costos, percentiles de corte, `HORIZONTE_DIAS=90` |
| `src/monitoreo/reproducibilidad.py` | `fijar_semillas(seed)`, `versiones()` para el README |
| `src/monitoreo/generador.py` | `generar(seed)` — flujo legítimo + f1/f2/f3 + ráfagas confusoras |
| `src/monitoreo/particion.py` | `asignar_split(df)`, `tabla(df, split)` — corte temporal global |
| `src/monitoreo/features_agregadas.py` | `construir(df)` → `X_A`, 30 agregados causales para el Modelo A |
| `src/monitoreo/ventanas.py` | `construir(df, K)` → `(win_idx, mask)`; `permutar(win_idx, mask, modo, rng)` |
| `src/monitoreo/features_evento.py` | Vocabularios, `E_num`, `E_cat`, escalado — entrada del Modelo B |

Suite: **73 tests en verde**. Corré `python -m pytest` desde la raíz.

## 2. El contrato que tenés que respetar

Todo lo que sigue se apoya en un **índice canónico único**. Si lo rompés, A y B dejan de ser comparables y el proyecto pierde su tesis.

```
eventos          N filas ordenadas por (card_id, ts), RangeIndex limpio
X_A   [N, 30]    agregados causales           -> Modelo A
E_num [N, 7]     float32, features por evento -> Modelo B
E_cat [N, 3]     int32 (mcc, channel, merchant)
win_idx [N, 20]  int32, indices posicionales dentro de eventos
mask    [N, 20]  bool, padding al inicio
```

Tres invariantes que conviene que el notebook verifique con `assert` a la vista del comité:

- `len(X_A) == len(win_idx) == len(mask) == N`
- `win_idx[:, -1] == np.arange(N)` — la última posición es siempre la transacción puntuada
- `X_A`, `E_num` y `win_idx` comparten el mismo vector de `split`

**Cuidado con el padding.** Las posiciones rellenadas guardan el índice de la propia fila con `mask=False`. Si hacés `gather` por `win_idx` sin aplicar la máscara, leés la transacción puntuada 20 veces en silencio, sin error. Aplicá siempre la máscara.

## 3. Cómo levantarlo

```python
from monitoreo import config as cfg, generador as gen, particion as part
from monitoreo import features_agregadas as fa, features_evento as fe, ventanas as ven

df = gen.generar(cfg.SEED_DATOS)              # 4000 tarjetas; usa n_tarjetas=400 para iterar
split = part.asignar_split(df)
es_train = (split == "train").to_numpy()

X_A = fa.construir(df)
vocab = fe.construir_vocabularios(df, es_train)
E_num, E_cat, scaler = fe.construir(df, vocab, es_train)
win, mask = ven.construir(df, cfg.K)
```

Para la prueba de permutación (§6.1 del enunciado), **no reentrenes**: reevaluá B con los mismos pesos sobre `ven.permutar(win, mask, "full", rng)` y `"history"`. El contenido de cada ventana se conserva por construcción, así que la caída de desempeño solo puede venir del orden.

## 4. Qué falta — tasks 9 a 16

Están **completamente especificados con código y tests** en `docs/superpowers/plans/2026-08-31-monitoreo-transaccional.md`. No hay que diseñar nada, hay que ejecutarlos.

| Task | Entrega |
|---|---|
| 9 | `metricas.py` — AUC-PR, métricas en umbral, desglose por mecanismo |
| 10 | `modelos_a.py` — logística + LightGBM |
| 11 | `modelos_b.py` — GRU y lotes por índice |
| 12 | `calibracion.py` y `economia.py` — isotónica, curva de costo, `u*` |
| 13 | `tests/test_integridad.py` — el checklist de penalizaciones, ejecutable |
| 14 | Notebook: EDA, partición, Modelos A y B (evidencias 1 y 2) |
| 15 | Notebook: apuesta C, permutación, curva de `K` (evidencias 3 y 4) |
| 16 | Corrida única de test, artefactos, README, informe |

El orden importa: 9 antes que 10 y 11, y 13 conviene correrlo antes de tocar el notebook.

## 5. Decisiones que tomé sobre la marcha

Estas se apartan de lo que decía el spec original. Están todas justificadas en `.superpowers/sdd/2026-08-31-monitoreo-transaccional/progress.md`, con lo que cuesta si me equivoqué. **Repasalas antes del viernes: la defensa se asigna al azar.**

**Los sondeos de f1 escalan de forma monótona, y hay ráfagas legítimas confusoras.**
Un f1 ingenuo no habría dependido del orden: en el momento del golpe, el Modelo A ve monto alto + muchas transacciones recientes + muchos comercios distintos, una firma agregada suficiente para ganar sin leer secuencia. Media, máximo, conteo y cardinalidad de un conjunto son invariantes a permutación; "estrictamente creciente" no lo es. Por eso los sondeos suben (Q5 → Q12 → Q25 → Q38, el atacante tanteando el límite) y por eso se inyectan ráfagas legítimas con montos desordenados y la misma firma agregada. Sin lo primero el generador estaría amañado a favor de A; sin lo segundo, a favor de B.

**f1 y las ráfagas son idénticas en todas sus marginales.** Canal, país y distribución de monto del evento grande. Un review encontró que los sondeos iban siempre por canal `online` mientras las ráfagas mezclaban POS/online — con eso A separaba los dos mecanismos sin leer orden. Hay un test que lo vigila midiendo los inyectores directamente.

**Todas las tarjetas viven sobre la misma ventana de 90 días.** Antes cada tarjeta acumulaba brechas exponenciales a su propia tasa, así que sus spans iban de 16 a 275 días y la línea de tiempo global se vaciaba al final: test quedaba 27 veces más ralo que train, con solo el 36 % de las tarjetas. Eso infla `Δt` en test justamente para el Modelo B, cuya feature central es `Δt`, y habría sesgado la comparación en contra de B por un artefacto del generador.

**El fraude se reparte por todo el horizonte.** Los episodios se anclaban en `uniform(0.1, 0.9)`, dejando el primer y el último 10 % sin fraude. Test es el último 15 % del tiempo: caía casi entero en la zona muerta, con 0.50 % de prevalencia contra 1.29 % de train. Como la AUC-PR depende de la prevalencia y el umbral se congela en validación, eso habría distorsionado las cifras del informe y el ahorro mensual.

## 6. Trampas conocidas

- **No toques test hasta el final.** Todo se decide en validación. Elegir umbral o arquitectura mirando test cuesta 10 pts. Cuando lo corras, imprimí la fecha y hora en la celda.
- **La exactitud no se reporta**, ni como nota al pie. AUC-PR es la primaria.
- **`fraud_type` y `fraud_subtype` son solo para análisis.** Nunca entran a una matriz de features; hay `assert` que lo verifican.
- **El umbral óptimo es ~0.043, no 0.5.** `p* = 180/4200`. Es la cifra más citable de la presentación: dejar pasar un fraude cuesta 23 veces más que molestar a un cliente legítimo.
- **Los puntajes de la red no están calibrados.** Aplicá isotónica ajustada en validación antes de convertir a quetzales, o el análisis económico compara peras con manzanas.
- El scaler y los vocabularios se ajustan **solo con train**. Hay tests que fallan si no.

## 7. Minor diferidos

Ninguno bloquea, pero conviene mirarlos antes de entregar:

- `dev_mode()` trata `MONITOREO_DEV=0` como verdadero
- `tasa_dia` quedó vestigial en `perfiles`: nada la lee aguas abajo
- El test de simetría de f1 acopla a funciones privadas (`_inyectar_*`)
- `generador.py` va por ~310 líneas haciendo perfiles, flujo legítimo y cuatro inyectores
- Un warning de `pytest-asyncio` ensucia la salida de la suite; viene de un plugin ajeno al proyecto

## 8. Dónde está todo

Todo esto está en el repo:

- **Plan con los tasks pendientes:** `docs/superpowers/plans/2026-08-31-monitoreo-transaccional.md`
- **Spec de diseño:** `docs/superpowers/specs/2026-08-31-monitoreo-transaccional-design.md`
- **Enunciado del curso:** `specs_proyecto1_monitoreo_transaccional.md`
- **Decisiones con su justificación:** `docs/DECISIONES.md` — las 14 decisiones tomadas
  durante la construcción, cada una con la evidencia que la inclinó y lo que cuesta si
  resultó equivocada. Es el insumo directo para la sección de decisiones técnicas del
  README y para la defensa en la presentación.

Solo en la máquina de Andres, no en el repo: los reportes de verificación por task
(`.superpowers/sdd/2026-08-31-monitoreo-transaccional/task-N-report.md`), con la salida
de pytest de cada ciclo y las mediciones. Pedímelos si hace falta el detalle.
