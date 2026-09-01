# Diseño — Proyecto 1: Monitoreo transaccional

**Autores:** Andres Mazariegos, June Herrera
**Fecha:** 31 de agosto de 2026
**Spec del curso:** `specs_proyecto1_monitoreo_transaccional.md`
**Entrega:** viernes 4 de septiembre de 2026, 23:59

---

## 1. La pregunta que se responde

> ¿El **orden** de las transacciones aporta información que las variables agregadas no capturan, bajo qué condiciones, y cuánto vale esa información en quetzales?

El proyecto no se califica por entrenar una red recurrente. Se califica por producir evidencia falsable sobre esa pregunta. De ahí que el diseño priorice, en este orden:

1. Que A y B sean comparables punto por punto **por construcción**, no por disciplina.
2. Que la prueba de permutación no pueda dar un falso positivo por un bug del pipeline.
3. Que un resultado negativo sea reportable sin rehacer nada.

## 2. Decisiones tomadas

| Decisión | Elegido | Alternativas descartadas y por qué |
|---|---|---|
| Ruta de datos | **A — generador sintético** | Ruta B pública: sin control sobre la dependencia del orden, riesgo alto de que la permutación no muestre nada, y 2 días de limpieza que no hay. `creditcard.csv` de ULB queda fuera por no tener ID de tarjeta. |
| Framework de B | **TensorFlow / Keras** | PyTorch (ya instalado) obligaría a entregar `.pt` en vez del `.keras` que exige §8. Keras 3 sobre backend torch resuelve el artefacto pero es un camino menos transitado si algo falla a 4 días de la entrega. |
| Representación de ventanas | **Índices `win_idx [N,K]` + `gather`** | Materializar `[N,K,d]` en float32 son ~1.2 GB y hace el shuffle sobre floats ya escalados, donde un bug de contenido es invisible. Construir ventanas en el `data generator` con pandas es demasiado lento en CPU. |
| Organización del código | **Paquete `src/` con tests + notebook delgado** | Notebook monolítico: sin tests, y un bug de fuga causal se descubre tarde o nunca. Justamente la lógica penalizada (§9) es la que necesita pruebas. |
| Apuesta C | **Híbrido A+B** | Robustez a fraude no visto y autoencoder cuestan un ciclo extra de entrenamiento; atención interpretable la marca el propio spec como costo alto. El híbrido reutiliza A y B ya entrenados como controles. |

### 2.1 Las tres decisiones técnicas del README (§8.3)

Cada una debe poder defenderse con la evidencia que la inclinó, no con la preferencia:

1. **GRU vs LSTM vs CNN 1D vs Transformer** para B.
2. **`K = 20`**, justificado por la curva de recorte de historia (§10.3), no por decreto.
3. **Ruta A sintética vs Ruta B pública**, justificada por la tabla de §1.1 del spec del curso y por el hecho de que f3 existe como control negativo.

## 3. Contrato de comparabilidad

Un único módulo produce el índice canónico. Todo lo demás lo consume.

```
eventos          N_ev filas ordenadas por (card_id, ts), con posición global entera
muestras         sample_id → (fila_evento, card_id, ts, y, fraud_type, split)
X_A   [N, d_a]   agregados causales                        ← Modelo A
E     [N_ev,d_e] features por evento (escaladas con train) ← Modelo B
win_idx [N, K]   int32, índices dentro de `eventos`         ← Modelo B
mask    [N, K]   bool, padding para historia < K
```

Invariantes que el notebook verifica con `assert` antes de entrenar nada:

- `len(X_A) == len(win_idx) == len(mask) == len(y)`
- `win_idx[:, -1]` apunta exactamente a la fila de la propia transacción puntuada
- `X_A`, `E` y `win_idx` comparten el mismo vector `split`

Consecuencias que se ganan gratis:

- **La permutación preserva el contenido por construcción.** Barajar es permutar enteros dentro de una fila de `win_idx`; es imposible que altere qué eventos hay en la ventana. Elimina el riesgo que §11 del spec marca como "verificar que el shuffle realmente se aplique".
- **A es invariante a la permutación estructuralmente.** A nunca toca `win_idx`. Si A se moviera, sería un bug del pipeline, y el `assert` numérico lo delata.
- **Memoria:** `win_idx` con 400k×20 int32 son 32 MB, contra ~1.2 GB del tensor materializado.

### 3.1 Unidad de predicción

Predicción a nivel de transacción. Para la transacción `t` de la tarjeta `c` con historia `t-K+1 … t`, ¿es `t` fraudulenta? Ambos modelos devuelven `p ∈ [0,1]`; el umbral se decide después y por costo.

Se puntúa **toda** transacción, incluidas las de tarjetas con menos de `K` de historia (ventana rellenada con padding). No se filtran: son exactamente el caso donde §2.4 del spec exige declarar y verificar un fallo esperado, y excluirlas sería esconder la evidencia.

## 4. Generador sintético

`generate(seed) → DataFrame`, reproducible bit a bit. Es un entregable en sí mismo (`artefactos/generador_datos.py`).

**Volumen:** 4,000 tarjetas × 60–200 transacciones, con el conteo por tarjeta muestreado de una lognormal truncada de **media ≈ 100** (una uniforme sobre 60–200 daría media 130 y ~520k eventos, fuera del objetivo). Total ≈ **400,000 eventos**. Tasa de fraude objetivo **~1.2 %** (≈4,800 transacciones fraudulentas), repartida f1 40 % / f2 35 % / f3 25 %.

**Esquema:** `card_id`, `ts`, `amount`, `merchant_id` (~300 comercios), `mcc` (~15 categorías), `channel` (POS/online/ATM/recurrente), `country` (GT + extranjero), `is_fraud`, `fraud_type`.

`fraud_type` es **solo para análisis** y nunca entra a una matriz de features. Verificado con `assert` sobre nombres de columnas.

### 4.1 Mecanismos de fraude

| Tipo | Patrón | ¿Depende del orden? | Rol en el argumento |
|---|---|---|---|
| **f1 — sondeo y golpe** | 3–6 compras de Q5–Q40 en comercios distintos en <2 h, seguidas de una compra grande | **Sí, fuertemente** | Donde B debe ganar |
| **f2 — ráfaga de cajero** | 3–5 retiros casi idénticos con Δt < 10 min | **Parcialmente** | Donde A debe competir de igual a igual |
| **f3 — monto atípico aislado** | Una transacción en el percentil 99.9 del cliente | **No** | Control negativo contra la acusación de circularidad |

La mezcla es la defensa explícita contra "usted construyó los datos para que B ganara": f3 está diseñado para que empaten.

### 4.2 Etiquetado de f1

En f1 se etiquetan como fraude **todas** las transacciones del atacante, sondeos incluidos, no solo el golpe. Es lo que un banco llama fraude, y es la opción honesta.

Efecto conocido y aceptado: los primeros sondeos son casi indetectables para A *y* para B, porque en ese momento no hay contexto acumulado. Arrastran igual a ambos modelos, así que no sesgan la comparación, y aportan un caso de fallo esperado verificable para §2.4.

Por eso el desglose por tipo reporta además **`f1-sondeo` vs `f1-golpe` por separado**: la ganancia de B debe concentrarse en el golpe, y esa es la evidencia más persuasiva del informe.

### 4.3 Casos donde se espera fallar

Declarados antes de medir, verificados empíricamente en el análisis de errores:

1. Clientes con menos de `K = 20` transacciones históricas (ventana con padding).
2. `f1` cuando el intervalo entre los sondeos y el golpe supera las 24 h, porque la ventana de 20 eventos no alcanza a contener ambos extremos del patrón.
3. Los sondeos iniciales de `f1`, por la razón de §4.2.

## 5. Partición temporal

Penalización de −20 pts por partición aleatoria. Protocolo:

```
Ordenar TODO el dataset por ts (global, no por tarjeta)
  Train: percentil   0 – 70  de ts
  Val:   percentil  70 – 85  de ts
  Test:  percentil  85 – 100 de ts
```

- El corte es por tiempo global. Una misma tarjeta puede aparecer en train y en test: es realista y correcto.
- Las secuencias **pueden** alcanzar hacia atrás a través del corte (la historia de una transacción de test puede venir del periodo de train). Jamás hacia adelante.
- Se documentan fechas de corte, tamaño y tasa de fraude por partición en la Tabla 1 del informe.
- **El test se toca una sola vez**, al final, con todas las decisiones tomadas, y la celda imprime fecha y hora de esa ejecución.

## 6. Anti-fuga: el checklist como tests que fallan

Penalización de −15 pts por usar estadísticas del conjunto completo. El checklist de §2.6 no va como comentarios: va como pruebas en `tests/`.

| Test | Qué falsifica |
|---|---|
| **Envenenamiento del futuro** | Se añade una transacción enorme en `t+1` y se afirma que las features de la fila `t` **no cambian**. Atrapa cualquier `groupby().mean()` sobre el histórico completo. |
| **Frontera de ventana** | Para toda fila `i` con máscara válida: `ts[win_idx[i]] <= ts[i]` y `card[win_idx[i]] == card[i]`. Ninguna ventana cruza de tarjeta ni mira adelante. |
| **Sin orden en A** | Los nombres de columna de `X_A` se contrastan contra una lista negra (`prev`, `lag`, `delta`, `diff`, `anterior`). Una feature de orden en A contaminaría la comparación. |
| **`fraud_type` fuera** | `assert` sobre los nombres de columnas de `X_A` y de `E`. |
| **Permutación preserva contenido** | El multiconjunto de índices de cada fila es idéntico antes y después del shuffle. |
| **Corte temporal** | `max(ts_train) <= min(ts_val)` y `max(ts_val) <= min(ts_test)`. |
| **Escalado solo con train** | El `scaler` se ajusta sobre filas de train; `transform` sobre val/test no reajusta. |
| **Vocabularios solo de train** | Una categoría presente únicamente en test mapea a `<UNK>`, no a un índice propio. |

Sin SMOTE ni sobremuestreo en validación ni en test. Si se usa en train, se declara.

## 7. Modelo A — línea base sin orden

Objetivo: representar honestamente el techo de lo alcanzable sin leer la secuencia. Una línea base débil invalida toda la comparación.

### 7.1 Features

`amt`, `amt_mean_24h`, `amt_std_24h`, `amt_max_24h`, `n_tx_1h`, `n_tx_24h`, `n_merchants_24h`, `amt_ratio_to_mean_7d`, `hour_sin`, `hour_cos`, `is_weekend`, y one-hot de `channel` y `mcc`.

Todas son invariantes al orden por construcción: medias, conteos y cardinalidades de conjunto.

### 7.2 `closed='left'` en los agregados de contexto

Se usa `closed='left'` en **todos** los agregados de ventana temporal. Así `amt` queda como la única feature que describe la transacción puntuada, y el resto describe solo lo previo.

Incluir la transacción actual también sería causal y legal, pero vuelve `amt_max_24h >= amt` trivialmente cierto y ensucia `amt_ratio_to_mean_7d`. La separación limpia entre "actual" y "contexto" es más fácil de defender ante el comité.

### 7.3 Nota de implementación

`n_merchants_24h` es un conteo de distintos sobre ventana temporal y pandas no ofrece `rolling.nunique`. Se resuelve con un barrido de dos punteros por tarjeta en numpy: exacto y O(n), en vez de una aproximación.

### 7.4 Algoritmos

- Primario: **LightGBM**, con fallback a `HistGradientBoostingClassifier` si la instalación no prospera.
- Piso obligatorio: **regresión logística** con `class_weight='balanced'`.
- Hiperparámetros ajustados **solo sobre validación**.
- Tres semillas, igual que B, para que la media ± σ sea comparable.

## 8. Modelo B — modelo secuencial

### 8.1 Representación

Ventana deslizante de `K = 20` eventos terminando en `t`, con padding **al inicio** si hay menos. Vector por evento:

```
log1p(amount)
log1p(Δt_segundos_respecto_al_evento_anterior)   + flag is_first
hour_sin, hour_cos
same_merchant_as_prev (0/1)
amount_ratio_to_prev
emb(mcc)      → dim 8
emb(channel)  → dim 4
emb(merchant) → dim 16
```

Vocabularios construidos **solo con train**, con `0 = PAD` y `1 = UNK` como índices reservados y distintos.

### 8.2 Qué mide exactamente la permutación

`Δt`, `same_merchant_as_prev` y `amount_ratio_to_prev` se calculan sobre el flujo original de eventos, antes de ventanear. Al barajar la ventana, **cada evento se lleva consigo su `Δt` original**.

Es decir: la permutación mide el valor del *arreglo secuencial* manteniendo fijas las features por evento. Sobrevive algo de información de orden dentro de cada vector. Es el montaje estándar, pero si B cae poco, la lectura queda ambigua entre "el orden no aporta" y "el orden ya está codificado en las features por evento".

Para desambiguar se añade una tercera comprobación barata: **ablación de `Δt`** — reentrenar B sin esa columna y comparar. El spec del curso la menciona como opcional en §4.1; aquí es la que cierra el argumento.

### 8.3 Arquitectura

```
Input (batch, K=20, d_feat)
  → Embeddings categóricos concatenados con numéricas
  → Masking del padding
  → GRU(64)
  → Dropout(0.3)
  → Dense(32, relu)
  → Dense(1, sigmoid)
```

**GRU sobre LSTM:** menos parámetros, converge más rápido, y con `K = 20` la memoria de largo plazo de la LSTM no aporta. **CNN 1D** captura patrones locales pero no el "sondeo y golpe", que requiere acumular estado a lo largo de la ventana. **Transformer** está sobredimensionado para `K = 20`. Las cuatro se documentan como consideradas.

- Pérdida: `binary_crossentropy` con `class_weight`; focal loss (γ=2) si el desbalance ahoga el gradiente.
- Adam `lr=1e-3`, `EarlyStopping` sobre **AUC-PR de validación** con `restore_best_weights`, `patience=5`. Nunca sobre `val_loss`, jamás sobre test.
- Tres semillas de modelo sobre el mismo dataset; se reporta media ± σ.

### 8.4 Cuidado con el padding al barajar

El padding va al inicio de la ventana. Al permutar hay que barajar **solo las posiciones válidas**: si el padding se mezcla al centro, la máscara miente y el resultado de la prueba de falsificación es basura. Test dedicado.

- **Full shuffle:** baraja las `K` posiciones válidas, evento actual incluido.
- **History shuffle:** baraja las `K-1` posiciones válidas previas y deja el evento objetivo en la última posición. Es la más limpia: aísla el aporte del orden *de la historia*.

## 9. Apuesta C — híbrido A+B

Frase de hipótesis, escrita y **fechada en una celda del notebook antes de la primera corrida de C**:

> "Creemos que **concatenar el estado oculto del GRU con el vector de agregados de A antes de la capa de salida** mejorará **la AUC-PR en validación** porque **f2 y f3 son mayormente capturables por agregados mientras que f1 requiere orden, y ningún modelo puro cubre ambos regímenes**. Lo consideraremos útil si **la AUC-PR en validación supera a la del mejor de A y B por al menos 0.02 absoluto, promediado sobre 3 semillas**."

Arquitectura: estado oculto del GRU (64) concatenado con `X_A` escalado → Dense(32, relu) → Dense(1, sigmoid). Mismos splits, mismas tres semillas.

Los controles (A solo, B solo) ya existen por diseño, así que la apuesta no puede quedarse sin control. **Se reporta aunque falle**: un "la hipótesis no se sostuvo, y esta es la evidencia" bien argumentado puntúa igual o mejor que un éxito sin control.

## 10. Evaluación y pruebas de falsificación

### 10.1 Métricas

- **Primaria: AUC-PR** (`average_precision_score`), independiente del umbral.
- En el umbral elegido: precisión, recall y F1.
- Media ± σ sobre 3 semillas.
- **La exactitud no aparece**, ni como nota al pie. Con 1.2 % de fraude, "todo legítimo" da 98.8 %.

### 10.2 Prueba 1 — permutación controlada (obligatoria)

Sobre validación primero, sobre test una sola vez al final. Se re-evalúa B con **los mismos pesos ya entrenados**; no se reentrena. No se recalculan los agregados de A.

Se reportan ambas variantes (full shuffle e history shuffle), más la ablación de `Δt` de §8.2.

| Resultado | Lectura |
|---|---|
| AUC-PR de B cae fuerte, A no se mueve | Evidencia de que B usa el orden |
| B cae poco o nada | Sin evidencia de aporte del orden; **se dice así** |
| A se mueve | Hay un bug: los agregados no son invariantes |

### 10.3 Prueba 2 — curva de recorte de historia

AUC-PR de B con `K ∈ {1, 3, 5, 10, 20}`. Produce una figura que el comité entiende de inmediato y responde una pregunta de negocio real: cuánta historia hay que guardar en producción. Con `K = 1` el modelo secuencial degenera en clasificador puntual y debería caer a la altura de A — otro control de sanidad.

Es también la evidencia que justifica `K = 20` como decisión técnica y no como número arbitrario.

**Complemento:** desglose de AUC-PR y recall **por `fraud_type`**, con `f1-sondeo` y `f1-golpe` separados. Se espera B > A en f1-golpe, empate en f2 y empate en f3. Si ese patrón aparece, es la evidencia más persuasiva del informe: la ganancia se concentra donde la teoría lo predice.

## 11. Calibración y decisión económica

### 11.1 Calibración

Los puntajes de una red con `class_weight` no están calibrados. Se aplica **calibración isotónica ajustada en validación** antes de cualquier conversión a quetzales. Sin esto, el análisis económico compara peras con manzanas entre A y B.

### 11.2 Umbral por costo

Costos: FN = **Q4,200**, FP = **Q180**.

```
Costo(u) = FN(u) · 4200 + FP(u) · 180
p* = 180 / 4200 = 0.0429
```

Bloquear a partir de ~4.3 % de probabilidad de fraude. El umbral óptimo está lejísimos de 0.5 porque dejar pasar un fraude cuesta 23 veces más que molestar a un cliente legítimo.

Procedimiento:

1. Barrer `u ∈ [0,1]` en 1,000 pasos **sobre validación**, elegir `u*` que minimiza el costo.
2. Verificar que `u*` empírico esté cerca de 0.0429; si no, la calibración falló o hay que explicar la interacción con el desbalance.
3. **Congelar `u*`** y aplicarlo a test sin reoptimizar. Elegir umbral mirando test cuesta −10 pts.
4. Reportar el ahorro mensual con el factor de escalamiento documentado, diciendo claramente que es una extrapolación.

**Figura obligatoria:** curva `Costo(u)` para A y para B en el mismo eje, con `u*` marcado en cada una.

### 11.3 Limitación que se declara

El calibrador isotónico y `u*` se ajustan **ambos** sobre validación, así que `u*` es levemente optimista. Es lo que pide el spec del curso, pero va explícito en la matriz de evidencias en vez de quedar escondido.

## 12. Estructura del repositorio

La **raíz del repo actual (`C:\dev\Proyecto-1-Deeplearning`) es la carpeta entregable**. Al empaquetar para la entrega se renombra o se comprime como `proyecto1_mazariegos_herrera`; no se crea un subdirectorio con ese nombre durante el desarrollo.

```
proyecto1_mazariegos_herrera/     (= raíz del repo)
├── src/monitoreo/
│   ├── config.py                # semillas, K, costos, percentiles, DEV_MODE
│   ├── reproducibilidad.py      # set_seeds, captura de versiones
│   ├── generador.py             # generate(seed) → DataFrame
│   ├── particion.py             # corte temporal + tabla de particiones
│   ├── features_agregadas.py    # X_A, agregados causales (Modelo A)
│   ├── ventanas.py              # win_idx, mask, permutaciones
│   ├── features_evento.py       # E, vocabularios, escalado
│   ├── modelos_a.py             # logística + LightGBM/HistGB
│   ├── modelos_b.py             # GRU e híbrido C
│   ├── calibracion.py           # isotónica
│   ├── metricas.py              # AUC-PR, curvas PR, desglose por tipo
│   └── economia.py              # curva de costo, u*, ahorro
├── tests/                       # espeja src/, con énfasis en los tests de §6
├── notebooks/
│   └── proyecto1_mazariegos_herrera.ipynb
├── artefactos/
│   ├── modelo_candidato.keras
│   ├── scaler.pkl
│   ├── vocab_embeddings.json
│   ├── config.json
│   └── generador_datos.py       # copia de src/monitoreo/generador.py
├── informe/
├── README.md
└── requirements.txt
```

El notebook importa de `src/` y muestra resultados y figuras. La lógica penalizable vive en módulos con tests; la narrativa vive en el notebook.

`artefactos/generador_datos.py` se genera copiando `src/monitoreo/generador.py` al final, para cumplir §8 sin duplicar código durante el desarrollo.

### 12.1 Parámetros congelados

| Parámetro | Valor |
|---|---|
| `SEED_DATOS` | 20260831 |
| `SEEDS_MODELO` | (7, 13, 29) |
| `K` | 20 |
| `N_TARJETAS` | 4,000 (400 en `DEV_MODE`) |
| `TX_POR_TARJETA` | 60–200 |
| `TASA_FRAUDE` | ~1.2 % |
| Percentiles de corte | 70 / 85 |
| `COSTO_FN` / `COSTO_FP` | Q4,200 / Q180 |
| Dims de embedding | mcc 8, channel 4, merchant 16 |

**Entorno verificado (31 ago 2026):** Windows 10 Pro 19045 · Python 3.12.3 · TensorFlow 2.21.0 · Keras 3.15.1 · LightGBM 4.7.0 · scikit-learn 1.8.0 · pandas 2.2.3 · numpy 2.4.6. CPU únicamente (sin GPU). Estas versiones van al README (§8.3 del spec del curso) y a `artefactos/config.json`.

El dataset se genera **una sola vez** con `SEED_DATOS`; las tres semillas varían solo la inicialización del modelo. Así la σ reportada mide ruido de entrenamiento y no ruido de datos, que es lo que §4.2 del spec pide separar.

## 13. Presupuesto de cómputo

400k muestras × 3 semillas × (B + C) × ~20 épocas en CPU: estimado **1–3 h**, más 5 ajustes para la curva de `K`.

`config.DEV_MODE` submuestrea a ~400 tarjetas para iterar rápido durante el desarrollo. Las cifras del informe salen únicamente de la corrida completa.

Si el tiempo se dispara, la palanca es quitar el embedding de `merchant` (la de mayor cardinalidad), tal como recomienda §11 del spec, o bajar a `batch_size=512` con menos unidades.

## 14. Entregables y las seis evidencias

| # | Evidencia | Dónde sale |
|---|---|---|
| 1 | Integridad de datos | Tabla 1 (particiones con fechas y tasas), checklist anti-fuga respaldado por `tests/` |
| 2 | Comparación común | Tabla 3 (A vs B: AUC-PR + P/R/F1 en `u*`), Fig. 2 (curvas PR superpuestas) |
| 3 | Valor del orden | Tabla 4 (permutación: original vs full vs history, A y B, + ablación de Δt), Fig. 3 (AUC-PR vs K) |
| 4 | Apuesta del equipo | Frase literal fechada, control, resultado, veredicto |
| 5 | Decisión económica | Fig. 4 (curva de costo con `u*` marcado), ahorro mensual, supuestos |
| 6 | Recomendación y límites | Complementar (no reemplazar) + patrones de error de §4.3 + condiciones de cambio |

El informe se escribe **para el comité de riesgos**: "el modelo detecta 8 de cada 10 fraudes bloqueando 1 de cada 400 compras legítimas", no "AUC-PR = 0.62". Máximo 7 páginas, sin código. La matriz de evidencias va en la última página, con una limitación honesta por fila.

**Recomendación esperada:** complementar. B debería ganar en f1-golpe y empatar en f2/f3; el motor actual sigue siendo más barato y explicable. Un ensamble es más defendible que "reemplácenlo todo".

## 15. Riesgos

| Riesgo | Señal temprana | Mitigación |
|---|---|---|
| ~~TensorFlow no instala limpio en Windows/Py3.12~~ | — | **Resuelto el 31 ago:** TF 2.21.0, Keras 3.15.1, LightGBM 4.7.0 instalados y verificados en Python 3.12.3 |
| B no supera a A | AUC-PR val casi idéntica tras 2 corridas | Verificar que `Δt` esté en las features de B; subir `K`; revisar prevalencia de f1 |
| La permutación no baja el desempeño | ΔAUC-PR < 0.01 | Resultado válido, se reporta honestamente. La ablación de Δt desambigua la causa |
| Sobreajuste del GRU | `val_loss` sube desde la época 3 | Dropout 0.3–0.5, bajar a 32 unidades, `patience=5` |
| AUC-PR inestable entre semillas | σ > 0.05 | Subir tasa de fraude del generador o volumen de datos |
| Entrenamiento excesivo en CPU | >10 min por época | Quitar embedding de merchant, `batch_size=512`, reducir `K` |
| El informe pasa de 7 páginas | — | Figuras al ancho de página, matriz compacta, detalles al notebook |

## 16. Definición de "hecho"

1. El notebook corre de principio a fin desde kernel limpio con la semilla fija y reproduce las cifras del informe.
2. Cada una de las seis evidencias es localizable en el informe en menos de 15 segundos.
3. La matriz de evidencias tiene una limitación honesta por fila; ninguna casilla dice "ninguna".
4. Ambos integrantes pueden defender las tres decisiones técnicas sin consultar notas.
5. El checklist de penalizaciones (§9 del spec del curso) está verificado punto por punto:
   - −20 partición aleatoria → corte por percentil de `ts` global
   - −15 estadísticas del conjunto completo → `fit` solo en train, agregados causales, tests que lo prueban
   - −15 exactitud como métrica principal → AUC-PR primaria, exactitud ausente
   - −10 decidir mirando test → todo en validación, test ejecutado una vez y fechado
   - −10 afirmar aporte del orden sin permutación → ambas variantes reportadas, más ablación de Δt
