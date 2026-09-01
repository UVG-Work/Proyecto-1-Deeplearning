
# Especificación técnica — Proyecto 1: Monitoreo transaccional

**Curso:** Deep Learning y Sistemas Inteligentes 2026 (UVG) — Kevin Recinos
**Modalidad:** parejas · **Peso:** 8 pts sobre la nota final
**Entrega:** viernes 4 de septiembre de 2026, 23:59
**Presentación:** viernes 4 de septiembre, sesión virtual — 8 min + 4 de preguntas

---

## 0. Lectura del encargo (qué se califica realmente)

El proyecto **no** se califica por entrenar una LSTM. Se califica por responder, con evidencia falsable, esta pregunta:

> ¿El **orden** de las transacciones aporta información que las variables agregadas no capturan, bajo qué condiciones, y cuánto vale esa información en quetzales?

Tres consecuencias de diseño que gobiernan todo lo demás:

1. **A y B deben ser comparables punto por punto.** Mismos datos, misma partición, mismo horizonte de predicción, misma etiqueta. La única diferencia permitida es *cómo* se representa la entrada (agregados vs. secuencia ordenada).
2. **Una mejora de métricas no prueba nada por sí sola.** Por eso son obligatorias las dos pruebas de falsificación. La permutación es la que convierte "B ganó" en "B ganó *porque leyó el orden*".
3. **El resultado negativo es aceptable; la conclusión deshonesta no.** Si al barajar el orden el desempeño no cae, se reporta eso y se concluye que no hay evidencia de aporte del orden.

---

## 1. Decisiones de arquitectura del proyecto

### 1.1 Ruta de datos: **Ruta A (generador sintético)** — recomendada

| Criterio | Ruta A (sintética) | Ruta B (pública real) |
|---|---|---|
| Control sobre dependencia del orden | Total: se inyecta por diseño | Nulo: puede no existir |
| Riesgo de que la permutación no muestre nada | Bajo | Alto |
| Costo de limpieza/EDA | Bajo | Alto (2 días fácilmente) |
| Evaluación por mecanismo de fraude | Trivial (etiqueta de tipo) | Casi imposible |
| Crítica esperable del comité | "Es circular, usted lo construyó" | Ninguna |
| Tiempo disponible (4 días) | Compatible | Apretado |

**Recomendación:** Ruta A. Con 4 días y una prueba de permutación obligatoria, garantizar que la señal de orden exista es más valioso que el realismo. La circularidad se mitiga explícitamente (ver §2.4): se incluye un tipo de fraude *sin* dependencia de orden y un caso donde se espera fallar, lo que demuestra que el generador no está amañado a favor de B.

**Si se elige Ruta B:** el dataset debe tener identificador de tarjeta/cliente + timestamp. Verificar antes de comprometerse. El clásico `creditcard.csv` (ULB, V1–V28 PCA) **no sirve**: no tiene ID de tarjeta, por lo que no se pueden formar secuencias por cliente.

### 1.2 Unidad de predicción

**Predicción a nivel de transacción, no de secuencia.**

> Para la transacción `t` de la tarjeta `c`, con la historia `t-K+1 … t` disponible, ¿es `t` fraudulenta?

Esto fija el "mismo horizonte de predicción" que exige el enunciado y hace que A y B respondan literalmente la misma pregunta con la misma información disponible en el momento `t`.

- Modelo A ve: vector de agregados calculados causalmente hasta `t`.
- Modelo B ve: los `K` eventos ordenados que terminan en `t`.
- Ambos devuelven `p ∈ [0,1]` (puntaje continuo de riesgo). El umbral se decide después.

### 1.3 Reutilización del Lab 4

| Pieza del Proyecto 1 | Base en el Lab 4 |
|---|---|
| A — línea base sin orden | Bloque MLP sobre agregados |
| B — modelo secuencial | Bloque LSTM |
| C — apuesta | Bloque autoencoder LSTM (anomalías) |
| Ventaneo y representación | Bloque de datos/representaciones |

No copiar y pegar sin revisar: el Lab 4 usaba `assert` fijos y umbral por percentil 95. Aquí el umbral se decide por **costo**, no por percentil.

---

## 2. Especificación de datos

### 2.1 Esquema del evento

| Campo | Tipo | Notas |
|---|---|---|
| `card_id` | int | agrupador de secuencia |
| `ts` | datetime | orden estricto dentro de la tarjeta |
| `amount` | float | Q |
| `merchant_id` | int | cardinalidad ~200–500 |
| `mcc` / `category` | cat | ~15 categorías |
| `channel` | cat | POS / online / ATM / recurrente |
| `country` | cat | GT + extranjero |
| `is_fraud` | 0/1 | etiqueta objetivo |
| `fraud_type` | cat | `none` / `f1` / `f2` / `f3` — **solo para análisis, nunca como feature** |

### 2.2 Generador sintético — requisitos

- **Reproducible por semilla.** `generate(seed) → DataFrame` idéntico. Semilla fija en el notebook y documentada en el README.
- **Es un entregable en sí mismo**, no un script escondido.
- Tasa de fraude objetivo: **0.5 %–2 %** (desbalance realista; por debajo de 0.3 % la AUC-PR se vuelve inestable con pocos datos).
- Volumen sugerido: **~3,000–5,000 tarjetas × 60–200 transacciones** ≈ 300k–600k eventos. Suficiente para entrenar un GRU en CPU/GPU modesta en minutos.

### 2.3 Tipos de fraude (mínimo 3, al menos 1 dependiente del orden)

| Tipo | Patrón | ¿Depende del orden? |
|---|---|---|
| **f1 — Sondeo y golpe** | 3–6 compras muy pequeñas (Q5–Q40) en comercios distintos en <2 h, seguidas de una compra grande | **Sí, fuertemente.** Barajar destruye la señal: los mismos montos en otro orden no forman el patrón |
| **f2 — Ráfaga de cajero** | 3–5 retiros consecutivos casi idénticos con Δt < 10 min | **Parcialmente.** El conteo agregado ya lo captura casi todo → A debería competir bien aquí |
| **f3 — Monto atípico aislado** | Una sola transacción con monto en el percentil 99.9 del cliente | **No.** Control negativo: A y B deberían empatar |

Esta mezcla es deliberada. Que **f3 no dependa del orden** es la defensa contra la acusación de circularidad: el generador no está diseñado para que B gane siempre.

### 2.4 Caso donde se espera fallar (exigido por el enunciado)

Declarar explícitamente, por ejemplo:

> "Esperamos que el modelo falle en clientes con menos de `K` transacciones históricas (secuencia rellenada con padding) y en f1 cuando el intervalo entre las compras pequeñas y la grande supera las 24 h, porque la ventana de `K=20` eventos no alcanza a contener ambos extremos del patrón."

Y **verificarlo empíricamente** en el análisis de errores (§6.2).

### 2.5 Partición temporal — la regla que más puntos cuesta

Penalización de **−20 pts** por partición aleatoria. Protocolo:

```
Ordenar TODO el dataset por ts (global, no por tarjeta)
  Train: percentil   0 – 70  de ts
  Val:   percentil  70 – 85  de ts
  Test:  percentil  85 – 100 de ts
```

- El corte es por **tiempo global**, no por tarjeta. Una misma tarjeta puede aparecer en train y test — eso es realista y correcto.
- Documentar las fechas de corte y el tamaño y tasa de fraude de cada partición en una tabla del informe.
- **El test se toca una sola vez**, al final, con todas las decisiones ya tomadas. Escribir la fecha/hora de esa ejecución en el notebook.

### 2.6 Controles contra fuga de información

Penalización de **−15 pts** por normalizar o ventanear con estadísticas del conjunto completo. Checklist a incluir literalmente en el notebook:

- [ ] `StandardScaler` / `MinMaxScaler` con `.fit()` **solo en train**, `.transform()` en val y test.
- [ ] Agregados de A calculados con ventana **causal** (`rolling` hacia atrás, `closed='left'` para excluir la transacción actual si el agregado no debe verse a sí mismo). Nunca `groupby(card).mean()` sobre todo el histórico.
- [ ] Vocabularios de embeddings (merchant, mcc, channel) construidos **solo con train**; categorías nuevas en val/test → token `<UNK>`.
- [ ] Ninguna secuencia cruza el corte temporal hacia adelante: los `K` eventos de contexto de una transacción de test pueden venir de antes, pero jamás de después de `t`.
- [ ] `fraud_type` excluido de las matrices de features (verificar con un `assert` sobre los nombres de columnas).
- [ ] Sin SMOTE ni sobremuestreo en validación ni en test. Si se usa en train, declararlo.

---

## 3. Modelo A — línea base sin orden

**Objetivo:** representar honestamente el techo de lo alcanzable sin leer la secuencia. Una línea base débil hace que toda la comparación pierda valor y el comité lo notará.

### 3.1 Features (agregados causales, ventana de 24 h y 7 d)

| Feature | Descripción |
|---|---|
| `amt` | monto de la transacción actual |
| `amt_mean_24h`, `amt_std_24h`, `amt_max_24h` | estadísticos del monto |
| `n_tx_1h`, `n_tx_24h` | conteo de transacciones |
| `n_merchants_24h` | diversidad de comercios (cardinalidad distinta) |
| `amt_ratio_to_mean_7d` | monto actual / promedio de 7 días |
| `hour_sin`, `hour_cos`, `is_weekend` | tiempo cíclico |
| `channel`, `mcc` | one-hot |

Estas replican las que el banco ya usa ("monto promedio de últimas 24 h, número de transacciones por hora, monto máximo del día y diversidad de comercios") más un par razonables. **Ninguna feature debe codificar orden** (nada tipo "monto anterior", "Δt respecto de la previa") — eso contaminaría la comparación.

### 3.2 Algoritmo

- Primario: **Gradient Boosting** (`LightGBM` o `sklearn.ensemble.HistGradientBoostingClassifier`).
- Secundario obligatorio: **Regresión logística** con `class_weight='balanced'` como piso de referencia.
- Salida: `predict_proba(X)[:,1]`.
- Hiperparámetros ajustados **solo en validación**.

---

## 4. Modelo B — modelo secuencial

### 4.1 Representación

Ventana deslizante de `K = 20` eventos terminando en `t` (padding al inicio si hay menos). Cada evento es un vector:

```
[ log1p(amount),
  log1p(Δt_segundos_respecto_al_evento_anterior),
  hour_sin, hour_cos,
  emb(mcc)        → dim 8,
  emb(channel)    → dim 4,
  emb(merchant)   → dim 16,   # opcional; si la cardinalidad explota, usar hashing
  same_merchant_as_prev (0/1),
  amount_ratio_to_prev ]
```

`Δt` es la variable que hace que el orden *importe físicamente*. Ojo: es también la que se retira en una de las pruebas de falsificación opcionales.

### 4.2 Arquitectura

```
Input (batch, K=20, d_feat)
  → Embeddings categóricos concatenados con numéricas
  → GRU(64, return_sequences=False)      # o LSTM(64); GRU entrena más rápido
  → Dropout(0.3)
  → Dense(32, relu)
  → Dense(1, sigmoid)
```

- **GRU sobre LSTM** salvo que haya evidencia de lo contrario: menos parámetros, converge más rápido, y con `K=20` la memoria de largo plazo de la LSTM no aporta. *Esta es una de las tres decisiones técnicas a declarar en el README (§8).*
- Pérdida: `binary_crossentropy` con `class_weight`, o **focal loss** (`γ=2`) si el desbalance ahoga el gradiente.
- Optimizador: Adam, `lr=1e-3`, `EarlyStopping` sobre **AUC-PR de validación** (no sobre `val_loss`, y jamás sobre test).
- Semilla fija; reportar media ± desviación de **3 corridas** para no confundir ruido con mejora.

### 4.3 Justificación a documentar

Por qué GRU y no CNN 1D / Transformer: con `K=20` y ~500k muestras, un Transformer está sobredimensionado y una CNN 1D captura patrones locales pero no el "sondeo y golpe" que requiere acumular estado a lo largo de la ventana. Ambas alternativas deben mencionarse como *consideradas y descartadas con razón*, que es exactamente lo que pide §7 del enunciado.

---

## 5. Apuesta C

### 5.1 Frase de hipótesis (escribir en el notebook ANTES de entrenar)

Plantilla obligatoria del enunciado:

> "Creemos que ___ mejorará ___ porque ___. Lo consideraremos útil si ___."

**Opción recomendada — Modelo híbrido (agregados + secuencia):**

> "Creemos que **concatenar el estado oculto del GRU con el vector de agregados de A antes de la capa de salida** mejorará **la AUC-PR en validación** porque **f2 y f3 son mayormente capturables por agregados mientras que f1 requiere orden, y ningún modelo puro cubre ambos regímenes**. Lo consideraremos útil si **la AUC-PR en validación supera a la del mejor de A y B por al menos 0.02 absoluto, promediado sobre 3 semillas**."

- **Control experimental:** A solo y B solo, con la misma partición y semillas. Sin ese control, la apuesta no puntúa.
- **Métrica de éxito declarada antes de ver test.** Fechar la celda.

**Alternativas viables (elegir una sola):**

| Opción | Hipótesis | Control | Costo |
|---|---|---|---|
| Híbrido A+B | ver arriba | A solo, B solo | Bajo — **recomendada** |
| Robustez a fraude no visto | Entrenar excluyendo `f1`; ¿detecta B el mecanismo nuevo mejor que A? | B entrenado con las 3 clases | Medio, muy vistoso |
| Autoencoder LSTM no supervisado | Error de reconstrucción como puntaje de anomalía detecta fraude nuevo sin etiquetas | B supervisado | Medio (reutiliza Lab 4) |
| Atención interpretable | Capa de atención sobre la secuencia; los pesos señalan las compras de sondeo | GRU sin atención | Alto |

### 5.2 Veredicto

Se reporta **aunque falle**. Un "la hipótesis no se sostuvo, y esta es la evidencia" bien argumentado puntúa igual o mejor que un éxito sin control.

---

## 6. Las dos pruebas de falsificación

### 6.1 Prueba 1 — Permutación controlada (obligatoria)

Omitirla y aun así afirmar que el orden aporta cuesta **−10 pts**.

**Procedimiento:**

1. Tomar el conjunto de validación (y luego test, una sola vez).
2. Para cada secuencia, **barajar el orden de los `K` eventos** sin alterar sus contenidos.
3. **No recalcular** las variables agregadas: por construcción son invariantes a la permutación, así que A no debe moverse en absoluto.
4. Re-evaluar B con los mismos pesos ya entrenados. **No reentrenar.**

**Dos variantes, reportar ambas:**

- **Full shuffle:** barajar los `K` eventos incluyendo el actual.
- **History shuffle:** barajar los `K-1` eventos previos, dejando el evento objetivo en la última posición. Esta es la más limpia: aísla el aporte del *orden de la historia* del efecto de mover la transacción que se está clasificando.

**Interpretación:**

| Resultado | Lectura |
|---|---|
| AUC-PR de B cae fuerte, A no se mueve | Evidencia de que B usa el orden |
| B cae poco o nada | Sin evidencia de aporte del orden; **decirlo así** |
| A se mueve | Hay un bug: los agregados no son invariantes → hay fuga de orden en A |

Ese último caso es un control de sanidad gratuito: si A cambia al permutar, el pipeline está mal.

### 6.2 Prueba 2 — Elegida por el equipo

**Recomendada: curva de recorte de historia.** Evaluar B con `K ∈ {1, 3, 5, 10, 20}` y graficar AUC-PR vs. `K`.

Por qué es la mejor opción con este calendario: es barata, produce **una figura** que el comité entiende de inmediato, y responde una pregunta de negocio real ("¿cuánta historia necesitamos guardar en producción?"). Con `K=1` el modelo secuencial degenera en un clasificador puntual y debería caer a la altura de A — otro control de sanidad.

**Complemento de bajo costo (recomendado añadirlo):** desglosar AUC-PR y recall **por `fraud_type`**. Se espera que B > A en f1, empate en f2 y empate en f3. Si ese patrón aparece, es la evidencia más persuasiva del informe entero — muestra que la ganancia se concentra exactamente donde la teoría predice.

---

## 7. Métricas, umbral y decisión económica

### 7.1 Métricas

- **Primaria: AUC-PR** (average precision). Independiente del umbral.
- En el umbral elegido: **precisión, exhaustividad (recall) y F1**.
- **Nunca reportar exactitud como métrica principal** → **−15 pts**. Con 1 % de fraude, predecir "todo legítimo" da 99 % de exactitud.
- Reportar media ± desviación sobre 3 semillas.

### 7.2 Calibración

Los puntajes de una red con `class_weight` **no están calibrados**. Antes de convertir puntaje en decisión de costo, aplicar **calibración isotónica o Platt ajustada en validación** (`sklearn.calibration.CalibratedClassifierCV`). Sin esto, el análisis económico compara peras con manzanas entre A y B.

### 7.3 Umbral óptimo por costo

Costos dados: fraude no detectado (FN) = **Q4,200**; bloquear transacción legítima (FP) = **Q180**.

```
Costo(u) = FN(u) · 4200 + FP(u) · 180
```

**Umbral teórico (regla de Bayes con puntaje calibrado):** conviene bloquear cuando el costo esperado de dejar pasar supera el de bloquear:

```
p · 4200 > 180   →   p* = 180 / 4200 = 0.0429
```

Es decir: **bloquear a partir de ~4.3 % de probabilidad de fraude**. Contraintuitivo y muy citable en la presentación — el umbral óptimo está lejísimos de 0.5, porque dejar pasar un fraude cuesta 23 veces más que molestar a un cliente legítimo.

**Procedimiento:**

1. Barrer `u ∈ [0, 1]` en 1,000 pasos **sobre validación**, calcular `Costo(u)`, elegir `u*` que lo minimiza.
2. Verificar que `u*` empírico esté cerca de 0.0429; si no, la calibración falló o los costos interactúan con el desbalance de forma que hay que explicar.
3. **Congelar `u*`.** Aplicarlo a test sin volver a optimizar. Elegir umbral mirando test cuesta **−10 pts**.
4. Reportar el ahorro mensual, con supuestos explícitos:

```
Ahorro_mensual = Costo_A(u*_A) − Costo_B(u*_B)   escalado a transacciones/mes
```

Documentar el factor de escalamiento (transacciones en test → transacciones/mes) y decir claramente que es una extrapolación.

**Figura obligatoria:** curva `Costo(u)` para A y para B en el mismo eje, con `u*` marcado en cada una. Es la diapositiva que convence al comité.

---

## 8. Entregables

```
proyecto1_<apellido1>_<apellido2>/
├── proyecto1_<apellidos>.ipynb      # ejecutado, con salidas visibles
├── artefactos/
│   ├── modelo_candidato.keras       # pesos del modelo elegido
│   ├── scaler.pkl                   # ajustado solo con train
│   ├── vocab_embeddings.json        # mcc, channel, merchant → índice
│   ├── config.json                  # K, semilla, umbral u*, versiones
│   └── generador_datos.py           # si Ruta A
├── informe.pdf                      # máx. 7 páginas, SIN código
├── presentacion.pdf                 # máx. 8 diapositivas
└── README.md
```

### 8.1 Informe — las seis evidencias

Estructura libre, pero el comité debe **localizar sin adivinar**:

| # | Evidencia | Contenido mínimo |
|---|---|---|
| 1 | Integridad de datos | Origen, tamaño, tasa de fraude, construcción de secuencias, tabla de partición temporal con fechas, checklist anti-fuga |
| 2 | Comparación común | Tabla A vs B: AUC-PR + precisión/recall/F1 en `u*`. Curvas PR superpuestas |
| 3 | Valor del orden | Tabla de permutación (original vs shuffle, A y B) + curva AUC-PR vs `K` |
| 4 | Apuesta del equipo | Frase de hipótesis literal, control, resultado, veredicto |
| 5 | Decisión económica | Curva Costo(u), `u*`, ahorro mensual, supuestos |
| 6 | Recomendación y límites | Reemplazar/complementar/conservar + ≥1 patrón de error concreto + condiciones de cambio |

**Escrito para el comité de riesgos.** Sin jerga innecesaria: "el modelo detecta 8 de cada 10 fraudes bloqueando 1 de cada 400 compras legítimas", no "AUC-PR = 0.62".

**Recomendación esperada:** casi con seguridad **complementar**, no reemplazar. B suele ganar en f1 y empatar en f2/f3; el motor actual sigue siendo más barato y explicable. Un ensamble (máximo de los dos puntajes, o motor actual como primer filtro y B como segundo) es defendible y más honesto que "reemplácenlo todo".

### 8.2 Matriz de evidencias (última página, ≤1 página)

Es también la guía de calificación. Formato:

| Evidencia | Figura/Tabla | Conclusión | Limitación |
|---|---|---|---|
| Partición temporal | Tabla 1 | Sin fuga temporal; test posterior a val | Una sola ventana; sin validación rolling |
| A vs B | Tabla 3, Fig. 2 | B mejora AUC-PR de 0.xx a 0.yy | Diferencia dentro de ±1σ en f2 y f3 |
| Permutación | Tabla 4 | AUC-PR de B cae 0.zz; A invariante | No prueba *qué* orden usa, solo que lo usa |
| Recorte de historia | Fig. 3 | La ganancia satura en K≈10 | K>20 no explorado |
| Apuesta C | Tabla 5 | Híbrido supera/no supera el umbral declarado | Una sola configuración probada |
| Umbral y costo | Fig. 4 | u*=0.04x, ahorro ~Qn/mes | Costos fijos y uniformes; datos sintéticos |

### 8.3 README

Secciones obligatorias:

1. **Reproducción:** comandos exactos, semillas, tiempo aproximado de ejecución.
2. **Versiones:** Python, TensorFlow/Keras, scikit-learn, pandas, numpy, sistema operativo.
3. **Declaración de uso de IA:** para qué se usó y **qué verificó cada quien**.
4. **Tres decisiones técnicas** — para cada una: alternativas consideradas y **la evidencia que inclinó la decisión**. Candidatas:
   - GRU vs LSTM vs CNN 1D vs Transformer para B
   - `K = 20`: elegido por la curva de recorte de historia, no arbitrariamente
   - Ruta A sintética vs Ruta B pública
   - Focal loss vs `class_weight`
   - Umbral por costo vs por F1 máximo
5. **Candidato al Proyecto Final:**
   - Qué modelo se conserva y dónde está su artefacto
   - Quién usaría el puntaje y qué decisión tomaría (analista de riesgos: bloquear / revisar / dejar pasar)
   - Contrato preliminar de entrada y salida, p. ej.:
     ```
     Entrada:  {card_id, últimas 20 transacciones [{ts, amount, mcc, channel, merchant_id}]}
     Salida:   {risk_score: float ∈ [0,1], decision: "block"|"review"|"allow", model_version: str}
     Latencia objetivo: < 100 ms p95
     ```
   - Límites, riesgos y datos que aún faltan

> **En la presentación se escoge al azar una de las tres decisiones y cualquiera de los dos debe defenderla.** Repasar las tres juntos antes del viernes. Si no se puede explicar por qué existe una parte del código, esa evidencia no recibe crédito.

---

## 9. Penalizaciones — checklist de verificación final

Revisar una por una antes de entregar:

- [ ] **−20** Partición aleatoria → *confirmado: corte por percentil de `ts` global*
- [ ] **−15** Normalizar/seleccionar/ventanear con estadísticas del conjunto completo → *confirmado: `fit` solo en train, agregados causales*
- [ ] **−15** Exactitud como métrica principal → *confirmado: AUC-PR primaria; exactitud no aparece o aparece solo como nota al pie*
- [ ] **−10** Elegir arquitectura, umbral o apuesta mirando test → *confirmado: todo decidido en validación, test ejecutado una sola vez y fechado*
- [ ] **−10** Afirmar que el orden aporta sin permutación → *confirmado: permutación ejecutada, ambas variantes reportadas*

---

## 10. Cronograma (31 ago – 4 sep)

| Día | Hito | Criterio de "listo" |
|---|---|---|
| **Lun 31 ago** | Generador sintético + partición temporal + EDA | `generate(seed)` reproducible, tabla de particiones con tasas de fraude, checklist anti-fuga firmado |
| **Mar 1 sep** | Modelo A completo + features agregados causales | AUC-PR de A en validación registrada; logística y GBM comparados |
| **Mié 2 sep** | Modelo B + calibración | B entrenado con 3 semillas; AUC-PR val vs A; scaler y vocabs guardados |
| **Jue 3 sep** | Apuesta C + ambas pruebas de falsificación + análisis económico | Frase de hipótesis fechada antes de entrenar C; tabla de permutación; curva de costo; **test ejecutado una sola vez al final del día** |
| **Vie 4 sep AM** | Informe + presentación + README + artefactos | Matriz de evidencias completa; 7 pág / 8 slides respetadas |
| **Vie 4 sep PM** | Ensayo de presentación | 8 min cronometrados; ambos capaces de defender las 3 decisiones |

**Punto de no retorno:** el conjunto de test no se toca hasta el jueves por la noche. Si algo va tarde, se recorta la apuesta C (15 pts) antes que las pruebas de falsificación (20 pts) o el protocolo temporal (15 pts).

### 10.1 División sugerida del trabajo en pareja

| Persona 1 | Persona 2 |
|---|---|
| Generador de datos + partición temporal | Features agregados + Modelo A |
| Modelo B + calibración | Pruebas de falsificación (permutación, recorte) |
| Apuesta C | Análisis económico y umbral |
| Notebook y artefactos | Informe y presentación |

Ambos deben poder explicar **todo**, porque la defensa en la presentación se asigna al azar.

---

## 11. Riesgos técnicos y mitigaciones

| Riesgo | Señal temprana | Mitigación |
|---|---|---|
| B no supera a A | AUC-PR val casi idéntica tras 2 corridas | Verificar que `Δt` esté en las features de B; subir `K`; revisar que f1 tenga suficiente prevalencia |
| Permutación no baja el desempeño | ΔAUC-PR < 0.01 | Es un resultado válido — **reportarlo honestamente**. Verificar antes que el shuffle realmente se aplique (imprimir una secuencia antes/después) |
| Sobreajuste del GRU | `val_loss` sube desde la época 3 | Dropout 0.3–0.5, reducir a 32 unidades, `EarlyStopping` con `patience=5` |
| AUC-PR muy inestable entre semillas | σ > 0.05 | Subir la tasa de fraude del generador o el volumen de datos |
| Tiempo de entrenamiento excesivo en Windows/Anaconda | >10 min por época | Reducir `K`, usar `batch_size=512`, quitar el embedding de merchant (el de mayor cardinalidad) |
| El informe se pasa de 7 páginas | — | Las figuras al ancho de página y la matriz de evidencias en tabla compacta; mover detalles al notebook |

---

## 12. Definición de "hecho"

El proyecto está terminado cuando:

1. El notebook corre de principio a fin desde kernel limpio, con la semilla fija, y reproduce las cifras del informe.
2. Cada una de las seis evidencias es localizable en el informe en menos de 15 segundos.
3. La matriz de evidencias tiene una limitación honesta por fila (ninguna casilla dice "ninguna").
4. Ambos integrantes pueden defender las tres decisiones técnicas sin consultar notas.
5. El checklist de penalizaciones (§9) está verificado punto por punto.
