# Proyecto 1 — Monitoreo transaccional

**Curso:** Deep Learning y Sistemas Inteligentes 2026 (UVG) — Kevin Recinos
**Autores:** Andres Mazariegos · June Herrera

La pregunta que responde este proyecto:

> ¿El **orden** de las transacciones aporta información que las variables
> agregadas no capturan, bajo qué condiciones, y cuánto vale esa información
> en quetzales?

---

## 1. Reproducción

```bash
python -m pip install -r requirements.txt

# Suite completa: 129 tests, ~2 min. Incluye el checklist de penalizaciones.
python -m pytest

# Notebook completo, de principio a fin.
python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=14400 \
    notebooks/proyecto1_mazariegos_herrera.ipynb
```

**Semillas.** `SEED_DATOS = 20260831` genera el único dataset. Las tres
semillas de modelo `(7, 13, 29)` varían solo la inicialización de la red;
los datos no cambian entre corridas. Todo está en `src/monitoreo/config.py`,
que es el único lugar donde vive un número mágico.

**Tiempo.** La primera ejecución entrena 11 modelos secuenciales en CPU y
tarda unas **2–3 horas**. Las siguientes tardan **minutos**: los pesos
quedan en `artefactos/modelos/` y `experimentos.correr_b_cacheado` los
recupera en vez de reentrenar. Las cifras son idénticas porque las semillas
están fijas. **Para forzar el reentrenamiento completo, borrar
`artefactos/modelos/`.**

Para iterar rápido, `MONITOREO_DEV=1` reduce el generador de 4,000 a 400
tarjetas. La corrida final se hace **sin** esa variable.

### Estructura

```
src/monitoreo/     lógica con tests — nada penalizable vive en el notebook
tests/             espeja src/; test_integridad.py es el checklist de §9
notebooks/         el entregable ejecutado
tools/             construir_notebook.py regenera el notebook desde texto plano
artefactos/        modelo candidato, scaler, vocabularios, config.json
informe/           informe.md y figuras/
docs/              spec de diseño, plan de implementación, DECISIONES.md
```

---

## 2. Versiones

| Componente | Versión |
|---|---|
| Python | 3.14.6 |
| Sistema | Windows 11 (10.0.26200) |
| numpy | 2.3.5 |
| pandas | 2.3.3 |
| scikit-learn | 1.8.0 |
| Keras | 3.15.1 |
| torch (backend de Keras) | 2.13.0+cpu |
| LightGBM | 4.7.0 |

> **Por qué torch y no TensorFlow.** El spec original pedía TensorFlow
> 2.21. TensorFlow no publica ruedas para Python 3.14, que es el intérprete
> de esta máquina, y no había un 3.12 disponible. Keras 3 es multi-backend
> y el Modelo B está escrito en API de Keras pura — `keras.utils.PyDataset`,
> `layers.GRU`, `keras.metrics.AUC` — así que corre sin un solo cambio sobre
> torch y guarda el mismo artefacto `.keras`. `monitoreo/__init__.py` fija
> `KERAS_BACKEND=torch` antes de que Keras se importe; **si se importa Keras
> antes que `monitoreo`, la importación falla**, porque Keras intenta cargar
> TensorFlow por defecto.

`reproducibilidad.versiones()` imprime esta tabla en la primera celda del
notebook y la deja grabada en `artefactos/config.json`.

---

## 3. Declaración de uso de IA

*(A completar a mano por cada integrante antes de entregar — el enunciado
pide para qué se usó y **qué verificó cada quien**.)*

**Andres Mazariegos** — Usé Claude Code para: _______. Verifiqué
personalmente: _______.

**June Herrera** — Usé Claude Code para: _______. Verifiqué personalmente:
_______.

Lo que sí quedó registrado de forma verificable: `docs/DECISIONES.md` lleva
las decisiones tomadas durante la construcción, cada una con la evidencia
que la inclinó y **qué cuesta si resultó equivocada**. Varias de ellas
salieron de revisiones que encontraron errores reales en el generador — por
ejemplo, que los sondeos de f1 iban siempre por canal `online` mientras las
ráfagas legítimas mezclaban POS/online, lo que dejaba a A separar los dos
mecanismos sin leer el orden y habría amañado el experimento a su favor.

---

## 4. Tres decisiones técnicas

### 4.1 GRU, y no LSTM ni CNN 1D ni Transformer

**Alternativas consideradas.** LSTM (la del Lab 4), CNN 1D, Transformer.

**La evidencia que inclinó la decisión.** Con `K = 20` no hay dependencia
de largo plazo que recordar, que es lo único que la compuerta de olvido
extra de la LSTM compra sobre la GRU; a cambio la LSTM trae ~33 % más
parámetros por unidad y converge más lento, y el presupuesto era de 4 días
en CPU. El Transformer está sobredimensionado para 20 pasos: su ventaja es
la atención sobre secuencias largas, y con 20 posiciones el costo
cuadrático no compra nada que un estado recurrente no dé más barato. La
CNN 1D fue el descarte más interesante y el que más se discutió: captura
patrones locales muy bien, pero el mecanismo f1 — "sondeo y golpe" —
requiere **acumular estado a lo largo de toda la ventana** para notar que
los montos pequeños vienen *creciendo* antes del golpe. Un filtro
convolucional de ancho fijo ve pares o tríos contiguos, no una tendencia
monótona de longitud variable.

**Qué cuesta si nos equivocamos.** Si la señal de f1 fuera realmente local,
una CNN 1D entrenaría más rápido y llegaría igual de lejos. La curva de
recorte de historia (Fig. 3) es lo que discrimina: si la AUC-PR saturara ya
en `K = 3`, la elección de GRU estaría sobredimensionada.

### 4.2 `K = 20`, elegido por la curva de recorte y no a dedo

**Alternativas consideradas.** `K ∈ {1, 3, 5, 10, 20}`.

**La evidencia que inclinó la decisión.** Se entrenó un modelo por cada
valor de `K` con la semilla 7, sobre la misma partición, y se graficó la
AUC-PR de validación contra `K` (Fig. 3, §11 del notebook). `K = 20` es el
valor que el enunciado sugiere, pero aquí está *justificado*: la figura
muestra dónde satura la ganancia, que es la respuesta a una pregunta de
negocio real — cuánta historia hay que guardar en producción para puntuar
una transacción.

La curva trae además un control de sanidad gratis: con `K = 1` el modelo
secuencial no tiene historia que leer y degenera en un clasificador
puntual, así que **debe** caer a la altura de A. Si con `K = 1` siguiera
ganando, la ventaja no vendría de la secuencia sino de la representación
por evento, y toda la tesis del proyecto estaría mal atribuida.

**Qué cuesta si nos equivocamos.** No se exploró `K > 20`. Si la curva aún
subiera en 20, estaríamos dejando señal sobre la mesa y subestimando el
valor del orden.

### 4.3 Ruta A (generador sintético) sobre Ruta B (dataset público)

**Alternativas consideradas.** El `creditcard.csv` de ULB quedó descartado
de entrada: no tiene identificador de tarjeta, así que es literalmente
imposible formar secuencias por cliente y el proyecto no tendría objeto.
La alternativa viva era buscar un dataset público con `card_id` + `ts`.

**La evidencia que inclinó la decisión.** Con la permutación como prueba
obligatoria, el riesgo dominante no es el realismo sino **quedarse sin
señal que medir**: si el orden no aporta nada en el dataset elegido, no hay
evidencia que presentar, solo un resultado nulo que no distingue "el orden
no importa" de "el modelo no lo encontró". Con 4 días, garantizar que la
señal exista vale más.

La circularidad — "es que usted lo construyó" — se mitiga explícitamente,
y esta es la parte que hay que poder defender:

- **f3** (monto atípico aislado) **no depende del orden**. Es un control
  negativo: A y B deben empatar, y si B ganara ahí habría un bug.
- **f2** (ráfaga de cajero) lo captura casi entero un conteo agregado.
- Los sondeos de f1 **escalan de forma monótona** (Q5 → Q12 → Q25 → Q38, el
  atacante tanteando el límite). Sin eso, f1 tampoco habría dependido del
  orden: media, máximo, conteo y cardinalidad de un conjunto son
  invariantes a la permutación, así que A habría visto "monto alto + muchas
  transacciones recientes + muchos comercios distintos" y ganado sin leer
  secuencia. "Estrictamente creciente" es lo único que no es invariante.
- Se inyectan **ráfagas legítimas confusoras** con la misma firma agregada
  que f1 e **idénticas en todas sus marginales** — canal, país, mcc y
  distribución de monto del evento grande — difiriendo solo en que sus
  montos van desordenados. Sin ellas el generador estaría amañado a favor
  de B; sin la monotonía, a favor de A.

**Qué cuesta si nos equivocamos.** Es la limitación honesta del proyecto y
está en la matriz de evidencias: lo que el experimento demuestra es que *si*
existe un patrón dependiente del orden como f1, un GRU lo encuentra y los
agregados no. **No** demuestra que ese patrón exista en el flujo real del
banco, ni con qué prevalencia.

### Decisiones adicionales (por si sale una al azar)

- **Umbral por costo, no por F1 máximo ni por percentil 95.** `p* = 180 /
  4200 = 0.0429`: bloquear a partir de ~4.3 % de probabilidad de fraude, no
  de 50 %, porque dejar pasar un fraude cuesta 23 veces más que molestar a
  un cliente legítimo. El Lab 4 usaba percentil 95, que aquí no tiene
  ninguna justificación económica.
- **`class_weight`, no SMOTE ni focal loss.** No hay sobremuestreo en
  ninguna partición. `class_weight` es reversible y no inventa
  transacciones que nunca ocurrieron; con 1.2 % de fraude y lotes de 2,048,
  cada lote lleva ~24 positivos, suficiente para que el gradiente de la
  minoritaria no se ahogue.
- **AUC-PR sobre el puntaje crudo, calibración solo para la decisión de
  costo.** La isotónica es monótona no decreciente, así que no invierte
  ningún par — pero al aplanar tramos **crea empates**, y la precisión
  promedio sí cambia con los empates (en una prueba, 5,000 puntajes
  distintos colapsaron a 5 mesetas y la AUC-PR se movió 0.024). Medir el
  ranking sobre el puntaje calibrado castigaría al modelo por un artefacto
  del calibrador.

---

## 5. Candidato al Proyecto Final

**Qué se conserva.** El modelo de menor costo esperado en test, guardado en
`artefactos/modelo_candidato.keras`. `artefactos/config.json` registra cuál
fue, su `u*` congelado, las AUC-PR de validación y test, el ahorro mensual
estimado y las versiones exactas.

Acompañan al modelo: `scaler.pkl` (los dos `StandardScaler`, el de eventos
y el de agregados, ambos ajustados **solo con train**),
`vocab_embeddings.json` (mcc / canal / comercio → índice, construidos solo
con train) y `generador_datos.py` (copia byte a byte del generador que
produjo los datos; hay un test que lo verifica).

**Quién lo usa y qué decide.** Un analista de riesgos, sobre la cola de
transacciones en vuelo. El puntaje calibrado se compara contra `u*`:

- `p ≥ u*` → **bloquear** y notificar al tarjetahabiente
- `p` en la banda inmediatamente inferior → **revisar** manualmente
- resto → **dejar pasar**

La banda de revisión existe porque el umbral óptimo por costo supone que
las únicas dos acciones son bloquear o no. En operación hay una tercera
más barata que un bloqueo erróneo.

**Contrato preliminar.**

```
Entrada:  {card_id, últimas 20 transacciones [{ts, amount, mcc, channel, merchant_id}]}
Salida:   {risk_score: float ∈ [0,1], decision: "block"|"review"|"allow", model_version: str}
Latencia objetivo: < 100 ms p95
```

**Límites, riesgos y datos que faltan.**

- **Los datos son sintéticos.** Es la limitación que gobierna todas las
  demás. Antes de cualquier decisión de producción hace falta repetir el
  experimento sobre el flujo real del banco, con `card_id` y `ts`.
- **Una sola ventana temporal**, sin validación rolling. No sabemos cómo se
  degrada el modelo mes a mes ni cada cuánto habría que reentrenarlo.
- **El calibrador y `u*` se ajustan ambos en validación**, así que el
  umbral hereda el optimismo de esa partición.
- **Los costos Q4,200 / Q180 son fijos y uniformes.** En la realidad el
  costo de un falso negativo escala con el monto y el de un falso positivo
  depende del cliente; un tarjetahabiente premium bloqueado no cuesta Q180.
- **Sin deriva ni adversario adaptativo.** El generador no modela que el
  atacante cambie de táctica al ver que lo bloquean, que es exactamente lo
  que pasa en producción.
- **La cardinalidad de comercios es de 300.** Un banco real tiene órdenes
  de magnitud más, y el embedding de comercio es la pieza que peor
  escalaría; habría que pasarlo a hashing.

---

## 6. Checklist de penalizaciones (§9 del enunciado)

Cada casilla es un test que corre, no una promesa.

| | Penalización | Estado | Verificado por |
|---|---|---|---|
| −20 | Partición aleatoria | Corte por percentil de `ts` **global** | `test_integridad.py::test_penalizacion_20_particion_no_es_aleatoria` |
| −15 | Estadísticas del conjunto completo | `fit` solo en train; agregados causales `closed='left'` | `test_penalizacion_15_vocabularios_y_scaler_solo_train`, `test_features_agregadas.py` |
| −15 | Exactitud como métrica principal | No se calcula ni se expone | `test_metricas.py::test_no_expone_exactitud` |
| −10 | Decidir mirando test | Todo en validación; test en una sola celda fechada | §15 del notebook |
| −10 | Afirmar aporte del orden sin permutación | Ambas variantes reportadas | §9 del notebook, `test_permutacion.py` |
