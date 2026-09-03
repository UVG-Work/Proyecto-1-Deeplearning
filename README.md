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

# Suite completa: 140 tests (133 + 7 que esperan al notebook), ~3 min.
# Incluye el checklist de penalizaciones como aserciones ejecutables.
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

### 4.2 `K = 20` — la curva de recorte **no** respalda esta elección

**Alternativas consideradas.** `K ∈ {1, 3, 5, 10, 20}`.

**La evidencia, y lo que dice en contra nuestra.** Se entrenó un modelo por
cada valor de `K` con la semilla 7, sobre la misma partición (Fig. 3, §11
del notebook). El resultado no es el que esperábamos:

| K | AUC-PR val |
|---:|---:|
| 1 | 0.7146 |
| 3 | **0.7491** ← óptimo |
| 5 | 0.7470 |
| 10 | 0.6346 |
| 20 | 0.6360 |

`K = 20` es el valor que fijamos al principio, siguiendo la sugerencia del
enunciado, y sobre él está construido todo lo demás: el Modelo B, el
híbrido C, la permutación y los artefactos. **La curva dice que fue una mala
elección**: el óptimo está en `K = 3` y `K = 20` rinde 0.11 por debajo.

Lo decimos así porque es exactamente lo que la curva de recorte existe para
detectar, y porque presentarla como si hubiera confirmado nuestra decisión
sería el tipo de conclusión deshonesta que el enunciado penaliza. La
decisión defendible aquí no es "elegimos K=20 con evidencia" sino "fijamos
K=20 a priori y nuestra propia prueba de falsificación lo desmintió".

Hay además un segundo hallazgo incómodo: con `K = 1` el modelo **no** tiene
historia que leer y aun así rinde 0.7146, por encima de `K = 20`. Ese punto
estaba puesto como piso de sanidad. Que el piso supere a la configuración
completa apunta a un problema de optimización —30 épocas fijas, sin
programación de tasa de aprendizaje, y más padding conforme crece `K`— y no
a que la historia larga carezca de información.

**Qué cuesta que nos hayamos equivocado.** El Modelo B está evaluado en su
peor configuración, así que la comparación A vs B lo trata injustamente y
el valor del orden queda **subestimado**. La primera prueba a correr con más
tiempo es reentrenar B y C en `K = 3` y rehacer la comparación económica.
No lo hicimos aquí por presupuesto de cómputo: son ~2 h más de CPU.

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

**Qué se conserva.** El mejor modelo secuencial **según la AUC-PR de
validación** —nunca según el costo en test, que sería la penalización de
−10 pts—, guardado en `artefactos/modelo_candidato.keras`. Resultó ser
**B (GRU)**, con 0.7138 de AUC-PR en validación contra 0.6939 del híbrido C.
`artefactos/config.json` registra el criterio de selección, el `u*`
congelado, las AUC-PR de validación y test, el impacto económico y las
versiones exactas.

Se conserva también `artefactos/modelo_a_lightgbm.pkl`, el motor de
agregados. No es un extra: es el modelo que **gana** en este proyecto, y la
recomendación al comité es seguir decidiendo con él. El candidato secuencial
se guarda porque es la pieza que el Proyecto Final tendría que retomar y
mejorar —empezando por reentrenarlo en `K = 3`, ver §4.2—, no porque hoy sea
el mejor clasificador de los dos.

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
