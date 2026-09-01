# Decisiones de diseño y su justificación

**Proyecto 1 — Monitoreo transaccional** · Mazariegos / Herrera

Cada decisión que se apartó del spec original, con la evidencia que la
inclinó y lo que cuesta si resultó equivocada. En la presentación se elige
una al azar y cualquiera de los dos debe poder defenderla.

---

## Ruling C: `_timestamps` garantiza timestamps únicos por tarjeta (desempate por
microsegundos) en vez de relajar el test. — El `drop_duplicates` podía bajar el
conteo por debajo de `TX_MIN` y romper el contrato del generador. — Si me
equivoco: los Δt de eventos colisionados quedan artificialmente en microsegundos,
lo que inflaría levemente la señal de f2; visible en el EDA.

## Ruling D: `_inyectar_f3` precalcula los cuantiles por tarjeta con un solo
`groupby` en vez de escanear `base` por episodio. — 3,600 escaneos de 400k filas
son ~1.4B operaciones; el generador tardaría minutos en vez de segundos. — Si me
equivoco: ninguno, es equivalencia exacta.

## Ruling E: f3 se inyecta DESPUÉS de las ráfagas legítimas y calcula el percentil
sobre `base + ráfagas`. — Una compra grande legítima de una ráfaga podía superar
el p99.9 que f3 usó como referencia, volviendo el test flaky. — Si me equivoco:
los montos de f3 suben un poco en tarjetas con ráfaga, lo que hace f3 más fácil
para ambos modelos por igual; no sesga la comparación A vs B.

## Ruling I: se reescribe el assert tautológico de `test_primera_transaccion_sin_contexto`.
— `np.isnan(x) is False` es siempre falso; el test no probaba nada. — Si me
equivoco: ninguno.

## Ruling K: `test_scaler_ajustado_solo_en_train` compara ajustar con `es_train`
contra ajustar con todo, y exige que difieran. — La versión original reajustaba
el mismo scaler y comparaba consigo mismo. Es la penalización de −15 pts: el test
tiene que morder. — Si me equivoco: ninguno.

## Ruling N: el test comprueba los nombres en `m.inputs` en vez de asumir que
`m.input_shape` es un dict. — Depende de un detalle de Keras 3 que no controlamos.
— Si me equivoco: ninguno, se verifica lo mismo.

## Ruling T3-1: los montos de sondeo de f1 se fuerzan estrictamente crecientes DESPUES
del redondeo a centavos, en vez de bajar el umbral del test. — Dos draws contiguos
pueden redondear al mismo centavo y un empate destruye exactamente la propiedad en la
que descansa la tesis del proyecto. Corregir el generador es preferible a aflojar el
test. — Si me equivoco: en los pares empatados el escalamiento queda en Q0.01, una
senal debil; afecta ~1 de cada 36 episodios y no cambia la distribucion de montos.

## Ruling T3-2: f1 y f2 eligen tarjeta SIN reemplazo (una tarjeta se compromete una sola
vez por mecanismo). — El test agrupa por tarjeta y dos atracos con niveles de monto
distintos lo rompen legitimamente; drenar la misma tarjeta dos veces por el mismo
mecanismo tampoco es realista. — Si me equivoco: se pierde el caso real pero raro de
reincidencia. A escala completa hay 4000 tarjetas para ~350 episodios f1 y ~420 f2,
asi que no hay presion de espacio.

## Ruling T3-3: f1 y las rafagas legitimas deben ser identicas en TODAS sus marginales
(canal, pais, mcc y distribucion de monto del evento grande), difiriendo unicamente en
el escalamiento monotono de los montos chicos. — El revisor detecto que los sondeos van
siempre channel="online" mientras las rafagas mezclan POS/online; ademas el golpe usa
monto_base*U(8,30) y pais GT/US mientras la compra grande de la rafaga usa U(600,3000)
y pais GT. El Modelo A puntua esa transaccion viendo su propio one-hot de canal y su
amt_ratio_to_mean_7d, asi que separa f1 del confusor SIN leer orden. La seccion 4.1.1
del spec exige explicitamente misma firma agregada, de modo que esto es incumplimiento
del spec y no mejora opcional. El defecto viene del plan, no del implementador.
— Si me equivoco: la compra grande legitima pasa a ser 8-30x el gasto base del cliente
en vez de Q600-3000 fijos, un poco menos realista para tarjetas de gasto muy alto, pero
es el precio de que el control negativo sea honesto.

## Ruling T4-1: todas las tarjetas deben estar activas sobre la MISMA ventana calendario.
Se introduce HORIZONTE_DIAS = 90 en config; _timestamps sortea los n_tx eventos
uniformemente dentro de ese horizonte y los ordena (proceso de Poisson homogeneo
condicionado al conteo) en vez de acumular brechas exponenciales a la tasa propia de
cada tarjeta. tasa_dia pasa a ser derivada (n_tx / HORIZONTE_DIAS), conservando la
columna y su significado. — Con el diseno anterior el span por tarjeta iba de 16 a 275
dias, dejando test 27x mas ralo que train (15.2 vs 405.7 filas/dia) y con solo el 36%
de las tarjetas. Eso infla Delta_t en test justamente para el Modelo B, cuya feature
central es Delta_t, sesgando la comparacion A vs B en contra de B por un artefacto del
generador; y vuelve absurda la extrapolacion mensual con dias_test=216. Es defecto del
plan, no del implementador. — Si me equivoco: la densidad baja de ~2.25 tx/dia a ~1.1,
lo que hace f2 (rafaga de cajero) mas facil de detectar por agregados; eso favorece al
Modelo A, no a B, asi que si erro lo hago en la direccion conservadora para la tesis.

## Ruling T4-2: test_f1_y_rafagas_tienen_la_misma_firma_agregada se reescribe para medir los
inyectores directamente (_inyectar_f1 / _inyectar_rafagas_legitimas con 300 episodios) en
vez de detectar rachas por heuristica sobre el flujo ya mezclado. — El test fallo con
0.579 vs 0.470 (umbral 0.1), pero la medicion directa da 0.481 vs 0.526 en los chicos y
0.500 vs 0.516 en los grandes: el codigo de produccion es correcto y las cuatro llamadas
son literalmente rng.choice(("POS","online")). El fallo venia de la heuristica, que
arrastra ~20% de clusters casuales del flujo ordinario (mix 0.28 online) y hunde el
estimador, sobre una muestra de solo ~152 filas. Subir el umbral habria escondido el
problema del estimador; medir en la fuente lo elimina y de paso cierra el minor que el
re-review habia diferido. — Si me equivoco: el test pasa a depender de funciones privadas
del modulo, asi que un refactor de los inyectores lo rompe; a cambio gana ~10x de muestra
y cero contaminacion.

## Ruling T5-1: _anclas pasa de u=uniform(0.1, 0.9) a uniform(0.01, 0.99) para que el fraude
se reparta por todo el horizonte. — Medido por deciles: los deciles 0 y 9 tienen 0.00% de
fraude por construccion, y test (ultimo 15% del tiempo) cae casi entero en la zona muerta,
quedando en 0.50% contra 1.29% de train. La AUC-PR es sensible a la prevalencia, asi que
las cifras de test saldrian deprimidas para A y para B; y el umbral u* congelado en val
(1.46%) aplicado a test (0.50%) distorsiona el analisis de costo y la cifra de ahorro
mensual, que es la evidencia central del informe. El revisor lo marco Minor; lo elevo a
bloqueante porque golpea la corrida unica de test. — Si me equivoco: los episodios
anclados cerca del dia 89 pueden extenderse mas alla del horizonte nominal, alargando
levemente el rango de esas tarjetas; inocuo.
Task 6: implementado (commit 089fcbb). 8/8 y suite 49/49. win_idx (42849,20) int32 = 3.27 MB, 82.3% ventanas completas.
Fix T5-1: despachado (haiku, BASE 089fcbb)

## Ruling T8-1: se corrige el off-by-one de _n_distintos_causal (ts[izq] <= limite pasa a
ts[izq] < limite). — pandas rolling(closed='left') incluye el punto que cae exactamente
en el borde izquierdo; el barrido de dos punteros lo excluia, asi que n_merchants_24h y
n_tx_24h discrepaban sobre que hay en la ventana cuando dos transacciones distan
exactamente 24h. El brief pedia explicitamente que ambos caminos usaran los mismos
limites. — Si me equivoco: ninguno, es alinear dos definiciones que ya debian coincidir.

## Ruling T8-2: el test de reparto de fraude deja de apoyarse solo en la banda de 2.5x sobre
deciles y agrega una asercion directa de presencia de fraude en el primer y el ultimo 5%
del horizonte. — El revisor demostro que uniform(0.05,0.95), una version mas suave del
mismo defecto, pasa en silencio con la banda actual. Con ~30-80 fraudes por decil el
ruido de 1 sigma es ~14%, asi que apretar la banda produciria flakes; medir presencia en
los extremos es la propiedad que realmente importa y es casi libre de ruido. — Si me
equivoco: la asercion de extremos podria fallar si la tasa global bajara mucho; a la
escala actual hay margen holgado.
Fix T8-1 + T8-2 aplicado (commit eeda254). Suite 73/73. Pre-fix n_merchants_24h en el borde
  daba 0.0 donde n_tx_24h daba 1.0: el off-by-one era real.
Traspaso escrito y commiteado: docs/TRASPASO.md
Task 7 + Task 8: complete (commits 24989e0..e6e340b, review clean, sin Critical/Important)
Task 7: minor: caracter no-ASCII (§) en un docstring de test_permutacion.py, heredado del brief
Task 8: minor: construir() no verifica que df venga ordenado por (card_id, ts); un frame
  desordenado produce Delta_t y same_merchant_as_prev basura EN SILENCIO. Se eleva a arreglo
  porque June va a llamar esa funcion y el fallo no da error.
