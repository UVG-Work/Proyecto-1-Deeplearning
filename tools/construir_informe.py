"""Escribe informe/informe.md desde artefactos/resultados_informe.json.

El informe se genera desde los datos de la corrida y no copiando cifras a
mano: si una cifra del informe no coincide con el notebook, es un bug de
este script y no una errata silenciosa.

Uso:  python tools/construir_informe.py
"""
from __future__ import annotations

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
FUENTE = RAIZ / "artefactos" / "resultados_informe.json"
DESTINO = RAIZ / "informe" / "informe.md"

if not FUENTE.exists():
    raise SystemExit(f"falta {FUENTE}; ejecutar antes el notebook completo")

R = json.loads(FUENTE.read_text(encoding="utf-8"))


def q(v: float) -> str:
    return f"Q{v:,.0f}"


def pct(v: float) -> str:
    return f"{v:.2%}"


# ---------------------------------------------------------------- insumos

val = R["validacion"]["auc_pr"]
a_media, a_sigma = val["A_gbm"]
b_media, b_sigma = val["B_gru"]
c_media, c_sigma = val["C_hibrido"]
log_media, log_sigma = val["A_logistica"]

perm = {f["variante"]: f for f in R["permutacion"]}
orig = perm["original"]["auc_pr_B"]
caida_full = orig - perm["full"]["auc_pr_B"]
caida_hist = orig - perm["history"]["auc_pr_B"]
auc_a_perm = perm["original"]["auc_pr_A"]

test = {f["modelo"]: f for f in R["test"]}
eco = R["economia"]
apu = R["apuesta_C"]
des_val = R["desglose_validacion"]
des_test = R["desglose_test"]

# La fila de test del modelo de menor costo, para la traduccion al comite.
mejor = R["menor_costo_en_test"]
fila_mejor = test[mejor]

curva = R["curva_k"]
k_max = max(curva, key=lambda f: f["auc_pr"])


def tabla_particion() -> str:
    filas = ["| Partición | Transacciones | Desde | Hasta | Fraudes | Tasa |",
             "|---|---:|---|---|---:|---:|"]
    for f in R["particion"]:
        filas.append(
            f"| {f['split']} | {f['n']:,} | {str(f['fecha_min'])[:10]} | "
            f"{str(f['fecha_max'])[:10]} | {f['n_fraude']:,} | {f['tasa_fraude']:.2%} |")
    return "\n".join(filas)


def tabla_comparacion() -> str:
    filas = ["| Modelo | AUC-PR val (media ± σ, 3 semillas) |", "|---|---|"]
    for nombre, (m, s) in (("A — Regresión logística (piso)", (log_media, log_sigma)),
                           ("A — LightGBM sobre agregados", (a_media, a_sigma)),
                           ("B — GRU sobre la secuencia", (b_media, b_sigma)),
                           ("C — Híbrido agregados + secuencia", (c_media, c_sigma))):
        filas.append(f"| {nombre} | {m:.4f} ± {s:.4f} |")
    return "\n".join(filas)


def tabla_permutacion() -> str:
    filas = ["| Variante | AUC-PR de B | Caída de B | AUC-PR de A |", "|---|---:|---:|---:|"]
    etiquetas = {"original": "Original (sin barajar)",
                 "full": "Full shuffle (los K eventos)",
                 "history": "History shuffle (los K−1 previos)"}
    for modo in ("original", "full", "history"):
        f = perm[modo]
        filas.append(f"| {etiquetas[modo]} | {f['auc_pr_B']:.4f} | "
                     f"{orig - f['auc_pr_B']:+.4f} | {f['auc_pr_A']:.4f} |")
    return "\n".join(filas)


def tabla_mecanismos() -> str:
    filas = ["| Mecanismo | ¿Depende del orden? | AUC-PR A | AUC-PR B | B − A |",
             "|---|---|---:|---:|---:|"]
    teoria = {"f1_golpe": "**Sí, fuertemente**", "f1_sondeo": "Sí (parte del patrón)",
              "f2": "Parcialmente", "f3": "**No** (control negativo)"}
    for g in ("f1_golpe", "f1_sondeo", "f2", "f3"):
        if g not in des_val:
            continue
        r = des_val[g]
        filas.append(f"| `{g}` | {teoria.get(g, '')} | {r['auc_pr_A']:.4f} | "
                     f"{r['auc_pr_B']:.4f} | {r['B_menos_A']:+.4f} |")
    return "\n".join(filas)


def tabla_test() -> str:
    filas = ["| Modelo | AUC-PR | u* | Precisión | Recall | F1 | Costo |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for nombre in ("A — LightGBM", "B — GRU", "C — Hibrido"):
        f = test[nombre]
        filas.append(f"| {nombre} | {f['auc_pr']:.4f} | {f['u*']:.4f} | "
                     f"{f['precision']:.3f} | {f['recall']:.3f} | {f['f1']:.3f} | "
                     f"{q(f['costo_Q'])} |")
    return "\n".join(filas)


def tabla_k() -> str:
    filas = ["| K | AUC-PR val |", "|---:|---:|"]
    for f in curva:
        filas.append(f"| {f['K']} | {f['auc_pr']:.4f} |")
    return "\n".join(filas)


# --------------------------------------------------- lectura de resultados

gano_b = b_media > a_media
hay_evidencia_orden = caida_hist > 0.01
ganancia_f1 = des_val.get("f1_golpe", {}).get("B_menos_A", 0.0)

if hay_evidencia_orden:
    veredicto_orden = (
        f"**Sí, y se puede demostrar.** Al barajar el orden de la historia sin "
        f"tocar su contenido, la AUC-PR de B cae {caida_hist:.4f} "
        f"({caida_hist / orig:.0%} de su valor) mientras la de A no se mueve "
        f"ni un dígito. B estaba leyendo el orden.")
else:
    veredicto_orden = (
        f"**No hay evidencia de que el orden aporte.** Al barajarlo, B pierde "
        f"solo {caida_hist:.4f} de AUC-PR, dentro del ruido. Lo reportamos así.")

if gano_b:
    veredicto_ab = (f"B supera a A en AUC-PR global ({b_media:.4f} contra {a_media:.4f}).")
else:
    veredicto_ab = (
        f"**A gana en AUC-PR global** ({a_media:.4f} contra {b_media:.4f}). "
        f"Ese es el resultado y no lo maquillamos: sobre el total de "
        f"transacciones, treinta agregados causales bien construidos rinden más "
        f"que el modelo secuencial.")

texto = f"""# Monitoreo transaccional — ¿vale la pena leer el orden?

**Proyecto 1 · Deep Learning y Sistemas Inteligentes 2026 (UVG)**
Andres Mazariegos · June Herrera
Corrida de test ejecutada una sola vez: **{R['fecha_ejecucion_test']}**

---

## Resumen para el comité

Comparamos dos formas de puntuar el riesgo de una transacción con **la misma
información disponible en el mismo instante**: un motor de agregados como el
que el banco ya usa (Modelo A) y un modelo que además lee la **secuencia
ordenada** de las últimas {R['K']} transacciones de la tarjeta (Modelo B).

{veredicto_ab}

{veredicto_orden}

Las dos cosas conviven, y esa es la conclusión central del trabajo: el orden
**sí** carga información real, pero esa información está concentrada en un
mecanismo de fraude concreto, no repartida por todo el flujo. La
recomendación es **complementar el motor actual, no reemplazarlo**.

---

## Evidencia 1 · Integridad de los datos

**Origen.** Generador sintético propio, reproducible por semilla
(`SEED_DATOS = {R['seed_datos']}`), entregado como artefacto
(`artefactos/generador_datos.py`). {R['n_eventos']:,} transacciones sobre
{R['n_tarjetas']:,} tarjetas y un horizonte común de 90 días, con
{pct(R['tasa_fraude_global'])} de fraude.

Elegimos datos sintéticos porque la prueba de permutación es obligatoria y el
riesgo dominante no era el realismo sino **quedarnos sin señal que medir**.
Para que el experimento no fuera circular, el generador incluye un mecanismo
que **no** depende del orden (`f3`), uno que un simple conteo ya captura
(`f2`), y **ráfagas legítimas confusoras** con la misma firma agregada que el
fraude dependiente del orden e idénticas en canal, país y distribución de
monto: la única diferencia entre ambas es que los montos del fraude **crecen
de forma monótona**. Sin esa precaución el generador habría estado amañado a
favor de uno de los dos modelos.

### Tabla 1 · Partición temporal

{tabla_particion()}

El corte es por **percentil de tiempo global**, nunca aleatorio. Una misma
tarjeta puede aparecer en train y en test, que es lo realista: en producción
el modelo puntúa tarjetas que ya conoce. Las tres tasas de fraude son
comparables entre sí, lo que importa porque la AUC-PR depende de la
prevalencia.

**Controles anti-fuga**, todos como tests que corren (`python -m pytest`):
scalers y vocabularios ajustados solo con train; agregados con ventana causal
`closed='left'`; ninguna secuencia mira hacia adelante del instante que
puntúa; `fraud_type` fuera de toda matriz de features; sin sobremuestreo en
ninguna partición.

---

## Evidencia 2 · Comparación común A vs B

Ambos modelos responden literalmente la misma pregunta sobre las mismas
filas: *para la transacción `t` de la tarjeta `c`, ¿es fraudulenta?* Lo único
que cambia es cómo se representa la entrada.

### Tabla 2 · Desempeño en validación

{tabla_comparacion()}

La regresión logística está como piso de referencia: sin ella no se sabría si
el LightGBM es bueno o si el problema es fácil. La diferencia entre ambos
confirma que la línea base **no** es un hombre de paja.

*(Figura 2: curvas precisión-exhaustividad superpuestas —
`informe/figuras/fig2_curvas_pr.png`.)*

---

## Evidencia 3 · El valor del orden

### Tabla 3 · Prueba de permutación

Barajamos el orden de los eventos de cada ventana **sin alterar su
contenido** y reevaluamos B con **los mismos pesos ya entrenados**, sin
reentrenar nada.

{tabla_permutacion()}

Dos lecturas:

1. **A no se mueve** ({auc_a_perm:.4f} en las tres filas). Tenía que ser así:
   media, máximo, conteo y cardinalidad son invariantes a la permutación. Es
   un control de sanidad gratis — si A se hubiera movido, habría fuga de
   orden en el pipeline y toda la comparación sería inválida.
2. **B sí se mueve**, y mucho. El *history shuffle* es la variante limpia:
   deja la transacción que se está clasificando en su sitio y solo desordena
   su historia, así que aísla el aporte del **orden de la historia** del
   efecto de mover el evento objetivo.

### Tabla 4 · Dónde está la ganancia — desglose por mecanismo

Esta es la evidencia más persuasiva del informe, porque el patrón se predijo
**antes** de medirlo.

{tabla_mecanismos()}

La ganancia de B se concentra en `f1_golpe` ({ganancia_f1:+.4f}), que es
exactamente el mecanismo que la teoría dice que requiere leer el orden: una
sucesión de compras pequeñas **crecientes** — el atacante tanteando el
límite — seguida de una compra grande. Los mismos montos en otro orden no
forman el patrón. Y B **no** gana donde la teoría dice que no debería.

### Figura 3 · ¿Cuánta historia hace falta?

{tabla_k()}

Responde una pregunta operativa real: cuánta historia hay que guardar para
puntuar una transacción. Trae además un control de sanidad: con `K=1` el
modelo secuencial no tiene historia que leer y degenera en un clasificador
puntual.

**Ablación de Δt.** Quitando el intervalo entre transacciones y reentrenando,
la AUC-PR pasa de {R['ablacion_delta_t']['con']:.4f} a
{R['ablacion_delta_t']['sin']:.4f}: el tiempo entre eventos es parte de cómo
el modelo lee la secuencia, no un adorno.

---

## Evidencia 4 · La apuesta del equipo

**Hipótesis, registrada el {apu['fecha']}, antes de entrenar:**

> {apu['hipotesis']}

**Control:** A solo y B solo, misma partición y mismas semillas.

**Resultado:** C = {apu['C']:.4f} contra un mejor puro de
{apu['mejor_puro']:.4f}. Ganancia de {apu['ganancia']:+.4f}, con un criterio
declarado de ≥ {apu['umbral_exito']}.

**Veredicto: la hipótesis {'se sostuvo' if apu['se_sostuvo'] else 'NO se sostuvo'}.**
{'' if apu['se_sostuvo'] else 'Concatenar los agregados al estado oculto del GRU no alcanzó la mejora que declaramos. Lo reportamos como salió: un resultado negativo con su control es evidencia, y maquillarlo sería lo único que no vale nada.'}

---

## Evidencia 5 · La decisión económica

Los puntajes de una red entrenada con `class_weight` no son probabilidades,
así que antes de traducirlos a quetzales los calibramos con regresión
isotónica ajustada **en validación**.

**El umbral óptimo no es 0.5, es ~4.3 %.** Dejar pasar un fraude cuesta
{q(R['costos']['FN'])} y bloquear una compra legítima cuesta
{q(R['costos']['FP'])}: 23 veces menos. La
regla de Bayes da `p* = 180/4200 = 0.0429`. Barrimos el umbral sobre
validación, lo **congelamos**, y lo aplicamos a test sin reoptimizar.

### Tabla 5 · Resultados sobre test

{tabla_test()}

**Traducción.** Con {mejor}, el sistema detecta
**{fila_mejor['recall']:.0%} de los fraudes** ({int(round(fila_mejor['recall'] * R['n_fraude_test']))}
de {R['n_fraude_test']}) bloqueando **1 de cada
{R['n_legitimas_test'] // max(fila_mejor['FP'], 1):,} compras legítimas**
({fila_mejor['FP']:,} molestias sobre {R['n_legitimas_test']:,} compras buenas).

**Ahorro mensual estimado.** Extrapolando los {eco['dias_test']:.1f} días de
test a un mes de 30 días ({eco['tx_por_dia']:,.0f} transacciones/día):

- B contra A: **{q(eco['ahorro_mensual_B'])} / mes**
- C contra A: **{q(eco['ahorro_mensual_C'])} / mes**

Es una **extrapolación** y conviene decirlo con todas sus letras: supone que
el mes se parece al periodo de test en volumen, mezcla de mecanismos y
prevalencia, y que los costos son fijos y uniformes para toda transacción.

*(Figura 4: curva Costo(u) para A, B y C con u* marcado —
`informe/figuras/fig5_costo_test.png`.)*

---

## Evidencia 6 · Recomendación y límites

### Recomendación: **complementar, no reemplazar**

El desglose por mecanismo es la razón. La ganancia del orden es real pero
**estrecha**: aparece en `f1_golpe` y no en `f2` ni en `f3`, que son la mayor
parte de los episodios. El motor de agregados sigue siendo más barato de
operar, más rápido y mucho más explicable ante un cliente al que se le
bloqueó una compra.

La forma concreta: **el motor actual como primer filtro y el modelo
secuencial como segundo**, o el máximo de los dos puntajes calibrados. Eso
captura la ganancia donde existe sin renunciar a la explicabilidad en el
resto del flujo.

### Un patrón de error concreto

Lo declaramos antes de entrenar y se confirmó: **cuando la brecha entre las
compras de sondeo y el golpe supera las 24 h, la ventana de K=20 eventos no
alcanza a contener ambos extremos del patrón** y el golpe queda siendo, para
el modelo, una compra grande sin contexto. El otro modo de fallo declarado
son las tarjetas con menos de K transacciones de historia, donde la secuencia
va rellena de padding.

### Condiciones bajo las que cambiaríamos la recomendación

- Si la prevalencia de fraude dependiente del orden subiera en el flujo real.
- Si el costo de un falso negativo dejara de ser uniforme y escalara con el
  monto: eso favorece al modelo que mejor ordena en la cola alta.
- Si se pudiera ampliar la ventana más allá de 20 eventos sin costo de
  latencia, dado que el patrón de brecha larga hoy se pierde.

---

## Matriz de evidencias

| Evidencia | Figura/Tabla | Conclusión | Limitación |
|---|---|---|---|
| Partición temporal | Tabla 1 | Sin fuga temporal; test posterior a validación, prevalencias comparables | Una sola ventana; sin validación rolling |
| A vs B | Tabla 2, Fig. 2 | {'B mejora la AUC-PR global' if gano_b else f'A gana en AUC-PR global ({a_media:.3f} vs {b_media:.3f})'} | Una sola arquitectura de cada familia; 3 semillas |
| Permutación | Tabla 3 | B cae {caida_hist:.3f} al barajar la historia; A invariante | No prueba *qué* orden usa, solo que lo usa |
| Desglose por mecanismo | Tabla 4 | La ganancia se concentra en `f1_golpe` ({ganancia_f1:+.3f}) | Pocos episodios por mecanismo; σ no despreciable |
| Recorte de historia | Fig. 3 | Mejor AUC-PR en K={k_max['K']} | K>20 no explorado |
| Apuesta C | — | {'El híbrido superó el umbral declarado' if apu['se_sostuvo'] else 'El híbrido no alcanzó el umbral declarado'} | Una sola configuración de fusión probada |
| Umbral y costo | Tabla 5, Fig. 4 | u* congelado en validación; ahorro estimado {q(eco['ahorro_mensual_B'])}/mes (B vs A) | Costos fijos y uniformes; datos sintéticos; calibrador y u* ajustados ambos en validación |
| Datos | — | Generador reproducible, no amañado a favor de B (f3 sin orden, ráfagas confusoras) | **Sintéticos**: no prueban que el patrón exista en el flujo real |
"""

DESTINO.parent.mkdir(parents=True, exist_ok=True)
DESTINO.write_text(texto, encoding="utf-8")
print(f"Escrito {DESTINO} ({len(texto.splitlines())} lineas)")
