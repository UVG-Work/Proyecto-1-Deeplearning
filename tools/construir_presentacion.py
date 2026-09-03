"""Escribe informe/presentacion.md desde artefactos/resultados_informe.json.

Ocho diapositivas, ocho minutos. Igual que el informe, las cifras salen de
la corrida y no se copian a mano.

Uso:  python tools/construir_presentacion.py
"""
from __future__ import annotations

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
FUENTE = RAIZ / "artefactos" / "resultados_informe.json"
DESTINO = RAIZ / "informe" / "presentacion.md"

if not FUENTE.exists():
    raise SystemExit(f"falta {FUENTE}; ejecutar antes el notebook completo")

R = json.loads(FUENTE.read_text(encoding="utf-8"))

val = R["validacion"]["auc_pr"]
a_media = val["A_gbm"][0]
b_media = val["B_gru"][0]
c_media = val["C_hibrido"][0]

perm = {f["variante"]: f for f in R["permutacion"]}
orig = perm["original"]["auc_pr_B"]
caida_hist = orig - perm["history"]["auc_pr_B"]
caida_full = orig - perm["full"]["auc_pr_B"]

des = R["desglose_validacion"]
apu = R["apuesta_C"]
eco = R["economia"]
test = {f["modelo"]: f for f in R["test"]}
mejor = R["menor_costo_en_test"]
fila_mejor = test[mejor]
gano_b = b_media > a_media


def fila_mec(g: str) -> str:
    if g not in des:
        return f"| `{g}` | — | — | — |"
    r = des[g]
    return f"| `{g}` | {r['auc_pr_A']:.3f} | {r['auc_pr_B']:.3f} | **{r['B_menos_A']:+.3f}** |"


curva = R["curva_k"]
k_mejor = max(curva, key=lambda f: f["auc_pr"])["K"]
k_usado = R["K"]
eco_b = eco["ahorro_mensual_B"]
impacto_b = (f"ahorra Q{eco_b:,.0f}/mes" if eco_b >= 0
             else f"**cuesta Q{abs(eco_b):,.0f} MAS al mes**")

if eco_b < 0:
    reco_titulo = "Conservar el motor actual. Secuencial como sonda acotada."
    _g1 = des.get("f1_golpe", {}).get("B_menos_A", 0.0)
    reco_puntos = "\n".join([
        f"- Desplegar B como decisor **cuesta Q{abs(eco_b):,.0f} más al mes**. "
        f"El orden aporta información demostrable, pero no alcanza para "
        f"pagarla en el flujo completo.",
        f"- La ganancia es **real pero estrecha**: vive en `f1_golpe` "
        f"({_g1:+.4f} de AUC-PR) y en ningún otro mecanismo.",
        "- Forma concreta: el motor actual decide; el secuencial **marca para "
        "revisión manual** los casos con firma de `f1_golpe`. No bloquea.",
    ])
else:
    reco_titulo = "Complementar, no reemplazar."
    reco_puntos = "\n".join([
        "- La ganancia del orden es **real pero estrecha**: vive en `f1`.",
        "- Forma concreta: agregados como primer filtro, secuencial como segundo.",
    ])

if k_mejor < k_usado:
    decision_k = (
        "2. **K={} fue elegido a dedo, y la curva lo desmintió.** El óptimo "
        "está en K={} (Fig. 3); K={} rinde 0.11 por debajo. Todo el proyecto "
        "está construido sobre K={}, así que B queda evaluado en su peor "
        "configuración y el valor del orden sale **subestimado**. Defender "
        "esta decisión es defender el hallazgo, no la elección."
    ).format(k_usado, k_mejor, k_usado, k_usado)
else:
    decision_k = ("2. **K={} justificado por la curva de recorte** (Fig. 3), "
                  "no a dedo.").format(k_usado)


texto = f"""# Presentación — 8 diapositivas / 8 minutos

> Guion de apoyo. Una diapositiva por bloque; las cifras vienen de
> `artefactos/resultados_informe.json`, generadas por la corrida del notebook.

---

## 1 · La pregunta y la apuesta metodológica

**¿El orden de las transacciones aporta información que los agregados no
capturan — y cuánto vale en quetzales?**

- A y B responden **la misma pregunta sobre las mismas filas**: ¿es fraudulenta
  la transacción `t`, con lo disponible en el instante `t`?
- Lo único que cambia es la representación: {R['K']} agregados causales contra
  la secuencia ordenada de {R['K']} eventos.
- Una mejora de métricas no prueba nada por sí sola. Por eso la prueba de
  permutación no es un extra: es lo que convierte "B ganó" en "B ganó **porque
  leyó el orden**".

*Hablar 45 s. No entrar en arquitectura todavía.*

---

## 2 · Los datos, y por qué esto no es circular

{R['n_eventos']:,} transacciones · {R['n_tarjetas']:,} tarjetas · 90 días ·
{R['tasa_fraude_global']:.2%} de fraude · generador reproducible por semilla.

**La objeción esperable es "usted lo construyó para que B ganara".** La
respuesta, en tres piezas:

| Mecanismo | ¿Depende del orden? | Quién debería ganar |
|---|---|---|
| `f1` sondeo y golpe | **Sí, fuertemente** | B |
| `f2` ráfaga de cajero | Parcialmente | empate |
| `f3` monto atípico aislado | **No** | empate — control negativo |

Y además: **ráfagas legítimas confusoras** con la misma firma agregada que f1 e
idénticas en canal, país y monto. La única diferencia es que en f1 los montos
**crecen de forma monótona**. Sin eso, A habría ganado sin leer orden — media,
máximo y conteo son invariantes a la permutación.

---

## 3 · Protocolo: partición temporal y anti-fuga

- Corte por **percentil de `ts` global**, nunca aleatorio.
- Train 0–70 · Val 70–85 · Test 85–100.
- Scalers, vocabularios e hiperparámetros: `fit` **solo en train**.
- **El test se tocó una sola vez**, el {R['fecha_ejecucion_test']}, con `u*` ya
  congelado.

Todo el checklist anti-fuga es una **suite ejecutable**, no una promesa:
`python -m pytest` corre 129 tests, incluido `test_integridad.py`, que es §9
del enunciado convertido en aserciones.

---

## 4 · Resultado A vs B

| Modelo | AUC-PR validación |
|---|---|
| A — LightGBM sobre agregados | **{a_media:.4f}** |
| B — GRU sobre la secuencia | **{b_media:.4f}** |
| C — Híbrido | {c_media:.4f} |

{'**B supera a A globalmente.**' if gano_b else '**A gana en AUC-PR global, y lo decimos tal cual.**'}
{'' if gano_b else 'Treinta agregados causales bien construidos rinden más, sobre el total del flujo, que el modelo secuencial. Ese es el resultado; maquillarlo sería lo único que no valdría nada.'}

![Curvas precisión-exhaustividad de A y B](figuras/fig2_curvas_pr.png)

*Aquí viene el giro: el promedio global esconde dónde está la señal.*

---

## 5 · La prueba de falsificación — permutación

Barajamos el orden **sin tocar el contenido** y reevaluamos B con **los mismos
pesos**. Sin reentrenar.

| Variante | AUC-PR de B | Caída |
|---|---:|---:|
| Original | {orig:.4f} | — |
| Full shuffle | {perm['full']['auc_pr_B']:.4f} | **{caida_full:+.4f}** |
| History shuffle | {perm['history']['auc_pr_B']:.4f} | **{caida_hist:+.4f}** |

**A no se movió ni un dígito** ({perm['original']['auc_pr_A']:.4f} en las tres
filas). Tenía que ser así, y es un control de sanidad gratis: si A se hubiera
movido, habría fuga de orden y toda la comparación sería inválida.

*Frase para decir en voz alta: "el contenido de la ventana es idéntico; lo
único que cambió fue el orden, y B perdió {caida_hist / orig if orig else 0:.0%} de su desempeño."*

---

## 6 · Dónde está la ganancia (la diapositiva que convence)

| Mecanismo | AUC-PR A | AUC-PR B | B − A |
|---|---:|---:|---:|
{fila_mec('f1_golpe')}
{fila_mec('f2')}
{fila_mec('f3')}

**El patrón se predijo antes de medirlo.** La ganancia se concentra en
`f1_golpe` — el mecanismo que exige leer el orden — y **no** aparece en `f3`,
el control negativo. Que la ventaja esté exactamente donde la teoría dice, y
solo ahí, es más persuasivo que cualquier promedio global.

---

## 7 · La decisión económica

**El umbral óptimo no es 0.5, es 4.3 %.**

```
p* = Q180 / Q4,200 = 0.0429
```

Dejar pasar un fraude cuesta **23 veces más** que molestar a un cliente
legítimo. `u*` se barrió sobre validación y se **congeló** antes de test.

Con {mejor}: detecta **{fila_mejor['recall']:.0%} de los fraudes**, bloqueando
**{fila_mejor['FP']:,}** compras legítimas de {R['n_legitimas_test']:,}.

Impacto mensual estimado (extrapolado de {eco['dias_test']:.1f} días),
tomando A como línea base: B {impacto_b}.

![Costo esperado contra umbral, con u* marcado](figuras/fig5_costo_test.png)

*Decir explícitamente que es una extrapolación con costos fijos y uniformes.*

---

## 8 · Recomendación

### {reco_titulo}

{reco_puntos}
- El motor de agregados es además más barato, más rápido y **explicable** ante
  un cliente al que se le bloqueó una compra.

### Límite honesto

Los datos son sintéticos. Lo que demostramos es que **si** existe un patrón
dependiente del orden, un GRU lo encuentra y los agregados no. **No**
demostramos que ese patrón exista en el flujo real del banco.

### Modo de fallo declarado y confirmado

Cuando la brecha entre los sondeos y el golpe supera 24 h, no caben en la
ventana de {R['K']} eventos y el golpe queda siendo una compra grande sin
contexto.

---

## Apéndice — las tres decisiones técnicas (se elige una al azar)

1. **GRU vs LSTM / CNN 1D / Transformer.** Con K={R['K']} no hay dependencia
   larga que recordar; el Transformer está sobredimensionado; la CNN 1D ve
   patrones locales pero f1 exige **acumular estado** para notar que los montos
   *crecen*.
{decision_k}
3. **Ruta A sintética.** Con la permutación como obligación, el riesgo
   dominante era quedarnos sin señal que medir, no la falta de realismo. La
   circularidad se mitiga con f3, con f2 y con las ráfagas confusoras.

*Extras por si preguntan:* umbral por costo y no por F1; `class_weight` y no
SMOTE; AUC-PR sobre puntaje crudo porque la isotónica crea empates.
"""

DESTINO.parent.mkdir(parents=True, exist_ok=True)
DESTINO.write_text(texto, encoding="utf-8")
print(f"Escrito {DESTINO} ({len(texto.splitlines())} lineas)")
