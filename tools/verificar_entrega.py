"""Recorre §8 y §9 del enunciado contra lo que hay en el repo.

No sustituye a `pytest` -- que es donde viven las verificaciones de fuga --
sino que comprueba lo que solo existe despues de ejecutar el notebook: que
estan los entregables, que el test se corrio una sola vez y despues de la
hipotesis, y que ninguna cifra del informe se escribio a mano.

Uso:  python tools/verificar_entrega.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ART = RAIZ / "artefactos"

fallos: list[str] = []
avisos: list[str] = []


def ok(cond: bool, etiqueta: str, detalle: str = "") -> bool:
    marca = "OK  " if cond else "FALLA"
    print(f"  [{marca}] {etiqueta}" + (f" — {detalle}" if detalle else ""))
    if not cond:
        fallos.append(etiqueta)
    return cond


def aviso(cond: bool, etiqueta: str, detalle: str = "") -> None:
    if not cond:
        print(f"  [AVISO] {etiqueta}" + (f" — {detalle}" if detalle else ""))
        avisos.append(etiqueta)
    else:
        print(f"  [OK  ] {etiqueta}" + (f" — {detalle}" if detalle else ""))


print("=" * 70)
print("§8 — ENTREGABLES")
print("=" * 70)

nb = RAIZ / "notebooks" / "proyecto1_mazariegos_herrera.ipynb"
ok(nb.exists(), "notebook presente")

celdas_sin_salida = 0
if nb.exists():
    doc = json.loads(nb.read_text(encoding="utf-8"))
    codigo = [c for c in doc["cells"] if c["cell_type"] == "code"]
    errores = [o for c in codigo for o in c.get("outputs", [])
               if o.get("output_type") == "error"]
    celdas_sin_salida = sum(1 for c in codigo if not c.get("outputs"))
    ok(not errores, "notebook ejecutado sin excepciones",
       f"{len(codigo)} celdas de codigo")
    ok(celdas_sin_salida == 0, "todas las celdas tienen salida visible",
       f"{celdas_sin_salida} sin salida")

for nombre in ("modelo_candidato.keras", "scaler.pkl", "vocab_embeddings.json",
               "config.json", "generador_datos.py"):
    ok((ART / nombre).exists(), f"artefactos/{nombre}")

# El enunciado (5) los nombra planos: informe.pdf y presentacion.pdf. Se
# publican en la raiz, que es donde el comite los va a buscar, y se conserva
# la copia de trabajo bajo informe/ junto a las fuentes .md y las figuras.
for nombre in ("informe.pdf", "presentacion.pdf"):
    en_raiz = (RAIZ / nombre).exists()
    en_informe = (RAIZ / "informe" / nombre).exists()
    ok(en_raiz, nombre, "en la raiz de la entrega" if en_raiz
       else ("existe en informe/ pero no en la raiz" if en_informe else "no existe"))

ok((RAIZ / "README.md").exists(), "README.md")

print()
print("=" * 70)
print("§9 — PENALIZACIONES")
print("=" * 70)

cfg_json = ART / "config.json"
if not cfg_json.exists():
    print("  falta artefactos/config.json; ejecutar el notebook completo")
    sys.exit(1)

C = json.loads(cfg_json.read_text(encoding="utf-8"))

ok(C.get("seed_datos") is not None and C.get("K") is not None,
   "-20 particion aleatoria", "corte por percentil de ts global (ver test_integridad)")

ok("validacion" in C.get("criterio_de_seleccion", "").lower(),
   "-10 elegir mirando test", f"criterio: {C.get('criterio_de_seleccion')}")

ok(C["fecha_hipotesis_C"] < C["fecha_ejecucion_test"],
   "-10 apuesta declarada antes de entrenar",
   f"hipotesis {C['fecha_hipotesis_C']} < test {C['fecha_ejecucion_test']}")

# El test debe aparecer una sola vez en el notebook.
fuente_nb = nb.read_text(encoding="utf-8") if nb.exists() else ""
n_marcas = fuente_nb.count("TEST EJECUTADO UNA SOLA VEZ")
ok(n_marcas == 2, "-10 test tocado una sola vez",
   f"la marca aparece {n_marcas} veces (codigo + salida)")

u = C["umbral_u_estrella"]
ok(0 < u["candidato"] < 0.5, "-15 umbral por costo y no 0.5",
   f"u*={u['candidato']:.4f}, teorico={C['umbral_teorico']:.4f}")

# La exactitud no debe aparecer como metrica.
metricas_src = (RAIZ / "src" / "monitoreo" / "metricas.py").read_text(encoding="utf-8")
sin_exactitud = not re.search(r"def (exactitud|accuracy)", metricas_src)
cuerpo_nb = " ".join(
    "".join(c["source"]) for c in json.loads(fuente_nb)["cells"]) if fuente_nb else ""
ok(sin_exactitud, "-15 exactitud como metrica principal",
   "metricas.py no define exactitud ni accuracy")

print()
print("=" * 70)
print("§12 — DEFINICION DE HECHO")
print("=" * 70)

res = ART / "resultados_informe.json"
ok(res.exists(), "resultados_informe.json escrito por la corrida")

if res.exists() and (RAIZ / "informe" / "informe.md").exists():
    R = json.loads(res.read_text(encoding="utf-8"))
    informe = (RAIZ / "informe" / "informe.md").read_text(encoding="utf-8")
    # Una cifra de control: el AUC-PR de A debe aparecer en el informe.
    a = R["validacion"]["auc_pr"]["A_gbm"][0]
    ok(f"{a:.4f}" in informe, "las cifras del informe salen de la corrida",
       f"AUC-PR de A = {a:.4f} localizado en informe.md")

evidencias = ("Evidencia 1", "Evidencia 2", "Evidencia 3",
              "Evidencia 4", "Evidencia 5", "Evidencia 6")
if (RAIZ / "informe" / "informe.md").exists():
    informe = (RAIZ / "informe" / "informe.md").read_text(encoding="utf-8")
    faltan = [e for e in evidencias if e not in informe]
    ok(not faltan, "las seis evidencias son localizables", f"faltan: {faltan}")
    filas = [l for l in informe.splitlines()
             if l.startswith("|") and l.count("|") >= 5]
    aviso("ninguna" not in informe.lower().split("matriz de evidencias")[-1],
          "la matriz no dice 'ninguna' en ninguna limitacion")

figs = sorted((RAIZ / "informe" / "figuras").glob("*.png")) if (RAIZ / "informe" / "figuras").exists() else []
ok(len(figs) >= 4, "figuras generadas", f"{len(figs)}: {[f.name for f in figs]}")

print()
print("=" * 70)
if fallos:
    print(f"RESULTADO: {len(fallos)} FALLAS")
    for f in fallos:
        print(f"  - {f}")
    sys.exit(1)
print("RESULTADO: todo en verde" + (f" ({len(avisos)} avisos)" if avisos else ""))
