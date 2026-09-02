"""Convierte un Markdown del informe a PDF.

Markdown -> HTML con estilo de impresion -> PDF con Edge en modo headless.
No hay pandoc ni una WeasyPrint funcional en esta maquina; Edge si esta y
imprime a PDF sin dependencias extra.

Uso:  python tools/a_pdf.py informe/informe.md [--figuras]
"""
from __future__ import annotations

import base64
import re
import subprocess
import sys
from pathlib import Path

import markdown

RAIZ = Path(__file__).resolve().parents[1]
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

CSS = """
@page { size: letter; margin: 1.6cm 1.7cm; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 10pt; line-height: 1.45; color: #1a1a1a; margin: 0;
}
h1 { font-size: 17pt; margin: 0 0 .35em; color: #10243e; letter-spacing: -.01em; }
h2 { font-size: 12.5pt; margin: 1.25em 0 .4em; color: #10243e;
     border-bottom: 1.5px solid #d7dee8; padding-bottom: .18em; }
h3 { font-size: 10.8pt; margin: 1em 0 .3em; color: #24425f; }
p { margin: .45em 0; }
strong { color: #0d1b2a; }
code { font-family: "Cascadia Mono", Consolas, monospace; font-size: 8.8pt;
       background: #eef2f7; padding: .1em .32em; border-radius: 3px; }
pre { background: #f5f7fa; border-left: 3px solid #4a7fb5; padding: .55em .8em;
      border-radius: 3px; overflow-x: auto; }
pre code { background: none; padding: 0; font-size: 8.6pt; }
table { border-collapse: collapse; width: 100%; margin: .6em 0; font-size: 8.9pt;
        page-break-inside: avoid; }
th { background: #10243e; color: #fff; text-align: left; padding: .38em .5em;
     font-weight: 600; }
td { padding: .32em .5em; border-bottom: 1px solid #dde3ea; }
tr:nth-child(even) td { background: #f7f9fb; }
blockquote { margin: .6em 0; padding: .4em .9em; border-left: 3px solid #4a7fb5;
             background: #f2f6fa; color: #2b3d50; }
blockquote p { margin: .2em 0; }
hr { border: none; border-top: 1px solid #d7dee8; margin: 1.1em 0; }
ul, ol { margin: .4em 0; padding-left: 1.4em; }
li { margin: .16em 0; }
img { max-width: 100%; page-break-inside: avoid; }
h1, h2, h3 { page-break-after: avoid; }
"""


def incrustar_figuras(html: str, base: Path) -> str:
    """Mete los PNG como data: URI para que el PDF no dependa de rutas."""
    def sub(m: re.Match) -> str:
        src = m.group(1)
        ruta = (base / src).resolve() if not Path(src).is_absolute() else Path(src)
        if not ruta.exists():
            return m.group(0)
        b64 = base64.b64encode(ruta.read_bytes()).decode()
        return f'src="data:image/png;base64,{b64}"'

    return re.sub(r'src="([^"]+)"', sub, html)


def convertir(md_path: Path) -> Path:
    texto = md_path.read_text(encoding="utf-8")
    cuerpo = markdown.markdown(
        texto, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
    cuerpo = incrustar_figuras(cuerpo, md_path.parent)
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>{CSS}</style></head><body>{cuerpo}</body></html>")

    html_path = md_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")

    pdf_path = md_path.with_suffix(".pdf")
    if not EDGE.exists():
        raise SystemExit(f"no encuentro Edge en {EDGE}; queda el HTML en {html_path}")

    subprocess.run(
        [str(EDGE), "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={pdf_path}", html_path.as_uri()],
        check=True, capture_output=True, timeout=180)
    return pdf_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.is_absolute():
            p = RAIZ / p
        salida = convertir(p)
        kb = salida.stat().st_size / 1024
        print(f"{p.name} -> {salida.relative_to(RAIZ)} ({kb:,.0f} KB)")
