#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
locate_table.py

Find table candidates in a SPA PDF that contain CANON labels and print
ready-to-copy selectors:
    PREFERRED_PAGE, STRATEGY, CANDIDATE_INDEX

Usage:
  python locate_table.py /path/to/spa_xxxx_*.pdf
  python locate_table.py /path/to/file.pdf --min-hits 8 --max 5 --no-preview
"""

import argparse
import os
import re
import unicodedata
from typing import List, Dict, Optional, Tuple

import pandas as pd
import pdfplumber

# -------- CANON labels (key, Icelandic, English) ----------
CANON: List[Tuple[str, str, str]] = [
    ("einkaneysla", "Einkaneysla", "Private final consumption"),
    ("samneysla", "Samneysla", "Government final consumption"),
    ("fjarmunamyndun", "Fjármunamyndun", "Gross fixed capital formation"),
    ("atvinnuvegafjarfesting", "Atvinnuvegafjárfesting", "Business investment"),
    ("fjarfesting i ibudarhusnaedi", "Fjárfesting í íbúðarhúsnæði", "Housing investment"),
    ("fjarfesting hins opinbera", "Fjárfesting hins opinbera", "Public investment"),
    ("thjodarutgjold alls", "Þjóðarútgjöld alls", "National final expenditure"),
    ("utflutningur voru og thjonustu", "Útflutningur vöru og þjónustu", "Exports of goods and services"),
    ("innflutningur voru og thjonustu", "Innflutningur vöru og þjónustu", "Import of goods and services"),
    ("verg landsframleidsla", "Verg landsframleiðsla", "Gross domestic product"),
    ("voru- og thjonustujofnudur", "Vöru- og þjónustujöfnuður (% af VLF)", "Goods and services balance (% of GDP)"),
    ("vidskiptajofnudur", "Viðskiptajöfnuður (% af VLF)", "Current account balance (% of GDP)"),
    ("vlf a verdlagi hvers ars", "VLF á verðlagi hvers árs, ma. kr.", "Nominal GDP, billion ISK"),
    ("visitala neysluverds", "Vísitala neysluverðs", "Consumer price index"),
    ("gengisvisitala", "Gengisvísitala", "Exchange rate index"),
    ("raungengi", "Raungengi", "Real exchange rate"),
    ("launavisitala m.v. fast verdlag", "Launavísitala m.v. fast verðlag", "Real wage rate index"),
    ("hagvoxtur i helstu vidskiptalondum", "Hagvöxtur í helstu viðskiptalöndum", "GDP growth in main trading partners"),
    ("althjodleg verdbolga", "Alþjóðleg verðbólga", "World CPI inflation"),
    ("verd utflutts als", "Verð útflutts áls", "Export price of aluminum"),
    ("oliuverd", "Olíuverð", "Oil price"),
    ("atvinnuleysi", "Atvinnuleysi (% af vinnuafli)", "Unemployment rate (% of labour force)"),
]

# -------- table extraction strategies to try ----------
TABLE_SETTINGS = [
    ("lines_loose", dict(vertical_strategy="lines", horizontal_strategy="lines",
                         snap_tolerance=3, join_tolerance=3, edge_min_length=3)),
    ("lines_tight", dict(vertical_strategy="lines", horizontal_strategy="lines",
                         snap_tolerance=1, join_tolerance=1, edge_min_length=5)),
    ("text_loose",  dict(vertical_strategy="text",  horizontal_strategy="text",
                         text_tolerance=3, snap_tolerance=3, join_tolerance=3)),
    ("mixed",       dict(vertical_strategy="lines", horizontal_strategy="text",
                         text_tolerance=3, snap_tolerance=2, join_tolerance=2)),
]

# -------- helpers ----------
def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize("NFKD", str(s))
                   if not unicodedata.combining(c))

def norm(s: str) -> str:
    s = strip_accents(s).lower()
    s = re.sub(r"[^0-9a-záðéíóúýþæö ]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

CANON_VARIANTS = [(norm(k), norm(isl), norm(en)) for k, isl, en in CANON]

def cell_matches_any_label(cell_text: str) -> Optional[str]:
    """Return canonical key if cell contains any Icelandic/English/key variant as substring."""
    t = norm(cell_text)
    if not t:
        return None
    for k, isl, en in CANON_VARIANTS:
        if k and k in t or isl and isl in t or en and en in t:
            return k  # return canonical key (normalized)
    return None

def extract_years(strings: List[str]) -> List[int]:
    ys = []
    for s in strings:
        ys += [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", s or "")]
    # dedupe keep order
    seen, out = set(), []
    for y in ys:
        if y not in seen:
            seen.add(y); out.append(y)
    return out

# -------- core ----------
def locate_candidates(pdf_path: str, min_hits: int, max_show: int, show_preview: bool):
    results = []
    base = os.path.basename(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            for name, ts in TABLE_SETTINGS:
                try:
                    tables = page.extract_tables(table_settings=ts) or []
                except Exception:
                    continue

                for i, t in enumerate(tables):
                    # count unique CANON labels across the table
                    matched_keys = set()
                    flat_cells = []
                    for row in t:
                        if not row: continue
                        for cell in row:
                            if cell:
                                flat_cells.append(str(cell))
                                mk = cell_matches_any_label(str(cell))
                                if mk:
                                    matched_keys.add(mk)

                    if len(matched_keys) < min_hits:
                        continue

                    # normalize ragged rows for preview
                    maxw = max((len(r) for r in t if r), default=0)
                    norm_rows = [(r + [None]*(maxw-len(r))) if r and len(r)<maxw else (r or []) for r in t]
                    df_preview = pd.DataFrame(norm_rows[:4])  # first 4 rows for quick look

                    # gather years seen anywhere; show span
                    ys = extract_years(flat_cells)
                    yr_span = f"{min(ys)}–{max(ys)}" if ys else "n/a"

                    results.append({
                        "pdf": base,
                        "page": pno,
                        "strategy": name,
                        "candidate": i,
                        "hits": len(matched_keys),
                        "years_span": yr_span,
                        "shape": (len(t), maxw),
                        "preview": df_preview if show_preview else None,
                        "matched_keys": sorted(list(matched_keys)),
                    })

    # sort: most label hits first, then by page
    results.sort(key=lambda r: (-r["hits"], r["pdf"], r["page"], r["candidate"]))
    if not results:
        print("[!] No candidates found that match the CANON labels with current settings.")
        return

    print(f"[+] Top {min(len(results), max_show)} candidates in {base} (min_hits={min_hits})")
    for r in results[:max_show]:
        print(f"\nPDF: {r['pdf']} | Page: {r['page']} | Strategy: {r['strategy']} | Candidate: {r['candidate']}")
        print(f"  Hits: {r['hits']} | Years: {r['years_span']} | Shape: {r['shape']}")
        print("  Selector:")
        print(f"    PREFERRED_PAGE  = {r['page']}")
        print(f"    STRATEGY        = \"{r['strategy']}\"")
        print(f"    CANDIDATE_INDEX = {r['candidate']}")
        print("  Matched labels (sample):", ", ".join(r["matched_keys"][:10]))
        if r["preview"] is not None:
            with pd.option_context('display.max_columns', None, 'display.width', 180):
                print("  Preview (first rows):")
                print(r["preview"])

def main():
    ap = argparse.ArgumentParser(description="Locate SPA table candidates that contain CANON labels.")
    ap.add_argument("pdf", help="Path to PDF (e.g., data/spa_pdf/spa_2025_juli_sumar.pdf)")
    ap.add_argument("--min-hits", type=int, default=6, help="Minimum number of distinct CANON labels required")
    ap.add_argument("--max", type=int, default=8, help="Max number of candidates to print")
    ap.add_argument("--no-preview", action="store_true", help="Do not print table row previews")
    args = ap.parse_args()

    if not os.path.exists(args.pdf):
        print(f"[!] Not found: {args.pdf}")
        return

    locate_candidates(args.pdf, min_hits=args.min_hits, max_show=args.max, show_preview=not args.no_preview)

if __name__ == "__main__":
    main()
