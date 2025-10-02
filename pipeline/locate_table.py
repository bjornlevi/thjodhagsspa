#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
locate_table.py

Find SPA table candidates that contain CANON labels and optionally
dump the raw text of the best candidate for further processing.

Usage:
  python pipeline/locate_table.py data/spa_pdf/spa_2020_juni_sumar.pdf
  python pipeline/locate_table.py data/spa_pdf/spa_2020_juni_sumar.pdf --dump-raw
"""

import argparse
import os
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import pdfplumber

# import canonical label definitions
from pipeline.canon import CANON

# -------- table extraction strategies ----------
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
        if (k and k in t) or (isl and isl in t) or (en and en in t):
            return k
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
def locate_candidates(pdf_path: str, min_hits: int, max_show: int, show_preview: bool, dump_raw: bool):
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
                    matched_keys = set()
                    flat_cells = []
                    for row in t:
                        if not row:
                            continue
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
                        "table": t,
                    })

    # sort
    results.sort(key=lambda r: (-r["hits"], r["pdf"], r["page"], r["candidate"]))
    if not results:
        print("[!] No candidates found that match CANON labels.")
        return

    print(f"[+] Top {min(len(results), max_show)} candidates in {base} (min_hits={min_hits})")
    for r in results[:max_show]:
        print(f"\nPDF: {r['pdf']} | Page: {r['page']} | Strategy: {r['strategy']} | Candidate: {r['candidate']}")
        print(f"  Hits: {r['hits']} | Years: {r['years_span']} | Shape: {r['shape']}")
        print("  Selector:")
        print(f"    PREFERRED_PAGE  = {r['page']}")
        print(f"    STRATEGY        = \"{r['strategy']}\"")
        print(f"    CANDIDATE_INDEX = {r['candidate']}")
        print("  Matched labels:", ", ".join(r["matched_keys"]))
        if r["preview"] is not None:
            with pd.option_context('display.max_columns', None, 'display.width', 180):
                print("  Preview (first rows):")
                print(r["preview"])

    if dump_raw:
        best = results[0]
        base_name = Path(pdf_path).stem
        raw_dir = Path("data/extracted/raw"); raw_dir.mkdir(parents=True, exist_ok=True)
        out_file = raw_dir / f"{base_name}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            for row in best["table"]:
                f.write(" | ".join(str(c) if c else "" for c in row) + "\n")
        print(f"\n[+] Raw dump of best candidate saved → {out_file}")

def main():
    ap = argparse.ArgumentParser(description="Locate SPA table candidates containing CANON labels.")
    ap.add_argument("pdf", help="Path to SPA PDF (e.g., data/spa_pdf/spa_2020_juni_sumar.pdf)")
    ap.add_argument("--min-hits", type=int, default=6, help="Minimum distinct CANON labels required")
    ap.add_argument("--max", type=int, default=5, help="Max number of candidates to print")
    ap.add_argument("--no-preview", action="store_true", help="Do not print table previews")
    ap.add_argument("--dump-raw", action="store_true", help="Dump best candidate table to data/extracted/raw/")
    args = ap.parse_args()

    if not os.path.exists(args.pdf):
        print(f"[!] Not found: {args.pdf}")
        return

    locate_candidates(args.pdf, args.min_hits, args.max, not args.no_preview, args.dump_raw)

if __name__ == "__main__":
    main()
