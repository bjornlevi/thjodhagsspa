#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Map the located SPA macro table into an operable DataFrame.

Usage:
  python map_table_to_df.py
"""

import re
import unicodedata
import pdfplumber
import pandas as pd
import difflib

# === Selectors for the known correct table ===
PDF_PATH        = "data/spa_pdf/spa_2025_mars_vor.pdf"
PREFERRED_PAGE  = 11
STRATEGY        = "text_loose"
CANDIDATE_INDEX = 0

TABLE_SETTINGS = {
    "lines_loose": dict(vertical_strategy="lines", horizontal_strategy="lines",
                        snap_tolerance=3, join_tolerance=3, edge_min_length=3),
    "text_loose": dict(vertical_strategy="text", horizontal_strategy="text",
                       text_tolerance=3, snap_tolerance=3, join_tolerance=3),
}

# === Canonical labels ===
CANON = [
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

# --- helpers ---
def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", strip_accents(s).lower()).strip()

CANON_MAP = {norm(k): (k, isl, en) for k, isl, en in CANON}
CANON_KEYS = set(CANON_MAP.keys())

def best_match(label_text: str) -> tuple[str, float]:
    n = norm(label_text)
    n = re.sub(r"[^0-9a-záðéíóúýþæö ]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    # find closest canonical key
    matches = difflib.get_close_matches(n, CANON_KEYS, n=1, cutoff=0.0)
    if matches:
        ratio = difflib.SequenceMatcher(None, n, matches[0]).ratio()
        return matches[0], ratio
    return "", 0.0

def norm(s: str) -> str:
    # strip accents, lower-case, collapse spaces
    s = ''.join(c for c in unicodedata.normalize("NFKD", str(s))
                if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^0-9a-záðéíóúýþæö ]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def match_label(text: str) -> str | None:
    t = norm(text)

    # first: try direct substring match against any of the 3 fields
    for k, isl, en in CANON:
        if norm(k) in t or norm(isl) in t or norm(en) in t:
            return k

    # fallback: fuzzy match against all 3 fields of each label
    best_key, best_score = None, 0.0
    for k, isl, en in CANON:
        for cand in (k, isl, en):
            score = difflib.SequenceMatcher(None, t, norm(cand)).ratio()
            if score > best_score:
                best_score, best_key = score, k

    if best_key and best_score >= 0.35:   # accept if reasonably close
        return best_key

    return None

def parse_num(tok):
    """Parse Icelandic-style numbers.
       '.' = thousands separator
       ',' = decimal separator
       e.g. '4.616' -> 4616.0
            '2,5'   -> 2.5
            '-3,8'  -> -3.8
    """
    if tok is None:
        return None

    s = str(tok).strip()
    s = s.replace("\u2212", "-")                 # normalize minus
    s = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]", "", s)           # drop footnote markers
    s = s.replace(" ", "")                        # remove internal spaces

    # detect decimal comma
    if "," in s:
        # treat '.' as thousands separator
        s = s.replace(".", "")
        # replace ',' with '.' for Python float
        s = s.replace(",", ".")
    else:
        # no decimal comma -> just strip thousands separators
        s = s.replace(".", "")

    # final sanity: keep digits, minus and dot
    s = re.sub(r"[^0-9\.\-]", "", s)
    if s == "" or s == "-" or s == ".":
        return None

    try:
        return float(s)
    except ValueError:
        return None


def find_header_row(tbl):
    best, count = None, -1
    for i, row in enumerate(tbl):
        ys = set(re.findall(r"\b(19\d{2}|20\d{2})\b", " ".join(c or "" for c in row)))
        if len(ys) > count:
            best, count = i, len(ys)
    return best

def map_years(header_row):
    col2year = {}
    for j, cell in enumerate(header_row):
        if not cell:
            continue
        txt = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]", "", str(cell))
        m = re.search(r"\b(19\d{2}|20\d{2})\b", txt)
        if m:
            col2year[j] = int(m.group(1))
    return col2year

# --- main ---
def main():
    with pdfplumber.open(PDF_PATH) as pdf:
        page = pdf.pages[PREFERRED_PAGE-1]
        tables = page.extract_tables(table_settings=TABLE_SETTINGS[STRATEGY]) or []
        table = tables[CANDIDATE_INDEX]

    maxw = max(len(r) for r in table if r)
    tbl = [(r + [None]*(maxw-len(r))) if r and len(r) < maxw else (r or []) for r in table]

    hidx = find_header_row(tbl)
    header = tbl[hidx]
    col2year = map_years(header)
    first_year_col = min(col2year.keys())

    print("Header row (detected):", hidx)
    print("Header row contents:")
    print(header)
    print("Detected year columns:", col2year)
    print("First year column index:", first_year_col)


    rows = []
    for ri in range(hidx + 1, len(tbl)):
        row = tbl[ri]
        label_text = " ".join(c for c in row[:first_year_col] if c and str(c).strip())
        key = match_label(label_text)
        if not key:
            cand, score = best_match(label_text)
            print(f"[unmatched] {repr(label_text)}  →  best={cand}  score={score:.2f}")
            continue
        k, isl, en = CANON_MAP[key]
        for cj, yr in col2year.items():
            val = parse_num(row[cj] if cj < len(row) else None)
            if val is not None:
                rows.append({"label_key": k, "label_is": isl, "label_en": en, "year": yr, "value": val})

    tidy = pd.DataFrame(rows).sort_values(["label_key","year"]).reset_index(drop=True)
    pivot = tidy.pivot_table(index="label_key", columns="year", values="value", aggfunc="last")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
        print("\n=== Pivot (labels × years) ===")
        print(pivot)

if __name__ == "__main__":
    main()
