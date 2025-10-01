#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, unicodedata, difflib
import pdfplumber
import pandas as pd

# -------- SELECTORS (use locate_table.py output) --------
PDF_PATH        = "data/spa_pdf/spa_2022_november_vetur_haust.pdf"
PREFERRED_PAGE  = 15
STRATEGY        = "text_loose"
CANDIDATE_INDEX = 0

TABLE_SETTINGS = {
    "lines_loose": dict(vertical_strategy="lines", horizontal_strategy="lines",
                        snap_tolerance=3, join_tolerance=3, edge_min_length=3),
    "text_loose": dict(vertical_strategy="text", horizontal_strategy="text",
                       text_tolerance=3, snap_tolerance=3, join_tolerance=3),
}

# -------- CANON (keep yours) --------
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

# -------- helpers you already have (kept concise) --------
def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))

def norm(s: str) -> str:
    s = strip_accents(s).lower()
    s = re.sub(r"[^0-9a-záðéíóúýþæö %]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

CANON_MAP = {norm(k): (k, isl, en) for k, isl, en in CANON}
CANON_ALL = [(k, norm(k), norm(isl), norm(en)) for k, isl, en in CANON]

def match_label(text: str) -> str | None:
    t = norm(text)
    if not t: return None
    for k, nk, nisl, nen in CANON_ALL:
        if nk in t or nisl in t or nen in t:
            return k
    best_key, best_score = None, 0.0
    for k, nk, nisl, nen in CANON_ALL:
        for cand in (nk, nisl, nen):
            sc = difflib.SequenceMatcher(None, t, cand).ratio()
            if sc > best_score:
                best_score, best_key = sc, k
    return best_key if best_score >= 0.35 else None

def parse_num(tok):
    """PDF-specific parser (for spa_2022_november_vetur_haust.pdf).

    Rules:
      - If ',' present: treat as decimal; strip any '.' thousands → '1.234,56' -> 1234.56
      - If only '.' present:
          keep as decimal iff it looks like -?\d{1,3}\.\d{1,3}
          otherwise treat '.' as thousands (strip) → '4.616' -> 4616
      - Else: plain integer
    """
    if tok is None:
        return None
    s = str(tok).strip()
    if not s:
        return None

    # normalize minus + drop footnote superscripts + internal spaces
    s = s.replace("\u2212", "-")
    s = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]", "", s)
    s = s.replace(" ", "")

    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "." in s:
        if s.count(".") == 1 and re.fullmatch(r"-?\d{1,3}\.\d{1,3}", s):
            pass  # keep dot as decimal (e.g., -6.116, 2.5, 12.3)
        else:
            s = s.replace(".", "")  # treat as thousands
    # else: digits only

    s = re.sub(r"[^0-9.\-]", "", s)
    if s in {"", "-", "."}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def to_rect(table):
    maxw = max(len(r) for r in table if r)
    return [(r + [None]*(maxw - len(r))) if r and len(r) < maxw else (r or []) for r in table]

def forecast_year_from_filename(path: str):
    m = re.search(r"spa[_\-]?(\d{4})", os.path.basename(path))
    return int(m.group(1)) if m else None

# -------- robust year detection (multi-pass) --------
def header_band_years(tbl, band=10):
    """Scan top/bottom bands for years per column (tolerates split digits)."""
    col2year = {}
    W = len(tbl[0])
    def scan_rows(rows):
        for i in rows:
            row = tbl[i]
            for j in range(W):
                cell = row[j]
                if not cell: continue
                # look in a 3-col window to tolerate split tokens
                win = []
                for jj in range(max(0,j-1), min(W, j+2)):
                    if tbl[i][jj]:
                        win.append(str(tbl[i][jj]))
                txt = " ".join(win)
                txt = re.sub(r"(?<=\d)\s+(?=\d)", "", txt)  # join split digits
                m = re.search(r"\b(19\d{2}|20\d{2})\b", txt)
                if m:
                    col2year[j] = int(m.group(1))
    scan_rows(range(0, min(band, len(tbl))))
    scan_rows(range(max(0, len(tbl)-band), len(tbl)))
    return col2year

def numeric_density_cols(tbl):
    W = len(tbl[0])
    dens = [0]*W
    for i in range(len(tbl)):
        for j in range(W):
            if parse_num(tbl[i][j]) is not None:
                dens[j] += 1
    return dens

def longest_contiguous(indices):
    if not indices: return []
    best, cur = [], [indices[0]]
    for a,b in zip(indices, indices[1:]):
        if b == a+1:
            cur.append(b)
        else:
            if len(cur) > len(best): best = cur
            cur = [b]
    if len(cur) > len(best): best = cur
    return best

def infer_years(tbl, filename_year=None):
    W = len(tbl[0])

    # Pass A: numeric density
    dens = numeric_density_cols(tbl)
    # progressively lower thresholds
    for thr in (8, 6, 4, 3, 2):
        candidate = [j for j,c in enumerate(dens) if c >= thr]
        run = longest_contiguous(candidate)
        if len(run) >= 4:
            year_cols = run
            break
    else:
        year_cols = []

    # Pass B: header band scan for explicit years
    col2year = header_band_years(tbl)
    # If we have year_cols but missing labels, try to align using nearest known year col
    if year_cols:
        # fill gaps using explicit years if any
        explicit = sorted((c,y) for c,y in col2year.items() if c in year_cols)
        if explicit:
            # anchor on first explicit and fill sequentially
            cols_sorted = sorted(year_cols)
            first_c, first_y = explicit[0]
            base_idx = cols_sorted.index(first_c)
            # left
            y = first_y
            for k in range(base_idx-1, -1, -1):
                y -= 1
                col2year[cols_sorted[k]] = y
            # right
            y = first_y
            for k in range(base_idx+1, len(cols_sorted)):
                y += 1
                col2year[cols_sorted[k]] = y
        else:
            # no explicit: try anchoring by filename year (if present) at the last column (common)
            if filename_year is None:
                filename_year = forecast_year_from_filename(PDF_PATH)
            if filename_year is not None:
                for idx,k in enumerate(sorted(year_cols)):
                    # assume increasing left->right; try to place filename year at the last col
                    offset = len(year_cols) - 1 - idx
                    col2year[k] = filename_year - offset

    # Pass C: if we still have no year_cols, use the most-numeric row to define them
    if not year_cols:
        best_row, best_idxs = None, []
        for i,row in enumerate(tbl):
            idxs = [j for j,v in enumerate(row) if parse_num(v) is not None]
            if len(idxs) > len(best_idxs):
                best_row, best_idxs = i, idxs
        run = longest_contiguous(best_idxs)
        if len(run) >= 4:
            year_cols = run
            if not col2year:
                # try header years again; else filename anchor
                col2year = header_band_years(tbl)
            if not col2year:
                fy = filename_year if filename_year is not None else forecast_year_from_filename(PDF_PATH)
                if fy is not None:
                    cols_sorted = sorted(year_cols)
                    for idx,k in enumerate(cols_sorted):
                        # place filename year at the last (forecast) column
                        offset = len(cols_sorted) - 1 - idx
                        col2year[k] = fy - offset

    # Final sanity
    if not year_cols or not col2year:
        return None, [], {}

    # ensure all year_cols have a year via sequential fill
    cols_sorted = sorted(year_cols)
    # find any anchor
    anchors = [c for c in cols_sorted if c in col2year]
    if anchors:
        base_c = anchors[0]
        base_y = col2year[base_c]
        # left
        y = base_y
        for k in range(cols_sorted.index(base_c)-1, -1, -1):
            y -= 1
            col2year[cols_sorted[k]] = y
        # right
        y = base_y
        for k in range(cols_sorted.index(base_c)+1, len(cols_sorted)):
            y += 1
            col2year[cols_sorted[k]] = y
    first_year_col = min(year_cols)
    return first_year_col, year_cols, col2year

# -------- main build --------
def main():
    with pdfplumber.open(PDF_PATH) as pdf:
        page = pdf.pages[PREFERRED_PAGE - 1]
        tables = page.extract_tables(table_settings=TABLE_SETTINGS[STRATEGY]) or []
        if not tables or CANDIDATE_INDEX >= len(tables):
            print("[!] Table not found with given selectors."); return
        raw = tables[CANDIDATE_INDEX]

    tbl = to_rect(raw)

    # robust year inference
    fyc, year_cols, col2year = infer_years(tbl, filename_year=forecast_year_from_filename(PDF_PATH))
    if not year_cols or not col2year:
        print("[!] Could not infer year columns. Debug:")
        print("   first_year_col:", fyc)
        print("   year_cols:", year_cols)
        print("   col2year:", col2year)
        return

    # build tidy
    rows, unmatched = [], []
    for ri, row in enumerate(tbl):
        # require at least one numeric value in year columns
        if not any(parse_num(row[c]) is not None for c in year_cols if c < len(row)):
            continue
        # label is left of first year col
        label_text = " ".join(c for c in row[:fyc] if c and str(c).strip()).strip()
        if not label_text:
            continue
        k = match_label(label_text)
        if not k:
            unmatched.append(label_text)
            continue
        key, isl, en = CANON_MAP[norm(k)]
        for cj in year_cols:
            yr = col2year.get(cj)
            if yr is None: continue
            val = parse_num(row[cj])
            if val is not None:
                rows.append({"label_key": key, "label_is": isl, "label_en": en, "year": yr, "value": val})

    if not rows:
        print("[!] No data rows matched after year inference.")
        print("   year_cols:", year_cols)
        print("   col2year:", col2year)
        if unmatched:
            print("   Unmatched labels (sample):", unmatched[:12])
        return

    tidy = pd.DataFrame(rows).sort_values(["label_key","year"]).reset_index(drop=True)
    pivot = tidy.pivot_table(index="label_key", columns="year", values="value", aggfunc="last").sort_index()

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 220):
        print("\n=== Pivot (labels × years) ===")
        print(pivot)

    expected = {k for k,_,_ in CANON}
    missing = sorted(expected - set(tidy["label_key"].unique()))
    if missing:
        print("\n[warn] Missing labels:", ", ".join(missing))

if __name__ == "__main__":
    main()
