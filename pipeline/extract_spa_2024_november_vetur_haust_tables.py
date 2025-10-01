#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re, unicodedata, difflib, os
import pdfplumber, pandas as pd

# ---- SELECTORS (from locate_table.py) ----
PDF_PATH        = "data/spa_pdf/spa_2024_november_vetur_haust.pdf"
PREFERRED_PAGE  = 6
STRATEGY        = "text_loose"
CANDIDATE_INDEX = 0

TABLE_SETTINGS = {
    "lines_loose": dict(vertical_strategy="lines", horizontal_strategy="lines",
                        snap_tolerance=3, join_tolerance=3, edge_min_length=3),
    "text_loose": dict(vertical_strategy="text", horizontal_strategy="text",
                       text_tolerance=3, snap_tolerance=3, join_tolerance=3),
}

# ---- CANON ----
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

# -------- helpers --------
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
    # fuzzy fallback for OCR damage
    best_key, best_score = None, 0.0
    for k, nk, nisl, nen in CANON_ALL:
        for cand in (nk, nisl, nen):
            sc = difflib.SequenceMatcher(None, t, cand).ratio()
            if sc > best_score:
                best_score, best_key = sc, k
    return best_key if best_score >= 0.35 else None

def parse_num(tok):
    """Icelandic number parsing: '.' thousands, ',' decimal."""
    if tok is None: return None
    s = str(tok).strip().replace("\u2212", "-")
    s = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]", "", s)
    s = s.replace(" ", "")
    if "," in s:
        s = s.replace(".", "")
        s = s.replace(",", ".")
    else:
        s = s.replace(".", "")
    s = re.sub(r"[^0-9.\-]", "", s)
    if s in {"", "-", "."}: return None
    try:
        return float(s)
    except ValueError:
        return None

def to_rect(table):
    maxw = max(len(r) for r in table if r)
    return [(r + [None]*(maxw - len(r))) if r and len(r) < maxw else (r or []) for r in table]

def forecast_year_from_filename(path: str) -> int | None:
    m = re.search(r"spa[_\-]?(\d{4})", os.path.basename(path))
    return int(m.group(1)) if m else None

# ---- HARDENED YEAR INFERENCE (data-row driven with header fallback) ----
def infer_year_block_and_map(tbl, min_nums_in_row=4, min_cols=5, max_cols=12):
    """
    1) Find data rows: rows having >= min_nums_in_row numeric cells.
    2) First-year column = mode of the minimum numeric index among those rows.
    3) Expand right to include a contiguous block where most of those rows have numbers.
    4) Map each column to a year by scanning nearby header/footer for 4-digit numbers;
       if missing, infer sequentially anchored on forecast year from filename.
    """
    # 1) rows with many numeric cells
    num_idx_per_row = []
    for i, row in enumerate(tbl):
        idxs = [j for j, v in enumerate(row) if parse_num(v) is not None]
        if len(idxs) >= min_nums_in_row:
            num_idx_per_row.append((i, idxs))
    if not num_idx_per_row:
        return None, [], {}

    # 2) choose first_year_col as the most common min index
    from collections import Counter
    mins = [idxs[0] for _, idxs in num_idx_per_row]
    first_year_col = Counter(mins).most_common(1)[0][0]

    # 3) expand to the right: keep columns where at least half the data rows have numbers
    candidate_cols = range(first_year_col, len(tbl[0]))
    year_cols = []
    for j in candidate_cols:
        hits = sum(1 for _, idxs in num_idx_per_row if j in idxs)
        if hits >= max(2, len(num_idx_per_row)//2):
            year_cols.append(j)
        elif year_cols:
            # stop at first gap after we started collecting
            break

    # enforce sensible bounds
    if len(year_cols) < min_cols:
        # try being more permissive: include next few columns even if some rows are NaN
        extra = []
        for j in range((year_cols[-1] + 1) if year_cols else first_year_col, min(len(tbl[0]), first_year_col + max_cols)):
            extra.append(j)
        year_cols = sorted(set(year_cols + extra))
    else:
        year_cols = year_cols[:max_cols]

    # 4) map columns -> years by scanning around top and bottom for 4-digit tokens
    col2year = {}
    for j in year_cols:
        texts = []
        for i in range(0, min(10, len(tbl))):
            if tbl[i][j]: texts.append(str(tbl[i][j]))
        for i in range(max(0, len(tbl)-8), len(tbl)):
            if tbl[i][j]: texts.append(str(tbl[i][j]))
        txt = " ".join(texts)
        txt = re.sub(r"(?<=\d)\s+(?=\d)", "", txt)  # join split digits: '2 025' -> '2025'
        m = re.search(r"\b(19\d{2}|20\d{2})\b", txt)
        if m: col2year[j] = int(m.group(1))

    # If not all columns mapped, infer sequentially using the first mapped year or the filename year
    if len(col2year) != len(year_cols):
        anchor_year = next(iter(sorted(col2year.items(), key=lambda x: x[0])), (None, None))[1]
        if anchor_year is None:
            anchor_year = forecast_year_from_filename(PDF_PATH)
        if anchor_year is not None:
            # assume left-to-right increasing by 1
            start_idx = year_cols.index(next((c for c in year_cols if col2year.get(c, None) is not None), year_cols[0]))
            # fill left
            y = anchor_year
            for k in range(start_idx-1, -1, -1):
                y -= 1
                col2year[year_cols[k]] = y
            # reset and fill right
            y = anchor_year
            for k in range(start_idx+1, len(year_cols)):
                y += 1
                col2year[year_cols[k]] = y

    return first_year_col, year_cols, col2year

# ---- main ----
def main():
    with pdfplumber.open(PDF_PATH) as pdf:
        page = pdf.pages[PREFERRED_PAGE - 1]
        tables = page.extract_tables(table_settings=TABLE_SETTINGS[STRATEGY]) or []
        if not tables or CANDIDATE_INDEX >= len(tables):
            print("[!] Table not found with given selectors."); return
        raw = tables[CANDIDATE_INDEX]

    tbl = to_rect(raw)

    fyc, year_cols, col2year = infer_year_block_and_map(tbl)
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
        # label is left of first year column
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
        if unmatched:
            print("   Unmatched labels (sample):", unmatched[:10])
        print("   year_cols:", year_cols)
        print("   col2year:", col2year)
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
