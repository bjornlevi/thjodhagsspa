#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re, unicodedata, difflib
import pdfplumber, pandas as pd

PDF_PATH        = "data/spa_pdf/spa_2021_mars_vor.pdf"
PREFERRED_PAGE  = 25
STRATEGY        = "text_loose"
CANDIDATE_INDEX = 0

EXPECTED_YEARS  = [2020, 2021, 2022, 2023, 2024, 2025, 2026]

TABLE_SETTINGS = {
    "text_loose": dict(vertical_strategy="text", horizontal_strategy="text",
                       snap_tolerance=3, join_tolerance=3, text_tolerance=3),
}

# ========== CANON ==========
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

KEY_TO_IS = {k: isl for k, isl, en in CANON}
KEY_TO_EN = {k: en  for k, isl, en in CANON}

# ========== HELPERS ==========
def strip_accents(s): 
    return ''.join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))

def norm(s):
    s = strip_accents(s).lower()
    s = re.sub(r"[^0-9a-záðéíóúýþæö %]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

CANON_ALL = [(k, norm(k), norm(isl), norm(en)) for k, isl, en in CANON]

def match_label(txt):
    t = norm(txt)
    if not t: return None
    for k, nk, nisl, nen in CANON_ALL:
        if nk in t or nisl in t or nen in t:
            return k
    # fallback fuzzy
    best_key, best_sc = None, 0
    for k, nk, nisl, nen in CANON_ALL:
        for cand in (nk, nisl, nen):
            sc = difflib.SequenceMatcher(None, t, cand).ratio()
            if sc > best_sc:
                best_sc, best_key = sc, k
    return best_key if best_sc >= 0.35 else None

def parse_num(s):
    if s is None: return None
    s = str(s).strip()
    if not s or s in {"-"}: return None
    s = s.replace(" ","").replace(" ","")
    s = s.replace(".", "").replace(",", ".")  # treat , as decimal
    try:
        return float(s)
    except ValueError:
        return None

def to_rect(table):
    W = max(len(r) for r in table)
    return [r + [None]*(W-len(r)) for r in table]

# ========== MAIN ==========
def main():
    with pdfplumber.open(PDF_PATH) as pdf:
        page = pdf.pages[PREFERRED_PAGE-1]
        tables = page.extract_tables(table_settings=TABLE_SETTINGS[STRATEGY])
        if not tables or CANDIDATE_INDEX >= len(tables):
            print("[!] table not found")
            return
        tbl = to_rect(tables[CANDIDATE_INDEX])

    # ---- Year mapping from header (robust) ----
    YEAR_RX = re.compile(r"20\d{2}")

    def _clean_year_token(s: str):
        # turn '2020¹' or '2020  ' into 2020 (int)
        t = re.sub(r"[^\d]", "", s or "")
        if len(t) >= 4:
            y = int(t[-4:])  # last 4 digits
            if 2000 <= y <= 2035:
                return y
        return None

    def detect_year_cols_from_header(tbl, expected_years):
        """
        Scan the top ~15 rows for a run of 7 year labels.
        Return (year_cols, col2year, header_row_index).
        """
        H = min(15, len(tbl))
        W = len(tbl[0]) if tbl else 0
        target_len = len(expected_years)

        for r in range(H):
            # collect (col_index, year_int) where a year-like token exists
            found = []
            for c in range(W):
                cell = tbl[r][c]
                if not cell:
                    continue
                s = str(cell)
                if YEAR_RX.search(s):
                    y = _clean_year_token(s)
                    if y is not None:
                        found.append((c, y))

            # try to find a contiguous slice of length target_len
            if len(found) >= target_len:
                # build a dense vector of just the year ints per column
                cols = [None] * W
                for c, y in found:
                    cols[c] = y

                # slide a window of width target_len and check if they form a strictly increasing sequence of years
                for start in range(0, W - target_len + 1):
                    window = cols[start:start + target_len]
                    if any(v is None for v in window):
                        continue
                    # e.g., should equal [2020,2021,...] OR [2021..2027]; we’ll just take them as-is
                    if all(window[i] < window[i+1] for i in range(target_len - 1)):
                        year_cols = list(range(start, start + target_len))
                        col2year = {c: y for c, y in zip(year_cols, window)}
                        return year_cols, col2year, r

        # fallback: use the first numeric-heavy column as start and map to expected_years
        dens = [sum(1 for row in tbl if c < len(row) and row[c] and re.search(r"\d", str(row[c]))) for c in range(W)]
        start_col = max(range(W), key=lambda c: dens[c])
        year_cols = list(range(start_col, start_col + len(expected_years)))
        col2year = {c: y for c, y in zip(year_cols, expected_years)}
        return year_cols, col2year, None

    # ---- use it (drop-in) ----
    EXPECTED_YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]

    year_cols, col2year, header_idx = detect_year_cols_from_header(tbl, EXPECTED_YEARS)
    if not year_cols:
        print("[!] Failed to detect year columns"); return
    first_year_col = min(year_cols)

    print("\n[debug] header-based year mapping")
    print("  header row idx:", header_idx if header_idx is not None else "(fallback)")
    print("  year_cols:     ", year_cols)
    print("  year labels:   ", [col2year[c] for c in year_cols])

    # proceed with your existing tidy build, but use 'first_year_col', 'year_cols', 'col2year'
    # e.g.:
    rows, unmatched = [], []
    for r in tbl:
        label_txt = " ".join(x for x in r[:first_year_col] if x and str(x).strip()).strip()
        if not label_txt:
            continue
        k = match_label(label_txt)
        if not k:
            unmatched.append(label_txt); continue
        for c in year_cols:
            raw = r[c] if c < len(r) else None
            v = parse_num(raw) if raw is not None else None
            if v is not None:
                rows.append({
                    "label_key": k,
                    "label_is": KEY_TO_IS[k],
                    "label_en": KEY_TO_EN[k],
                    "year": col2year[c],
                    "value": v
                })


    tidy = pd.DataFrame(rows).sort_values(["label_key", "year"]).reset_index(drop=True)
    pivot = tidy.pivot_table(index="label_key", columns="year", values="value", aggfunc="last")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 180):
        print("\n=== Pivot (rounded) ===")
        print(pivot.round(1))

    if unmatched:
        print("\n[warn] Unmatched labels:")
        for u in unmatched[:10]:
            print("  ", u)


if __name__ == "__main__":
    main()
