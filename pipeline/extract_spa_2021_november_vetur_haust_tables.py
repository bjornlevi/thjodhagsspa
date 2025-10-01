#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re, unicodedata, difflib
import pdfplumber, pandas as pd

# ====== CONFIG (this PDF) =====================================================
PDF_PATH        = "data/spa_pdf/spa_2021_november_vetur_haust.pdf"
PREFERRED_PAGE  = 23
STRATEGY        = "text_loose"
CANDIDATE_INDEX = 0
EXPECTED_YEARS  = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]

DEBUG_RAW_GRID  = False  # set True to print the fixed raw grid

TABLE_SETTINGS = {
    "text_loose": dict(vertical_strategy="text", horizontal_strategy="text",
                       text_tolerance=3, snap_tolerance=3, join_tolerance=3),
    "lines_loose": dict(vertical_strategy="lines", horizontal_strategy="lines",
                        snap_tolerance=3, join_tolerance=3, edge_min_length=3),
}

# ====== CANON =================================================================
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

# ====== Helpers ===============================================================
def strip_accents(s): return ''.join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
def norm(s):
    s = strip_accents(s).lower()
    s = re.sub(r"[^0-9a-záðéíóúýþæö %()\-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

CANON_ALL = [(k, norm(k), norm(isl), norm(en)) for k, isl, en in CANON]
def match_label(text):
    t = norm(text)
    if not t: return None
    for k, nk, nisl, nen in CANON_ALL:
        if nk in t or nisl in t or nen in t:
            return k
    # fuzzy (helps with hyphen/split words)
    best_key, best_score = None, 0.0
    for k, nk, nisl, nen in CANON_ALL:
        for cand in (nk, nisl, nen):
            sc = difflib.SequenceMatcher(None, t, cand).ratio()
            if sc > best_score:
                best_score, best_key = sc, k
    return best_key if best_score >= 0.38 else None

def to_rect(table):
    maxw = max(len(r) for r in table if r)
    return [(r + [None]*(maxw - len(r))) if r and len(r) < maxw else (r or []) for r in table]

# ----- CID decode: digit = (cid % 10 - 1) mod 10 ------------------------------
CID_TOKEN = re.compile(r"\(cid:(\d+)\)")
def decode_cid_numbers(cell):
    if cell is None: return None
    s = str(cell)
    def repl(m):
        n = int(m.group(1)); d = (n % 10) - 1
        if d < 0: d = 9
        return str(d)
    s = CID_TOKEN.sub(repl, s)
    s = s.replace("\u2212", "-").replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ----- Icelandic numeric parsing (prefer RIGHTMOST token when multiple) -------
RE_THOUSANDS = re.compile(r"^(-?\d{1,3}(?:\.\d{3})+)(?:,(\d+))?$")
RE_COMMA_DEC = re.compile(r"^(-?\d+),(\d+)$")
RE_SIMPLE    = re.compile(r"^(-?\d+(?:[.,]\d+)?)$")

def _tok_to_float(tok: str):
    tok = tok.strip()
    m = RE_THOUSANDS.match(tok)
    if m:
        left, dec = m.group(1), m.group(2)
        left = left.replace(".", "")
        s = left + ("," + dec if dec else "")
        return float(s.replace(",", "."))
    m = RE_COMMA_DEC.match(tok)
    if m:
        return float(m.group(1) + "." + m.group(2))
    m = RE_SIMPLE.match(tok)
    if m:
        s = m.group(1)
        if "," in s:
            s = s.replace(".", "").replace(",", ".")
        else:
            if not (s.count(".") == 1 and re.fullmatch(r"-?\d{1,3}\.\d{1,3}", s)):
                s = s.replace(".", "")
        try:
            return float(s)
        except ValueError:
            return None
    return None

def parse_cell(cell):
    if cell is None: return None
    s = decode_cid_numbers(cell)
    if not s: return None
    # tokens: allow "3.565", "2,1", "-0,7", "2", etc.
    parts = re.findall(r"-?\d+(?:\.\d{3})*(?:,\d+)?|-\d+|\d+", s)
    for p in reversed(parts):  # pick RIGHTMOST sensible token
        if re.fullmatch(r"\d{4}", p) and 2000 <= int(p) <= 2035:
            continue
        v = _tok_to_float(p)
        if v is not None:
            return v
    return None

def looks_numericish(cell) -> bool:
    if cell is None: return False
    return bool(re.search(r"\d|\(cid:\d+\)", str(cell)))

def best_numeric_run(tbl, run_len):
    W = len(tbl[0]); dens = [0]*W
    for row in tbl:
        for j in range(W):
            if looks_numericish(row[j]): dens[j] += 1
    best_sum, best_start = -1, None
    for start in range(0, W - run_len + 1):
        s = sum(dens[start:start+run_len])
        if s > best_sum: best_sum, best_start = s, start
    return [] if best_start is None else list(range(best_start, best_start + run_len))

# ----- Row collapsing (Icelandic+English lines → one row) ---------------------
def left_text(row, first_year_col):
    return " ".join(c for c in row[:first_year_col] if c and str(c).strip()).strip()

def numeric_count(row, year_cols):
    return sum(1 for cj in year_cols if cj < len(row) and parse_cell(row[cj]) is not None)

def collapse_rows(tbl, year_cols, first_year_col):
    out, i = [], 0
    while i < len(tbl):
        row = tbl[i]; n_this = numeric_count(row, year_cols)
        if n_this <= 1 and i+1 < len(tbl):
            nxt = tbl[i+1]; n_next = numeric_count(nxt, year_cols)
            if n_next >= max(2, int(0.6*len(year_cols))):
                merged = " ".join([t for t in (left_text(row, first_year_col), left_text(nxt, first_year_col)) if t])
                new_row = list(nxt)
                new_row[:first_year_col] = [merged] + [None]*(first_year_col-1)
                out.append(new_row); i += 2; continue
        out.append(row); i += 1
    return out

# ====== PDF-specific column surgery (this table!) =============================
def _concat(a, b):
    sa = "" if a is None else str(a).strip()
    sb = "" if b is None else str(b).strip()
    if not sa: return sb
    if not sb: return sa
    return sa + " " + sb

import re

def _merge_6_7(a, b):
    """
    Smart join for columns 6+7 (1-based).
    Handles: '-' + '8,7' -> '-8,7'; '2.' + '941' -> '2.941'; '-1' + '4,1' -> '-14,1';
             '3' + ',2' -> '3,2'; '12' + '345' -> '12345'
    """
    sa = "" if a is None else str(a).strip()
    sb = "" if b is None else str(b).strip()
    if not sa: return sb
    if not sb: return sa

    # 1) Bare minus on the left
    if re.fullmatch(r"[–—-]", sa):
        return "-" + sb

    # 2) '2.' + '941'  (thousands split)
    if re.fullmatch(r"-?\d+\.", sa) and re.fullmatch(r"\d+", sb):
        return sa + sb

    # 3) '-1' + '4,1'  OR  '3' + ',2'
    if re.fullmatch(r"-?\d+", sa) and re.fullmatch(r"[,\.\d].*", sb):
        return sa + sb

    # 4) '12' + '345'  → '12345' (rare, but seen)
    if re.fullmatch(r"-?\d+", sa) and re.fullmatch(r"\d{3,}", sb):
        return sa + sb

    # default: glue without space so we don't lose signs/digits
    return sa + sb


def _split_merged_9_10_into_three(merged: str):
    """
    After merging cols 9+10, reconstruct THREE values.
    Examples to fix:
      '4,9 3 ,2 2,7' -> ['4,9','3,2','2,7']
      '3.565 3.75 3.932' -> ['3.565','3.75','3.932']
      '-1 ,4 -6,8' -> ['-1,4','-6,8', '']  (if only two present)
    """
    if not merged:
        return ["", "", ""]

    toks = re.split(r"\s+", str(merged).strip())
    out = []
    i = 0
    while i < len(toks):
        t = toks[i]

        # Join bare '-' with next token
        if re.fullmatch(r"[–—-]", t) and i + 1 < len(toks):
            out.append("-" + toks[i + 1])
            i += 2
            continue

        # Join '3' + ',2'  OR  '-1' + ',4'
        if re.fullmatch(r"-?\d+", t) and i + 1 < len(toks) and re.fullmatch(r",[0-9]+", toks[i + 1]):
            out.append(t + toks[i + 1])
            i += 2
            continue

        # Join '2.' + '941'
        if re.fullmatch(r"-?\d+\.", t) and i + 1 < len(toks) and re.fullmatch(r"\d+", toks[i + 1]):
            out.append(t + toks[i + 1])
            i += 2
            continue

        # Otherwise take token as-is
        out.append(t)
        i += 1

    # Normalize to exactly 3 slots
    while len(out) < 3:
        out.append("")
    if len(out) > 3:
        # keep first two, collapse the rest into the last
        out = out[:2] + [" ".join(out[2:])]

    return out


def fix_split_columns_2021_november(tbl):
    """
    PDF-specific repair:
      - merge (6,7) -> col 6 (preserving signs/digits)
      - merge (9,10) -> then split that into THREE numeric columns
    Returns a NEW rect-like table.
    """
    fixed = []
    for row in tbl:
        r = list(row)

        # Step 1: merge col 6+7 (1-based) -> 0-based [5] with [6]
        if len(r) >= 7:
            r[5] = _merge_6_7(r[5], r[6])
            del r[6]

        # Step 2: merge col 9+10 (1-based) -> 0-based [8] with [9], then split into 3
        if len(r) >= 10:
            merged_9_10 = ""
            s8 = "" if r[8] is None else str(r[8]).strip()
            s9 = "" if r[9] is None else str(r[9]).strip()
            merged_9_10 = (s8 + " " + s9).strip() if s8 or s9 else ""
            del r[9]  # drop old col 10 (1-based)

            p1, p2, p3 = _split_merged_9_10_into_three(merged_9_10)
            r[8] = p1
            r.insert(9, p2)
            r.insert(10, p3)

        fixed.append(r)

    return fixed

# ====== Year columns selection (post-fix) =====================================
def pick_year_cols_with_auto_offset(tbl, expected_years, search_offsets=(-2,-1,0,+1,+2)):
    base = best_numeric_run(tbl, run_len=len(expected_years))
    if not base or len(base) != len(expected_years):
        return [], 0, 0
    best = ([], 0, -1)
    for off in search_offsets:
        cols = [c + off for c in base]
        if min(cols) < 0: continue
        parsed = 0
        for row in tbl:
            for cj in cols:
                if cj < len(row) and parse_cell(row[cj]) is not None:
                    parsed += 1
        if parsed > best[2]:
            best = (cols, off, parsed)
    return best  # (cols, offset, coverage)

# ====== MAIN ==================================================================
def main():
    with pdfplumber.open(PDF_PATH) as pdf:
        page = pdf.pages[PREFERRED_PAGE - 1]
        tables = page.extract_tables(table_settings=TABLE_SETTINGS[STRATEGY]) or []
        if not tables or CANDIDATE_INDEX >= len(tables):
            print("[!] Table not found."); return
        raw = tables[CANDIDATE_INDEX]
        

    # 1) Rectify and apply PDF-specific join/split fix
    raw = tables[CANDIDATE_INDEX]
    tbl = to_rect(raw)

    # PDF-specific surgery
    tbl = fix_split_columns_2021_november(tbl)
    tbl = to_rect(tbl)  # re-rectify widths after mutation

    # ...then your year-cols detection, collapse_rows, parse_cell, pivot, etc.

    if DEBUG_RAW_GRID:
        print("\n=== RAW after join/split fix ===")
        for ri, row in enumerate(tbl):
            print(f"{ri:02d}: " + " | ".join(str(c) if c else "" for c in row))

    # 2) Choose year columns (densest block + small offset scan)
    year_cols, YEAR_OFFSET, coverage = pick_year_cols_with_auto_offset(tbl, EXPECTED_YEARS)
    if not year_cols:
        print("[!] Could not lock year columns."); return
    first_year_col = min(year_cols)
    col2year = {c: yr for c, yr in zip(year_cols, EXPECTED_YEARS)}
    print(f"[debug] year cols={year_cols} (offset {YEAR_OFFSET}, coverage={coverage})")

    # 3) Collapse bilingual/split rows *after* year columns are fixed
    tbl2 = collapse_rows(tbl, year_cols, first_year_col)

    # 4) Build tidy rows
    rows, unmatched = [], []
    for row in tbl2:
        label = left_text(row, first_year_col)
        if not label: continue
        k = match_label(label)
        if not k: unmatched.append(label); continue
        if not any(parse_cell(row[c]) is not None for c in year_cols if c < len(row)):
            continue
        for cj in year_cols:
            yr = col2year[cj]
            raw_cell = row[cj] if cj < len(row) else None
            val = parse_cell(raw_cell)
            if val is not None:
                rows.append({
                    "label_key": k,
                    "label_is": KEY_TO_IS[k],
                    "label_en": KEY_TO_EN[k],
                    "year": yr,
                    "value": val,
                })

    if not rows:
        print("[!] No parsed values.")
        if unmatched: print("   Unmatched sample:", unmatched[:10])
        return

    tidy  = pd.DataFrame(rows).sort_values(["label_key","year"]).reset_index(drop=True)
    pivot = tidy.pivot_table(index="label_key", columns="year", values="value", aggfunc="last").sort_index()

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 220):
        print("\n=== Pivot (labels × years) — rounded 1 dp ===")
        print(pivot.round(1))

    expected = {k for k,_,_ in CANON}
    missing  = sorted(expected - set(tidy["label_key"].unique()))
    if missing:
        print("\n[warn] Missing labels:", ", ".join(missing))

if __name__ == "__main__":
    main()
