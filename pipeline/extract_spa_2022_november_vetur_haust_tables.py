#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re, unicodedata, difflib
import pdfplumber, pandas as pd

# === SELECTORS (this PDF) ===
PDF_PATH        = "data/spa_pdf/spa_2022_november_vetur_haust.pdf"
PREFERRED_PAGE  = 15
STRATEGY        = "text_loose"
CANDIDATE_INDEX = 0

EXPECTED_YEARS = list(range(2021, 2029))  # 2021..2028

TABLE_SETTINGS = {
    "lines_loose": dict(vertical_strategy="lines", horizontal_strategy="lines",
                        snap_tolerance=3, join_tolerance=3, edge_min_length=3),
    "text_loose": dict(vertical_strategy="text", horizontal_strategy="text",
                       text_tolerance=3, snap_tolerance=3, join_tolerance=3),
}

# === CANON ===
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

# === Helpers ===
def strip_accents(s): return ''.join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
def norm(s):
    s = strip_accents(s).lower()
    s = re.sub(r"[^0-9a-záðéíóúýþæö %]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

CANON_ALL = [(k, norm(k), norm(isl), norm(en)) for k, isl, en in CANON]
def match_label(text):
    t = norm(text)
    if not t: return None
    for k, nk, nisl, nen in CANON_ALL:
        if nk in t or nisl in t or nen in t:
            return k
    # fuzzy fallback
    best_key, best_score = None, 0.0
    for k, nk, nisl, nen in CANON_ALL:
        for cand in (nk, nisl, nen):
            sc = difflib.SequenceMatcher(None, t, cand).ratio()
            if sc > best_score:
                best_score, best_key = sc, k
    return best_key if best_score >= 0.35 else None

def to_rect(table):
    maxw = max(len(r) for r in table if r)
    return [(r + [None]*(maxw - len(r))) if r and len(r) < maxw else (r or []) for r in table]

# === CID decode (rule: digit = (cid % 10 - 1) mod 10) ===
CID_TOKEN = re.compile(r"\(cid:(\d+)\)")
def decode_cid_numbers(cell):
    """Decode '(cid:61x)' digits using the observed +1 offset; preserve commas and minus."""
    if cell is None:
        return None
    s = str(cell)
    def repl(m):
        n = int(m.group(1))
        d = (n % 10) - 1
        if d < 0: d = 9
        return str(d)
    s = CID_TOKEN.sub(repl, s)
    s = s.replace("\u2212", "-").replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# === parse (Icelandic) ===
def parse_num(tok):
    """',' = decimal; '.' = thousands unless small decimal pattern."""
    if tok is None: return None
    s = str(tok).strip()
    if not s: return None
    s = s.replace("\u2212", "-")
    s = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]", "", s)
    s = s.replace(" ", "")
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "." in s:
        if s.count(".") == 1 and re.fullmatch(r"-?\d{1,3}\.\d{1,3}", s):
            pass
        else:
            s = s.replace(".", "")
    s = re.sub(r"[^0-9.\-]", "", s)
    if s in {"", "-", "."}: return None
    try:
        return float(s)
    except ValueError:
        return None

# === choose densest 8-col numeric block ===
def looks_numericish(cell) -> bool:
    if cell is None: return False
    s = str(cell)
    return bool(re.search(r"\d|\(cid:\d+\)", s))

def best_numeric_run(tbl, run_len):
    W = len(tbl[0])
    dens = [0]*W
    for row in tbl:
        for j in range(W):
            if looks_numericish(row[j]):
                dens[j] += 1
    best_sum, best_start = -1, None
    for start in range(0, W - run_len + 1):
        s = sum(dens[start:start+run_len])
        if s > best_sum:
            best_sum, best_start = s, start
    if best_start is None: return []
    return list(range(best_start, best_start + run_len))

# === main ===
def main():
    with pdfplumber.open(PDF_PATH) as pdf:
        page = pdf.pages[PREFERRED_PAGE - 1]
        tables = page.extract_tables(table_settings=TABLE_SETTINGS[STRATEGY]) or []
        if not tables or CANDIDATE_INDEX >= len(tables):
            print("[!] Table not found with given selectors."); return
        raw = tables[CANDIDATE_INDEX]

    tbl = to_rect(raw)

    # 1) year columns 2021..2028
    year_cols = best_numeric_run(tbl, run_len=len(EXPECTED_YEARS))
    if len(year_cols) != len(EXPECTED_YEARS):
        print("[!] Could not lock a contiguous 8-column numeric block.")
        print("densest block:", year_cols)
        return
    col2year = {c: yr for c, yr in zip(sorted(year_cols), EXPECTED_YEARS)}
    first_year_col = min(year_cols)

    # 2) build tidy
    rows, unmatched = [], []
    for row in tbl:
        label_text = " ".join(c for c in row[:first_year_col] if c and str(c).strip()).strip()
        if not label_text:
            continue
        k = match_label(label_text)
        if not k:
            unmatched.append(label_text)
            continue

        # must have at least one numeric after decoding/parsing
        if not any(parse_num(decode_cid_numbers(row[c])) is not None for c in year_cols if c < len(row)):
            continue

        for cj in year_cols:
            yr = col2year[cj]
            raw = row[cj]
            dec = decode_cid_numbers(raw)
            val = parse_num(dec)
            if val is not None:
                rows.append({
                    "label_key": k,
                    "label_is": KEY_TO_IS[k],
                    "label_en": KEY_TO_EN[k],
                    "year": yr,
                    "value": val,
                })

    if not rows:
        print("[!] No data rows matched.")
        if unmatched:
            print("   Unmatched labels (sample):", unmatched[:10])
        return

    tidy = pd.DataFrame(rows).sort_values(["label_key","year"]).reset_index(drop=True)
    pivot = tidy.pivot_table(index="label_key", columns="year", values="value", aggfunc="last").sort_index()

    # 3) print rounded (1 dp) pivot
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 220):
        print("\n=== Pivot (labels × years) — rounded 1 dp ===")
        print(pivot.round(1))

    # 4) debug if a key is missing
    target = "atvinnuvegafjarfesting"
    if target not in tidy["label_key"].unique():
        print(f"\n[debug] '{target}' missing — dumping raw/decoded/parsed candidates containing 'atvinnu':")
        for ri, row in enumerate(tbl):
            label_text = " ".join(c for c in row[:first_year_col] if c and str(c).strip()).strip()
            if not label_text:
                continue
            if "atvinnu" in strip_accents(label_text).lower():
                print(f"\n[raw label guess @ row {ri}]: {label_text}")
                for cj in year_cols:
                    yr = col2year[cj]
                    raw = row[cj] if cj < len(row) else None
                    dec = decode_cid_numbers(raw)
                    print(f"  {yr}: raw={repr(raw)} dec={repr(dec)} parsed={parse_num(dec)}")

    # warn on missing labels overall
    expected = {k for k,_,_ in CANON}
    missing = sorted(expected - set(tidy["label_key"].unique()))
    if missing:
        print("\n[warn] Missing labels:", ", ".join(missing))

if __name__ == "__main__":
    main()
