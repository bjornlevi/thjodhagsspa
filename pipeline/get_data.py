# pipeline/extract_tafla1.py
import re
import argparse
import unicodedata
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pandas as pd

# ---------------- helpers ----------------
def fold_is(text: str) -> str:
    if text is None:
        return ""
    t = (text
         .replace("þ", "th").replace("Þ", "th")
         .replace("ð", "d").replace("Ð", "d")
         .replace("æ", "ae").replace("Æ", "ae")
         .replace("ö", "o").replace("Ö", "o"))
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

def norm_text(s: Optional[str]) -> str:
    if s is None: return ""
    s = re.sub(r"-\s*\n\s*", "", str(s))  # un-hyphenate across line breaks
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def parse_val(x):
    if x is None: return None
    s = str(x).strip()
    if s in {"", "-"}: return None
    s = s.replace("\u2212", "-").replace("\u00a0", " ")
    s = re.sub(r"[.\s]", "", s)   # thousands
    s = s.replace(",", ".")       # decimal
    try: return float(s)
    except Exception: return None

# --- clean year tokens like '20201', '2020¹', '2020 1' -> 2020
def clean_year_token(cell: str) -> Optional[int]:
    if not cell: return None
    c = norm_text(cell)
    # common footnote formats: 20201, 2020¹, 2020 1, 2020* etc.
    m = re.match(r"^(19|20)\d{2}(?:\D?\d)?$", c)
    if m:
        year = int(c[:4])
        return year
    # plain year anywhere in cell
    m2 = re.search(r"\b(19|20)\d{2}\b", c)
    if m2:
        return int(m2.group(0))
    return None

# ---------------- canonical indicators ----------------
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
    ("atvinnuleysi", "Atvinnuleysi (% af vinnuafli)", "Unemployment rate (% of labour force)"),
    ("kaupmattur launa", "Kaupmáttur launa", "Real wage rate index"),
    ("hagvoxtur i helstu vidskiptalondum", "Hagvöxtur í helstu viðskiptalöndum", "GDP growth in main trading partners"),
    ("althjodleg verdbolga", "Alþjóðleg verðbólga", "World CPI inflation"),
    ("verd utflutts als", "Verð útflutts áls", "Export price of aluminum"),
    ("oliuverd", "Olíuverð", "Oil price"),
]
TARGET_ORDER = [c[1] for c in CANON]
CANON_KEYS = [c[0] for c in CANON]
KEY_SET = set(CANON_KEYS)

def canon_indicator(label_is: str) -> Tuple[Optional[str], Optional[str]]:
    f = fold_is(label_is)
    for key, canon_is, canon_en in CANON:
        if key in f:  # contains-match (handles IS+EN label in same cell)
            return canon_is, canon_en
    return None, None

# ---------------- header/year inference ----------------
def infer_year_map(rows: List[List[str]]) -> Tuple[List[int], List[int]]:
    """
    Scan first ~10 rows: collect distinct (year -> first column index seen),
    preserving left-to-right order. If none found, fall back to rightmost 4–5 numeric cols.
    """
    year_to_idx: Dict[int, int] = {}
    max_scan = min(10, len(rows))
    for i in range(max_scan):
        row = rows[i]
        for j, cell in enumerate(row):
            y = clean_year_token(cell)
            if isinstance(y, int) and y not in year_to_idx:
                year_to_idx[y] = j

    if year_to_idx:
        years = sorted(year_to_idx.keys())
        # keep the last 4 if more found
        years = years[-4:] if len(years) > 4 else years
        idxs = [year_to_idx[y] for y in years]
        return years, idxs

    # Fallback: choose the rightmost numeric-looking columns
    # Use the row with most numeric cells
    best_cols = []
    for r in rows[:max_scan]:
        cols = [j for j,c in enumerate(r) if re.search(r"[0-9]", norm_text(c))]
        if len(cols) > len(best_cols):
            best_cols = cols
    if best_cols:
        cols = best_cols[-5:] if len(best_cols) >= 5 else best_cols[-4:] if len(best_cols) >= 4 else best_cols[-3:] if len(best_cols) >= 3 else best_cols[-2:]
        return [None]*len(cols), cols
    return [], []

def best_four_years(years: List[int], idxs: List[int]) -> Tuple[List[int], List[int]]:
    if len(years) >= 4:
        return years[-4:], idxs[-4:]
    return years, idxs

# ---------------- table extraction ----------------
def extract_tables(pdf_path: Path) -> List[List[List[str]]]:
    import pdfplumber
    out = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        strategies = [
            {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
            {"vertical_strategy": "text",  "horizontal_strategy": "text"},
            {"vertical_strategy": "lines", "horizontal_strategy": "text"},
            {"vertical_strategy": "text",  "horizontal_strategy": "lines"},
        ]
        for page in pdf.pages:
            for s in strategies:
                try:
                    for t in page.extract_tables(table_settings=s) or []:
                        if not t or len(t) < 3:  # too small
                            continue
                        tbl = [[norm_text(c) for c in row] for row in t]
                        w = max(len(r) for r in tbl)
                        tbl = [r + [""]*(w-len(r)) for r in tbl]
                        out.append(tbl)
                except Exception:
                    continue
            try:
                t = page.extract_table()
                if t and len(t) >= 3:
                    tbl = [[norm_text(c) for c in row] for row in t]
                    w = max(len(r) for r in tbl)
                    tbl = [r + [""]*(w-len(r)) for r in tbl]
                    out.append(tbl)
            except Exception:
                pass
    return out

def score_table(tbl: List[List[str]]) -> Tuple[int,int,int,int]:
    if not tbl: return (0,0,0,0)
    labels = []
    for r in tbl:
        left = " ".join(r[:2]) if len(r) >= 2 else (r[0] if r else "")
        labels.append(fold_is(left))
    hits = sum(any(key in lab for key in KEY_SET) for lab in labels)
    years = sum(1 for c in tbl[0] if clean_year_token(c) is not None)
    nrows = len(tbl)
    ncols = max(len(r) for r in tbl)
    return (hits, nrows, ncols, years)

def melt_table(tbl: List[List[str]], pdf_name: str) -> List[dict]:
    years, idxs = infer_year_map(tbl[:10])
    years, idxs = best_four_years(years, idxs)

    out_rows = []
    for r in tbl:
        label = " ".join(r[:2]) if len(r) >= 2 else (r[0] if r else "")
        ci, ce = canon_indicator(label)
        if not ci:
            continue
        unit = ""
        m = re.search(r"\(([^)]+)\)", label)
        if m: unit = m.group(1)
        for y, j in zip(years, idxs):
            val = parse_val(r[j] if j < len(r) else None)
            out_rows.append({
                "year": y,
                "indicator_is": ci,
                "indicator_en": ce,
                "value": val,
                "unit": unit,
                "table_id": "Tafla 1",
                "source_pdf_file": pdf_name,
                "notes": "",
            })

    # Stabilize: ensure every indicator appears once per detected year, even if missing
    detected_years = sorted({yy for yy in years if yy is not None})
    if not detected_years and idxs:
        detected_years = [None] * len(idxs)

    have = {(r["indicator_is"], r["year"]) for r in out_rows}
    for _, ci, ce in CANON:
        for y in detected_years:
            if (ci, y) not in have:
                out_rows.append({
                    "year": y,
                    "indicator_is": ci,
                    "indicator_en": ce,
                    "value": None,
                    "unit": "",
                    "table_id": "Tafla 1",
                    "source_pdf_file": pdf_name,
                    "notes": "filled-missing",
                })

    # Deduplicate preferring non-null values
    ded: Dict[Tuple[str, Optional[int]], dict] = {}
    for r in out_rows:
        k = (r["indicator_is"], r["year"])
        if k not in ded or (ded[k]["value"] is None and r["value"] is not None):
            ded[k] = r
    out = list(ded.values())
    out.sort(key=lambda r: (TARGET_ORDER.index(r["indicator_is"]) if r["indicator_is"] in TARGET_ORDER else 999, r["year"] or 0))
    return out

def extract_one(pdf_path: Path) -> List[dict]:
    tables = extract_tables(pdf_path)
    if not tables:
        return []
    best = max(tables, key=score_table)
    return melt_table(best, pdf_path.name)

def main(in_dir: Path, out_csv: Path):
    all_rows = []
    for p in sorted(in_dir.glob("*.pdf")):
        rows = extract_one(p)
        if rows:
            print(f"✓ {p.name}: {len(rows)} rows (indicators x years incl. missing)")
            all_rows.extend(rows)
        else:
            print(f"! {p.name}: Tafla 1 not found")
            all_rows.append({
                "year": None, "indicator_is": "", "indicator_en": "", "value": None, "unit": "",
                "table_id": "Tafla 1", "source_pdf_file": p.name, "notes": "No Tafla 1 detected"
            })
    df = pd.DataFrame(all_rows, columns=[
        "year","indicator_is","indicator_en","value","unit","table_id","source_pdf_file","notes"
    ])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved → {out_csv} ({len(df)} rows)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/spa_pdf", help="Folder with PDFs")
    ap.add_argument("--out-csv", default="data/derived/thjodhagsspa_tafla1.csv")
    args = ap.parse_args()
    main(Path(args.in_dir), Path(args.out_csv))
