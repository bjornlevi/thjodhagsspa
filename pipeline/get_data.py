# pipeline/extract_spa_targets.py
# -*- coding: utf-8 -*-
import re
import argparse
import unicodedata
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pandas as pd

# =========================
# Helpers & Normalization
# =========================
def norm_text(s: Optional[str]) -> str:
    """Collapse whitespace, normalize hyphens, and un-hyphenate line breaks."""
    if s is None:
        return ""
    s = str(s)
    # remove soft hyphen
    s = s.replace("\u00AD", "")
    # unify all dash/hyphen-like characters to a single space
    s = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212\-]+", " ", s)
    # un-hyphenate across line breaks (after dash normalization)
    s = re.sub(r"\s*\n\s*", " ", s)
    # collapse spaces
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def fold_is(text: str) -> str:
    """Accent/ligature fold + dash normalization for robust matching."""
    if text is None:
        return ""
    t = text.replace("\u00AD", "")
    t = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212\-]+", " ", t)  # all hyphens → space
    t = (t.replace("þ","th").replace("Þ","th")
           .replace("ð","d").replace("Ð","d")
           .replace("æ","ae").replace("Æ","ae")
           .replace("ö","o").replace("Ö","o"))
    t = unicodedata.normalize("NFKD", t).encode("ascii","ignore").decode("ascii")
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


# parse single number (Icelandic)
def parse_icelandic_number(cell: Optional[str]) -> Optional[float]:
    if cell is None: return None
    s = str(cell).strip()
    if s in {"", "-", "."}: return None
    s = s.replace("\u2212","-").replace("\u00a0"," ")
    s = re.sub(r"[.\s]","", s)  # drop thousands & spaces
    s = s.replace(",",".")      # decimal comma
    try:
        return float(s)
    except Exception:
        return None

# tokenization to split concatenated Icelandic numbers
_IS_NUM_RE = re.compile(r"[−-]?\d{1,3}(?:\.\d{3})*(?:,\d+)?")

def find_all_icelandic_numbers(cell: Optional[str]) -> list:
    if cell is None:
        return []
    return _IS_NUM_RE.findall(norm_text(cell))

def parse_icelandic_token(tok: str) -> Optional[float]:
    s = tok.replace("\u2212","-").replace("\u00a0"," ")
    s = s.replace(".","").replace(",",".")
    try:
        return float(s)
    except Exception:
        return None

def clean_year_token(cell: str) -> Optional[int]:
    c = norm_text(cell)
    if re.match(r"^(?:19|20)\d{2}(?:\D?\d)?$", c):
        return int(c[:4])
    m = re.search(r"\b(?:19|20)\d{2}\b", c)
    return int(m.group(0)) if m else None

def extract_target_year_from_filename(name: str) -> Optional[int]:
    base = Path(name).name
    years = re.findall(r"(?:19|20)\d{2}", base)
    return int(years[0]) if years else None

# =========================
# Canonical indicators (IS → EN)
# =========================
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
    ("kaupmattur launa", "Launavísitala m.v. fast verðlag", "Real wage rate index"),
    ("hagvoxtur i helstu vidskiptalondum", "Hagvöxtur í helstu viðskiptalöndum", "GDP growth in main trading partners"),
    ("althjodleg verdbolga", "Alþjóðleg verðbólga", "World CPI inflation"),
    ("verd utflutts als", "Verð útflutts áls", "Export price of aluminum"),
    ("oliuverd", "Olíuverð", "Oil price"),
]
TARGET_ORDER = [c[1] for c in CANON]
KEY_SET = {c[0] for c in CANON}

def canon_indicator(label: str) -> Tuple[Optional[str], Optional[str]]:
    f = fold_is(label)
    for key, is_name, en_name in CANON:
        if key in f:
            return is_name, en_name
    return None, None

# =========================
# Year/column inference
# =========================
def infer_year_map(rows: List[List[str]]) -> Tuple[List[Optional[int]], List[int]]:
    year_to_idx: Dict[int,int] = {}
    scan = min(50, len(rows))
    for i in range(scan):
        for j, cell in enumerate(rows[i]):
            y = clean_year_token(cell)
            if isinstance(y,int) and y not in year_to_idx:
                year_to_idx[y] = j
    if year_to_idx:
        years = sorted(year_to_idx.keys())
        idxs = [year_to_idx[y] for y in years]
        return years, idxs

    # fallback: numeric density per column across table
    ncols = max((len(r) for r in rows), default=0)
    scores = []
    for j in range(ncols):
        nums = 0
        for r in rows:
            c = r[j] if j < len(r) else ""
            if re.search(r"[0-9]", norm_text(c)): nums += 1
        scores.append((nums, j))
    scores.sort(key=lambda t: (t[0], t[1]))
    cols = [j for c,j in scores if c > 0]
    if not cols: return [], []
    cols = sorted(cols)
    return [None]*len(cols), cols

def backfill_year_labels(rows: List[List[str]], years: List[Optional[int]], idxs: List[int]) -> List[Optional[int]]:
    out = list(years)
    for k,(y,j) in enumerate(zip(years,idxs)):
        if y is not None: continue
        for i in range(min(30, len(rows))):
            cell = rows[i][j] if j < len(rows[i]) else ""
            cy = clean_year_token(cell)
            if cy is not None:
                out[k] = cy
                break
    return out

# =========================
# Table extraction (scan ALL; score by labels, “Tafla N”, year coverage, percentiness)
# =========================
def extract_tables(pdf_path: Path):
    import pdfplumber
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for pi, page in enumerate(pdf.pages):
            tables = []
            strategies = [
                {"vertical_strategy":"lines","horizontal_strategy":"lines"},
                {"vertical_strategy":"text","horizontal_strategy":"text"},
                {"vertical_strategy":"lines","horizontal_strategy":"text"},
                {"vertical_strategy":"text","horizontal_strategy":"lines"},
            ]
            for s in strategies:
                try:
                    for t in page.extract_tables(table_settings=s) or []:
                        if not t or len(t) < 3: continue
                        tbl = [[norm_text(c) for c in row] for row in t]
                        w = max(len(r) for r in tbl)
                        tbl = [r + [""]*(w-len(r)) for r in tbl]
                        tables.append(tbl)
                except Exception:
                    continue
            # single fallback
            try:
                t = page.extract_table()
                if t and len(t) >= 3:
                    tbl = [[norm_text(c) for c in row] for row in t]
                    w = max(len(r) for r in tbl)
                    tbl = [r + [""]*(w-len(r)) for r in tbl]
                    tables.append(tbl)
            except Exception:
                pass
            pages.append((pi, tables))
    return pages

def table_label_hits(tbl: List[List[str]]) -> Tuple[int, set]:
    present = set()
    for r in tbl:
        left = " ".join(r[:3]) if len(r) >= 3 else " ".join(r)
        for key, is_name, en_name in CANON:
            if key in fold_is(left):
                present.add(is_name)
    return len(present), present

def table_caption_number(tbl: List[List[str]]) -> int:
    """Read a caption like 'Tafla 4.' if extraction included it as the first row."""
    if not tbl: return -1
    top = " ".join(tbl[0])
    m = re.search(r"\bTafla\s+(\d+)\b", top, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return -1
    return -1

def table_percentiness(tbl: List[List[str]], years: List[Optional[int]], idxs: List[int]) -> float:
    if not idxs: return 0.0
    total, perc = 0, 0
    for r in tbl:
        label = " ".join(r[:3]) if len(r) >= 3 else " ".join(r)
        is_name, _ = canon_indicator(label)
        is_nominal_gdp = (is_name == "VLF á verðlagi hvers árs, ma. kr.")
        for j in idxs:
            if j >= len(r): continue
            raw = r[j]
            if raw is None or norm_text(raw) in {"","-"}: continue
            total += 1
            if is_nominal_gdp:
                continue
            raw_nospace = norm_text(raw)
            looks_percent_token = bool(re.fullmatch(r"[−-]?\d{1,3}(?:,\d+)?", raw_nospace)) and "." not in raw_nospace
            val = parse_icelandic_number(raw)
            reasonable_mag = (val is not None and abs(val) <= 100.0)
            if looks_percent_token or reasonable_mag:
                perc += 1
    return 0.0 if total == 0 else perc/total

def score_table(tbl: List[List[str]], target_year: Optional[int]) -> Tuple[float, dict]:
    hits, present = table_label_hits(tbl)
    years, idxs = infer_year_map(tbl[:30])
    years = backfill_year_labels(tbl, years, idxs)
    years_detected = [y for y in years if isinstance(y,int)]
    yr_coverage = 0
    includes_target = 0
    if years_detected:
        yr_coverage = len(set(years_detected))
        if target_year in years_detected:
            includes_target = 1
    perciness = table_percentiness(tbl, years, idxs)
    tafla_no = table_caption_number(tbl)

    # weighted score:
    #   labels (x200), includes target year (x100), percentiness (x20),
    #   year coverage (x5), tafla number (x1)
    score = hits*200 + includes_target*100 + perciness*20 + yr_coverage*5 + max(tafla_no,0)*1
    meta = {"hits":hits, "includes_target":includes_target, "perciness":perciness,
            "yr_coverage":yr_coverage, "tafla_no":tafla_no, "years":years, "idxs":idxs}
    return score, meta

def choose_best_table(pages, target_year: Optional[int]):
    best = None
    best_meta = None
    best_score = None
    for _, tables in pages:
        for tbl in tables:
            s, meta = score_table(tbl, target_year)
            if best_score is None or s > best_score:
                best, best_meta, best_score = tbl, meta, s
    return best, best_meta

# =========================
# Year windowing (make sure target year is present)
# =========================
def select_year_window(years: List[Optional[int]], idxs: List[int],
                       target_year: Optional[int], max_cols: int = 8):
    """
    Keep a window (<= max_cols) that includes the target year if present.
    If all concrete years fit in <= max_cols, keep the FULL span (e.g. 2009–2015).
    """
    pairs = list(zip(years, idxs))
    pairs.sort(key=lambda p: (9999 if p[0] is None else p[0], p[1]))
    ys = [p[0] for p in pairs]
    js = [p[1] for p in pairs]

    concrete = [y for y in ys if isinstance(y, int)]
    if not concrete:
        # no year labels at all; just keep the last up to max_cols columns
        if len(ys) <= max_cols:
            return ys, js
        return ys[-max_cols:], js[-max_cols:]

    y_min, y_max = min(concrete), max(concrete)
    span_len = len({y for y in concrete})

    # If everything fits, keep the full span (e.g., 2009..2015)
    if span_len <= max_cols:
        # select exactly those columns whose year lies in [y_min, y_max]
        sel = [(y, j) for y, j in zip(ys, js) if (isinstance(y, int) and y_min <= y <= y_max)]
        ys_sel, js_sel = [y for y, _ in sel], [j for _, j in sel]
        return ys_sel, js_sel

    # Otherwise, build a target-centered window
    if target_year is not None and target_year in concrete:
        tpos = next(i for i, y in enumerate(ys) if y == target_year)
        half = max_cols // 2
        start = max(0, tpos - half)
        end = min(len(ys), start + max_cols)
        start = max(0, end - max_cols)  # re-align if at the end
        return ys[start:end], js[start:end]

    # Fallback: most recent max_cols
    return ys[-max_cols:], js[-max_cols:]

# =========================
# Melting + labelling
# =========================
def status_for_year(table_year: Optional[int], target_year: Optional[int]) -> str:
    if table_year is None or target_year is None:
        return "unknown"
    if table_year < target_year: return "historical"
    if table_year == target_year: return "prediction"
    return "extrapolation"

def melt_table(tbl: List[List[str]], pdf_name: str, target_year: Optional[int],
               years_all: List[Optional[int]], idxs_all: List[int]) -> List[dict]:
    years, idxs = select_year_window(years_all, idxs_all, target_year, max_cols=7)

    out = []
    for r in tbl:
        label = " ".join(r[:3]) if len(r) >= 3 else " ".join(r)
        is_name, en_name = canon_indicator(label)
        if not is_name: 
            continue
        unit = ""
        m = re.search(r"\(([^)]+)\)", label)
        if m: unit = m.group(1)

        value_type = "percent" if is_name != "VLF á verðlagi hvers árs, ma. kr." else "level_billion_isk"

        # gather raw cells
        raw_cells = [(j, (r[j] if j < len(r) else "")) for j in idxs]
        non_empty = [(j, c) for j, c in raw_cells if norm_text(c) not in {"","-"}]

        # concatenation split: one cell holds several values
        if len(non_empty) == 1:
            j_only, cell = non_empty[0]
            toks = find_all_icelandic_numbers(cell)
            vals = [parse_icelandic_token(t) for t in toks]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 2:
                # align counts with number of columns we selected
                if len(vals) > len(idxs):
                    vals = vals[-len(idxs):]   # keep most recent-looking
                elif len(vals) < len(idxs):
                    # right-align shorter sequences
                    idxs_use = idxs[-len(vals):]
                    years_use = years[-len(vals):]
                else:
                    idxs_use = idxs
                    years_use = years

                if len(vals) < len(idxs):
                    # (computed above)
                    pass
                else:
                    idxs_use = idxs
                    years_use = years

                for y, j, v in zip(years_use, idxs_use, vals):
                    out.append({
                        "source_pdf_file": pdf_name,
                        "target_year": target_year,
                        "table_year": y,
                        "status": status_for_year(y, target_year),
                        "indicator_is": is_name,
                        "indicator_en": en_name,
                        "unit": unit,
                        "value_type": value_type,
                        "value": v
                    })
                continue

        # normal per-cell parse (also handles cells with multiple tokens → take first)
        for y, j in zip(years, idxs):
            raw = r[j] if j < len(r) else None
            toks = find_all_icelandic_numbers(raw)
            v = parse_icelandic_token(toks[0]) if toks else parse_icelandic_number(raw)
            out.append({
                "source_pdf_file": pdf_name,
                "target_year": target_year,
                "table_year": y,
                "status": status_for_year(y, target_year),
                "indicator_is": is_name,
                "indicator_en": en_name,
                "unit": unit,
                "value_type": value_type,
                "value": v
            })

    # Ensure stable shape for every detected year
    detected_years = [yy for yy in years if isinstance(yy,int)]
    have = {(x["indicator_is"], x["table_year"]) for x in out}
    for _, is_name, en_name in CANON:
        for y in detected_years:
            if (is_name, y) not in have:
                out.append({
                    "source_pdf_file": pdf_name,
                    "target_year": target_year,
                    "table_year": y,
                    "status": status_for_year(y, target_year),
                    "indicator_is": is_name,
                    "indicator_en": en_name,
                    "unit": "",
                    "value_type": "percent" if is_name != "VLF á verðlagi hvers árs, ma. kr." else "level_billion_isk",
                    "value": None
                })

    # dedupe preferring non-null
    ded: Dict[Tuple[str, Optional[int]], dict] = {}
    for r in out:
        k = (r["indicator_is"], r["table_year"])
        if k not in ded or (ded[k]["value"] is None and r["value"] is not None):
            ded[k] = r
    res = list(ded.values())
    res.sort(key=lambda r: (TARGET_ORDER.index(r["indicator_is"]) if r["indicator_is"] in TARGET_ORDER else 999,
                            r["table_year"] if r["table_year"] is not None else 0))
    return res

# =========================
# Driver
# =========================
def extract_one(pdf_path: Path) -> List[dict]:
    target_year = extract_target_year_from_filename(pdf_path.name)
    pages = extract_tables(pdf_path)
    best_tbl, meta = choose_best_table(pages, target_year)

    if not best_tbl:
        return []

    # Use the table's full year/column mapping, THEN choose a target-centered window
    years_all, idxs_all = infer_year_map(best_tbl[:30])
    years_all = backfill_year_labels(best_tbl, years_all, idxs_all)

    return melt_table(best_tbl, pdf_path.name, target_year, years_all, idxs_all)

def main(in_dir: Path, out_csv: Path):
    all_rows = []
    for p in sorted(in_dir.glob("*.pdf")):
        try:
            rows = extract_one(p)
            if rows:
                print(f"✓ {p.name}: {len(rows)} rows (best-match table; target-centered years)")
                all_rows.extend(rows)
            else:
                print(f"! {p.name}: no suitable table found")
                all_rows.append({
                    "source_pdf_file": p.name,
                    "target_year": extract_target_year_from_filename(p.name),
                    "table_year": None,
                    "status": "unknown",
                    "indicator_is": "",
                    "indicator_en": "",
                    "unit": "",
                    "value_type": "",
                    "value": None
                })
        except Exception as e:
            print(f"✗ {p.name}: {e}")
            all_rows.append({
                "source_pdf_file": p.name,
                "target_year": extract_target_year_from_filename(p.name),
                "table_year": None,
                "status": "error",
                "indicator_is": "",
                "indicator_en": "",
                "unit": "",
                "value_type": "",
                "value": None
            })

    df = pd.DataFrame(all_rows, columns=[
        "source_pdf_file","target_year","table_year","status",
        "indicator_is","indicator_en","unit","value_type","value"
    ])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved → {out_csv} ({len(df)} rows)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/spa_pdf", help="Folder with PDFs")
    ap.add_argument("--out-csv", default="data/derived/thjodhagsspa_targets.csv")
    args = ap.parse_args()
    main(Path(args.in_dir), Path(args.out_csv))
