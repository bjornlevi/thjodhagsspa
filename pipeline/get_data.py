# pipeline/extract_spa_with_recipes.py
# -*- coding: utf-8 -*-
import argparse
import re, yaml, sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import pdfplumber

# ---------------------------
# Canonical indicators (IS -> EN)
# ---------------------------
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
]
TARGET_ORDER = [c[1] for c in CANON]

# ---------------------------
# Normalization & parsing
# ---------------------------
def norm_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u00AD", "")  # soft hyphen
    s = re.sub(r"[\u2010-\u2015\u2212\-]+", " ", s)  # unify hyphens/dashes
    s = re.sub(r"[\u00B9\u00B2\u00B3\u2070-\u2079]", "", s)  # strip superscripts ¹²³…
    s = re.sub(r"\s*\n\s*", " ", s)  # newlines -> space
    s = re.sub(r"\s+", " ", s)  # collapse
    return s.strip()

def fold_is(text: str) -> str:
    if text is None: return ""
    t = norm_text(text)
    t = (t.replace("þ","th").replace("Þ","th")
           .replace("ð","d").replace("Ð","d")
           .replace("æ","ae").replace("Æ","ae")
           .replace("ö","o").replace("Ö","o"))
    t = unicodedata.normalize("NFKD", t).encode("ascii","ignore").decode("ascii")
    return t.lower()

def canon_indicator(label: str) -> Tuple[Optional[str], Optional[str]]:
    f = fold_is(label)
    for key, is_name, en_name in CANON:
        if key in f:
            return is_name, en_name
    return None, None

_NUM_RE = re.compile(r"[−-]?\d{1,3}(?:\.\d{3})*(?:,\d+)?")

def find_all_icelandic_numbers(cell: Optional[str]) -> List[str]:
    if cell is None: return []
    return _NUM_RE.findall(norm_text(cell))

def parse_icelandic_token(tok: str) -> Optional[float]:
    s = tok.replace("\u2212","-")
    s = s.replace(".","").replace(",",".")
    try:
        return float(s)
    except Exception:
        return None

def parse_icelandic_number(cell: Optional[str]) -> Optional[float]:
    toks = find_all_icelandic_numbers(cell)
    if not toks: return None
    return parse_icelandic_token(toks[0])

def clean_year_token(cell: str) -> Optional[int]:
    c = norm_text(cell)
    m = re.search(r"\b((?:19|20)\d{2})\b", c)
    return int(m.group(1)) if m else None

def extract_target_year_from_filename(name: str) -> Optional[int]:
    years = re.findall(r"(?:19|20)\d{2}", Path(name).name)
    return int(years[0]) if years else None

# ---------------------------
# Year header detection
# ---------------------------
def detect_year_row(rows: List[List[str]], max_scan=60) -> Optional[int]:
    best_i, best_cnt = None, 0
    for i in range(min(max_scan, len(rows))):
        ys = set()
        for cell in rows[i]:
            y = clean_year_token(cell)
            if y: ys.add(y)
        if len(ys) > best_cnt:
            best_cnt = len(ys); best_i = i
    return best_i if best_cnt >= 3 else None

def infer_year_map(rows: List[List[str]]) -> Tuple[List[Optional[int]], List[int]]:
    hdr = detect_year_row(rows, max_scan=60)
    year_to_idx: Dict[int,int] = {}
    if hdr is not None:
        for j, cell in enumerate(rows[hdr]):
            y = clean_year_token(cell)
            if y is not None and y not in year_to_idx:
                year_to_idx[y] = j
    if len(year_to_idx) < 3:
        for i in range(min(60, len(rows))):
            for j, cell in enumerate(rows[i]):
                y = clean_year_token(cell)
                if y is not None and y not in year_to_idx:
                    year_to_idx[y] = j
    if year_to_idx:
        years = sorted(year_to_idx.keys())
        idxs = [year_to_idx[y] for y in years]
        return years, idxs
    # fallback: numeric columns
    ncols = max((len(r) for r in rows), default=0)
    cols = []
    for j in range(ncols):
        if any(re.search(r"\d", norm_text(r[j] if j < len(r) else "")) for r in rows):
            cols.append(j)
    return [None]*len(cols), cols

def backfill_year_labels(rows: List[List[str]], years: List[Optional[int]], idxs: List[int]) -> List[Optional[int]]:
    out = list(years)
    for k,(y,j) in enumerate(zip(years, idxs)):
        if y is not None: continue
        for i in range(min(60, len(rows))):
            cy = clean_year_token(rows[i][j] if j < len(rows[i]) else "")
            if cy:
                out[k] = cy; break
    return out

# ---------------------------
# Table scoring
# ---------------------------
def table_label_hits(tbl: List[List[str]]) -> Tuple[int, set, bool]:
    present = set()
    has_gdp = False
    for r in tbl:
        left = " ".join((r + ["","","",""])[:4])  # use first 4 cells
        is_name, _ = canon_indicator(left)
        if is_name:
            present.add(is_name)
            if is_name == "Verg landsframleiðsla":
                has_gdp = True
    return len(present), present, has_gdp

def table_percentiness(tbl: List[List[str]], years: List[Optional[int]], idxs: List[int]) -> float:
    if not idxs: return 0.0
    total = perc = 0
    for r in tbl:
        left = " ".join((r + ["","","",""])[:4])
        is_name, _ = canon_indicator(left)
        is_nominal = (is_name == "VLF á verðlagi hvers árs, ma. kr.")
        for j in idxs:
            if j >= len(r): continue
            raw = r[j]
            if raw is None or norm_text(raw) in {"","-"}: continue
            total += 1
            if not is_nominal:
                m = parse_icelandic_number(raw)
                if m is not None and abs(m) <= 100:  # plausible % change
                    perc += 1
    return 0 if total == 0 else perc/total

def score_table_generic(tbl: List[List[str]], target_year: Optional[int]) -> float:
    hits, _, has_gdp = table_label_hits(tbl)
    years, idxs = infer_year_map(tbl[:60])
    years = backfill_year_labels(tbl, years, idxs)
    ys = [y for y in years if isinstance(y,int)]
    includes_target = int(target_year in ys) if target_year else 0
    span = len(set(ys))
    perc = table_percentiness(tbl, years, idxs)
    if not has_gdp or hits < 10 or span < 4:
        return -1_000_000 + hits
    return hits*300 + includes_target*150 + perc*30 + span*10

# ---------------------------
# Recipes
# ---------------------------
def load_recipes(path: str = None):
    try:
        import yaml
    except ImportError:
        print("[recipes] PyYAML not installed; running with 0 recipes.")
        return []

    # search order
    candidates = []
    if path: candidates.append(Path(path))
    here = Path(__file__).resolve().parent
    candidates += [
        here / "recipes.yaml",
        here.parent / "pipeline" / "recipes.yaml",
        Path.cwd() / "pipeline" / "recipes.yaml",
        Path.cwd() / "recipes.yaml",
    ]

    for p in candidates:
        if p.is_file():
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                recs = data.get("recipes", [])
                for r in recs:
                    m = (r.get("match") or "").strip()
                    r["_is_fallback"] = (m == "*" or m == "")
                    if "name" not in r:
                        r["name"] = m if m else "(unnamed)"
                print(f"[recipes] loaded {len(recs)} recipe(s) from {p}")
                return recs
            except Exception as e:
                print(f"[recipes] failed to load {p}: {e}", file=sys.stderr)
                break
    print("[recipes] no recipes file found; running with 0 recipes.")
    return []

RECIPES = load_recipes()

def find_recipe_for(filename: str):
    """
    Return the matching recipe dict or None.
    Preference order:
      - explicit matches (substring/stem/exact)
      - fallback recipe where match == "*" (if present)
    """
    base = Path(filename).name
    stem = Path(filename).stem

    explicit = None
    fallback = None
    for r in RECIPES:
        pat = (r.get("match") or "").strip()
        if not pat:
            continue
        if pat == "*" or r.get("_is_fallback"):
            fallback = fallback or r
            continue
        if pat == base or pat == stem or pat in base or pat in stem:
            explicit = r
            break

    return explicit or fallback

# ---------------------------
# Forecast metadata
# ---------------------------
_MONTHS = {
    "jan":1,"feb":2,"februar":2,"mars":3,"apr":4,"april":4,
    "mai":5,"maí":5,"jun":6,"juni":6,"júní":6,"juli":7,"júlí":7,
    "aug":8,"sep":9,"sept":9,"okt":10,"oktober":10,"nov":11,"nóv":11,"nóvember":11,
}
def fold_ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii").lower()

def forecast_meta_from_filename(fname: str) -> Dict[str, Any]:
    b = fold_ascii(Path(fname).stem)
    month = None
    for k,m in _MONTHS.items():
        if re.search(rf"\b{k}\b", b):
            month = m; break
    season = None
    if re.search(r"\b(vetur|haust)\b", b): season = "vetur/haust"
    elif re.search(r"\b(sumar|vor)\b", b): season = "sumar/vor"
    if season is None and month is not None:
        season = "vetur/haust" if month in (10,11,12,1,2) else "sumar/vor"
    ty = extract_target_year_from_filename(fname)
    fid = f"{ty or 'NA'}_{month or 'xx'}"
    return {"forecast_id": fid, "forecast_month": month, "forecast_season": season}

def status_for_year(table_year: Optional[int], target_year: Optional[int]) -> str:
    if table_year is None or target_year is None: return "unknown"
    if table_year < target_year: return "historical"
    if table_year == target_year: return "prediction"
    return "extrapolation"

# ---------------------------
# Table extraction (per page, multi strategy)
# ---------------------------
def extract_tables_from_pdf(pdf: pdfplumber.PDF):
    tables_by_page: Dict[int, List[List[List[str]]]] = {}
    text_by_page: Dict[int, str] = {}
    for pi, page in enumerate(pdf.pages, start=1):  # 1-based
        text_by_page[pi] = norm_text(page.extract_text() or "")
        page_tables: List[List[List[str]]] = []

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
                    page_tables.append(tbl)
            except Exception:
                pass

        # single generic
        try:
            t = page.extract_table()
            if t and len(t) >= 3:
                tbl = [[norm_text(c) for c in row] for row in t]
                w = max(len(r) for r in tbl)
                tbl = [r + [""]*(w-len(r)) for r in tbl]
                page_tables.append(tbl)
        except Exception:
            pass

        tables_by_page[pi] = page_tables
    return tables_by_page, text_by_page

# ---------------------------
# Recipe-aware table selection
# ---------------------------
def select_table_with_recipe(filename: str, tables_by_page, text_by_page, target_year: Optional[int]):
    """
    Returns: (tbl, years, idxs, used_recipe:boolean, recipe_name or None)
    used_recipe is True only for an explicit per-PDF recipe (not for catch-all "*").
    """
    recipe = find_recipe_for(filename)
    used_recipe = False
    recipe_name = recipe.get("name") if isinstance(recipe, dict) else None

    def recipe_score(page_no: int, tbl):
        if not recipe:
            return None
        years, idxs = infer_year_map(tbl[:60])
        years = backfill_year_labels(tbl, years, idxs)
        hits, present, has_gdp = table_label_hits(tbl)
        score = 0

        cap = recipe.get("caption_regex")
        if cap and re.search(cap, text_by_page.get(page_no, ""), flags=re.IGNORECASE):
            score += 300
        ph = set(recipe.get("page_hints", []))
        if ph and page_no in ph:
            score += 150
        req = set(recipe.get("required_years", []))
        if req:
            have = set(y for y in years if isinstance(y, int))
            score += 40 * len(have & req)
        score += 15 * hits
        if has_gdp:
            score += 50
        if recipe.get("prefer_percenty"):
            score += int(table_percentiness(tbl, years, idxs) * 50)
        if recipe.get("min_label_hits", 0) and hits < recipe["min_label_hits"]:
            return -10_000
        return score

    # 1) Try recipe-guided selection
    if isinstance(recipe, dict):
        candidates = []
        pages = recipe.get("page_hints") or sorted(tables_by_page.keys())
        for p in pages:
            for tbl in tables_by_page.get(p, []):
                s = recipe_score(p, tbl)
                if s is not None:
                    candidates.append((s, p, tbl))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_s, best_p, best_tbl = candidates[0]
            if best_s > -10_000:
                years, idxs = infer_year_map(best_tbl[:60])
                years = backfill_year_labels(best_tbl, years, idxs)
                # mark "used" only if the recipe is not a fallback ("*")
                used_recipe = not recipe.get("_is_fallback", False)
                return best_tbl, years, idxs, used_recipe, recipe_name

    # 2) Fallback to generic scoring
    best = None
    for p in sorted(tables_by_page.keys()):
        for tbl in tables_by_page[p]:
            s = score_table_generic(tbl, target_year)
            if best is None or s > best[0]:
                best = (s, p, tbl)
    if not best:
        return None
    _, p, tbl = best
    years, idxs = infer_year_map(tbl[:60])
    years = backfill_year_labels(tbl, years, idxs)
    return tbl, years, idxs, False, None

# ---------------------------
# Melting (robust per recipe-selected table)
# ---------------------------
def stitch_and_harvest(cells: List[str], idxs: List[int]) -> List[Optional[float]]:
    # collect tokens per col
    col_toks = []
    for j in idxs:
        raw = cells[j] if j < len(cells) else ""
        toks = [t for t in find_all_icelandic_numbers(raw) if t.strip() != ""]
        col_toks.append(toks)
    all_vals = [parse_icelandic_token(t) for toks in col_toks for t in toks]
    all_vals = [v for v in all_vals if v is not None]
    n = len(idxs)
    # distribute if needed
    if len(all_vals) >= 2 and len(all_vals) != n:
        vals = list(all_vals)
        if len(vals) > n:
            vals = vals[-n:]
        else:
            vals = [None]*(n-len(vals)) + vals
        return vals
    # first token per col + neighbor repair
    vals = [parse_icelandic_token(t[0]) if t else None for t in col_toks]
    for k in range(n):
        if vals[k] is None:
            if k-1 >=0 and len(col_toks[k-1])==1:
                v = parse_icelandic_token(col_toks[k-1][0]); 
                if v is not None: vals[k]=v
            if vals[k] is None and k+1<n and len(col_toks[k+1])==1:
                v = parse_icelandic_token(col_toks[k+1][0]); 
                if v is not None: vals[k]=v
    return vals

def melt_table(tbl: List[List[str]], filename: str, target_year: Optional[int],
               years_all: List[Optional[int]], idxs_all: List[int],
               stitch_continuations: int = 2) -> List[dict]:
    # Full year span, but cap at 10 cols
    pairs = sorted(zip(years_all, idxs_all), key=lambda p: (9999 if p[0] is None else p[0], p[1]))
    ys = [p[0] for p in pairs]
    js = [p[1] for p in pairs]
    concrete = [y for y in ys if isinstance(y,int)]
    if concrete and len(set(concrete)) <= 10:
        want = set(range(min(concrete), max(concrete)+1))
        sel = [(y,j) for y,j in zip(ys,js) if (isinstance(y,int) and y in want)]
        years, idxs = [y for y,_ in sel], [j for _,j in sel]
    else:
        years, idxs = ys[-10:], js[-10:]

    meta = forecast_meta_from_filename(filename)

    out = []
    T = [[norm_text(c) for c in row] for row in tbl]
    i = 0
    current_is = current_en = None
    current_unit = ""

    def is_numeric_only(r: List[str]) -> bool:
        left4 = " ".join((r + ["","","",""])[:4])
        has_lab, _ = canon_indicator(left4)
        has_num = any(re.search(r"\d", c) for c in r)
        return (not has_lab) and has_num

    while i < len(T):
        row = T[i]
        left4 = " ".join((row + ["","","",""])[:4])
        ind_is, ind_en = canon_indicator(left4)
        if ind_is:
            current_is, current_en = ind_is, ind_en
            m = re.search(r"\(([^)]+)\)", left4)
            current_unit = m.group(1) if m else ""
            # stitch with next K numeric-only rows
            rows = [row]
            for k in range(1, stitch_continuations+1):
                if i+k < len(T) and is_numeric_only(T[i+k]): rows.append(T[i+k])
                else: break
            # merge columns by concatenating strings (to recover split numbers)
            width = max(len(r) for r in rows)
            stitched = []
            for col in range(width):
                parts = [r[col] if col < len(r) else "" for r in rows]
                merged = " ".join([p for p in parts if p]).strip()
                stitched.append(merged)
            vals = stitch_and_harvest(stitched, idxs)
            vtype = "percent" if current_is != "VLF á verðlagi hvers árs, ma. kr." else "level_billion_isk"
            for y, v in zip(years, vals):
                out.append({
                    "source_pdf_file": filename,
                    "target_year": target_year,
                    "table_year": y,
                    "status": status_for_year(y, target_year),
                    "indicator_is": current_is,
                    "indicator_en": current_en,
                    "unit": current_unit,
                    "value_type": vtype,
                    "value": v,
                    "forecast_id": meta["forecast_id"],
                    "forecast_month": meta["forecast_month"],
                    "forecast_season": meta["forecast_season"],
                })
            i += len(rows)
            continue

        # numeric-only extra line for current indicator
        if current_is and is_numeric_only(row):
            vals = stitch_and_harvest(row, idxs)
            vtype = "percent" if current_is != "VLF á verðlagi hvers árs, ma. kr." else "level_billion_isk"
            for y, v in zip(years, vals):
                out.append({
                    "source_pdf_file": filename,
                    "target_year": target_year,
                    "table_year": y,
                    "status": status_for_year(y, target_year),
                    "indicator_is": current_is,
                    "indicator_en": current_en,
                    "unit": current_unit,
                    "value_type": vtype,
                    "value": v,
                    "forecast_id": meta["forecast_id"],
                    "forecast_month": meta["forecast_month"],
                    "forecast_season": meta["forecast_season"],
                })
            i += 1
            continue

        i += 1

    # fill grid (indicator × year)
    det_years = [yy for yy in years if isinstance(yy,int)]
    have = {(r["indicator_is"], r["table_year"]) for r in out if r["table_year"] is not None}
    for _, is_name, en_name in CANON:
        for y in det_years:
            if (is_name, y) not in have:
                out.append({
                    "source_pdf_file": filename,
                    "target_year": target_year,
                    "table_year": y,
                    "status": status_for_year(y, target_year),
                    "indicator_is": is_name,
                    "indicator_en": en_name,
                    "unit": "",
                    "value_type": "percent" if is_name != "VLF á verðlagi hvers árs, ma. kr." else "level_billion_isk",
                    "value": None,
                    "forecast_id": meta["forecast_id"],
                    "forecast_month": meta["forecast_month"],
                    "forecast_season": meta["forecast_season"],
                })

    # dedupe preferring non-null
    ded: Dict[Tuple[str, Optional[int]], dict] = {}
    for r in out:
        k = (r["indicator_is"], r["table_year"])
        if k not in ded or (ded[k]["value"] is None and r["value"] is not None):
            ded[k] = r

    res = list(ded.values())
    res.sort(key=lambda r: (TARGET_ORDER.index(r["indicator_is"]) if r["indicator_is"] in TARGET_ORDER else 999,
                            r["table_year"] if r["table_year"] is not None else -1))
    return res

# ---------------------------
# Driver
# ---------------------------
def main(in_dir: Path, out_csv: Path):
    rows_all = []
    pdfs = sorted(in_dir.glob("*.pdf"))
    for pdf_path in pdfs:
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                tables_by_page, text_by_page = extract_tables_from_pdf(pdf)
            target_year = extract_target_year_from_filename(pdf_path.name)
            sel = select_table_with_recipe(pdf_path.name, tables_by_page, text_by_page, target_year)
            if not sel:
                print(f"! {pdf_path.name}: no tables found")
                continue
            tbl, years_all, idxs_all, used_recipe, recipe_name = sel
            recipe = find_recipe_for(pdf_path.name)
            stitch_k = int(recipe.get("stitch_continuations", 2)) if recipe else 2
            melted = melt_table(tbl, pdf_path.name, target_year, years_all, idxs_all, stitch_continuations=stitch_k)
            rows_all.extend(melted)
            print(
                f"✓ {pdf_path.name}: {len(melted)} rows "
                f"(recipe used: {'yes' if used_recipe else 'no'}"
                + (f", id: {recipe_name}" if used_recipe and recipe_name else "")
                + ")"
            )
        except Exception as e:
            print(f"✗ {pdf_path.name}: {e}")

    df = pd.DataFrame(rows_all, columns=[
        "source_pdf_file","target_year","table_year","status",
        "indicator_is","indicator_en","unit","value_type","value",
        "forecast_id","forecast_month","forecast_season",
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
