#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tailored SPA extractor.

- Scans data/spa_pdf/ but ONLY processes files that have a registered extractor.
- Writes merged output to data/extracted/spa_extracted.csv

Deps:
  pip install pdfplumber pandas
"""

import os, re, unicodedata, sys
from typing import List, Dict, Optional, Tuple, Callable
import pandas as pd
import pdfplumber

# ========================== CANON ==========================

CANON: List[Tuple[str, str, str]] = [
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

def _strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", _strip_accents(str(s)).lower()).strip()

CANON_MAP = {norm(k): (k, isl, eng) for k, isl, eng in CANON}

ALIASES = {
    "voru og thjonusta": "utflutningur voru og thjonustu",
    "voru- og thjonustujofnudur (% af vlf)": "voru- og thjonustujofnudur",
    "vidskiptajofnudur (% af vlf)": "vidskiptajofnudur",
    "gdp": "verg landsframleidsla",
    "hagvoxtur": "verg landsframleidsla",
    "cpi": "visitala neysluverds",
}

def key_from_label(label: str) -> Optional[str]:
    n = norm(label)
    if n in CANON_MAP: return CANON_MAP[n][0]
    if n in ALIASES: return ALIASES[n]
    for c,(k,_,_) in CANON_MAP.items():
        if c in n or n in c:
            return k
    return None

def parse_number(tok: str) -> Optional[float]:
    tok = tok.strip().replace("\u2212","-")
    if not re.search(r"\d", tok): return None
    if re.search(r"[.,]", tok):
        last = max(tok.rfind("."), tok.rfind(","))
        int_part = re.sub(r"[^\d-]", "", tok[:last])
        dec_part = re.sub(r"[^\d]", "", tok[last+1:])
        tok = f"{int_part}.{dec_part}" if dec_part else int_part
    else:
        tok = re.sub(r"[^\d-]", "", tok)
    try:
        return float(tok)
    except:
        return None

def extract_years(line: str) -> List[int]:
    ys = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", line)]
    out, seen = [], set()
    for y in ys:
        if y not in seen:
            seen.add(y); out.append(y)
    return out

def classify(y: int, forecast_year: Optional[int]) -> Optional[str]:
    if forecast_year is None: return None
    if y < forecast_year: return "historical"
    if y == forecast_year: return "forecast"
    return "extrapolation"

# ===================== Shared parsing helpers =====================

def _find_best_block(page_text: str, min_labels=6, min_years=3, window=35) -> Optional[Dict]:
    lines = [l for l in page_text.splitlines() if l.strip()]
    best = None
    for i in range(len(lines)):
        win = lines[i:i+window]
        labels = {key_from_label(l) for l in win if key_from_label(l)}
        years = set()
        for l in win: years.update(extract_years(l))
        if len(labels) >= min_labels and len(years) >= min_years:
            cand = {"text":"\n".join(win), "labels":labels, "years":sorted(years)}
            if (best is None) or (len(labels), len(years)) > (len(best["labels"]), len(best["years"])):
                best = cand
    return best

def _parse_block_rows(block_text: str, years: List[int]) -> Dict[str, List[Optional[float]]]:
    out = {}
    for line in [l for l in block_text.splitlines() if l.strip()]:
        k = key_from_label(line)
        if not k: continue
        # Try column splits first
        nums = []
        for tok in re.split(r"\s{2,}|\t+|\s\|\s", line.strip()):
            if key_from_label(tok):  # skip label pieces
                continue
            if re.search(r"\d", tok):
                v = parse_number(tok)
                if v is not None: nums.append(v)
        # Fallback: any numeric-like spans
        if len(nums) < 2:
            for m in re.findall(r"[-−]?\d[\d\s.,]*", line):
                v = parse_number(m)
                if v is not None: nums.append(v)
        if nums:
            nums = nums[-len(years):]
            if len(nums) < len(years):
                nums = [None]*(len(years)-len(nums)) + nums
            out[k] = nums
    return out

def _extract_core(path: str, preferred_pages: Optional[List[int]] = None,
                  min_labels=6, min_years=3, window=35) -> pd.DataFrame:
    """Core routine used by tailored extractors with optional page pinning/tuning."""
    m = re.search(r"spa_(\d{4})", os.path.basename(path))
    forecast_year = int(m.group(1)) if m else None
    rows = []

    with pdfplumber.open(path) as pdf:
        page_indices = range(len(pdf.pages)) if not preferred_pages else [p-1 for p in preferred_pages if 1 <= p <= len(pdf.pages)]
        for idx in page_indices:
            page = pdf.pages[idx]
            text = page.extract_text() or ""
            if not text.strip():
                continue

            block = _find_best_block(text, min_labels=min_labels, min_years=min_years, window=window)
            if not block: 
                continue

            years = [y for y in block["years"] if 1990 <= y <= 2100]
            parsed = _parse_block_rows(block["text"], years)
            for k, vals in parsed.items():
                k_norm = norm(k)
                k_can, isl, eng = CANON_MAP.get(k_norm, (k, k, k))
                for yr, val in zip(years, vals):
                    rows.append({
                        "label_key": k_can,
                        "label_is": isl,
                        "label_en": eng,
                        "year": yr,
                        "value": val,
                        "page": idx+1,
                        "source_pdf": os.path.basename(path),
                        "type": classify(yr, forecast_year)
                    })

    df = pd.DataFrame(rows)
    if df.empty: return df
    canon_keys = {k for k,_,_ in CANON}
    df = df[df["label_key"].isin(canon_keys)].copy()
    if not df.empty:
        df.sort_values(["label_key","year"], inplace=True)
        df.drop_duplicates(subset=["label_key","year","source_pdf"], keep="last", inplace=True)
    return df

# ===================== Tailored extractors =====================

def extract_spa_2025_mars_vor(path: str) -> pd.DataFrame:
    """
    SPA 2025 mars/vor: tailored parameters.
    - Usually the macro table sits on 1–3; we pin a small set and use slightly stricter label threshold.
    - Tune window to catch wide tables.
    """
    # If you confirm exact pages later, e.g., preferred_pages=[2], lock it down here.
    return _extract_core(path, preferred_pages=None, min_labels=7, min_years=3, window=38)

def extract_spa_2025_juli_sumar(path: str) -> pd.DataFrame:
    """
    SPA 2025 júlí/sumar: tailored parameters.
    - Slightly different layout; allow a tad smaller window but insist on many labels.
    """
    return _extract_core(path, preferred_pages=None, min_labels=7, min_years=3, window=34)

# ===================== Registry (filename pattern -> extractor) =====================

# Compile patterns that must match the filename (case-insensitive).
EXTRACTORS: List[Tuple[re.Pattern, Callable[[str], pd.DataFrame]]] = [
    (re.compile(r"^spa_2025_.*mars.*vor.*\.pdf$", re.I), extract_spa_2025_mars_vor),
    (re.compile(r"^spa_2025_.*j[uú]li.*sumar.*\.pdf$", re.I), extract_spa_2025_juli_sumar),
]

# ===================== Main =====================

def main():
    pdf_dir = "data/spa_pdf"
    out_dir = "data/extracted"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "spa_extracted.csv")

    if not os.path.isdir(pdf_dir):
        print(f"[!] Missing directory: {pdf_dir}", file=sys.stderr)
        sys.exit(2)

    to_process = []
    for fname in os.listdir(pdf_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        for pat, _ in EXTRACTORS:
            if pat.match(fname):
                to_process.append(os.path.join(pdf_dir, fname))
                break  # stop at first matching extractor

    if not to_process:
        print("[!] No PDFs matched any registered extractor. Nothing to do.")
        sys.exit(0)

    all_rows = []
    for path in sorted(to_process):
        base = os.path.basename(path)
        # find the first matching extractor
        extractor = None
        for pat, fn in EXTRACTORS:
            if pat.match(base):
                extractor = fn
                break
        if extractor is None:
            continue
        print(f"[+] Extracting with tailored method: {base}")
        try:
            df = extractor(path)
        except Exception as e:
            print(f"[!] Failed {base}: {e}", file=sys.stderr)
            continue
        if not df.empty:
            all_rows.append(df)
        else:
            print(f"[!] No rows parsed from {base}", file=sys.stderr)

    if not all_rows:
        print("[!] No data extracted from matched PDFs.")
        sys.exit(1)

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(out_csv, index=False)
    print(f"[✓] Wrote {len(out)} rows → {out_csv}")

if __name__ == "__main__":
    main()
