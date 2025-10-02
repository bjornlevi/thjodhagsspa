#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_parse.py

Pipeline:
- Read raw page dumps (_page.txt)
- Parse into tidy CSV with year classification
"""

import re
import unicodedata
from pathlib import Path
import pandas as pd
from rapidfuzz import process, fuzz

from pipeline.canon import CANON

RAW_DIR = Path("data/extracted/raw")
CSV_DIR = Path("data/extracted/csv")
CSV_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Helpers ----------

def normalize(s: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    if not s:
        return ""
    s = str(s).lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s).strip()

# Prebuild lookup: Icelandic + English → stable key
CANON_CHOICES = []
for key, isl, eng in CANON:
    CANON_CHOICES.append((normalize(isl), key))
    CANON_CHOICES.append((normalize(eng), key))
CANON_LOOKUP = dict(CANON_CHOICES)

def best_match_canon(label_text: str, threshold=75):
    norm = normalize(label_text)
    if not norm:
        return None

    # Try splitting Icelandic vs English
    parts = norm.split(" ", 1)  # crude split, first word vs rest
    candidates = [norm]
    if len(parts) > 1:
        candidates.extend(parts)

    best_key, best_score = None, 0
    for cand in candidates:
        match = process.extractOne(
            cand,
            list(CANON_LOOKUP.keys()),
            scorer=fuzz.partial_ratio,
            score_cutoff=threshold
        )
        if match and match[1] > best_score:
            best_key = CANON_LOOKUP[match[0]]
            best_score = match[1]

    return best_key

def extract_years_from_header(line: str) -> list[int]:
    years = []
    for token in line.split():
        if token.startswith("20"):
            try:
                y = int(token[:4])  # only first 4 digits
                years.append(y)
            except ValueError:
                pass
    # deduplicate preserving order
    seen, uniq = set(), []
    for y in years:
        if y not in seen:
            seen.add(y)
            uniq.append(y)
    return uniq

def find_year_header(lines):
    """Find the last header line that contains ≥2 valid years."""
    header, years = None, []
    for ln in lines:
        yrs = extract_years_from_header(ln)
        if len(yrs) >= 2:
            header, years = ln, yrs
    return header, years

NUM_RE = re.compile(r"^-?\d+(?:[.,]\d+)?$")

# ---------- Core ----------

def parse_page(raw_file: Path, forecast_year: int, pdf_name: str) -> pd.DataFrame:
    lines = [ln.strip() for ln in raw_file.read_text(encoding="utf-8").splitlines() if ln.strip()]

    header_line, years = find_year_header(lines)
    if not years:
        print(f"[!] No year header found in {raw_file}")
        return pd.DataFrame()
    n_years = len(years)

    rows, carry_label, in_table = [], "", False
    for line in lines:
        if not in_table:
            if line == header_line:
                in_table = True   # start after header
            continue

        parts = line.split()
        idx = next((i for i, p in enumerate(parts) if NUM_RE.match(p)), None)
        if idx is None:
            carry_label += " " + line
            continue

        label_text = (carry_label + " " + " ".join(parts[:idx])).strip()
        carry_label = ""

        canon_key = best_match_canon(label_text)
        if not canon_key:
            continue

        values = []
        for token in parts[idx:]:
            token = token.replace(",", ".")
            try:
                values.append(float(token))
            except ValueError:
                pass

        if len(values) != n_years:
            print(f"[!] Mismatch in {pdf_name}: {label_text}")
            print(f"    years: {years}")
            print(f"    values: {values}")
            continue


        if len(values) != n_years:
            print(f"[!] Mismatch {label_text} ({len(values)} vs {n_years}) in {pdf_name}")
            continue

        for y, val in zip(years, values):
            if y == forecast_year:
                dtype = "forecast"
            elif y < forecast_year:
                dtype = "history"
            else:
                dtype = "extrapolation"
            rows.append({
                "pdf_name": pdf_name,
                "label_key": canon_key,
                "year": y,
                "value": val,
                "type": dtype,
            })

    return pd.DataFrame(rows)

# ---------- Entry ----------

def main():
    raw_pages = list(RAW_DIR.glob("*_page.txt"))
    if not raw_pages:
        print("[!] No _page.txt files found in", RAW_DIR)
        return

    for raw_file in raw_pages:
        pdf_name = raw_file.stem.replace("_page", "")
        m = re.search(r"spa_(\d{4})", pdf_name)
        if not m:
            print(f"[!] Could not parse forecast year from {pdf_name}")
            continue
        forecast_year = int(m.group(1))

        print(f"=== Parsing {raw_file} (forecast_year={forecast_year}) ===")
        tidy = parse_page(raw_file, forecast_year, pdf_name)

        if tidy.empty:
            continue

        out_file = CSV_DIR / f"{pdf_name}.csv"
        tidy.to_csv(out_file, index=False)
        print(f"[✓] Saved tidy CSV → {out_file}")

if __name__ == "__main__":
    main()
