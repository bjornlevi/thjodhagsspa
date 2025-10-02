#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import unicodedata
from pathlib import Path
import pandas as pd
from rapidfuzz import fuzz

from pipeline.canon import CANON

RAW_DIR = Path("data/extracted/raw")
CSV_DIR = Path("data/extracted/csv")
CSV_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Normalization helpers ----------------

# Accept ASCII minus -, Unicode minus − (U+2212), hyphen ‐ (U+2010), NB hyphen (U+2011)
NUM_RE = re.compile(r"^[\-\u2212\u2010\u2011]?\d{1,3}(?:[.,]\d+)?$")

YEAR_TOKEN_RE = re.compile(r"^20\d{2}$")

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # drop parentheses contents entirely (they’re noisy in labels)
    s = re.sub(r"\([^)]*\)", "", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_token(tok: str) -> str:
    """Normalize a single token for numeric detection/parsing."""
    if not tok:
        return ""
    # unify unicode minus variants to ASCII '-'
    tok = (tok
           .replace("\u2212", "-")   # −
           .replace("\u2010", "-")   # ‐
           .replace("\u2011", "-"))  # non-breaking hyphen
    return tok

def is_number_token(tok: str) -> bool:
    tok = normalize_token(tok)
    return bool(NUM_RE.match(tok))

def parse_float(tok: str):
    tok = normalize_token(tok).replace(",", ".")
    return float(tok)

# ---------------- Canon prep ----------------

# list of (label_text_norm, key) for isl, eng, and combined
CANON_CHOICES = []
for key, isl, eng in CANON:
    CANON_CHOICES.append((normalize_text(isl), key))
    CANON_CHOICES.append((normalize_text(eng), key))
    CANON_CHOICES.append((normalize_text(f"{isl} {eng}"), key))

def best_row_to_canon(label_text: str, remaining):
    """Greedy best match of this row's label to any remaining canon entry."""
    lt = normalize_text(label_text)
    if not lt:
        return None, 0.0, None
    best = None
    best_score = -1
    best_tpl = None
    for tpl in remaining:
        key, isl, eng = tpl
        for candidate in (isl, eng, f"{isl} {eng}"):
            score = fuzz.token_sort_ratio(lt, normalize_text(candidate))
            if score > best_score:
                best_score = score
                best = key
                best_tpl = tpl
    return best, float(best_score), best_tpl

# ---------------- Header & pre-join ----------------

HEADER_KEYWORDS_RE = re.compile(
    r"(volume growth from previous year|magnbreyting frá fyrra ári)",
    re.IGNORECASE
)

def extract_years_from_header(line: str) -> list[int]:
    years = []
    for raw in line.split():
        digits = re.sub(r"\D", "", raw)
        if digits.startswith("20") and len(digits) > 4:
            digits = digits[:4]  # e.g. 20101 -> 2010
        if re.fullmatch(r"20\d{2}", digits):
            years.append(int(digits))
    # preserve order & uniq
    seen, out = set(), []
    for y in years:
        if y not in seen:
            seen.add(y)
            out.append(y)
    return out

def score_year_sequence(years: list[int], forecast_year: int, line: str) -> float:
    if len(years) < 2:
        return -1e9  # not a header

    score = 0.0

    # 1) Keyword bonus
    if HEADER_KEYWORDS_RE.search(line):
        score += 100.0

    # 2) Length: reasonable table header usually 4–10 years
    if 3 <= len(years) <= 10:
        score += 20.0
    else:
        score -= 30.0

    # 3) Consecutive by 1 (no gaps)
    diffs = {b - a for a, b in zip(years, years[1:])}
    if diffs == {1}:
        score += 80.0
    else:
        # heavy penalty if looks like every 2 years etc.
        if any(d > 1 for d in diffs):
            score -= 60.0

    # 4) Proximity to forecast year
    if forecast_year in years:
        score += 40.0
    elif any(y in (forecast_year - 1, forecast_year + 1) for y in years):
        score += 20.0

    # 5) Chronological order (in case of noise)
    if years != sorted(years):
        score -= 25.0

    return score

def find_best_year_header(raw_lines: list[str], forecast_year: int) -> tuple[int, list[int]]:
    """
    Scan RAW (unjoined) lines. Return (header_index, years) for the best-scoring line.
    """
    best_idx, best_years, best_score = -1, [], -1e9
    for i, ln in enumerate(raw_lines):
        yrs = extract_years_from_header(ln)
        if len(yrs) < 2:
            continue
        s = score_year_sequence(yrs, forecast_year, ln)
        if s > best_score:
            best_idx, best_years, best_score = i, yrs, s
    return best_idx, best_years


def find_year_header_on_raw(raw_lines: list[str]) -> tuple[int, list[int]]:
    """
    Return (header_index, years) scanning the unjoined raw lines.
    Picks the LAST line with ≥2 valid years.
    If none found, returns (-1, []).
    """
    header_idx, years = -1, []
    for i, ln in enumerate(raw_lines):
        yrs = extract_years_from_header(ln)
        if len(yrs) >= 2:
            header_idx, years = i, yrs
    return header_idx, years

# --- join with provenance (keep first contributing raw index) ---
def join_label_lines_with_index(raw_lines: list[str]):
    """
    Merge consecutive label-only lines into the following numeric line.
    Return tuples (merged_line, numeric_idx_if_any_else_label_idx).
    """
    merged = []
    buffer = []
    buffer_first_idx = None

    def flush_with_numeric(numeric_line: str, numeric_idx: int):
        nonlocal merged, buffer, buffer_first_idx
        if buffer:
            merged.append((" ".join(buffer + [numeric_line]), numeric_idx))  # <-- use numeric idx
            buffer, buffer_first_idx = [], None
        else:
            merged.append((numeric_line, numeric_idx))

    for idx, ln in enumerate(raw_lines):
        toks = [normalize_token(t) for t in ln.split()]
        has_number = any(NUM_RE.match(t) for t in toks)
        if has_number:
            flush_with_numeric(ln, idx)
        else:
            if not buffer:
                buffer_first_idx = idx
            buffer.append(ln)

    # trailing label-only buffer (no numeric line after): keep as-is; will be skipped later
    if buffer:
        merged.append((" ".join(buffer), buffer_first_idx))

    return merged


def join_label_lines(lines):
    """
    Merge consecutive lines without numbers into one,
    so Icelandic+English labels (possibly two lines) become a single row.
    We use robust number detection (after normalizing tokens).
    """
    merged, buffer = [], []
    for ln in lines:
        toks = [normalize_token(t) for t in ln.split()]
        has_number = any(NUM_RE.match(t) for t in toks)
        if has_number:
            if buffer:
                merged.append(" ".join(buffer + [ln]))
                buffer = []
            else:
                merged.append(ln)
        else:
            buffer.append(ln)
    # flush any trailing label-only buffer (keep it — parser will ignore if no numbers)
    if buffer:
        merged.extend(buffer)
    return merged

def find_year_header(lines):
    header, years = None, []
    for ln in lines:
        yrs = extract_years_from_header(ln)
        if len(yrs) >= 2:
            header, years = ln, yrs
    return header, years

# ---------------- Value parsing ----------------

def parse_numeric_tokens(parts, idx, n_years):
    """Parse at most n_years floats from tokens starting at idx."""
    vals = []
    for token in parts[idx:]:
        if is_number_token(token):
            try:
                vals.append(parse_float(token))
            except ValueError:
                pass
        if len(vals) == n_years:
            break
    return vals

# ---------------- Core ----------------
def parse_page(raw_file: Path, forecast_year: int, pdf_name: str) -> pd.DataFrame:
    """
    Parse a single *_page.txt into tidy rows.

    Assumes these helpers exist in this module:
      - find_best_year_header(raw_lines, forecast_year) -> (header_idx:int, years:list[int])
      - join_label_lines_with_index(raw_lines) -> list[(merged_line:str, numeric_src_idx:int)]
      - is_number_token(token:str) -> bool
      - parse_numeric_tokens(parts:list[str], start_idx:int, n_years:int) -> list[float]
      - best_row_to_canon(label_text:str, remaining:list[tuple]) -> (canon_key:str|None, score:float, tpl|None)
      - CANON  # list of (key, isl, eng)
    """
    # 1) Read RAW lines (no joining yet)
    raw_lines = [ln.strip() for ln in raw_file.read_text(encoding="utf-8").splitlines() if ln.strip()]

    # 2) Pick the correct header on RAW lines
    header_idx, years = find_best_year_header(raw_lines, forecast_year)
    if not years:
        print(f"[!] No year header found in {raw_file}")
        return pd.DataFrame()
    n_years = len(years)
    print(f"[i] {pdf_name} years: {years} (header at raw idx {header_idx})")

    # 3) Join label-only lines to following numeric line, keeping the NUMERIC line's index
    joined = join_label_lines_with_index(raw_lines)  # -> [(line, numeric_idx_or_label_idx), ...]

    # 4) Only parse rows whose numeric line originated AFTER the header
    joined = [(line, idx0) for (line, idx0) in joined if idx0 > header_idx]

    remaining_labels = list(CANON)  # [(key, isl, eng), ...]
    rows: list[dict] = []

    for line, _src_idx in joined:
        parts = line.split()

        # Find the first numeric token index using robust numeric detector
        first_num_i = next((i for i, tok in enumerate(parts) if is_number_token(tok)), None)
        if first_num_i is None:
            # label-only or footer junk
            continue

        # Label text is everything before the first numeric token
        label_text = " ".join(parts[:first_num_i]).strip()

        # Greedy best match against remaining CANON labels (isl / eng / combined)
        canon_key, score, matched_tpl = best_row_to_canon(label_text, remaining_labels)
        if not canon_key:
            # No reasonable match – skip
            # print(f"[debug] Unmatched label: {label_text}")
            continue

        # Parse up to n_years numeric values
        values = parse_numeric_tokens(parts, first_num_i, n_years)
        if len(values) != n_years:
            print(f"[!] Mismatch in {pdf_name}: {label_text} ({len(values)} vs {n_years})")
            # Uncomment for deeper debugging:
            # print("    tokens:", parts[first_num_i:])
            continue

        # Emit tidy rows
        for y, val in zip(years, values):
            dtype = "history" if y < forecast_year else "forecast" if y == forecast_year else "extrapolation"
            rows.append({
                "pdf_name": pdf_name,
                "label_key": canon_key,
                "year": y,
                "value": val,
                "type": dtype,
            })

        # Consume this canon key so no other row can reuse it
        remaining_labels = [tpl for tpl in remaining_labels if tpl[0] != canon_key]

    if remaining_labels:
        # Report which canon rows we didn't find on this page (useful for debugging)
        isl_left = [isl for _, isl, _ in remaining_labels]
        print(f"[!] {len(remaining_labels)} canon labels not matched in {pdf_name}: {isl_left}")

    return pd.DataFrame(rows)

# ---------------- Entry ----------------

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
