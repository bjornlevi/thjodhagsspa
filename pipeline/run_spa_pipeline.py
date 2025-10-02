#!/usr/bin/env python3
import re
import unicodedata
from pathlib import Path
import pdfplumber
import yaml
from rapidfuzz import process

from pipeline.canon import CANON
from pipeline.locate_table import TABLE_SETTINGS, cell_matches_any_label

PDF_DIR = Path("data/spa_pdf")
RAW_DIR = Path("data/extracted/raw")
MANIFEST = Path("data/extracted/spa_tables.yaml")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# --- CANON matching helpers ---
def normalize(s: str) -> str:
    if not s:
        return ""
    s = str(s).lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s).strip()

CANON_LABELS = [isl for _, isl, _ in CANON] + [en for _, _, en in CANON]

CID_PATTERN = re.compile(r"\(cid:(\d+)\)")

def replace_cid_numbers(text: str) -> str:
    """Replace (cid:NNN) artifacts with digits, using 611 → 0 .. 620 → 9 mapping."""
    def repl(m):
        val = int(m.group(1)) - 611
        if 0 <= val <= 9:
            return str(val)
        return ""  # drop unexpected values
    return CID_PATTERN.sub(repl, text)

def best_match(line: str, threshold=70):
    result = process.extractOne(normalize(line), CANON_LABELS, score_cutoff=threshold)
    return result[0] if result else None

# --- YAML manifest ---
def load_manifest():
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}

def save_manifest(manifest: dict):
    MANIFEST.write_text(
        yaml.dump(manifest, sort_keys=True, allow_unicode=True), encoding="utf-8"
    )

# --- page dump (with filtering) ---
def dump_filtered_page(pdf_file: Path, page_num: int):
    base_name = pdf_file.stem
    out_file = RAW_DIR / f"{base_name}_page.txt"

    with pdfplumber.open(pdf_file) as pdf:
        page = pdf.pages[page_num - 1]
        text = page.extract_text() or ""

    lines = [replace_cid_numbers(ln.strip()) for ln in text.splitlines() if ln.strip()]

    filtered = []
    for ln in lines:
        if re.search(r"\b20\d{2}", ln):        # year headers
            filtered.append(ln)
            continue
        if best_match(ln):                     # canon-matching rows
            filtered.append(ln)

    out_file.write_text("\n".join(filtered), encoding="utf-8")
    print(f"  [✓] Dumped & filtered page {page_num} → {out_file}")
    return out_file

# --- locate best table ---
def locate_table(pdf_file: Path, min_hits=6):
    """Locate the last table in the PDF that matches CANON labels."""
    best = None
    with pdfplumber.open(pdf_file) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            for name, ts in TABLE_SETTINGS:
                try:
                    tables = page.extract_tables(table_settings=ts) or []
                except Exception:
                    continue
                for i, t in enumerate(tables):
                    matched_keys = set()
                    for row in t or []:
                        for cell in row or []:
                            if cell and cell_matches_any_label(str(cell)):
                                matched_keys.add(cell)
                    if len(matched_keys) >= min_hits:
                        best = dict(page=pno, strategy=name, candidate=i, hits=len(matched_keys))
    return best

# --- pipeline run ---
def main():
    manifest = load_manifest()
    for pdf_file in PDF_DIR.glob("*.pdf"):
        pdf_name = pdf_file.stem
        m = re.search(r"spa_(\d{4})", pdf_name)
        if not m:
            continue

        print(f"\n=== {pdf_name} ===")

        if pdf_name in manifest:
            page_num = manifest[pdf_name]["page"]
            print(f"  Using manifest page {page_num}")
        else:
            best = locate_table(pdf_file)
            if not best:
                print("  [!] No candidate found")
                continue
            page_num = best["page"]
            manifest[pdf_name] = best
            save_manifest(manifest)
            print(f"  Auto-detected page {page_num} (saved to manifest)")

        dump_filtered_page(pdf_file, page_num)
        print(f"  [i] Raw page ready for parsing")

if __name__ == "__main__":
    main()
