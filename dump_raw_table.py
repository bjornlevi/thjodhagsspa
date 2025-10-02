#!/usr/bin/env python3
"""
Dump raw text lines from a PDF page.
- Raw lines: /data/extracted/raw/<pdf_name>.txt
"""

import pdfplumber
from pathlib import Path

# --- Config ---
PDF_PATH = "data/spa_pdf/spa_2020_juni_sumar.pdf"
PAGE = 17  # 1-based page number

RAW_DIR = Path("data/extracted/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def dump_lines(pdf_path, page_num, out_file):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num-1]
        text = page.extract_text()
    if not text:
        raise ValueError("No text extracted from page")
    lines = text.splitlines()
    with open(out_file, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")
    print(f"Raw text lines saved → {out_file}")

if __name__ == "__main__":
    pdf_file = Path(PDF_PATH)
    base_name = pdf_file.stem   # e.g. spa_2020_juni_sumar
    out_file = RAW_DIR / f"{base_name}.txt"
    dump_lines(PDF_PATH, PAGE, out_file)
