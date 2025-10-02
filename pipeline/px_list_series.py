#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
List PX series for a given table prefix and show quick stats.

Examples:
  python3 -m pipeline.px_list_series THJ07000
  python3 -m pipeline.px_list_series THJ07100 --grep "VLF|GDP"
  PX_TABLE=THJ07000 python3 -m pipeline.px_list_series
"""

import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional

DB_PATH = Path("data/spa.sqlite3")

def which_table(argv) -> str:
    if len(argv) >= 2 and argv[1]:
        return argv[1].strip()
    env = os.getenv("PX_TABLE")
    if env:
        return env.strip()
    print("Usage: python -m pipeline.px_list_series <TABLE> [--grep REGEX]")
    sys.exit(2)

def parse_args(argv):
    table = which_table(argv)
    grep = None
    if "--grep" in argv:
        i = argv.index("--grep")
        if i+1 < len(argv):
            grep = argv[i+1]
        else:
            print("Missing pattern after --grep")
            sys.exit(2)
    return table, grep

def main():
    table, grep = parse_args(sys.argv)
    if not DB_PATH.exists():
        print(f"[!] DB missing: {DB_PATH}. Run px-fetch first.")
        sys.exit(1)

    rx: Optional[re.Pattern] = re.compile(grep, re.IGNORECASE) if grep else None

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        # Use correlated subqueries for latest_year and latest_value (no alias reuse).
        cur.execute("""
            SELECT
                s.series_code,
                s.series_label,
                MIN(o.year) AS y_min,
                MAX(o.year) AS y_max,
                COUNT(o.year) AS n_obs,
                (
                  SELECT o2.year
                  FROM px_obs o2
                  WHERE o2.series_code = s.series_code
                  ORDER BY o2.year DESC
                  LIMIT 1
                ) AS latest_year,
                (
                  SELECT o3.value
                  FROM px_obs o3
                  WHERE o3.series_code = s.series_code
                  ORDER BY o3.year DESC
                  LIMIT 1
                ) AS latest_value
            FROM px_series s
            JOIN px_obs o ON o.series_code = s.series_code
            WHERE s.series_code LIKE ? || '|%'
            GROUP BY s.series_code, s.series_label
            ORDER BY s.series_code;
        """, (table,))
        rows = cur.fetchall()

    if not rows:
        print(f"[i] No series found for table {table}. Did you run px-fetch-{table.lower()}?")
        sys.exit(0)

    print(f"== {table}: {len(rows)} series ==")
    for code, label, y_min, y_max, n, y_latest, v_latest in rows:
        if rx and not (rx.search(code) or rx.search(label or "")):
            continue
        yrs = f"{y_min}→{y_max}, n={n}"
        latest = f"{y_latest}={v_latest}" if y_latest is not None else "n/a"
        print(f"- {code}\n    {label}\n    years: {yrs} | latest: {latest}")

if __name__ == "__main__":
    main()
