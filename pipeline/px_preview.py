#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Print a small preview from px_* tables.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path("data/spa.sqlite3")

def main():
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        print("== Series ==")
        for code, label in cur.execute("SELECT series_code, series_label FROM px_series ORDER BY series_code LIMIT 20;"):
            print(" ", code, "→", label)

        print("\n== Sample obs (latest 10 rows) ==")
        for code, year, value in cur.execute("""
            SELECT series_code, year, value
            FROM px_obs
            ORDER BY year DESC, series_code
            LIMIT 10;
        """):
            print(f"  {year}  {code}: {value}")

if __name__ == "__main__":
    main()
