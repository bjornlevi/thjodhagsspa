#!/usr/bin/env python3
# load_spa_sqlite.py
import csv
import sqlite3
from pathlib import Path

from pipeline.canon import CANON  # list[tuple(label_key, isl, eng)]

DB_PATH = Path("data/spa.sqlite3")
CSV_DIR = Path("data/extracted/csv")
SCHEMA_PATH = Path("schema.sql")

def init_db(conn: sqlite3.Connection):
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

def upsert_canon(conn: sqlite3.Connection):
    sql = """
    INSERT INTO canon (label_key, isl, eng)
    VALUES (?, ?, ?)
    ON CONFLICT(label_key) DO UPDATE SET
      isl=excluded.isl,
      eng=excluded.eng;
    """
    conn.executemany(sql, CANON)

def load_csv_file(conn: sqlite3.Connection, csv_path: Path):
    # csv columns: pdf_name,label_key,year,value,type
    sql = """
    INSERT INTO observations (pdf_name, label_key, year, value, type)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(pdf_name, label_key, year) DO UPDATE SET
      value=excluded.value,
      type=excluded.type;
    """
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            t = r["type"].strip().lower()
            if t not in ("history","forecast","extrapolation"):
                # hard fail so you notice bad data
                raise ValueError(f"Bad type '{t}' in {csv_path}")
            rows.append((
                r["pdf_name"].strip(),
                r["label_key"].strip(),
                int(r["year"]),
                float(r["value"]),
                t,
            ))
        conn.executemany(sql, rows)

def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")

        init_db(conn)
        upsert_canon(conn)

        csv_files = sorted(CSV_DIR.glob("*.csv"))
        if not csv_files:
            print(f"[!] No CSVs found in {CSV_DIR}")
            return

        for p in csv_files:
            print(f"[i] Loading {p.name}")
            load_csv_file(conn, p)

        conn.commit()
        print(f"[✓] Loaded {len(csv_files)} file(s) into {DB_PATH}")

if __name__ == "__main__":
    main()
