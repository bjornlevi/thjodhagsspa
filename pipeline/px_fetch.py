#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load THJ07100 PXWeb data into SQLite with proper Flokkun labels.

- Fetches JSON data (POST) OR uses cached data/px/thj07100.json.
- Fetches metadata (GET) to resolve Flokkun code -> label (valueTexts).
- Writes to:
    px_series(series_code TEXT PRIMARY KEY, series_label TEXT)
    px_obs(series_code TEXT, year INTEGER, value REAL,
           PRIMARY KEY(series_code, year),
           FOREIGN KEY(series_code) REFERENCES px_series(series_code))
"""

import json
import sqlite3
import urllib.request
from pathlib import Path

DB_PATH  = Path("data/spa.sqlite3")
RAW_DIR  = Path("data/px")
RAW_DIR.mkdir(parents=True, exist_ok=True)
RAW_JSON = RAW_DIR / "thj07100.json"

PX_URL = "https://px.hagstofa.is/pxis/api/v1/is/Efnahagur/thjodhagsspa/THJ07100.px"
PX_PAYLOAD = {"query": [], "response": {"format": "json"}}

# ---------- HTTP helpers ----------

def http_get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))

def http_post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))

# ---------- Fetchers ----------

def fetch_px_data() -> dict:
    """Return PXWeb 'columns' + 'data'. Cache to RAW_JSON if missing."""
    if RAW_JSON.exists():
        return json.loads(RAW_JSON.read_text(encoding="utf-8"))
    data = http_post_json(PX_URL, PX_PAYLOAD)
    RAW_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[i] cached → {RAW_JSON}")
    return data

def fetch_px_metadata() -> dict:
    """
    GET the table metadata:
    {
      "title": "...",
      "variables":[
        {"code":"Flokkun","text":"Flokkun","values":["0","1",...],"valueTexts":["Label0","Label1",...]},
        {"code":"Ár","text":"Ár","values":[...],"valueTexts":[...]},
        ...
      ]
    }
    """
    return http_get_json(PX_URL)

def build_flokkun_label_map(meta: dict) -> dict[str, str]:
    """Return {'0': 'Label0', '1': 'Label1', ...} or {} if not found."""
    vars_ = meta.get("variables") or []
    for var in vars_:
        if var.get("code") == "Flokkun":
            values = var.get("values") or []
            texts  = var.get("valueTexts") or []
            # zip to dict; lengths can differ in weird cases, so guard
            n = min(len(values), len(texts))
            return {values[i]: texts[i] for i in range(n)}
    return {}

# ---------- DB ----------

def ensure_tables(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS px_series(
            series_code  TEXT PRIMARY KEY,
            series_label TEXT NOT NULL
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS px_obs(
            series_code TEXT NOT NULL,
            year        INTEGER NOT NULL,
            value       REAL NOT NULL,
            PRIMARY KEY(series_code, year),
            FOREIGN KEY(series_code) REFERENCES px_series(series_code)
        );
    """)

# ---------- Loader ----------

def load_pxweb(conn: sqlite3.Connection, payload: dict, flokkun_labels: dict[str, str]):
    cols = payload.get("columns", [])
    rows = payload.get("data", [])
    if len(cols) < 3 or not rows:
        raise RuntimeError("Unexpected PXWeb shape: need at least 3 columns and non-empty data.")

    # Identify indices (we saw 0=Flokkun, 1=Ár)
    try:
        flokkun_idx = next(i for i,c in enumerate(cols) if c.get("code") == "Flokkun")
        year_idx    = next(i for i,c in enumerate(cols) if c.get("code") == "Ár")
    except StopIteration:
        raise RuntimeError("Could not find 'Flokkun' and 'Ár' columns in PXWeb response.")

    series = {}  # key -> (series_code, series_label)
    obs = []     # (series_code, year, value)

    for r in rows:
        key = r.get("key") or []
        vals = r.get("values") or []
        if len(key) <= max(flokkun_idx, year_idx) or not vals:
            continue

        flokkun_code = str(key[flokkun_idx])
        year_token   = str(key[year_idx])

        # Parse year (first 4 digits)
        digits = "".join(ch for ch in year_token if ch.isdigit())
        if len(digits) < 4:
            continue
        year = int(digits[:4])

        val_raw = vals[0]
        if val_raw in (None, "", ".", ".."):
            continue
        try:
            value = float(str(val_raw).replace(",", "."))
        except Exception:
            continue

        # Build series
        code  = f"Flokkun={flokkun_code}"
        label = flokkun_labels.get(flokkun_code, code)
        if code not in series:
            series[code] = (code, label)

        obs.append((code, year, value))

    cur = conn.cursor()
    cur.execute("BEGIN;")
    cur.executemany(
        "INSERT INTO px_series(series_code, series_label) VALUES(?, ?) "
        "ON CONFLICT(series_code) DO UPDATE SET series_label=excluded.series_label",
        list(series.values())
    )
    cur.executemany(
        "INSERT INTO px_obs(series_code, year, value) VALUES(?, ?, ?) "
        "ON CONFLICT(series_code, year) DO UPDATE SET value=excluded.value",
        obs
    )
    cur.execute("COMMIT;")
    print(f"[✓] Stored {len(series)} series, {len(obs)} observations (PXWeb with labels).")

# ---------- Entry ----------

def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = fetch_px_data()
    meta    = fetch_px_metadata()
    flokkun_labels = build_flokkun_label_map(meta)
    if not flokkun_labels:
        print("[!] Warning: no Flokkun labels found; using codes as labels.")

    with sqlite3.connect(DB_PATH) as conn:
        ensure_tables(conn)
        if "columns" in payload and "data" in payload:
            load_pxweb(conn, payload, flokkun_labels)
        else:
            raise RuntimeError("Only PXWeb shape supported for this table.")

if __name__ == "__main__":
    main()
