#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic PXWeb loader for Hagstofa endpoints (e.g., THJ07100, THJ07000).

- Fetches metadata (GET) to identify:
    * time variable (type 't' OR by name/year-like)
    * value column (type 'c')
    * all other variables become series dimensions
    * valueTexts used for human-readable labels

- Fetches data (POST) with {"query": [], "response": {"format": "json"}}
  and caches raw JSON to data/px/<table>.json

- Stores into SQLite:
    px_series(series_code TEXT PRIMARY KEY, series_label TEXT)
    px_obs(series_code TEXT, year INTEGER, value REAL, PRIMARY KEY(series_code, year))

Series code is namespaced by table:
    <TABLE>|VarA=valA|VarB=valB
Series label is a readable string:
    <TABLE> — VarA=LabelA — VarB=LabelB

Usage:
  PX_TABLE=THJ07000 python3 -m pipeline.px_fetch_any
  python3 -m pipeline.px_fetch_any THJ07000
"""

import json
import os
import re
import sqlite3
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Any

DB_PATH = Path("data/spa.sqlite3")
PX_BASE = "https://px.hagstofa.is/pxis/api/v1/is/Efnahagur/thjodhagsspa"
RAW_DIR = Path("data/px")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def which_table() -> str:
    if len(sys.argv) >= 2 and sys.argv[1]:
        return sys.argv[1].strip()
    return os.getenv("PX_TABLE", "THJ07100").strip()

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

def fetch_metadata(table: str) -> dict:
    return http_get_json(f"{PX_BASE}/{table}.px")

def fetch_data(table: str) -> dict:
    raw_path = RAW_DIR / f"{table.lower()}.json"
    if raw_path.exists():
        return json.loads(raw_path.read_text(encoding="utf-8"))
    payload = {"query": [], "response": {"format": "json"}}
    data = http_post_json(f"{PX_BASE}/{table}.px", payload)
    raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[✓] Cached raw PX JSON → {raw_path}")
    return data

def find_time_var(meta_vars: List[dict]) -> str:
    # Prefer type 't'
    for v in meta_vars:
        if v.get("type") == "t":
            return v["code"]
    # Fallback by name heuristic
    for v in meta_vars:
        code = (v.get("code") or "").lower()
        text = (v.get("text") or "").lower()
        if any(t in code for t in ("ár", "ar", "time", "year")) or any(t in text for t in ("ár", "ar", "time", "year")):
            return v["code"]
    # Last resort: first variable whose valueTexts look like years
    yearish = re.compile(r"\b\d{4}\b")
    for v in meta_vars:
        for t in (v.get("valueTexts") or []):
            if yearish.search(str(t)):
                return v["code"]
    raise RuntimeError("Could not identify time variable in metadata.")

def parse_year(ystr: str) -> int | None:
    digits = "".join(ch for ch in str(ystr) if ch.isdigit())
    if len(digits) >= 4:
        y = int(digits[:4])
        if 1900 <= y <= 2100:
            return y
    return None

def load_pxweb_any(conn: sqlite3.Connection, table: str, meta: dict, payload: dict):
    cols = payload.get("columns", [])
    rows = payload.get("data", [])
    if not cols or not rows:
        raise RuntimeError("PXWeb payload missing columns/data.")

    # Build quick lookup for metadata vars
    vars_list: List[dict] = meta.get("variables") or []
    var_by_code: Dict[str, dict] = {v["code"]: v for v in vars_list if "code" in v}

    # Identify indices in payload columns
    col_codes = [c.get("code") for c in cols]
    # value column in PXWeb has type 'c'
    value_col_idx = next((i for i, c in enumerate(cols) if c.get("type") == "c" or c.get("code") == "value"), None)

    # time index via metadata time var code
    time_code = find_time_var(vars_list)
    try:
        time_idx = col_codes.index(time_code)
    except ValueError:
        # sometimes PXWeb renames; try by text matching
        time_idx = next((i for i,c in enumerate(cols) if c.get("type") == "t"), None)
    if time_idx is None:
        raise RuntimeError("Could not align time column between metadata and data.")

    # Series dimensions = all columns except time + value
    series_dim_idxs = [i for i in range(len(cols)) if i not in (time_idx, value_col_idx)]

    # Human labels map for each series var: {code_val -> label_text}
    label_maps: Dict[str, Dict[str, str]] = {}
    for i in series_dim_idxs:
        vcode = cols[i].get("code")
        var_meta = var_by_code.get(vcode, {})
        vals = var_meta.get("values") or []
        texts = var_meta.get("valueTexts") or []
        label_maps[vcode] = {vals[j]: texts[j] for j in range(min(len(vals), len(texts)))}

    # Upserts
    series_seen: Dict[Tuple[str, ...], Tuple[str, str]] = {}
    obs_rows: List[Tuple[str, int, float]] = []

    for r in rows:
        key = r.get("key") or []
        vals = r.get("values") or []
        if not key or not vals:
            continue

        # Parse year
        y = parse_year(key[time_idx])
        if y is None:
            continue

        # Value
        vraw = vals[0]
        if vraw in (None, "", ".", ".."):
            continue
        try:
            v = float(str(vraw).replace(",", "."))
        except Exception:
            continue

        # Series composite key on all series dims (string values as-is)
        series_tuple = tuple(f"{cols[i]['code']}={key[i]}" for i in series_dim_idxs) if series_dim_idxs else tuple(["ALL=all"])

        # Create series_code + label (namespaced by table)
        if series_tuple not in series_seen:
            code_parts = [f"{table}"] + list(series_tuple)
            series_code = "|".join(code_parts)

            # Pretty label from valueTexts where possible
            label_parts = [table]
            for i in series_dim_idxs:
                vcode = cols[i]['code']
                vtext = cols[i].get('text') or vcode
                kval  = key[i]
                kval_label = label_maps.get(vcode, {}).get(kval, kval)
                label_parts.append(f"{vtext}={kval_label}")
            series_label = " — ".join(label_parts)

            series_seen[series_tuple] = (series_code, series_label)

        series_code, _ = series_seen[series_tuple]
        obs_rows.append((series_code, y, v))

    cur = conn.cursor()
    cur.execute("BEGIN;")
    cur.executemany(
        "INSERT INTO px_series(series_code, series_label) VALUES(?, ?) "
        "ON CONFLICT(series_code) DO UPDATE SET series_label=excluded.series_label",
        list(series_seen.values())
    )
    cur.executemany(
        "INSERT INTO px_obs(series_code, year, value) VALUES(?, ?, ?) "
        "ON CONFLICT(series_code, year) DO UPDATE SET value=excluded.value",
        obs_rows
    )
    cur.execute("COMMIT;")
    print(f"[✓] [{table}] Stored {len(series_seen)} series, {len(obs_rows)} observations.")

def main():
    table = which_table()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    meta = fetch_metadata(table)
    data = fetch_data(table)

    if "columns" not in data or "data" not in data:
        raise RuntimeError("This loader expects PXWeb 'columns'+'data' shape.")

    with sqlite3.connect(DB_PATH) as conn:
        ensure_tables(conn)
        load_pxweb_any(conn, table, meta, data)

if __name__ == "__main__":
    main()
