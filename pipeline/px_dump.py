#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PX dump (no parsing): fetch THJ07100 and save raw JSON so we can inspect it.

- Saves to: data/px/thj07100.json
- Prints a short summary depending on payload shape:
    * PXWeb ("columns" + "data")
    * JSON-stat-ish ("dataset")
Set VERBOSE=1 for extra details.
"""

import json
import os
import unicodedata
from pathlib import Path
import urllib.request

PX_URL = "https://px.hagstofa.is/pxis/api/v1/is/Efnahagur/thjodhagsspa/THJ07100.px"
PX_PAYLOAD = {"query": [], "response": {"format": "json"}}

RAW_DIR = Path("data/px")
RAW_DIR.mkdir(parents=True, exist_ok=True)
RAW_JSON = RAW_DIR / "thj07100.json"

VERBOSE = os.getenv("VERBOSE", "0") == "1"
def vprint(*a, **k):
    if VERBOSE: print(*a, **k)

def _normalize(s: str) -> str:
    if not s: return ""
    s = s.lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    return s

def fetch_px():
    req = urllib.request.Request(
        PX_URL,
        data=json.dumps(PX_PAYLOAD).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    RAW_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[✓] Saved raw PX JSON → {RAW_JSON}")
    return data

def summarize_pxweb(payload: dict):
    cols = payload.get("columns", [])
    rows = payload.get("data", [])
    print(f"[i] PXWeb shape: columns={len(cols)}, rows={len(rows)}")
    if not cols or not rows:
        return
    # show columns
    head_cols = [(c.get("code"), c.get("text"), c.get("type")) for c in cols]
    print("    columns (code,text,type):")
    for c in head_cols:
        print(f"      - {c}")
    # show first 5 rows
    print("    first rows:")
    for r in rows[:5]:
        print(f"      key={r.get('key')} values={r.get('values')}")
    # quick guess of a time-like column
    time_idx = None
    for i, c in enumerate(cols):
        code = _normalize(c.get("code") or "")
        text = _normalize(c.get("text") or "")
        if any(t in code for t in ("timi", "ar", "time", "year")) or any(t in text for t in ("timi", "ar", "time", "year")):
            time_idx = i
            break
    if time_idx is not None:
        years = []
        seen = set()
        for r in rows[:200]:
            kv = (r.get("key") or [None])[time_idx]
            if kv is None: continue
            digits = "".join(ch for ch in str(kv) if ch.isdigit())
            if len(digits) >= 4:
                y = int(digits[:4])
                if y not in seen:
                    seen.add(y); years.append(y)
        if years:
            print(f"    time column index ~ {time_idx}, sample years: {years[:12]}{' ...' if len(years)>12 else ''}")

def summarize_dataset(payload: dict):
    ds = payload.get("dataset", {})
    dims = ds.get("dimension", {})
    vals = ds.get("value", [])
    print(f"[i] dataset shape: dims={len([k for k in dims if k not in ('id','size','__order__')])}, values={len(vals)}")

    # list dims
    names = [k for k in dims.keys() if k not in ("id", "size", "__order__")]
    print("    dimensions:")
    for n in names:
        cat = dims[n].get("category", {})
        labels = cat.get("label", {})
        print(f"      - {n}: {len(labels)} categories")

    # try to find years in any dimension labels
    def parse_years(labels: dict):
        out = []
        for k, v in (labels or {}).items():
            digits = "".join(ch for ch in str(v) if ch.isdigit())
            if len(digits) >= 4:
                out.append(int(digits[:4]))
            elif str(k).isdigit() and len(str(k)) == 4:
                out.append(int(k))
        return out

    for n in names:
        labels = dims[n].get("category", {}).get("label", {})
        yrs = parse_years(labels)
        if len(yrs) >= 2:
            print(f"    '{n}' looks like time, sample years: {yrs[:12]}{' ...' if len(yrs)>12 else ''}")
            break

def main():
    data = fetch_px()
    # Two possible shapes:
    if "columns" in data and "data" in data:
        summarize_pxweb(data)
    elif "dataset" in data:
        summarize_dataset(data)
    else:
        print("[!] Unrecognized PX shape. Top-level keys:", list(data.keys())[:10])
        print("    Saved raw for inspection.")

if __name__ == "__main__":
    main()
