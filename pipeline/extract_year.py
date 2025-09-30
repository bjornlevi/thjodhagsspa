# pipeline/extract_year.py
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd
import unicodedata, re

ORDER = [
    "Einkaneysla","Samneysla","Fjármunamyndun","Atvinnuvegafjárfesting",
    "Fjárfesting í íbúðarhúsnæði","Fjárfesting hins opinbera","Þjóðarútgjöld alls",
    "Útflutningur vöru og þjónustu","Innflutningur vöru og þjónustu",
    "Verg landsframleiðsla","Vöru- og þjónustujöfnuður (% af VLF)",
    "Viðskiptajöfnuður (% af VLF)","VLF á verðlagi hvers árs, ma. kr.",
    "Vísitala neysluverðs","Gengisvísitala","Raungengi",
    "Launavísitala m.v. fast verðlag",
    "Hagvöxtur í helstu viðskiptalöndum","Alþjóðleg verðbólga",
    "Verð útflutts áls","Olíuverð",
]

MONTHS = {
    "jan":1,"feb":2,"februar":2,"mars":3,"apr":4,"april":4,
    "mai":5,"maí":5,"jun":6,"juni":6,"júní":6,"juli":7,"júlí":7,
    "aug":8,"sep":9,"sept":9,"okt":10,"oktober":10,"nov":11,"nóv":11,"nóvember":11,
}
def fold_ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii").lower()

def infer_meta(df: pd.DataFrame) -> pd.DataFrame:
    if {"forecast_id","forecast_month","forecast_season"}.issubset(df.columns):
        return df
    base = df.copy()
    base["forecast_month"] = pd.NA
    base["forecast_season"] = pd.NA
    base["forecast_id"] = pd.NA
    for i, r in base.iterrows():
        fname = (r.get("source_pdf_file") or "")
        stem = fold_ascii(Path(fname).stem)
        month = None
        for k,m in MONTHS.items():
            if re.search(rf"\b{k}\b", stem): month = m; break
        season = None
        if re.search(r"\b(vetur|haust)\b", stem): season = "vetur/haust"
        elif re.search(r"\b(sumar|vor)\b", stem): season = "sumar/vor"
        if season is None and month is not None:
            season = "vetur/haust" if month in (10,11,12,1,2) else "sumar/vor"
        ty = r.get("target_year")
        fid = f"{int(ty) if pd.notnull(ty) else 'NA'}_{month or 'xx'}"
        base.at[i,"forecast_month"] = month
        base.at[i,"forecast_season"] = season
        base.at[i,"forecast_id"] = fid
    return base

def show_year(csv_path: Path, year: int, which: str = None, source: str = None):
    df = pd.read_csv(csv_path)
    for c in ("target_year","table_year","value","forecast_month"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = infer_meta(df)
    sub = df[df["target_year"] == year].copy()
    if which:
        sub = sub[sub["forecast_season"] == which]
    if source:
        sub = sub[sub["source_pdf_file"].str.contains(source, case=False, na=False)]
    if sub.empty:
        print(f"No data found for target_year={year}" + (f", season={which}" if which else "") + (f", source~={source}" if source else ""))
        return
    groups = list(sub.groupby(["forecast_id","source_pdf_file","forecast_season","forecast_month"], dropna=False))
    groups.sort(key=lambda item: ((item[0][3] if pd.notnull(item[0][3]) else 99), item[0][0]))
    for key, g in groups:
        fid, src, season, month = key
        pivot = g.pivot_table(index="indicator_is", columns="table_year", values="value", aggfunc="first")
        pivot = pivot.reindex(ORDER)
        cols = sorted([c for c in pivot.columns if pd.notnull(c)])
        pivot = pivot[cols]
        print(f"\n=== Þjóðhagsspá {year} — {season or '–'} (mánuður: {int(month) if pd.notnull(month) else '–'}) ===")
        print(f"source: {src}")
        print(pivot.fillna("–").to_string(float_format=lambda v: f"{v:,.2f}"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("--csv", default="data/derived/thjodhagsspa_targets.csv")
    ap.add_argument("--which", choices=["sumar/vor","vetur/haust"], help="Filter by season")
    ap.add_argument("--source", help="Substring to pick a specific PDF")
    args = ap.parse_args()
    show_year(Path(args.csv), args.year, args.which, args.source)
