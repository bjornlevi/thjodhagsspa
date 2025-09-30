# pipeline/extract_year.py
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd

def show_year(csv_path: Path, year: int):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Filter for chosen forecast year
    sub = df[df["target_year"] == year].copy()

    if sub.empty:
        print(f"No data found for target_year = {year}")
        return

    # Sort for neat display: indicator order, then table year ascending
    sub.sort_values(
        by=["indicator_is","table_year"], 
        inplace=True,
        key=lambda col: col.map(lambda x: x if pd.notnull(x) else 0)
    )

    # Pivot to wide: rows = indicator, cols = table_year, values = value
    pivot = sub.pivot_table(
        index=["indicator_is"],
        columns="table_year",
        values="value",
        aggfunc="first"
    )

    # Keep also status
    # For readability we print only the numeric values
    pivot = pivot.sort_index()

    order = [
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
    pivot = pivot.reindex(order)  # force all rows to appear

    print(f"\n=== Þjóðhagsspá {year} ===")
    print(pivot.fillna("–").to_string(float_format=lambda v: f"{v:,.2f}"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int, help="Target forecast year to display")
    ap.add_argument("--csv", default="data/derived/thjodhagsspa_targets.csv",
                    help="Path to extracted CSV")
    args = ap.parse_args()

    show_year(Path(args.csv), args.year)
