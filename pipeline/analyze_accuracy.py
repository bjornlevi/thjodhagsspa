#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze SPA-only forecast accuracy (no PX needed).

Logic:
- observations table has: pdf_name, label_key, year, value, type ∈ {history, forecast, extrapolation}
- Parse an edition ordering from pdf_name (year + month/season).
- "Realized" value for (label_key, year) = the value from the latest edition where type=='history'.
- Join all earlier forecast/extrapolation rows for the same (label_key, year) to realized.
- Compute error = forecast - realized, horizon = year - forecast_year (forecast_year parsed from pdf_name).

Outputs (data/analysis/):
- spa_joined_rows.csv                   (row-level joins)
- spa_metrics_by_label_horizon.csv      (RMSE/MAE/etc by label & horizon)
- spa_metrics_overall_horizon.csv       (… aggregated over labels)
- plots in data/analysis/plots/
"""

import re
import os
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

# --------- paths ---------
DB_PATH   = Path("data/spa.sqlite3")
OUT_DIR   = Path("data/analysis")
PLOTDIR   = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTDIR.mkdir(parents=True, exist_ok=True)

# --------- edition parsing from pdf_name ---------
# expected names like: spa_2019_februar_vetur_endurskodun
RE_YEAR = re.compile(r"spa_(\d{4})")
MONTHS_IS = {
    "jan":1,"janúar":1,"janvar":1,"janua":1,"januar":1,
    "feb":2,"febrúar":2,"februar":2,
    "mar":3,"mars":3,
    "apr":4,"apríl":4,"april":4,
    "maí":5,"mai":5,"mae":5,
    "jún":6,"juni":6,"júní":6,"juni":6,
    "júl":7,"juli":7,"júlí":7,
    "ágú":8,"agu":8,"ágúst":8,"agust":8,
    "sep":9,"september":9,
    "okt":10,"október":10,"oktober":10,
    "nóv":11,"nov":11,"nóvember":11,"november":11,
    "des":12,"desember":12
}
SEASON_HINTS = {
    "vor": 3,       # spring ≈ Mar
    "sumar": 7,     # summer ≈ Jul
    "haust": 10,    # autumn ≈ Oct
    "vetur": 1      # winter ≈ Jan
}

def safe_lower(s: str) -> str:
    try:
        return s.lower()
    except Exception:
        return s

def month_from_name(name: str) -> int | None:
    n = safe_lower(name)
    # try full token
    if n in MONTHS_IS:
        return MONTHS_IS[n]
    # try prefix matching
    for k,v in MONTHS_IS.items():
        if n.startswith(k):
            return v
    return None

def season_guess(tokens: list[str]) -> int | None:
    for t in tokens:
        tl = safe_lower(t)
        if tl in SEASON_HINTS:
            return SEASON_HINTS[tl]
    return None

def parse_edition_tuple(pdf_name: str) -> tuple[int,int,int,str]:
    """
    Return (year, month_guess, season_rank, full_name) to order editions chronologically.
    month_guess/season_rank are best-effort (0 if unknown) so sorting is stable enough.
    """
    toks = re.split(r"[_\W]+", pdf_name)
    year = 0
    m = RE_YEAR.search(pdf_name)
    if m:
        year = int(m.group(1))
    # try to find a month token
    mg = 0
    for t in toks:
        mm = month_from_name(t)
        if mm:
            mg = mm
            break
    # season fallback
    sr = season_guess(toks) or 0
    return (year, mg, sr, pdf_name)

def forecast_year_from_pdf(pdf_name: str) -> int | None:
    m = RE_YEAR.search(pdf_name)
    return int(m.group(1)) if m else None

# --------- metrics helper (robust) ---------
def compute_metrics(df: pd.DataFrame, by):
    print("[debug] compute_metrics got cols:", list(df.columns))

    d = df.copy()

    # Build error on the fly if needed
    if "error" not in d.columns:
        if {"pred_value", "realized_value"} <= set(d.columns):
            d["pred_value"]     = pd.to_numeric(d["pred_value"], errors="coerce")
            d["realized_value"] = pd.to_numeric(d["realized_value"], errors="coerce")
            d["error"]          = d["pred_value"] - d["realized_value"]
        else:
            raise KeyError(f"compute_metrics: need 'error' or both 'pred_value' and 'realized_value'. "
                           f"Got cols={list(d.columns)}")

    d = d.dropna(subset=["error"])
    if d.empty:
        return pd.DataFrame(columns=[*by, "n","mae","rmse","bias","p50","p95"])

    d["abs_err"] = d["error"].abs()
    g = d.groupby(list(by), dropna=False)

    out = g.agg(
        n    = ("error", "size"),
        mae  = ("abs_err", "mean"),
        rmse = ("error", lambda s: float(np.sqrt(np.mean(np.square(s))))),
        bias = ("error", "mean"),
        p50  = ("abs_err", "median"),
        p95  = ("abs_err", lambda s: s.quantile(0.95)),
    ).reset_index()

    if "horizon" in out.columns:
        # keep horizon ascending within other keys
        out = out.sort_values([*(k for k in by if k != "horizon"), "horizon"])
    return out

# ---------- plotting ----------

def plot_error_vs_horizon_metrics(df_metrics: pd.DataFrame, fn: Path):
    """
    Expects aggregated metrics with columns at least:
      ['horizon', 'mae', 'rmse', 'bias', 'n']
    """
    if plt is None or df_metrics is None or df_metrics.empty:
        print("[i] Skipping plot_error_vs_horizon_metrics: no data")
        return

    d = df_metrics.copy()
    # ensure numeric + sort
    for c in ("horizon","mae","rmse","bias","n"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["horizon"]).sort_values("horizon")

    fig, ax = plt.subplots(figsize=(7,4.5))
    # Plot MAE and RMSE; bias as thin line around 0 for reference
    if "mae" in d.columns:
        ax.plot(d["horizon"], d["mae"], marker="o", label="MAE")
    if "rmse" in d.columns:
        ax.plot(d["horizon"], d["rmse"], marker="s", label="RMSE")
    if "bias" in d.columns:
        ax.plot(d["horizon"], d["bias"], linestyle="--", alpha=0.6, label="Bias (mean error)")

    ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Horf (ár frá spáárinu)")        # Horizon (years from forecast year)
    ax.set_ylabel("Villa (prósentustig)")          # Error (percentage points)
    ax.set_title("Villa vs. horf (SPA)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fn, dpi=150)
    plt.close(fig)


def plot_calibration(df_eval: pd.DataFrame, fn: Path):
    """
    Hexbin calibration plot: realized vs predicted.
    Colors = point density (count per hex).
    """
    if plt is None or df_eval is None or df_eval.empty:
        print("[i] Skipping plot_calibration: no data")
        return

    d = df_eval.copy()
    for c in ("pred_value","realized_value"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["pred_value","realized_value"])

    import numpy as np
    x = d["pred_value"].to_numpy()
    y = d["realized_value"].to_numpy()

    # symmetric square limits around the diagonal
    lo = np.nanmin([x.min(), y.min()])
    hi = np.nanmax([x.max(), y.max()])
    pad = 0.04 * (hi - lo if np.isfinite(hi - lo) and hi != lo else 1.0)
    lo2, hi2 = lo - pad, hi + pad

    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    # HEXBIN — this *replaces* the scatter
    hb = ax.hexbin(
        x, y,
        gridsize=55,            # increase/decrease to taste
        cmap="viridis",         # visible cmap
        mincnt=1,               # hide empty bins
        linewidths=0,           # clean look
        extent=(lo2, hi2, lo2, hi2)  # fill the square
    )
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Fjöldi punkta (þéttleiki)")

    # identity line and ± bands
    ax.plot([lo2, hi2], [lo2, hi2], color="crimson", lw=2, label="y = x (full hit)")
    for band, ls in [(5, ":"), (10, "--")]:
        ax.plot([lo2, hi2], [lo2 + band, hi2 + band], color="gray", lw=1.0, ls=ls, alpha=0.7)
        ax.plot([lo2, hi2], [lo2 - band, hi2 - band], color="gray", lw=1.0, ls=ls, alpha=0.7)

    ax.set_xlim(lo2, hi2)
    ax.set_ylim(lo2, hi2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Spáð / framreiknað (predicted)")
    ax.set_ylabel("Raunmælt (realized)")
    ax.set_title("Kalibrering: spá vs raun (hexbin)")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(fn, dpi=150)
    plt.close(fig)


def ensure_eval_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the frame has columns:
    ['pred_value','realized_value','horizon'] (plus others we already carry).
    If 'error' is missing, compute it.
    """
    d = df.copy()

    # Coerce numerics if present
    for col in ("pred_value", "realized_value", "horizon"):
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    # Compute error if absent
    if "error" not in d.columns:
        if {"pred_value", "realized_value"} <= set(d.columns):
            d["error"] = d["pred_value"] - d["realized_value"]

    # Validate required columns for metrics
    required = {"pred_value", "realized_value", "horizon"}
    missing = required - set(d.columns)
    if missing:
        raise KeyError(f"ensure_eval_frame: missing {sorted(missing)} in columns={list(d.columns)}")

    return d

# ---------- Error plots ----------

def _prep_eval(df):
    d = df.copy()
    for c in ("pred_value","realized_value","horizon","error"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["pred_value","realized_value","horizon","error"])
    return d

def plot_error_hist(df, fn: Path):
    d = _prep_eval(df)
    if plt is None or d.empty:
        print("[i] Skipping plot_error_hist: no data"); return
    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(d["error"], bins=50, alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Villa (spáð − raun)")
    ax.set_ylabel("Fjöldi")
    ax.set_title("Dreifing villu")
    fig.tight_layout(); fig.savefig(fn, dpi=150); plt.close(fig)

def plot_bias_mae_vs_horizon(df, fn: Path):
    d = _prep_eval(df)
    if plt is None or d.empty:
        print("[i] Skipping plot_bias_mae_vs_horizon: no data"); return
    g = d.groupby("horizon")["error"]
    bias = g.mean()
    mae  = g.apply(lambda x: x.abs().mean())
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(bias.index, bias.values, marker="o", label="Meðalbias")
    ax.plot(mae.index,  mae.values,  marker="s", label="MAE")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Frá spá (ár)"); ax.set_ylabel("Gildi")
    ax.set_title("Bias og MAE eftir horni")
    ax.legend()
    fig.tight_layout(); fig.savefig(fn, dpi=150); plt.close(fig)

def plot_error_by_horizon_box(df, fn: Path):
    d = _prep_eval(df)
    if plt is None or d.empty:
        print("[i] Skipping plot_error_by_horizon_box: no data"); return
    # prepare data per horizon
    horizons = sorted(d["horizon"].unique())
    data = [d.loc[d["horizon"]==h, "error"].values for h in horizons]
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.boxplot(data, labels=horizons, showfliers=False)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Frá spá (ár)"); ax.set_ylabel("Villa (spáð − raun)")
    ax.set_title("Villudreifing eftir horni (boxplot)")
    fig.tight_layout(); fig.savefig(fn, dpi=150); plt.close(fig)

def plot_abs_error_vs_level(df, fn: Path):
    d = _prep_eval(df)
    if plt is None or d.empty:
        print("[i] Skipping plot_abs_error_vs_level: no data"); return
    fig, ax = plt.subplots(figsize=(6.5,6))
    ax.scatter(d["pred_value"], d["error"].abs(), s=10, alpha=0.5)
    ax.set_xlabel("Spáð gildi"); ax.set_ylabel("|Villa|")
    ax.set_title("Stærð villu m.t.t. spágildis")
    fig.tight_layout(); fig.savefig(fn, dpi=150); plt.close(fig)

def plot_error_qq(df, fn: Path):
    d = _prep_eval(df)
    if plt is None or d.empty:
        print("[i] Skipping plot_error_qq: no data"); return
    import scipy.stats as st
    fig, ax = plt.subplots(figsize=(6,6))
    st.probplot(d["error"], dist="norm", plot=ax)
    ax.set_title("Q–Q plot af villu")
    fig.tight_layout(); fig.savefig(fn, dpi=150); plt.close(fig)


# --------- plotting (optional; skip if matplotlib missing) ---------
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

def plot_error_vs_horizon(metrics_overall: pd.DataFrame, fn: Path):
    if plt is None or metrics_overall is None or metrics_overall.empty:
        print("[i] Skipping plot_error_vs_horizon: no data")
        return
    fig, ax = plt.subplots(figsize=(7,4))
    m = metrics_overall.sort_values("horizon")
    ax.plot(m["horizon"], m["rmse"], marker="o", label="RMSE")
    ax.plot(m["horizon"], m["mae"], marker="s", label="MAE")
    ax.axhline(0, lw=0.8, color="gray")
    ax.set_xlabel("Horizon (years ahead)")
    ax.set_ylabel("Error magnitude")
    ax.set_title("SPA-only forecast error vs horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fn, dpi=150)
    plt.close(fig)

def plot_calibration(df: pd.DataFrame, fn: Path):
    if plt is None or df is None or df.empty:
        print("[i] Skipping plot_calibration: no data")
        return

    # numeric + finite only
    x = pd.to_numeric(df["pred_value"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["realized_value"], errors="coerce").to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        print("[i] Skipping plot_calibration: no finite data")
        return

    # unified limits from data
    vmin = float(np.min(np.r_[x, y]))
    vmax = float(np.max(np.r_[x, y]))
    span = vmax - vmin
    if span <= 0 or not np.isfinite(span):
        # degenerate: force a small span around vmin
        span = 1.0
    pad = 0.07 * span
    lo, hi = vmin - pad, vmax + pad
    # print(f"[debug] calib limits: lo={lo:.3g}, hi={hi:.3g}")

    fig, ax = plt.subplots(figsize=(6, 6))

    # points below the line (so line is visible)
    ax.scatter(x, y, s=16, alpha=0.55, zorder=2)

    # identity line y = x on top, bold & bright
    ax.plot([lo, hi], [lo, hi],
            linestyle="-", linewidth=2.0, color="#d62728",  # red
            zorder=3, label="y = x (perfect)")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_xlabel("Predicted (forecast / extrapolation)")
    ax.set_ylabel("Realized (latest SPA history)")
    ax.set_title("Calibration: predicted vs realized (SPA-only)")
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    fig.savefig(fn, dpi=150, facecolor="white")
    plt.close(fig)

# ---------- small utils ----------
def _safe_slug(s: str) -> str:
    return (
        s.lower()
         .replace(" ", "-")
         .replace("/", "-")
         .replace("%", "pct")
         .replace("ð", "d")
         .replace("þ", "th")
         .replace("á", "a").replace("é","e").replace("í","i")
         .replace("ó", "o").replace("ú", "u").replace("ý","y")
         .replace("æ","ae").replace("ö","o")
    )

def _label_map_from_db(conn) -> dict:
    # prefer Icelandic pretty names for titles
    cur = conn.cursor()
    cur.execute("SELECT label_key, isl FROM canon;")
    return {row[0]: row[1] for row in cur.fetchall()}

# ---------- plotting helpers (pure matplotlib) ----------
def plot_calibration(df_eval: pd.DataFrame, fn: Path):
    """
    Hexbin calibration plot: realized vs predicted.
    Colors = point density (count per hex).
    """
    if plt is None or df_eval is None or df_eval.empty:
        print("[i] Skipping plot_calibration: no data")
        return

    d = df_eval.copy()
    for c in ("pred_value","realized_value"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["pred_value","realized_value"])

    import numpy as np
    x = d["pred_value"].to_numpy()
    y = d["realized_value"].to_numpy()

    # symmetric square limits around the diagonal
    lo = np.nanmin([x.min(), y.min()])
    hi = np.nanmax([x.max(), y.max()])
    pad = 0.04 * (hi - lo if np.isfinite(hi - lo) and hi != lo else 1.0)
    lo2, hi2 = lo - pad, hi + pad

    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    # HEXBIN — this *replaces* the scatter
    hb = ax.hexbin(
        x, y,
        gridsize=55,            # increase/decrease to taste
        cmap="viridis",         # visible cmap
        mincnt=1,               # hide empty bins
        linewidths=0,           # clean look
        extent=(lo2, hi2, lo2, hi2)  # fill the square
    )
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Fjöldi punkta (þéttleiki)")

    # identity line and ± bands
    ax.plot([lo2, hi2], [lo2, hi2], color="crimson", lw=2, label="y = x (full hit)")
    for band, ls in [(5, ":"), (10, "--")]:
        ax.plot([lo2, hi2], [lo2 + band, hi2 + band], color="gray", lw=1.0, ls=ls, alpha=0.7)
        ax.plot([lo2, hi2], [lo2 - band, hi2 - band], color="gray", lw=1.0, ls=ls, alpha=0.7)

    ax.set_xlim(lo2, hi2)
    ax.set_ylim(lo2, hi2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Spáð / framreiknað (predicted)")
    ax.set_ylabel("Raunmælt (realized)")
    ax.set_title("Kalibrering: spá vs raun (hexbin)")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(fn, dpi=150)
    plt.close(fig)


def plot_error_vs_horizon(df: pd.DataFrame, fn: Path, title: str = None):
    """Mean absolute error by horizon (0 = forecast year, 1 = +1, …)."""
    if plt is None or df is None or df.empty:
        print(f"[i] Skipping plot_error_vs_horizon: no data for {fn}")
        return
    d = df.copy()
    d = d[np.isfinite(pd.to_numeric(d["pred_value"], errors="coerce")) &
          np.isfinite(pd.to_numeric(d["realized_value"], errors="coerce"))]
    if d.empty or "horizon" not in d.columns:
        print(f"[i] Skipping plot_error_vs_horizon: missing horizon/data for {fn}")
        return
    d["abs_err"] = (pd.to_numeric(d["pred_value"], errors="coerce") -
                    pd.to_numeric(d["realized_value"], errors="coerce")).abs()
    g = d.groupby("horizon", as_index=False).agg(
        mae=("abs_err", "mean"),
        n=("abs_err", "size"),
        p95=("abs_err", lambda s: s.quantile(0.95)),
        p50=("abs_err", "median")
    ).sort_values("horizon")

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(g["horizon"], g["mae"], marker="o")
    # light band up to 95th percentile for context
    ax.fill_between(g["horizon"], g["p50"], g["p95"], alpha=0.15, step="mid")
    for xi, yi, ni in zip(g["horizon"], g["mae"], g["n"]):
        ax.annotate(str(int(ni)), (xi, yi), textcoords="offset points", xytext=(0,6),
                    ha="center", fontsize=8, alpha=0.8)
    ax.set_xlabel("Framreikningsbil (ár frá spá)")
    ax.set_ylabel("Meðal algild villa (MAE)")
    ax.set_title(title or "Villa vs framreikningsbil")
    ax.grid(True, lw=0.4, alpha=0.5)
    fig.tight_layout()
    fn.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fn, dpi=150, facecolor="white")
    plt.close(fig)

# ---------- main per-label driver ----------
def run_label_plots(conn, df_eval: pd.DataFrame, plotdir: Path):
    """
    For each label_key in df_eval:
      - calibration scatter (pred vs realized)
      - MAE vs horizon
    Saves to plots/labels/<slug>/...
    """
    if df_eval is None or df_eval.empty:
        print("[i] No evaluation data to plot per-label.")
        return
    pretty = _label_map_from_db(conn)
    outroot = Path(plotdir) / "labels"

    keys = sorted(df_eval["label_key"].dropna().unique())
    for k in keys:
        sub = df_eval[df_eval["label_key"] == k].copy()
        if sub.empty:
            continue
        title = pretty.get(k, k)
        slug = _safe_slug(title or k)
        subdir = outroot / slug
        # plots
        plot_calibration(sub, subdir / f"{slug}_calibration.png", title=f"{title} — Kalibrering")
        plot_error_vs_horizon(sub, subdir / f"{slug}_error_vs_horizon.png", title=f"{title} — Villa vs bil")
        print(f"[✓] Plotted {k} → {subdir}")

# --------- main ---------
def main():
    if not DB_PATH.exists():
        raise SystemExit(f"[!] DB not found: {DB_PATH}. Run `make db` first.")

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    # ---- Load SPA observations ----
    obs = pd.read_sql_query(
        "SELECT pdf_name, label_key, year, value, type FROM observations;",
        con
    )
    if obs.empty:
        print("[!] observations is empty. Run pipeline + parse + db first.")
        return

    # Edition metadata + forecast year
    obs[["edit_year","edit_month","edit_season","_nm"]] = (
        obs["pdf_name"].apply(lambda s: pd.Series(parse_edition_tuple(s)))
    )
    obs.drop(columns=["_nm"], inplace=True)
    obs["forecast_year"] = obs["pdf_name"].apply(forecast_year_from_pdf)

    # ---- Realized values (latest 'history' per label/year) ----
    hist = obs[obs["type"] == "history"].copy()
    if hist.empty:
        print("[!] No history rows in observations; cannot build realized values.")
        return

    hist["edition_order"] = hist[["edit_year","edit_month","edit_season"]].apply(
        lambda r: (int(r["edit_year"]), int(r["edit_month"]), int(r["edit_season"])),
        axis=1
    )
    idx_latest = hist.groupby(["label_key","year"])["edition_order"].idxmax()
    realized = (
        hist.loc[idx_latest, ["label_key","year","value"]]
            .rename(columns={"value":"realized_value"})
            .reset_index(drop=True)
    )

    # ---- Forecast/extrapolation candidates ----
    fx = obs[obs["type"].isin(["forecast","extrapolation"])].copy()
    if fx.empty:
        print("[!] No forecast/extrapolation rows found; nothing to evaluate.")
        return
    fx["pred_value"] = fx["value"]

    # ---- Join on (label_key, year) ----
    joined = pd.merge(
        fx,
        realized,
        on=["label_key","year"],
        how="inner",
        validate="many_to_one"
    )
    if joined.empty:
        print("[!] Join produced no rows (no overlap of years).")
        OUT_DIR.mkdir(exist_ok=True, parents=True)
        pd.DataFrame().to_csv(OUT_DIR / "spa_joined_rows.csv", index=False)
        return

    # Horizon & error
    joined["horizon"] = joined["year"].astype(int) - joined["forecast_year"].astype(int)
    joined["pred_value"]     = pd.to_numeric(joined["pred_value"], errors="coerce")
    joined["realized_value"] = pd.to_numeric(joined["realized_value"], errors="coerce")
    joined["error"]          = joined["pred_value"] - joined["realized_value"]

    # Save the raw joined rows for transparency
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    cols_joined = ["pdf_name","label_key","year","type","forecast_year",
                   "horizon","pred_value","realized_value","error"]
    joined.loc[:, cols_joined].to_csv(OUT_DIR / "spa_joined_rows.csv", index=False)

    # ---- Build evaluation frame (this is what metrics consume) ----
    df_eval = joined.loc[:, [
        "label_key","pdf_name","year",
        "pred_value","realized_value",
        "horizon","type","forecast_year","error"
    ]].copy()

    # Debug peek
    print("[debug] df_eval cols:", list(df_eval.columns))
    print("[debug] head:\n", df_eval.head(3))

    # Ensure required columns and numeric types
    df_eval = ensure_eval_frame(df_eval)

    # ---- Metrics ----
    m_label_h   = compute_metrics(df_eval, by=["label_key","horizon"])
    m_overall_h = compute_metrics(df_eval, by=["horizon"])

    m_label_h.to_csv(OUT_DIR / "spa_metrics_by_label_horizon.csv", index=False)
    m_overall_h.to_csv(OUT_DIR / "spa_metrics_overall_horizon.csv", index=False)

    # ---- Plots ----
    plot_error_vs_horizon_metrics(m_overall_h, PLOTDIR / "spa_error_vs_horizon.png")
    plot_calibration(df_eval, PLOTDIR / "spa_calibration.png")

    PLOTDIR.mkdir(parents=True, exist_ok=True)

    plot_error_hist(joined, PLOTDIR / "spa_error_hist.png")
    plot_bias_mae_vs_horizon(joined, PLOTDIR / "spa_bias_mae_vs_horizon.png")
    plot_error_by_horizon_box(joined, PLOTDIR / "spa_error_by_horizon_box.png")
    plot_abs_error_vs_level(joined, PLOTDIR / "spa_abs_error_vs_level.png")
    plot_error_qq(joined, PLOTDIR / "spa_error_qq.png")

    print("[✓] Wrote:")
    print("   - data/analysis/spa_joined_rows.csv")
    print("   - data/analysis/spa_metrics_by_label_horizon.csv")
    print("   - data/analysis/spa_metrics_overall_horizon.csv")
    print("   - data/analysis/plots/spa_error_vs_horizon.png")
    print("   - data/analysis/plots/spa_calibration.png")


if __name__ == "__main__":
    main()
