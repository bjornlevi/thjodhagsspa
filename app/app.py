import re
import sqlite3
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for

APP_ROOT = Path(__file__).resolve().parent
DB_PATH = APP_ROOT.parent / "data" / "spa.sqlite3"
DEFAULT_LABEL_KEY = "vlf a verdlagi hvers ars"  # VLF á verðlagi hvers árs, ma. kr.

app = Flask(__name__)

# --- Pretty labels for SPA editions -----------------------------------------
_MONTH_MAP = {
    "jan": "Janúar", "janúar": "Janúar", "januar": "Janúar",
    "feb": "Febrúar", "febrúar": "Febrúar", "februar": "Febrúar",
    "mar": "Mars", "mars": "Mars",
    "apr": "Apríl", "april": "Apríl",
    "mai": "Maí", "maí": "Maí",
    "jun": "Júní", "júní": "Júní", "juni": "Júní",
    "jul": "Júlí", "júlí": "Júlí", "juli": "Júlí",
    "ago": "Ágúst", "ágúst": "Ágúst", "agust": "Ágúst",
    "sep": "September",
    "okt": "Október", "október": "Október", "oktober": "Október",
    "nov": "Nóvember", "nóvember": "Nóvember", "november": "Nóvember",
    "des": "Desember", "desember": "Desember",
}

_MONTH_ORDER = {
    "Janúar": 1, "Febrúar": 2, "Mars": 3, "Apríl": 4, "Maí": 5, "Júní": 6,
    "Júlí": 7, "Ágúst": 8, "September": 9, "Október": 10, "Nóvember": 11, "Desember": 12
}

_SEASON_MAP = {
    "vor": "Vor",
    "sumar": "Sumar",
    "vetur": "Vetur",
    "haust": "Haust",
    "greining": "greining",
    "endurskodun": "endurskoðun",  # normalize without accent in filenames → with accent in label
}

_SPAREGEX = re.compile(
    r"""^spa_
        (?P<year>\d{4})
        (?:_(?P<month>[a-záðéíóúýþæö]+))?
        (?:_(?P<season1>vor|sumar|vetur|haust|greining|endurskodun))?
        (?:_(?P<season2>vor|sumar|vetur|haust))?
        $""",
    re.IGNORECASE | re.VERBOSE
)

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def format_spa_label(pdf_name: str) -> str:
    """
    spa_2024_november_vetur_haust → 'Nóvember 2024 · Vetur (haust)'
    spa_2019_februar_vetur_endurskodun → 'Febrúar 2019 · Vetur (endurskoðun)'
    spa_2012_juli_sumar → 'Júlí 2012 · Sumar'
    spa_2024_april_vor_greining → 'Apríl 2024 · Vor (greining)'
    fallback: 'YYYY' if parse fails.
    """
    stem = pdf_name.rsplit("/", 1)[-1]
    stem = stem.replace(".pdf", "")
    m = _SPAREGEX.match(stem)
    if not m:
        # loose fallback: try to find a year
        year = re.search(r"(\d{4})", stem)
        return year.group(1) if year else stem

    year = m.group("year")
    month_raw = _norm(m.group("month"))
    season1_raw = _norm(m.group("season1"))
    season2_raw = _norm(m.group("season2"))

    month = _MONTH_MAP.get(month_raw, None) if month_raw else None
    s1 = _SEASON_MAP.get(season1_raw, None)
    s2 = _SEASON_MAP.get(season2_raw, None)

    # Build “Month Year · Season (subseason)”
    left = f"{month} {year}" if month else year
    if s1 and s2:
        right = f"{s1} ({s2})"
    elif s1:
        right = s1
    else:
        right = None

    return f"{left} · {right}" if right else left

def spa_sort_key(pdf_name: str):
    """Sort by (year, month) for consistent latest-first lists."""
    stem = pdf_name.rsplit("/", 1)[-1].replace(".pdf", "")
    m = _SPAREGEX.match(stem)
    if not m:
        yr = re.search(r"(\d{4})", stem)
        return (int(yr.group(1)) if yr else 0, 0)
    year = int(m.group("year"))
    month_raw = _norm(m.group("month"))
    month = _MONTH_MAP.get(month_raw, None) if month_raw else None
    m_order = _MONTH_ORDER.get(month, 0)
    return (year, m_order)

# Make available to Jinja
app.jinja_env.filters['spa_label'] = format_spa_label


# --- DB helper
def q(sql, params=()):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(sql, params)
        rows = cur.fetchall()
    finally:
        con.close()
    return rows

def execsql(sql, params=()):
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute(sql, params)
        con.commit()
    finally:
        con.close()

# --- helpers
YEAR_RE = re.compile(r"spa_(\d{4})")

def forecast_year_from_pdf(pdf_name: str) -> int | None:
    """
    Extract the forecast year from a file stem like 'spa_2024_november_vetur_haust'.
    Returns int or None.
    """
    m = re.search(r"spa_(\d{4})", pdf_name)
    return int(m.group(1)) if m else None

def label_list():
    # list available labels (only those with data)
    return q("""
        SELECT c.label_key AS key, c.isl, c.eng
        FROM canon c
        WHERE EXISTS (SELECT 1 FROM spa_obs o WHERE o.label_key = c.label_key)
        ORDER BY c.isl COLLATE NOCASE
    """)

# --- label choices & defaults ---
def get_default_label_key() -> str:
    # VLF á verðlagi hvers árs, ma. kr.
    row = q("SELECT label_key FROM canon WHERE label_key LIKE 'vlf a verdlagi hvers ars' LIMIT 1")
    return row[0]["label_key"] if row else q("SELECT label_key FROM canon ORDER BY label_key LIMIT 1")[0]["label_key"]

def label_choices():
    # [(key, "isl — eng"), ...]
    rows = q("SELECT label_key, isl, eng FROM canon ORDER BY isl")
    return [(r["label_key"], f'{r["isl"]} — {r["eng"]}') for r in rows]

def parse_recent(n_str):
    if n_str is None: 
        return 3
    try:
        n = int(n_str)
        return max(1, n)
    except:
        return 3

def build_series(label_key: str):
    rows = q("""
        SELECT pdf_name, label_key, year, value, type
        FROM spa_obs
        WHERE label_key = ?
        ORDER BY year, pdf_name
    """, (label_key,))

    # group rows by pdf, compute forecast_year per pdf
    by_pdf = {}
    for r in rows:
        fy = forecast_year_from_pdf(r["pdf_name"])
        if fy is None:
            continue
        by_pdf.setdefault((r["pdf_name"], fy), []).append(r)

    # HISTORY (absolute years): pick the most recent vintage per year
    history_latest = {}
    for (pdf, fy), rs in by_pdf.items():
        for r in rs:
            if r["type"] != "history":
                continue
            y, v = int(r["year"]), float(r["value"])
            # choose the history from the LATEST forecast vintage <= that year
            prev = history_latest.get(y)
            if (prev is None) or (fy > prev["fy"]) and (fy <= y):
                history_latest[y] = {"fy": fy, "value": v}
    history_xy = sorted([(y, d["value"]) for y, d in history_latest.items()])

    # FORECAST point per vintage (x=forecast_year, y=value where type='forecast')
    forecast_xy = []
    for (pdf, fy), rs in by_pdf.items():
        for r in rs:
            if r["type"] == "forecast" and int(r["year"]) == fy:
                forecast_xy.append((fy, float(r["value"])))
                break
    forecast_xy.sort()

    # EXTRAPOLATIONS stitched by offset k: x=forecast_year, y=value at fy+k
    # Build per-vintage dict year->value for extrapolation
    ladders = {}  # offset k -> list of (fy, value)
    for (pdf, fy), rs in by_pdf.items():
        vals = {}
        for r in rs:
            if r["type"] == "extrapolation":
                vals[int(r["year"])] = float(r["value"])
        for y, v in vals.items():
            k = y - fy
            if k <= 0:
                continue
            ladders.setdefault(k, []).append((fy, v))

    # sort each ladder by forecast year
    ladders_sorted = []
    for k, pts in sorted(ladders.items()):
        pts.sort()
        ladders_sorted.append({"k": k, "points": pts})

    return {
        "history": history_xy,           # list[(year, value)]
        "forecast": forecast_xy,         # list[(forecast_year, value)]
        "ladders": ladders_sorted        # list[{k, points:[(forecast_year, value)]}]
    }


def build_abs_datasets(label_key: str, recent_n: int = 3):
    """
    Build absolute-year Chart.js datasets for a single canon label.
    Returns (datasets, x_min, x_max).
    """
    rows = q("""
        SELECT pdf_name, year, value, type
        FROM spa_obs
        WHERE label_key = ?
        ORDER BY pdf_name, year
    """, (label_key,))
    if not rows:
        return [], None, None

    # Group rows by PDF and capture forecast year (fy) from file name
    by_pdf: dict[str, dict] = {}
    for r in rows:
        pdf = r["pdf_name"]
        fy = forecast_year_from_pdf(pdf)
        if fy is None:
            # skip PDFs we can’t parse a forecast year from
            continue
        ent = by_pdf.setdefault(pdf, {"fy": fy, "rows": []})
        ent["rows"].append(r)

    if not by_pdf:
        return [], None, None

    # Order editions by forecast year; keep only the last N if requested
    editions = sorted(by_pdf.items(), key=lambda kv: kv[1]["fy"])
    if recent_n is not None:
        editions = editions[-recent_n:]

    datasets, x_min, x_max = [], None, None

    for pdf, meta in editions:
        pts = []
        for r in meta["rows"]:
            yr = int(r["year"])
            val = float(r["value"])
            typ = r["type"]  # history/forecast/extrapolation

            pts.append({
                "x": yr,        # numeric for linear axis
                "abs_year": yr,     # integer year for tooltip title
                "y": val,
                "t": typ            # marker style & dashed segments in JS
            })

            x_min = yr if x_min is None else min(x_min, yr)
            x_max = yr if x_max is None else max(x_max, yr)

        # Sort points by x then by type, so lines render left→right nicely
        pts.sort(key=lambda p: (p["x"], p["t"]))

        datasets.append({
            "label": format_spa_label(pdf),
            "data": pts,
            "showLine": True,
            "fill": False,
            "tension": 0.1
            # segment dash/width & point styles are attached in JS
        })

    return datasets, x_min, x_max

def build_rel_datasets(label_key: str, recent_n: int = 3):
    """
    Return (datasets, x_min, x_max) for 'útgáfuár' view.
    x is relative year: t = year - forecast_year, with small type-based jitter.
    """
    rows = q("""
        SELECT pdf_name, year, value, type
        FROM spa_obs
        WHERE label_key = ?
        ORDER BY pdf_name, year
    """, (label_key,))
    if not rows:
        return [], None, None

    by_pdf: dict[str, dict] = {}
    for r in rows:
        pdf = r["pdf_name"]
        fy = forecast_year_from_pdf(pdf)
        if fy is None:
            continue
        ent = by_pdf.setdefault(pdf, {"fy": fy, "rows": []})
        ent["rows"].append(r)

    editions = sorted(by_pdf.items(), key=lambda kv: kv[1]["fy"])
    if recent_n is not None:
        editions = editions[-recent_n:]

    datasets, x_min, x_max = [], None, None
    for pdf, meta in editions:
        fy = meta["fy"]
        pts = []
        for r in meta["rows"]:
            yr  = int(r["year"])
            t   = yr - fy
            val = float(r["value"])
            typ = r["type"]
            pts.append({
                "x": t,   # numeric for linear axis
                "t_rel": t,                 # integer t for tooltip
                "abs_year": yr,             # show true year in tooltip
                "y": val,
                "type": typ
            })
            x_min = t if x_min is None else min(x_min, t)
            x_max = t if x_max is None else max(x_max, t)

        pts.sort(key=lambda p: (p["x"], p["type"]))
        datasets.append({
            "label": format_spa_label(pdf),
            "data": pts,
            "showLine": True,
            "fill": False,
            "tension": 0.1
        })

    return datasets, x_min, x_max

# --- revision helpers ---

def available_target_years(label_key: str) -> list[int]:
    """All distinct years available for this label (sorted)."""
    rows = q("""
        SELECT DISTINCT year FROM spa_obs
        WHERE label_key = ?
        ORDER BY year
    """, (label_key,))
    return [int(r["year"]) for r in rows]

def build_revision_dataset(label_key: str, target_year: int):
    """
    For a fixed calendar year (target_year), build a single dataset showing
    the value across SPA editions sorted by their forecast year.
    Returns (datasets, x_min, x_max).
    """
    rows = q("""
        SELECT pdf_name, year, value, type
        FROM spa_obs
        WHERE label_key = ? AND year = ?
        ORDER BY pdf_name
    """, (label_key, target_year))

    # Group by edition (pdf) and attach forecast year (fy)
    pts = []
    for r in rows:
        pdf = r["pdf_name"]
        fy = forecast_year_from_pdf(pdf)
        if fy is None:
            continue
        pts.append({
            "x": int(fy),                    # edition (publication) year on x-axis
            "edition": pdf,                  # show in tooltip
            "abs_year": int(r["year"]),      # the fixed target year
            "y": float(r["value"]),
            "t": r["type"]                   # history / forecast / extrapolation
        })

    # sort by edition-year
    pts.sort(key=lambda p: p["x"])

    if not pts:
        return [], None, None

    x_min = min(p["x"] for p in pts)
    x_max = max(p["x"] for p in pts)

    datasets = [{
        "label": f"{target_year}",
        "data": pts,
        "showLine": True,
        "fill": False,
        "tension": 0.1
    }]

    return datasets, x_min, x_max

@app.route("/")
def index():
    label_key = request.args.get("label_key") or get_default_label_key()
    recent_n  = parse_recent(request.args.get("n"))
    datasets, x_min, x_max = build_abs_datasets(label_key, recent_n)
    return render_template(
        "index.html",
        mode="abs",
        label_key=label_key,
        choices=label_choices(),
        datasets=datasets,
        x_min=x_min,
        x_max=x_max,
        recent_n=recent_n,
    )

@app.route("/breyting")
def revision_view():
    # pick label + target year
    label_key = request.args.get("label_key") or get_default_label_key()
    years = available_target_years(label_key)
    if not years:
        return render_template("revision.html",
                               label_key=label_key, choices=label_choices(),
                               years=[], target_year=None,
                               datasets=[], x_min=None, x_max=None)

    # default: most recent year that exists for this label
    try:
        target_year = int(request.args.get("year")) if request.args.get("year") else max(years)
    except Exception:
        target_year = max(years)

    datasets, x_min, x_max = build_revision_dataset(label_key, target_year)

    return render_template(
        "revision.html",
        mode="rev",
        label_key=label_key,
        choices=label_choices(),
        years=years,
        target_year=target_year,
        datasets=datasets,
        x_min=x_min,
        x_max=x_max,
    )


@app.route("/utgafu")
def rel_view():
    label_key = request.args.get("label_key") or get_default_label_key()
    recent_n  = parse_recent(request.args.get("n"))
    datasets, x_min, x_max = build_rel_datasets(label_key, recent_n)
    return render_template(
        "relative.html",
        mode="rel",
        label_key=label_key,
        choices=label_choices(),
        datasets=datasets,
        x_min=x_min,
        x_max=x_max,
        recent_n=recent_n,
    )

@app.route("/label/<label_key>")
def label_view(label_key):
    labels = label_list()
    series = build_series(label_key)
    pretty = q("SELECT isl, eng FROM canon WHERE label_key = ?", (label_key,))
    title_isl, title_eng = (pretty[0]["isl"], pretty[0]["eng"]) if pretty else (label_key, "")
    return render_template("index.html",  # reuse same template
                           label_key=label_key,
                           title_isl=title_isl,
                           title_eng=title_eng,
                           labels=labels,
                           series=series)

@app.get("/series/<label_key>")
def series_page(label_key):
    # latest pdf for this label (by name ordering; tweak if you want by date)
    latest = q("SELECT MAX(pdf_name) AS pdf FROM observations")[0]["pdf"]
    pretty = q("SELECT isl, eng FROM canon WHERE label_key=?", (label_key,))
    if not pretty:
        abort(404)
    isl, eng = pretty[0]["isl"], pretty[0]["eng"]

    forecasts = q("""
        SELECT pdf_name, year, value, type
        FROM observations
        WHERE label_key=?
        ORDER BY pdf_name, year
    """, (label_key,))

    # actuals if mapped
    actual = q("""
        SELECT px.series_code, px.series_label
        FROM label_map lm
        JOIN px_series px ON px.series_code = lm.series_code
        WHERE lm.label_key=?
    """, (label_key,))
    actual_series = None
    actual_obs = []
    if actual:
        actual_series = actual[0]
        actual_obs = q("""
            SELECT year, value FROM px_obs WHERE series_code=? ORDER BY year
        """, (actual_series["series_code"],))

    return render_template(
        "series.html",
        label_key=label_key, isl=isl, eng=eng,
        latest=latest,
        forecasts=forecasts,
        actual_series=actual_series,
        actual_obs=actual_obs,
    )

# API to set mapping from label_key → series_code
@app.post("/map")
def set_map():
    label_key = request.form.get("label_key", "").strip()
    series_code = request.form.get("series_code", "").strip()
    if not label_key or not series_code:
        abort(400)
    # Ensure both exist
    if not q("SELECT 1 FROM canon WHERE label_key=?", (label_key,)):
        abort(400)
    if not q("SELECT 1 FROM px_series WHERE series_code=?", (series_code,)):
        abort(400)
    execsql("""
        INSERT INTO label_map(label_key, series_code)
        VALUES(?, ?)
        ON CONFLICT(label_key) DO UPDATE SET series_code=excluded.series_code
    """, (label_key, series_code))
    return redirect(url_for("series_page", label_key=label_key))

# Simple JSON endpoints (for charts, if you prefer fetching)
@app.get("/api/forecasts/<label_key>")
def api_forecasts(label_key):
    rows = q("""
        SELECT pdf_name, year, value, type
        FROM observations
        WHERE label_key=?
        ORDER BY pdf_name, year
    """, (label_key,))
    if not rows:
        abort(404)
    return jsonify(rows)

@app.get("/api/actuals/<label_key>")
def api_actuals(label_key):
    s = q("""SELECT px.series_code
             FROM label_map lm JOIN px_series px ON px.series_code=lm.series_code
             WHERE lm.label_key=?""", (label_key,))
    if not s:
        return jsonify([])
    rows = q("SELECT year, value FROM px_obs WHERE series_code=? ORDER BY year", (s[0]["series_code"],))
    return jsonify(rows)

@app.context_processor
def inject_helpers():
    return {"query": q}

def forecast_year_from_pdf(pdf_name: str) -> int | None:
    m = re.search(r"spa_(\d{4})", pdf_name)
    return int(m.group(1)) if m else None

@app.route("/chart/<label_key>")
def chart_rel(label_key):
    if not q("SELECT 1 FROM canon WHERE label_key = ?", (label_key,)):
        abort(404)
    n_raw = (request.args.get("n") or "3").lower()
    recent_n = None if n_raw in ("all", "allt") else int(n_raw)

    datasets_rel, rx_min, rx_max = build_rel_datasets(label_key, recent_n)
    meta = q("SELECT isl, eng FROM canon WHERE label_key = ?", (label_key,))
    isl = meta[0]["isl"] if meta else label_key
    eng = meta[0]["eng"] if meta else label_key

    return render_template("chart_rel.html",
                           label_key=label_key, n=n_raw,
                           isl=isl, eng=eng,
                           datasets_rel=datasets_rel,
                           rx_min=rx_min, rx_max=rx_max)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
