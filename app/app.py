from flask import Flask, render_template, jsonify, request, abort, redirect, url_for
import sqlite3

DB_PATH = "data/spa.sqlite3"

app = Flask(__name__)

def q(sql, params=()):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()

def execsql(sql, params=()):
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute(sql, params)
        con.commit()
    finally:
        con.close()

@app.get("/")
def index():
    labels = q("SELECT label_key, isl, eng FROM canon ORDER BY isl;")
    # which labels already mapped to PX?
    mapped = {r["label_key"]: r["series_code"] for r in q("SELECT * FROM label_map")}
    return render_template("index.html", labels=labels, mapped=mapped)

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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
