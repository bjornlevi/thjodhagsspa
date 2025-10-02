-- schema.sql
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS canon (
  label_key TEXT PRIMARY KEY,
  isl       TEXT NOT NULL,
  eng       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
  pdf_name  TEXT NOT NULL,
  label_key TEXT NOT NULL,
  year      INTEGER NOT NULL,
  value     REAL NOT NULL,
  type      TEXT NOT NULL CHECK (type IN ('history','forecast','extrapolation')),
  PRIMARY KEY (pdf_name, label_key, year),
  FOREIGN KEY (label_key) REFERENCES canon(label_key) ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_obs_label_year ON observations(label_key, year);
CREATE INDEX IF NOT EXISTS idx_obs_pdf ON observations(pdf_name);

-- === Actuals from Statistics Iceland (PX) ===

-- a generic series catalog; we don't assume codes won't change,
-- so keep both code and human label
CREATE TABLE IF NOT EXISTS px_series (
  series_code   TEXT PRIMARY KEY,
  series_label  TEXT NOT NULL
);

-- wide “year,value” observations for PX series
CREATE TABLE IF NOT EXISTS px_obs (
  series_code   TEXT NOT NULL,
  year          INTEGER NOT NULL,
  value         REAL NOT NULL,
  PRIMARY KEY (series_code, year),
  FOREIGN KEY (series_code) REFERENCES px_series(series_code) ON DELETE CASCADE
);

-- optional mapping from your canon label_key → PX series_code
-- lets you choose which PX series to compare to a forecast label
CREATE TABLE IF NOT EXISTS label_map (
  label_key     TEXT PRIMARY KEY,
  series_code   TEXT NOT NULL,
  FOREIGN KEY (label_key)  REFERENCES canon(label_key) ON DELETE CASCADE,
  FOREIGN KEY (series_code) REFERENCES px_series(series_code) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_px_obs_year ON px_obs(year);
