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
