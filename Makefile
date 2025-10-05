# -------- Variables --------
PYTHON      ?= python3
PIPELINE    = pipeline/run_spa_pipeline.py
PARSE       = pipeline/run_parse.py
RAW_DIR     = data/extracted/raw
CSV_DIR     = data/extracted/csv
PDF_DIR     = data/spa_pdf

# SQLite (module-based)
DB          = data/spa.sqlite3
SCHEMA      = schema.sql
LOADER_MOD  = pipeline.load_spa_sqlite

# -------- Targets --------
.PHONY: help pipeline parse clean clean-raw clean-csv db db-load db-reset db-clean db-stats fresh rebuild serve \
        px-dump px-fetch px-fetch-07100 px-fetch-07000 px-preview px-list px-list-07000 px-list-07100

help: ## Show this help
	@echo ""
	@echo "Available targets:"
	@echo "  make pipeline   - Clean RAW then locate/dump pages"
	@echo "  make parse      - Clean CSV then parse RAW -> CSV"
	@echo "  make db         - Reset DB then load CSVs"
	@echo "  make fresh      - Clean RAW+CSV, rebuild DB, then pipeline+parse+db"
	@echo "  make db-stats   - Row counts"
	@echo "  make db-clean   - Remove the SQLite DB"
	@echo "  make clean      - Clean RAW+CSV only"
	@echo ""

# ---- Cleaning helpers (used automatically) ----
clean-raw:
	@rm -rf $(RAW_DIR)
	@mkdir -p $(RAW_DIR)
	@echo "[✓] Cleaned $(RAW_DIR)"

clean-csv:
	@rm -rf $(CSV_DIR)
	@mkdir -p $(CSV_DIR)
	@echo "[✓] Cleaned $(CSV_DIR)"

# ---- Main steps (auto-clean first) ----
pipeline: clean-raw ## Locate and dump tables (raw page text only)
	$(PYTHON) -m pipeline.run_spa_pipeline

parse: clean-csv ## Parse raw dumps into tidy CSV
	$(PYTHON) -m pipeline.run_parse

# ---- DB: reset then load (no leftovers) ----
db-reset: ## Drop/recreate DB from schema.sql
	@rm -f $(DB)
	@mkdir -p $(dir $(DB))
	@sqlite3 $(DB) < $(SCHEMA)
	@echo "[✓] Reset $(DB) from $(SCHEMA)"

db-load: ## Load all CSVs into observations (upsert)
	$(PYTHON) -m $(LOADER_MOD)

db: db-reset db-load ## Clean DB then load CSVs

db-stats: ## Quick table counts
	@sqlite3 $(DB) "SELECT 'canon', COUNT(*) FROM canon UNION ALL SELECT 'observations', COUNT(*) FROM observations;"

db-clean: ## Remove the SQLite DB
	@rm -f $(DB)
	@echo "[✓] Removed $(DB)"

# ---- One-shot end-to-end clean rebuild ----
fresh: clean-raw clean-csv db-reset ## Clean RAW+CSV, reset DB, then run everything
	@$(MAKE) pipeline
	@$(MAKE) parse
	@$(MAKE) db-load
	@$(MAKE) db-stats

# ---- Old 'clean' for just files (no DB) ----
clean: clean-raw clean-csv

# ---- PX + Flask (unchanged) ----
px-dump:
	$(PYTHON) -m pipeline.px_dump

px-fetch:
	$(PYTHON) -m pipeline.px_fetch

px-fetch-07100:
	$(PYTHON) -m pipeline.px_fetch_any THJ07100

px-fetch-07000:
	$(PYTHON) -m pipeline.px_fetch_any THJ07000

px-preview:
	$(PYTHON) -m pipeline.px_preview

px-list:
	$(PYTHON) -m pipeline.px_list_series $(PX_TABLE)

px-list-07000:
	$(PYTHON) -m pipeline.px_list_series THJ07000

px-list-07100:
	$(PYTHON) -m pipeline.px_list_series THJ07100

serve: ## Run the Flask dev server
	FLASK_APP=app.app FLASK_ENV=development $(PYTHON) -m flask run -p 5000
