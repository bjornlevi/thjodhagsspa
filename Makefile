# -------- Variables --------
PYTHON      ?= python3
PIPELINE    = pipeline/run_spa_pipeline.py
PARSE       = pipeline/run_parse.py
RAW_DIR     = data/extracted/raw
CSV_DIR     = data/extracted/csv
PDF_DIR     = data/spa_pdf

# SQLite loader (run as module)
DB          = data/spa.sqlite3
SCHEMA      = schema.sql
LOADER_MOD  = pipeline.load_spa_sqlite

# -------- Targets --------
.PHONY: help pipeline parse clean db db-load db-clean db-stats

help: ## Show this help
	@echo ""
	@echo "Available targets:"
	@echo "  make pipeline   - Run spa pipeline: scan PDFs, locate tables, dump raw pages"
	@echo "  make parse      - Run parse: read raw dumps and create tidy CSVs"
	@echo "  make db         - Load all CSVs into SQLite (via module)"
	@echo "  make db-stats   - Show row counts in SQLite tables"
	@echo "  make db-clean   - Remove the SQLite DB"
	@echo "  make clean      - Remove extracted raw + csv outputs"
	@echo ""

pipeline: ## Locate and dump tables (raw page text only)
	$(PYTHON) -m pipeline.run_spa_pipeline

parse: ## Parse raw dumps into tidy CSV
	$(PYTHON) -m pipeline.run_parse

# --- SQLite (module-based) ---
db: db-load ## Load all CSVs into SQLite

db-load: $(SCHEMA)  ## Run the loader module; requires schema.sql present
	@echo "[i] Loading CSVs into $(DB)"
	@$(PYTHON) -m $(LOADER_MOD)

db-stats: ## Quick table counts
	@sqlite3 $(DB) "SELECT 'canon', COUNT(*) FROM canon UNION ALL SELECT 'observations', COUNT(*) FROM observations;"

db-clean: ## Remove the SQLite DB
	@rm -f $(DB)
	@echo "[✓] Removed $(DB)"

clean: ## Remove extracted files
	rm -rf $(RAW_DIR)/* $(CSV_DIR)/*
	@echo "Cleaned $(RAW_DIR) and $(CSV_DIR)"

# --- PX fetch + Flask ---
PX_FETCH_MOD = pipeline.px_fetch

px-dump: ## Fetch PX raw JSON only (no parsing), print quick summary
	$(PYTHON) -m pipeline.px_dump

px-fetch: ## Load THJ07100 PXWeb into SQLite (uses cached JSON if present)
	$(PYTHON) -m pipeline.px_fetch

px-fetch-07100: ## Load THJ07100 (actuals) into SQLite
	$(PYTHON) -m pipeline.px_fetch_any THJ07100

px-fetch-07000: ## Load THJ07000 (latest SPA table) into SQLite
	$(PYTHON) -m pipeline.px_fetch_any THJ07000

px-preview: ## Preview px_* tables
	$(PYTHON) -m pipeline.px_preview

px-list: ## List PX series for a table (use PX_TABLE or pass table name)
	$(PYTHON) -m pipeline.px_list_series $(PX_TABLE)

px-list-07000: ## List series in THJ07000
	$(PYTHON) -m pipeline.px_list_series THJ07000

px-list-07100: ## List series in THJ07100
	$(PYTHON) -m pipeline.px_list_series THJ07100


serve: ## Run the Flask dev server
	FLASK_APP=app.py FLASK_ENV=development $(PYTHON) app.py
