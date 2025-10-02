# -------- Variables --------
PYTHON      ?= python3
PIPELINE    = pipeline/run_spa_pipeline.py
PARSE       = pipeline/run_parse.py
RAW_DIR     = data/extracted/raw
CSV_DIR     = data/extracted/csv
PDF_DIR     = data/spa_pdf

# -------- Targets --------
.PHONY: help pipeline parse clean

help: ## Show this help
	@echo ""
	@echo "Available targets:"
	@echo "  make pipeline   - Run spa pipeline: scan PDFs, locate tables, dump raw pages"
	@echo "  make parse      - Run parse: read raw dumps and create tidy CSVs"
	@echo "  make clean      - Remove extracted raw + csv outputs"
	@echo ""

pipeline: ## Locate and dump tables (raw page text only)
	$(PYTHON) -m pipeline.run_spa_pipeline

parse: ## Parse raw dumps into tidy CSV
	$(PYTHON) -m pipeline.run_parse

clean: ## Remove extracted files
	rm -rf $(RAW_DIR)/* $(CSV_DIR)/*
	@echo "Cleaned $(RAW_DIR) and $(CSV_DIR)"
