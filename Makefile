.PHONY: help setup ingest dbt ml app

PYTHON ?= ./venv/bin/python
DBT ?= ./venv/bin/dbt
PIP ?= ./venv/bin/pip

help:
	@echo "FinShield-AI Commands:"
	@echo "  make setup   - Install python requirements into local venv"
	@echo "  make ingest  - Run ingestion/synthetic generator script into DuckDB"
	@echo "  make dbt     - Run dbt models and tests"
	@echo "  make ml      - Train XGBoost + SHAP explainer and run graph detection"
	@echo "  make app     - Launch FastAPI backend & interactive frontend"

setup:
	python3 -m venv venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

ingest:
	$(PYTHON) scripts/01_ingest_kaggle_data.py

dbt:
	cd dbt_finshield && ../venv/bin/dbt run --profiles-dir . && ../venv/bin/dbt test --profiles-dir .

ml:
	$(PYTHON) scripts/03_build_graph_network.py
	$(PYTHON) scripts/02_train_xgboost_shap.py

app:
	PYTHONPATH=. $(PYTHON) backend/main.py
