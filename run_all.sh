#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python src/run_experiment.py
PYTHONPATH=src python src/plot_results.py
pytest -q
