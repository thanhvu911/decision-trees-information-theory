#!/usr/bin/env bash
# Rebuild every result and figure in the study from scratch (about an hour on 2 cores).
set -euo pipefail
cd "$(dirname "$0")/.."
python study/data/fetch_adult.py
python -m pytest -q study/tests
python study/experiments/e1_many_levels_bias.py
python study/experiments/e3_alpha_path.py
python study/experiments/e2_pruning.py
python study/make_figures.py
