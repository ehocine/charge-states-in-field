# Tests, figures and generated tables from the saved results.
PY ?= python3

.PHONY: test figures

test:
	$(PY) -m pytest -q tests

figures:
	$(PY) scripts/f6_figures.py
	$(PY) scripts/h2_window.py > /dev/null
	$(PY) scripts/f7_tables.py > /dev/null
