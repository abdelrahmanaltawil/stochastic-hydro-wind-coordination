PYTHON ?= python

.PHONY: test run study paper

test:
	$(PYTHON) -m pytest -q

run:
	$(PYTHON) -m src.workflow --solver highs

study:
	$(PYTHON) -m src.experiments

paper:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/main.tex
