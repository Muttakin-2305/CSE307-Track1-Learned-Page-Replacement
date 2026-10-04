.PHONY: all experiment plots test clean

all: experiment plots test

experiment:
	PYTHONPATH=src python src/run_experiment.py

plots:
	PYTHONPATH=src python src/plot_results.py

test:
	pytest -q

clean:
	rm -f results/*.csv results/*.png results/*.txt results/config.json results/traces/*.csv
