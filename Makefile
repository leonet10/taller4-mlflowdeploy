# Makefile
# --------
# Define los comandos del pipeline MLOps.
# GitHub Actions llama a estos targets para mantener
# los pasos del CI/CD independientes de la implementación interna.
#
# Uso local:
#   make train      ← entrena y guarda model.pkl
#   make validate   ← evalúa model.pkl contra el umbral de calidad
#   make pipeline   ← ejecuta ambos pasos en orden

.PHONY: train validate pipeline

train:
	python src/train.py

validate:
	python src/validate.py

pipeline: train validate
