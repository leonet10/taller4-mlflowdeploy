"""
validate.py
===========
Carga model.pkl generado por train.py y evalúa su MSE sobre un
conjunto de prueba reproducible.  Si el MSE supera el umbral definido,
el proceso termina con código de error (sys.exit(1)), lo que hace
fallar el job de GitHub Actions y detiene el pipeline.

Conceptos MLOps que cubre:
  - Quality gate: umbral de métricas como criterio de aceptación
  - Reproducibilidad: mismo random_state que en entrenamiento
  - Salida de proceso como señal al orquestador (CI/CD)
"""

import os
import sys

import joblib
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ──────────────────────────────────────────────
# Umbral de aceptación: el MSE debe estar por debajo de este valor.
# Para LinearRegression sobre diabetes, valores típicos rondan 2 800-3 200.
# Ajusta este valor según el criterio del negocio / ejercicio.
THRESHOLD = 5_000.0

model_pkl_path = os.path.join(os.getcwd(), "model.pkl")

# ──────────────────────────────────────────────
# 2. CARGAR EL MODELO
# ──────────────────────────────────────────────
print(f"[validate] Buscando modelo en: {model_pkl_path}")

if not os.path.exists(model_pkl_path):
    print(f"[validate] ❌ ERROR: No se encontró model.pkl.")
    print(f"[validate]    Archivos en CWD: {os.listdir(os.getcwd())}")
    sys.exit(1)

model = joblib.load(model_pkl_path)
print(f"[validate] Modelo cargado. Tipo: {type(model).__name__}")
print(f"[validate] Features esperadas: {model.n_features_in_}")

# ──────────────────────────────────────────────
# 3. PREPARAR DATOS DE PRUEBA
# ──────────────────────────────────────────────
# IMPORTANTE: usar el MISMO random_state que en train.py
# para que X_test sea idéntico y la comparación sea válida.
X, y = load_diabetes(return_X_y=True, as_frame=True)
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"[validate] X_test shape: {X_test.shape}")

# ──────────────────────────────────────────────
# 4. PREDICCIÓN Y MÉTRICAS
# ──────────────────────────────────────────────
try:
    y_pred = model.predict(X_test)
except ValueError as exc:
    print(f"[validate] ❌ Error de dimensiones al predecir: {exc}")
    print(f"[validate]    Modelo espera {model.n_features_in_} features, "
          f"X_test tiene {X_test.shape[1]}.")
    sys.exit(1)

mse = mean_squared_error(y_test, y_pred)
print(f"[validate] 🔍 MSE obtenido : {mse:.4f}")
print(f"[validate] 🎯 Umbral       : {THRESHOLD:.4f}")

# ──────────────────────────────────────────────
# 5. QUALITY GATE
# ──────────────────────────────────────────────
if mse <= THRESHOLD:
    print("[validate] ✅ El modelo PASA el quality gate.")
    sys.exit(0)          # éxito → el pipeline continúa
else:
    print("[validate] ❌ El modelo NO supera el quality gate. "
          "Deteniendo pipeline.")
    sys.exit(1)          # falla → GitHub Actions marca el job como fallido
