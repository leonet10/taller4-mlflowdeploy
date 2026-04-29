"""
train.py
========
Entrena un modelo de regresión lineal sobre el dataset de diabetes,
registra métricas y el modelo en MLflow, y guarda model.pkl para
que el paso de validación pueda cargarlo directamente.

Conceptos MLOps que cubre:
  - Tracking de experimentos con MLflow (métricas, parámetros, artefactos)
  - Firma del modelo (ModelSignature) para detectar incompatibilidades en producción
  - Serialización del modelo con joblib para uso en validate.py
"""

import os
import sys
import traceback

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# 1. RUTAS  (todo relativo al CWD del runner)
# ──────────────────────────────────────────────
workspace_dir  = os.getcwd()
mlruns_dir     = os.path.join(workspace_dir, "mlruns")
tracking_uri   = "file://" + os.path.abspath(mlruns_dir)
artifact_loc   = tracking_uri          # experimentos y modelos en el mismo directorio
model_pkl_path = os.path.join(workspace_dir, "model.pkl")

print(f"[train] CWD            : {workspace_dir}")
print(f"[train] MLRuns dir     : {mlruns_dir}")
print(f"[train] Tracking URI   : {tracking_uri}")

os.makedirs(mlruns_dir, exist_ok=True)

# ──────────────────────────────────────────────
# 2. CONFIGURAR MLFLOW
# ──────────────────────────────────────────────
mlflow.set_tracking_uri(tracking_uri)

experiment_name = "CI-CD-Lab-MLflow"
try:
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location=artifact_loc,
    )
    print(f"[train] Experimento creado  → ID: {experiment_id}")
except mlflow.exceptions.MlflowException as exc:
    if "RESOURCE_ALREADY_EXISTS" not in str(exc):
        raise
    exp = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = exp.experiment_id
    print(f"[train] Experimento existente → ID: {experiment_id}")

# ──────────────────────────────────────────────
# 3. DATOS Y MODELO
# ──────────────────────────────────────────────
X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
mse   = mean_squared_error(y_test, preds)

print(f"[train] MSE en test set: {mse:.4f}")

# ──────────────────────────────────────────────
# 4. REGISTRAR EN MLFLOW
# ──────────────────────────────────────────────
try:
    with mlflow.start_run(experiment_id=experiment_id) as run:
        # Parámetros del modelo
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("test_size",  0.2)
        mlflow.log_param("random_state", 42)

        # Métricas
        mlflow.log_metric("mse", mse)

        # Artefacto: modelo con firma
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
        )

        print(f"[train] Run ID        : {run.info.run_id}")
        print(f"[train] Artifact URI  : {run.info.artifact_uri}")

except Exception:
    traceback.print_exc()
    sys.exit(1)

# ──────────────────────────────────────────────
# 5. GUARDAR model.pkl  (lo usa validate.py)
# ──────────────────────────────────────────────
joblib.dump(model, model_pkl_path)
print(f"[train] model.pkl guardado en: {model_pkl_path}")
print("[train] ✅ Entrenamiento completado exitosamente.")
