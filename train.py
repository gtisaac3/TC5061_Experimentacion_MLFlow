import argparse
import json
import random
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
  mean_absolute_error,
  mean_squared_error,
  r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Configuración base
DATA_DIR = Path(".")

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "store_sales_ridge_experiments"

NUMERIC_FEATURES = [
  "onpromotion",
  "dcoilwtico",
  "year",
  "month",
  "day",
  "day_of_week",
]

CATEGORICAL_FEATURES = [
  "store_nbr",
  "family",
  "city",
  "state",
  "type",
  "cluster",
]

TARGET = "sales"


def parse_args():
  """
  Define y obtiene los argumentos de línea de comandos.
  """
  parser = argparse.ArgumentParser(
    description=(
      "Entrena y evalúa un modelo Ridge para predecir "
      "las ventas del dataset Store Sales."
    )
  )

  parser.add_argument(
    "--alpha",
    type=float,
    default=1.0,
    help=(
      "Fuerza de regularización del modelo Ridge. "
      "Default: 1.0"
    ),
  )

  parser.add_argument(
    "--solver",
    type=str,
    default="lsqr",
    choices=[
      "lsqr",
      "sparse_cg",
      "sag",
      "saga",
    ],
    help=(
      "Algoritmo de optimización utilizado por Ridge. "
      "Default: lsqr"
    ),
  )

  parser.add_argument(
    "--tol",
    type=float,
    default=1e-4,
    help=(
      "Tolerancia utilizada como criterio de convergencia. "
      "Default: 0.0001"
    ),
  )

  parser.add_argument(
    "--random-state",
    type=int,
    default=42,
    help=(
      "Semilla aleatoria utilizada para reproducibilidad. "
      "Default: 42"
    ),
  )

  parser.add_argument(
    "--test-start-date",
    type=str,
    default="2017-08-01",
    help=(
      "Fecha inicial del conjunto de prueba en formato YYYY-MM-DD. "
      "Default: 2017-08-01"
    ),
  )

  return parser.parse_args()


def set_seed(random_state):
  """
  Fija las semillas utilizadas durante la ejecución.
  """
  random.seed(random_state)
  np.random.seed(random_state)


def load_data(data_dir):
  """
  Carga los archivos utilizados para entrenar el modelo.
  """
  train = pd.read_csv(
    data_dir / "train.csv",
    parse_dates=["date"],
  )
  stores = pd.read_csv(
    data_dir / "stores.csv",
  )
  oil = pd.read_csv(
    data_dir / "oil.csv",
    parse_dates=["date"],
  )
  return train, stores, oil


def prepare_oil_data(train, oil):
  """
  Construye una serie diaria del precio del petróleo y
  completa los valores faltantes.
  """
  oil_clean = oil.copy()
  oil_clean = oil_clean.sort_values("date")
  full_dates = pd.DataFrame(
    {
      "date": pd.date_range(
        start=train["date"].min(),
        end=train["date"].max(),
        freq="D",
      )
    }
  )

  oil_daily = full_dates.merge(
    oil_clean,
    on="date",
    how="left",
  )

  oil_daily["dcoilwtico"] = (
    oil_daily["dcoilwtico"]
    .interpolate(method="linear")
    .ffill()
    .bfill()
  )

  return oil_daily


def prepare_dataset(train, stores, oil_daily):
  """
  Integra los datasets y genera las variables temporales
  utilizadas por el modelo.
  """
  df = train.merge(
    stores,
    on="store_nbr",
    how="left",
  )
  df = df.merge(
    oil_daily,
    on="date",
    how="left",
  )
  df["year"] = df["date"].dt.year
  df["month"] = df["date"].dt.month
  df["day"] = df["date"].dt.day
  df["day_of_week"] = df["date"].dt.dayofweek
  
  return df


def split_data(df, test_start_date):
  """
  Divide los datos cronológicamente en entrenamiento y prueba.
  """
  test_start_date = pd.Timestamp(test_start_date)

  train_data = df[
    df["date"] < test_start_date
  ].copy()

  test_data = df[
    df["date"] >= test_start_date
  ].copy()

  features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

  X_train = train_data[features]
  y_train = train_data[TARGET]

  X_test = test_data[features]
  y_test = test_data[TARGET]

  return X_train, X_test, y_train, y_test


def build_model(alpha, solver, tol, random_state):
  """
  Construye el pipeline de preprocesamiento y el modelo Ridge.
  """
  numeric_transformer = Pipeline(
    steps=[
      ("scaler", StandardScaler()),
    ]
  )

  categorical_transformer = Pipeline(
    steps=[
      (
        "onehot",
        OneHotEncoder(
          handle_unknown="ignore",
        ),
      )
    ]
  )

  preprocessor = ColumnTransformer(
    transformers=[
      (
        "num",
        numeric_transformer,
        NUMERIC_FEATURES,
      ),
      (
        "cat",
        categorical_transformer,
        CATEGORICAL_FEATURES,
      ),
    ]
  )

  model = Pipeline(
    steps=[
      ("preprocessor", preprocessor),
      (
        "model",
        Ridge(
          alpha=alpha,
          solver=solver,
          tol=tol,
          random_state=random_state,
        ),
      ),
    ]
  )

  return model


def train_model(model, X_train, y_train):
  """
  Entrena el modelo utilizando el conjunto de entrenamiento.
  """
  model.fit(X_train, y_train)

  return model


def evaluate_model(model, X_test, y_test):
  """
  Evalúa el modelo sobre el conjunto de prueba.
  """
  y_pred = model.predict(X_test)

  mae = mean_absolute_error(
    y_test,
    y_pred,
  )

  rmse = np.sqrt(
    mean_squared_error(
      y_test,
      y_pred,
    )
  )

  r2 = r2_score(
    y_test,
    y_pred,
  )

  negative_predictions = int(
    (y_pred < 0).sum()
  )

  negative_predictions_percentage = (
    negative_predictions / len(y_pred)
  ) * 100

  metrics = {
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
    "negative_predictions": int(negative_predictions),
    "negative_predictions_percentage": float(
      negative_predictions_percentage
    ),
  }

  return metrics

def log_metrics_artifact(metrics):
  """
  Guarda las métricas en un archivo JSON y lo registra
  como artefacto de MLflow.
  """
  with tempfile.TemporaryDirectory() as temp_dir:
    metrics_path = Path(temp_dir) / "metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
      json.dump(
        metrics,
        file,
        indent=2,
      )

    mlflow.log_artifact(
      str(metrics_path),
      artifact_path="evaluation",
    )

def main(args):
  set_seed(args.random_state)

  mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
  )

  mlflow.set_experiment(
    MLFLOW_EXPERIMENT_NAME
  )

  print("Cargando datos...")

  train, stores, oil = load_data(
    DATA_DIR
  )

  print("Preparando datos...")
  oil_daily = prepare_oil_data(
    train,
    oil,
  )

  df = prepare_dataset(
    train,
    stores,
    oil_daily,
  )

  X_train, X_test, y_train, y_test = split_data(
    df,
    args.test_start_date,
  )

  print(
    f"Registros de entrenamiento: {len(X_train):,}"
  )
  print(
    f"Registros de prueba: {len(X_test):,}"
  )

  print("\nConfiguración")
  print("-------------")
  print(f"Alpha: {args.alpha}")
  print(f"Solver: {args.solver}")
  print(f"Tolerance: {args.tol}")
  print(f"Random state: {args.random_state}")
  print(f"Test start date: {args.test_start_date}")
  print()

  run_name = (
    f"ridge_"
    f"alpha-{args.alpha}_"
    f"solver-{args.solver}_"
    f"tol-{args.tol}_"
    f"seed-{args.random_state}"
  )

  with mlflow.start_run(
    run_name=run_name
  ) as run:
    mlflow.log_params(
      {
        "model": "Ridge",
        "alpha": args.alpha,
        "solver": args.solver,
        "tol": args.tol,
        "random_state": args.random_state,
        "test_start_date": args.test_start_date,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
      }
    )

    print("Construyendo modelo...")

    model = build_model(
      alpha=args.alpha,
      solver=args.solver,
      tol=args.tol,
      random_state=args.random_state,
    )

    print("Entrenando modelo...")

    model = train_model(
      model,
      X_train,
      y_train,
    )

    print("Evaluando modelo...")

    metrics = evaluate_model(
      model,
      X_test,
      y_test,
    )

    mlflow.log_metrics(
      metrics
    )

    log_metrics_artifact(
      metrics
    )

    mlflow.sklearn.log_model(
      sk_model=model,
      name="model",
    )

    print("\nResultados")
    print("----------")
    print(f"MAE:  {metrics['mae']:,.2f}")
    print(f"RMSE: {metrics['rmse']:,.2f}")
    print(f"R²:   {metrics['r2']:.4f}")

    print(
      "Predicciones negativas: "
      f"{metrics['negative_predictions']:,}"
    )

    print(
      "Porcentaje de predicciones negativas: "
      f"{metrics['negative_predictions_percentage']:.2f}%"
    )

    print("\nMLflow")
    print("------")
    print(
      f"Experimento: {MLFLOW_EXPERIMENT_NAME}"
    )
    print(
      f"Run name: {run_name}"
    )
    print(
      f"Run ID: {run.info.run_id}"
    )

if __name__ == "__main__":
  args = parse_args()
  main(args)