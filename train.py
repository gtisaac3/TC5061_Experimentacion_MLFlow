from pathlib import Path

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
TEST_START_DATE = "2017-08-01"

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


def build_model():
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
          alpha=1.0,
          solver="lsqr",
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
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
    "negative_predictions": negative_predictions,
    "negative_predictions_percentage":
      negative_predictions_percentage,
  }

  return metrics


def main():
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
    TEST_START_DATE,
  )

  print(
    f"Registros de entrenamiento: {len(X_train):,}"
  )
  print(
    f"Registros de prueba: {len(X_test):,}"
  )

  print("Construyendo modelo...")

  model = build_model()

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


if __name__ == "__main__":
  main()