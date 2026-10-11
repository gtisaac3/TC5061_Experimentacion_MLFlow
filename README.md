# Experimentación con MLflow - Store Sales

Proyecto desarrollado para la actividad de experimentación y seguimiento de modelos con MLflow.

El proyecto utiliza el dataset **Store Sales - Time Series Forecasting** de Kaggle y entrena un modelo de regresión Ridge para predecir la variable `sales`.

## Requisitos

- Python 3.12.3
- pip
- MLflow

Las versiones de las dependencias utilizadas se encuentran en `requirements.txt`.

## Configuración del entorno

Crear un entorno virtual:

```bash
python3 -m venv .venv
```

Activarlo en macOS o Linux:

```bash
source .venv/bin/activate
```

Instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

## Dataset

El proyecto utiliza el dataset **Store Sales - Time Series Forecasting** de Kaggle.

Para ejecutar el entrenamiento deben colocarse los siguientes archivos en la raíz del proyecto:

```text
train.csv
stores.csv
oil.csv
```

Los archivos del dataset no se incluyen en el repositorio debido a su tamaño.

## Iniciar MLflow

En una terminal, con el entorno virtual activo, ejecutar:

```bash
python -m mlflow server --host 127.0.0.1 --port 5000
```

La interfaz de MLflow estará disponible en:

```text
http://127.0.0.1:5000
```

Esta terminal debe permanecer abierta mientras se ejecutan los experimentos.

## Ejecutar el entrenamiento

En otra terminal, con el entorno virtual activo:

```bash
python train.py
```

El script puede ejecutarse sin argumentos utilizando los valores por defecto.

## Argumentos disponibles

Para consultar los argumentos disponibles:

```bash
python train.py --help
```

El script permite modificar los siguientes parámetros:

- `--alpha`: fuerza de regularización del modelo Ridge.
- `--solver`: algoritmo de optimización utilizado por Ridge.
- `--tol`: tolerancia utilizada como criterio de convergencia.
- `--random-state`: semilla utilizada para reproducibilidad.
- `--test-start-date`: fecha de inicio del conjunto de prueba.

Ejemplo de ejecución:

```bash
python train.py --alpha 10 --solver lsqr --tol 0.001 --random-state 42
```

## MLflow Tracking

Cada ejecución del script registra en MLflow:

- hiperparámetros del modelo;
- configuración de la corrida;
- MAE;
- RMSE;
- R²;
- número y porcentaje de predicciones negativas;
- archivo `metrics.json` como artefacto;
- modelo entrenado.

Todas las corridas se agrupan dentro del experimento:

```text
store_sales_ridge_experiments
```

## Reproducibilidad

La semilla se controla mediante:

```text
--random-state
```

El valor por defecto es:

```text
42
```

El script fija la semilla de Python y NumPy y utiliza el mismo `random_state` en el modelo.

Al ejecutar el script con los mismos datos, versiones de dependencias, hiperparámetros y semilla se deben obtener los mismos resultados.

## Configuración baseline

La configuración utilizada como baseline es:

```text
alpha = 1.0
solver = lsqr
tol = 0.0001
random_state = 42
test_start_date = 2017-08-01
```

Puede ejecutarse explícitamente con:

```bash
python train.py --alpha 1.0 --solver lsqr --tol 0.0001 --random-state 42 --test-start-date 2017-08-01
```

## Estructura principal del repositorio

```text
TC5061_Experimentacion_MLFlow/
├── .gitignore
├── README.md
├── requirements.txt
├── train.py
└── TC5061_Notebook_Baseline_GonzalezIsaac.ipynb
```

El notebook se conserva como evidencia del análisis exploratorio de datos, mientras que el entrenamiento, evaluación y registro de experimentos se realizan desde `train.py`.