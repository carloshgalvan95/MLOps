# A02. Código de experimentación base con MLflow

Actividad individual de TC5061.10 Operaciones de aprendizaje automático (MNA, Tec de Monterrey).
Carlos Humberto Galván Perales, A01797969.

## Qué hace

`train.py` migra el notebook baseline de la semana 3 a un script parametrizable. Usa el dataset
Heart Disease (Cleveland, UCI) con objetivo binario (`num != 0`) y una regresión logística
(misma familia que el notebook). Cada ejecución registra en MLflow los hiperparámetros, la semilla,
cinco métricas (accuracy, f1_macro, recall_pos, precision_pos, roc_auc), la matriz de confusión
en PNG, `metrics.json` y el modelo con firma e `input_example`, dentro del experimento
`tc5061-heart-disease-baseline-experimentation`.

El notebook en `notebooks/` se conserva solo para el análisis exploratorio.
Los datos (`data/processed.cleveland.data`, 18 KB) vienen incluidos.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

Con valores por defecto:

```bash
python train.py
```

Las cinco configuraciones que comparé:

```bash
python train.py
python train.py --C 0.01 --penalty l2
python train.py --C 0.1 --penalty l1 --scale yes
python train.py --C 1.0 --scale yes --class-weight balanced
python train.py --C 10 --penalty l1 --scale yes --test-size 0.3
```

`python train.py --help` muestra todos los argumentos (`--C`, `--penalty`, `--max-iter`,
`--class-weight`, `--scale`, `--test-size`, `--seed`, `--data-path`, `--experiment-name`,
`--tracking-uri`).

## MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Luego abrir http://localhost:5000, entrar al experimento, seleccionar las corridas y usar Compare.

## Reproducibilidad

La semilla (`--seed`, 42 por defecto) se fija en `random`, `numpy`, el `train_test_split` y el
`random_state` del modelo. Ejecuté dos veces `python train.py` y ambas corridas dieron exactamente
las mismas métricas (accuracy 0.8833, f1_macro 0.8817). Las versiones exactas están en
`requirements.txt`. `mlflow.db` y `mlruns/` no se versionan.
