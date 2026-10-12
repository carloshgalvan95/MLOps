"""Entrenamiento base con MLflow para el dataset Heart Disease (Cleveland).

Migracion del notebook baseline de la semana 3 a un script parametrizable.
Modelo: LogisticRegression (misma familia que el notebook).
Autor: Carlos Humberto Galvan Perales (A01797969), TC5061.10
"""
import argparse
import json
import os
import random
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

COLUMNAS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
DATA_DEFAULT = Path(__file__).parent / "data" / "processed.cleveland.data"


def parse_args():
    p = argparse.ArgumentParser(description="Regresion logistica sobre Heart Disease con tracking en MLflow.")
    p.add_argument("--C", type=float, default=1.0,
                   help="Inverso de la fuerza de regularizacion (valores menores regularizan mas). Default 1.0")
    p.add_argument("--penalty", choices=["l1", "l2"], default="l2",
                   help="Tipo de regularizacion de la regresion logistica. Default l2")
    p.add_argument("--max-iter", type=int, default=1000,
                   help="Maximo de iteraciones del optimizador (solver liblinear). Default 1000")
    p.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                   help="Ponderacion de clases; 'balanced' compensa el desbalance. Default none")
    p.add_argument("--scale", choices=["yes", "no"], default="no",
                   help="Aplicar StandardScaler antes del modelo (el notebook no escalaba). Default no")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Proporcion del conjunto de prueba en el split estratificado. Default 0.2")
    p.add_argument("--seed", type=int, default=42,
                   help="Semilla para random, numpy, el split y random_state del modelo. Default 42")
    p.add_argument("--data-path", type=str, default=str(DATA_DEFAULT),
                   help="Ruta al archivo processed.cleveland.data")
    p.add_argument("--experiment-name", type=str, default="tc5061-heart-disease-baseline-experimentation",
                   help="Nombre del experimento en MLflow")
    p.add_argument("--tracking-uri", type=str, default="sqlite:///mlflow.db",
                   help="URI de tracking de MLflow. Default sqlite:///mlflow.db")
    return p.parse_args()


def set_seed(seed):
    """Fija las semillas globales para que las corridas sean reproducibles."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data(path):
    """Lee el CSV crudo de Cleveland; '?' se interpreta como faltante."""
    return pd.read_csv(path, header=None, names=COLUMNAS, na_values="?")


def preprocess(df):
    """Binariza el objetivo (num != 0), elimina filas con NaN y separa X, y."""
    df = df.copy()
    df["target"] = (df["num"] != 0).astype(int)
    df = df.dropna()
    features = [c for c in COLUMNAS if c != "num"]
    X = df[features].astype(float)
    y = df["target"]
    return X, y


def split_data(X, y, test_size, seed):
    """Split estratificado train/test con semilla fija."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def build_model(args):
    """Construye el pipeline (escalado opcional + LogisticRegression)."""
    clf = LogisticRegression(
        C=args.C,
        penalty=args.penalty,
        solver="liblinear",
        max_iter=args.max_iter,
        class_weight=None if args.class_weight == "none" else "balanced",
        random_state=args.seed,
    )
    steps = [("scaler", StandardScaler())] if args.scale == "yes" else []
    return Pipeline(steps + [("clf", clf)])


def train(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    """Calcula metricas sobre el conjunto de prueba."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "recall_pos": recall_score(y_test, y_pred, pos_label=1),
        "precision_pos": precision_score(y_test, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return {k: float(v) for k, v in metrics.items()}, confusion_matrix(y_test, y_pred)


def log_artifacts(metrics, cm, outdir):
    """Guarda la matriz de confusion (PNG) y metrics.json y los registra en MLflow."""
    cm_path = Path(outdir) / "confusion_matrix.png"
    disp = ConfusionMatrixDisplay(cm, display_labels=["sin enf. (0)", "con enf. (1)"])
    disp.plot(cmap="Blues", colorbar=False)
    plt.title("Matriz de confusion (test)")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=120)
    plt.close()
    json_path = Path(outdir) / "metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2))
    mlflow.log_artifact(str(cm_path))
    mlflow.log_artifact(str(json_path))


def run_name(args):
    return (f"logreg_C{args.C}_{args.penalty}_cw-{args.class_weight}"
            f"_scale-{args.scale}_ts{args.test_size}_s{args.seed}")


def main():
    args = parse_args()
    set_seed(args.seed)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    X, y = preprocess(load_data(args.data_path))
    X_train, X_test, y_train, y_test = split_data(X, y, args.test_size, args.seed)

    with mlflow.start_run(run_name=run_name(args)):
        mlflow.log_params({
            "C": args.C, "penalty": args.penalty, "max_iter": args.max_iter,
            "class_weight": args.class_weight, "scale": args.scale,
            "test_size": args.test_size, "seed": args.seed,
            "model": "LogisticRegression", "solver": "liblinear",
            "n_train": len(X_train), "n_test": len(X_test),
        })
        model = train(build_model(args), X_train, y_train)
        metrics, cm = evaluate(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as tmp:
            log_artifacts(metrics, cm, tmp)
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, name="model", signature=signature,
                                 input_example=X_train.head(5))
        print(json.dumps({"run_name": run_name(args), **metrics}, indent=2))


if __name__ == "__main__":
    main()
