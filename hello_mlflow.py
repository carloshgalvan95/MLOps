from pathlib import Path
import mlflow


def main() -> None:
    artifact = Path("hello_artifact.txt")
    artifact.write_text(
        "Primer artefacto A01 TC5061 - Carlos Galvan\n",
        encoding="utf-8",
    )

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("A01_hello_mlflow")

    with mlflow.start_run(run_name="hello_world"):
        mlflow.log_param("model_type", "dummy")
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_param("dataset", "synthetic_v0")

        mlflow.log_metric("accuracy", 0.91)
        mlflow.log_metric("loss", 0.12)

        mlflow.log_artifact(str(artifact))

        run = mlflow.active_run()
        print(f"Run ID: {run.info.run_id}")
        print("Abre MLFlow UI: mlflow ui  ->  http://localhost:5000")


if __name__ == "__main__":
    main()
