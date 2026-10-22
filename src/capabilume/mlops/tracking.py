
import mlflow
import mlflow.sklearn
import yaml
from pathlib import Path


def configure_tracking(project_root: Path, config_path: Path) -> None:
    """Configure SQLite-backed MLflow tracking."""
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    database = (project_root / config["tracking"]["database"]).resolve()
    mlflow.set_tracking_uri(f"sqlite:///{database.as_posix()}")
    mlflow.set_experiment(config["experiment"]["name"])


def log_training_run(
    *,
    run_name: str,
    params: dict,
    metrics: dict,
    artifacts: dict,
    model,
    model_name: str,
    input_example=None,
) -> str:
    """Log a complete training run and return its MLflow run ID."""
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics({
            key: float(value)
            for key, value in metrics.items()
        })

        for artifact_path, local_path in artifacts.items():
            mlflow.log_artifact(
                str(local_path),
                artifact_path=artifact_path,
            )

        mlflow.sklearn.log_model(
            sk_model=model,
            name=model_name,
            input_example=input_example,
        )

        return run.info.run_id
