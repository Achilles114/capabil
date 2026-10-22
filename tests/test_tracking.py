
import mlflow

from capabilume.mlops.tracking import configure_tracking


def test_tracking_uses_configured_sqlite_database(tmp_path):
    config = tmp_path / "training.yaml"
    config.write_text(
        "experiment:\n"
        "  name: capabilume-test\n"
        "tracking:\n"
        "  backend: sqlite\n"
        "  database: test.db\n",
        encoding="utf-8",
    )

    configure_tracking(tmp_path, config)

    assert mlflow.get_tracking_uri().startswith("sqlite:///")
    assert mlflow.get_experiment_by_name("capabilume-test") is not None
