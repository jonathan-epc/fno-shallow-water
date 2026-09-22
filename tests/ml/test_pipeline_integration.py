from pathlib import Path

import pytest
import yaml

from ML.core.optimizer import HyperparameterOptimizer
from ML.core.trainer import ModelTrainer
from nconfig import Config, get_config

pytestmark = pytest.mark.integration

project_root = Path(__file__).resolve().parents[2]


@pytest.fixture
def override_config(monkeypatch):
    """A pytest fixture that cleans up old test databases, loads the main config,
    overrides it with test settings, and monkeypatches the global config safely.
    """
    config = get_config("config.yml")

    test_config_path = project_root / "tests" / "test_config.yml"
    with open(test_config_path) as f:
        test_overrides = yaml.safe_load(f)

    config_data = config.model_dump()

    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    updated_config_data = update_dict(config_data, test_overrides)
    updated_config_data["data"]["file_path"] = "data/test_data.hdf5"
    updated_config_data["data"]["parameters_path"] = "data/dataset_parameters.csv"

    test_config = Config(**updated_config_data)

    test_db_path = Path(test_config.optuna_storage_url.replace("sqlite:///", ""))
    if test_db_path.exists():
        test_db_path.unlink()

    monkeypatch.setattr("nconfig._config", test_config)

    yield test_config


def test_single_training_run_pipeline(override_config):
    """Tests if the single training run pipeline executes without crashing.
    This is a smoke test, not an accuracy test.
    """
    config = override_config
    trainer = ModelTrainer(config)
    avg_cv_loss = trainer.train()

    assert avg_cv_loss is not None
    assert isinstance(avg_cv_loss, float)


def test_hyperparameter_search_pipeline(override_config):
    """Tests if the hyperparameter search pipeline executes a single trial."""
    config = override_config
    config.optuna.n_trials = 1
    config.optuna.time_limit_per_trial = 30.0
    optimizer = HyperparameterOptimizer(config)
    results = optimizer.run_optimization()

    assert results is not None
    assert results.n_trials == 1
