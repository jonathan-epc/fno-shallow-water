# tests/ml/test_pipeline_integration.py


import sys
from pathlib import Path

import h5py
import pytest
import yaml

from ML.core.optimizer import HyperparameterOptimizer
from ML.core.trainer import ModelTrainer
from nconfig import _config as global_config
from nconfig import get_config

# Ensure test data exists
project_root = Path(__file__).resolve().parents[2]
test_data_path = project_root / "data" / "test_data.hdf5"
parameters_path = project_root / "data" / "dataset_parameters.csv"

needs_gen = not test_data_path.exists() or not parameters_path.exists()
if not needs_gen:
    try:
        with h5py.File(test_data_path, "r") as f:
            sample_shape = f["simulation_0"]["B"].shape
            cfg = get_config("config.yml")
            if sample_shape != (cfg.mesh.num_points_y, cfg.mesh.num_points_x):
                needs_gen = True
    except Exception:
        needs_gen = True

if needs_gen:
    print("Test data missing or dimension mismatch. Generating...")
    sys.path.append(str(project_root / "tests"))
    from generate_test_data import create_fake_test_data

    create_fake_test_data()

# --- Test Fixture to Load and Override Config ---


@pytest.fixture
def override_config():
    """
    A pytest fixture that cleans up old test databases, loads the main config,
    and overrides it with test settings.
    """
    # --- Setup ---
    # Load the main configuration first to know which DB to clean
    config = get_config("config.yml")

    # Load the test-specific overrides
    test_config_path = project_root / "tests" / "test_config.yml"
    with open(test_config_path) as f:
        test_overrides = yaml.safe_load(f)

    # Create a deep copy of the original config data to modify
    config_data = config.model_dump()

    # Recursively update the main config with test overrides
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

    from nconfig import Config

    test_config = Config(**updated_config_data)

    # --- CLEAN UP *BEFORE* THE TEST ---
    # This avoids the file-locking issue on Windows.
    test_db_path = Path(test_config.optuna_storage_url.replace("sqlite:///", ""))
    if test_db_path.exists():
        print(f"\n--- Cleaning up previous test database: {test_db_path} ---")
        test_db_path.unlink()

    # Monkeypatch the global config
    global global_config
    original_config = global_config
    global_config = test_config

    yield test_config  # Provide the test_config to the test function

    # --- Teardown ---
    # Restore the original global config
    global_config = original_config


# --- Integration Tests ---


def test_single_training_run_pipeline(override_config):
    """
    Tests if the single training run pipeline executes without crashing.
    This is a smoke test, not an accuracy test.

    Args:
        override_config (Config): The pytest fixture providing the test config.
    """
    print("--- Running Single Training Integration Test ---")
    config = override_config
    trainer = ModelTrainer(config)
    avg_cv_loss = trainer.train()

    # The test passes if the run completes and returns a loss (not None)
    assert avg_cv_loss is not None
    assert isinstance(avg_cv_loss, float)


def test_hyperparameter_search_pipeline(override_config):
    """
    Tests if the hyperparameter search pipeline executes a single trial.

    Args:
        override_config (Config): The pytest fixture providing the test config.
    """
    print("--- Running Hyperparameter Search Integration Test ---")
    config = override_config
    optimizer = HyperparameterOptimizer(config)
    results = optimizer.run_optimization()

    # The test passes if the optimization completes and returns a result object
    assert results is not None
    assert results.n_trials == 1
