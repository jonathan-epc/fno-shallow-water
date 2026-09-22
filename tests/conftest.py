import sys
from pathlib import Path

import h5py
import pytest

from nconfig import get_config


@pytest.fixture(scope="session", autouse=True)
def ensure_synthetic_test_fixtures():
    """
    Session-level autouse fixture to guarantee that synthetic test data
    (test_data.hdf5 and dataset_parameters.csv) exists before any test runs.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    test_data_path = data_dir / "test_data.hdf5"
    parameters_path = data_dir / "dataset_parameters.csv"

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
        tests_dir = project_root / "tests"
        if str(tests_dir) not in sys.path:
            sys.path.insert(0, str(tests_dir))
        from generate_test_data import create_fake_test_data

        create_fake_test_data()
