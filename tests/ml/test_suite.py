import sys
from unittest.mock import MagicMock, patch

import pytest

from ML.core.suite import (
    ALL_CASE_KEYS,
    apply_runtime_overrides,
    build_case_config,
    execute_case_batch,
    resolve_dataset_path,
    resolve_test_dataset_path,
)
from ML.modules.data import HDF5Dataset
from ML.scripts.run_hyperparameter_search import main as hparam_main
from ML.scripts.train_model import main as train_main

pytestmark = pytest.mark.unit


def test_all_case_keys_cover_12_configurations():
    """Verify the suite defines all 12 (direct/inverse x dim/adim x 3 geometries) cases."""
    assert len(ALL_CASE_KEYS) == 12
    assert set(ALL_CASE_KEYS) == {
        "ddb",
        "idb",
        "dab",
        "iab",
        "dds",
        "ids",
        "das",
        "ias",
        "ddn",
        "idn",
        "dan",
        "ian",
    }


def test_resolve_dataset_path_priority_and_fallback(base_config, tmp_path):
    """Verify dataset resolution prefers train_val_*, then config filename, then <GEOM>.hdf5."""
    cfg = base_config.model_copy(deep=True)
    cfg.paths.data_dir = str(tmp_path)

    with pytest.raises(FileNotFoundError):
        resolve_dataset_path(cfg, "b")

    geom_file = tmp_path / "BARS.hdf5"
    geom_file.write_bytes(b"placeholder")
    assert resolve_dataset_path(cfg, "b") == geom_file

    configured_file = tmp_path / cfg.datasets["b"].filename
    configured_file.write_bytes(b"placeholder")
    assert resolve_dataset_path(cfg, "b") == configured_file

    train_val_file = tmp_path / "train_val_BARS.hdf5"
    train_val_file.write_bytes(b"placeholder")
    assert resolve_dataset_path(cfg, "b") == train_val_file


def test_build_case_config_sets_inputs_outputs_and_overrides(base_config, tmp_path):
    """Verify build_case_config configures adimensional/dimensional cases cleanly."""
    cfg = base_config.model_copy(deep=True)
    cfg.paths.data_dir = str(tmp_path)
    (tmp_path / "train_val_BARS.hdf5").write_bytes(b"placeholder")

    dab_cfg = build_case_config(cfg, "dab", epochs=50, trials=10, device="cpu")
    assert dab_cfg.data.is_adimensional is True
    assert dab_cfg.data.inputs == ["Hr", "Fr", "M", "Re", "B*", "Ar", "Vr"]
    assert dab_cfg.data.outputs == ["H*", "U*", "V*"]
    assert dab_cfg.training.num_epochs == 50
    assert dab_cfg.optuna.n_trials == 10
    assert dab_cfg.device == "cpu"
    assert "DAB-BARS" in dab_cfg.model.name
    assert dab_cfg.optuna.study_name == "Study-DAB-BARS"


def test_build_case_config_rejects_unknown_case(base_config):
    """Verify unknown case key raises ValueError."""
    with pytest.raises(ValueError, match="Unknown case"):
        build_case_config(base_config, "invalid_case")


def test_apply_runtime_overrides_cpu_fallback(base_config):
    """Verify runtime override updates epochs, trials, and device safely."""
    cfg = base_config.model_copy(deep=True)
    apply_runtime_overrides(cfg, epochs=25, trials=15, device="cpu")
    assert cfg.training.num_epochs == 25
    assert cfg.optuna.n_trials == 15
    assert cfg.device == "cpu"


def test_execute_case_batch_collects_success_and_failure(base_config, tmp_path):
    """Verify execute_case_batch runs all requested cases and reports failures."""
    cfg = base_config.model_copy(deep=True)
    cfg.paths.data_dir = str(tmp_path)
    (tmp_path / "train_val_BARS.hdf5").write_bytes(b"placeholder")
    (tmp_path / "train_val_SLOPE.hdf5").write_bytes(b"placeholder")

    logger = MagicMock()

    def fake_runner(case_cfg):
        if "SLOPE" in case_cfg.model.name:
            raise RuntimeError("boom")
        return 0.01

    succeeded, failed = execute_case_batch(
        base_config=cfg,
        case_keys=["ddb", "dds"],
        runner_fn=fake_runner,
        logger=logger,
    )
    assert succeeded == ["ddb"]
    assert len(failed) == 1
    assert failed[0][0] == "dds"


def test_train_model_all_flag_dispatches_all_12_cases(
    base_config, tmp_path, monkeypatch
):
    """Verify train_model.py --all dispatches ModelTrainer across all 12 cases."""
    cfg = base_config.model_copy(deep=True)
    cfg.paths.data_dir = str(tmp_path)
    for geom in ("BARS", "SLOPE", "NOISE"):
        (tmp_path / f"train_val_{geom}.hdf5").write_bytes(b"placeholder")

    monkeypatch.setattr("ML.scripts.train_model.get_config", lambda: cfg)
    monkeypatch.setattr(sys, "argv", ["train_model.py", "--all", "--epochs", "3"])

    with patch("ML.scripts.train_model.ModelTrainer") as mock_trainer_cls:
        instance = MagicMock()
        instance.train.return_value = 0.02
        mock_trainer_cls.return_value = instance

        train_main()
        assert mock_trainer_cls.call_count == 12


def test_hparam_search_cases_flag_dispatches_subset(base_config, tmp_path, monkeypatch):
    """Verify run_hyperparameter_search.py --cases dispatches selected studies."""
    cfg = base_config.model_copy(deep=True)
    cfg.paths.data_dir = str(tmp_path)
    (tmp_path / "train_val_BARS.hdf5").write_bytes(b"placeholder")

    monkeypatch.setattr("ML.scripts.run_hyperparameter_search.get_config", lambda: cfg)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_hyperparameter_search.py", "--cases", "ddb", "iab", "--trials", "5"],
    )

    with patch(
        "ML.scripts.run_hyperparameter_search.HyperparameterOptimizer"
    ) as mock_opt_cls:
        instance = MagicMock()
        res = MagicMock()
        res.best_value = 0.01
        instance.run_optimization.return_value = res
        mock_opt_cls.return_value = instance

        hparam_main()
        assert mock_opt_cls.call_count == 2


def test_resolve_test_dataset_path_finds_sibling_test_file(tmp_path):
    """Verify resolve_test_dataset_path switches train_val_* to sibling test_* file."""
    train_file = tmp_path / "train_val_BARS.hdf5"
    train_file.write_bytes(b"placeholder")
    assert resolve_test_dataset_path(str(train_file)) == str(train_file)

    test_file = tmp_path / "test_BARS.hdf5"
    test_file.write_bytes(b"placeholder")
    assert resolve_test_dataset_path(str(train_file)) == str(test_file)


def test_hdf5_dataset_matches_numeric_parameter_ids(tmp_path):
    """Verify HDF5Dataset matches integer IDs in parameters.csv to simulation_<id> groups."""
    import h5py
    import numpy as np
    import pandas as pd

    h5_path = tmp_path / "train_val_SLOPE.hdf5"
    with h5py.File(h5_path, "w") as f:
        for sim_id in (0, 1):
            grp = f.create_group(f"simulation_{sim_id}")
            grp.attrs["H0"] = 0.2
            grp.attrs["Q0"] = 0.01
            grp.attrs["n"] = 0.02
            grp.attrs["nut"] = 1e-4
            grp.create_dataset("B", data=np.zeros((4, 8), dtype=np.float32))
            grp.create_dataset("H", data=np.ones((4, 8), dtype=np.float32))

    csv_path = tmp_path / "parameters.csv"
    pd.DataFrame(
        [
            {"id": 0, "set_type": "train_val"},
            {"id": 1, "set_type": "test"},
        ]
    ).to_csv(csv_path, index=False)

    ds = HDF5Dataset(
        file_path=str(h5_path),
        input_vars=["H0", "Q0", "n", "nut", "B"],
        output_vars=["H"],
        numpoints_x=8,
        numpoints_y=4,
        channel_length=12.0,
        channel_width=0.3,
        parameters_file_path=str(csv_path),
        filter_set_type="train_val",
        augment=False,
    )
    assert ds.keys == ["simulation_0"]
    assert len(ds) == 1
