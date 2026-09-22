import runpy
import sys
from unittest.mock import MagicMock, patch

import pytest

from ML.scripts.run_hyperparameter_search import main as hparam_main
from ML.scripts.run_transfer_learning import main as transfer_learning_main
from ML.scripts.train_model import main as train_main


@pytest.mark.parametrize(
    "module_name",
    [
        "ML.scripts.run_trial_repeat",
        "ML.scripts.measure_ml_times",
        "ML.scripts.generate_plots",
        "simulation_data_processor",
        "telemac.input_generator",
        "telemac.run_telemac_simulations",
    ],
)
def test_cli_scripts_help_smoke(module_name, monkeypatch, capsys):
    """Verify that CLI entry points parse --help and exit cleanly in-process."""
    monkeypatch.setattr(sys, "argv", [module_name, "--help"])
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module(module_name, run_name="__main__")
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert (
        "show this help message and exit" in captured.out
        or "--help" in captured.out
        or "-h" in captured.out
    )


def test_train_model_main_execution():
    """Verify train_model.main executes configuration and trainer invocation."""
    with patch("ML.scripts.train_model.ModelTrainer") as mock_trainer_cls:
        instance = MagicMock()
        instance.train.return_value = 0.05
        mock_trainer_cls.return_value = instance

        train_main()
        assert instance.train.called


def test_run_hyperparameter_search_main_execution():
    """Verify run_hyperparameter_search.main executes optimizer pipeline."""
    with patch(
        "ML.scripts.run_hyperparameter_search.HyperparameterOptimizer"
    ) as mock_opt_cls:
        instance = MagicMock()
        mock_res = MagicMock()
        mock_res.best_value = 0.042
        instance.run_optimization.return_value = mock_res
        mock_opt_cls.return_value = instance

        hparam_main()
        assert instance.run_optimization.called


def test_run_transfer_learning_main_execution(monkeypatch):
    """Verify run_transfer_learning.main executes fine-tuner pipeline."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_transfer_learning.py",
            "--source_model",
            "ddb",
            "--target_dataset",
            "data/test_data.hdf5",
            "--run_name",
            "test_transfer_cli",
        ],
    )
    with patch("ML.scripts.run_transfer_learning.ModelFineTuner") as mock_finetuner_cls:
        instance = MagicMock()
        instance.finetune.return_value = 0.035
        mock_finetuner_cls.return_value = instance

        transfer_learning_main()
        assert instance.finetune.called
