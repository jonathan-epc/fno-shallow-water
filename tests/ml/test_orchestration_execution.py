from pathlib import Path
from unittest.mock import MagicMock, patch

import optuna
import pytest
import torch

from ML.core.finetuner import TRAINED_MODELS_INFO, ModelFineTuner
from ML.core.repeater import TrialRepeater
from ML.core.results import ResultsLoader
from ML.modules.metrics import evaluate_predictions
from nconfig import get_config


@pytest.fixture
def base_config():
    return get_config().model_copy(deep=True)


def test_finetuner_execution_flow(base_config):
    """Verify ModelFineTuner.finetune completes CV and test evaluation with mocked weights."""
    sample_model_info = TRAINED_MODELS_INFO["ddb"]
    target_data_path = Path(__file__).resolve().parents[2] / "data" / "test_data.hdf5"
    cfg = base_config.model_copy(deep=True)
    cfg.data.file_path = str(target_data_path)
    cfg.training.kfolds = 2

    fine_tuner = ModelFineTuner(
        config=cfg,
        source_model_info=sample_model_info,
        target_dataset_path=str(target_data_path),
        finetune_run_name="test_ft_execution",
    )

    mock_hparams = {
        "n_modes_x": 4,
        "n_modes_y": 4,
        "hidden_channels": 8,
        "n_layers": 1,
        "learning_rate": 1e-3,
        "batch_size": 2,
    }
    mock_io_counts = {
        "field_inputs_n": 1,
        "scalar_inputs_n": 1,
        "field_outputs_n": 1,
        "scalar_outputs_n": 1,
    }

    with (
        patch.object(
            fine_tuner,
            "_load_source_model_hparams",
            return_value=(mock_hparams, mock_io_counts),
        ),
        patch(
            "ML.core.finetuner.cross_validation_procedure", return_value=0.0321
        ) as mock_cv,
        patch.object(Path, "exists", return_value=False),
    ):
        loss = fine_tuner.finetune()
        assert loss == 0.0321
        assert mock_cv.called


def test_repeater_execution_flow(base_config):
    """Verify TrialRepeater loads trial and invokes cross-validation procedure."""
    cfg = base_config.model_copy(deep=True)
    target_data_path = Path(__file__).resolve().parents[2] / "data" / "test_data.hdf5"
    cfg.data.file_path = str(target_data_path)
    repeater = TrialRepeater(config=cfg, trial_id=0)

    mock_trial = MagicMock()
    mock_trial.state = optuna.trial.TrialState.COMPLETE
    mock_trial.params = {
        "n_modes_x": 4,
        "n_modes_y": 4,
        "hidden_channels": 8,
        "n_layers": 1,
    }
    mock_trial.value = 0.045

    mock_study = MagicMock()
    mock_study.trials = {0: mock_trial}

    with (
        patch("optuna.load_study", return_value=mock_study),
        patch(
            "ML.core.repeater.cross_validation_procedure", return_value=0.042
        ) as mock_cv,
    ):
        result = repeater.run()
        assert result == 0.042
        assert mock_cv.called


def test_results_loader_setup_datasets(base_config):
    """Verify ResultsLoader._setup_datasets sets up full_dataset and test_loader properly."""
    cfg = base_config.model_copy(deep=True)
    target_data_path = Path(__file__).resolve().parents[2] / "data" / "test_data.hdf5"
    cfg.data.file_path = str(target_data_path)

    loader = ResultsLoader.__new__(ResultsLoader)
    loader.config = cfg
    loader.hparams = {"batch_size": 2}

    loader._setup_datasets()
    assert loader.test_loader is not None
    assert len(loader.test_loader) > 0
    assert hasattr(loader.full_dataset, "stats")


def test_evaluate_predictions_metrics():
    """Verify evaluate_predictions computes MSE, RMSE, MAE, R², and SMAPE."""
    pred_fields = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target_fields = torch.tensor([[[[1.0, 2.1], [2.9, 4.0]]]])

    metrics, df = evaluate_predictions(
        predictions=(pred_fields, None),
        targets=(target_fields, None),
        output_fields=["U"],
        output_scalars=[],
    )

    assert "U" in metrics
    assert "mse" in metrics["U"]
    assert "r2" in metrics["U"]
    assert metrics["U"]["mse"] >= 0.0
    assert not df.empty
    assert "case_id" in df.columns
