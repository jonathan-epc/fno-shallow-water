from pathlib import Path

import pytest

from ML.core.finetuner import (
    GEOM_NAMES,
    GEOMETRY_FILES,
    TRAINED_MODELS_INFO,
    ModelFineTuner,
)
from ML.core.repeater import TrialRepeater
from ML.core.results import ResultsLoader
from ML.modules.models import FNOnet
from nconfig import get_config


@pytest.fixture
def base_config():
    return get_config().model_copy(deep=True)


def test_metadata_dictionaries_integrity():
    """Verify metadata mappings and keys used for transfer learning and results."""
    assert len(TRAINED_MODELS_INFO) == 12
    for _model_key, info in TRAINED_MODELS_INFO.items():
        assert "trial_number" in info
        assert "study_name" in info
        assert "source_geom" in info
        assert "inputs" in info
        assert "outputs" in info
        assert len(info["inputs"]) > 0
        assert len(info["outputs"]) > 0

    assert isinstance(GEOM_NAMES, dict)
    assert isinstance(GEOMETRY_FILES, dict)


def test_finetuner_class_resolution(base_config):
    """Verify ModelFineTuner resolves model class dynamically."""
    sample_model_info = TRAINED_MODELS_INFO["ddb"]
    target_data_path = Path(__file__).resolve().parents[2] / "data" / "test_data.hdf5"
    cfg = base_config.model_copy(deep=True)

    fine_tuner = ModelFineTuner(
        config=cfg,
        source_model_info=sample_model_info,
        target_dataset_path=str(target_data_path),
        finetune_run_name="smoke_test_run",
    )

    resolved_class = fine_tuner._get_model_class()
    assert resolved_class is FNOnet

    # Test error handling on non-existent architecture
    fine_tuner.base_config.model.class_name = "NonExistentModelArchitecture"
    with pytest.raises(
        ValueError, match="Model class 'NonExistentModelArchitecture' not found"
    ):
        fine_tuner._get_model_class()


def test_repeater_class_resolution(base_config):
    """Verify TrialRepeater resolves model class dynamically."""
    cfg = base_config.model_copy(deep=True)
    repeater = TrialRepeater(config=cfg, trial_id=0)
    resolved_class = repeater._get_model_class()
    assert resolved_class is FNOnet

    repeater.config.model.class_name = "NonExistentModelArchitecture"
    with pytest.raises(
        ValueError, match="Model class 'NonExistentModelArchitecture' not found"
    ):
        repeater._get_model_class()


def test_results_loader_class_resolution(base_config):
    """Verify ResultsLoader resolves model class dynamically."""
    cfg = base_config.model_copy(deep=True)
    dummy_loader = ResultsLoader.__new__(ResultsLoader)
    dummy_loader.config = cfg

    resolved_class = dummy_loader._get_model_class()
    assert resolved_class is FNOnet

    dummy_loader.config.model.class_name = "NonExistentModelArchitecture"
    with pytest.raises(
        ValueError, match="Model class 'NonExistentModelArchitecture' not found"
    ):
        dummy_loader._get_model_class()


def test_finetuner_missing_target_file(base_config):
    """Verify ModelFineTuner throws FileNotFoundError if target dataset does not exist."""
    sample_model_info = TRAINED_MODELS_INFO["ddb"]
    non_existent_path = "non_existent_dataset_file.hdf5"

    with pytest.raises(FileNotFoundError, match="Target dataset not found"):
        ModelFineTuner(
            config=base_config,
            source_model_info=sample_model_info,
            target_dataset_path=non_existent_path,
        )
