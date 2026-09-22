from pathlib import Path

import pytest
import torch

from ML.modules.data import HDF5Dataset
from ML.modules.models import FNOnet

pytestmark = pytest.mark.unit


def test_fno_net_forward_shapes():
    """Verify FNOnet forward pass produces expected field and scalar tensor shapes."""
    batch_size = 2
    height, width = 11, 401

    # Network with 1 field input, 2 scalar inputs, 2 field outputs, 1 scalar output
    model = FNOnet(
        field_inputs_n=1,
        scalar_inputs_n=2,
        field_outputs_n=2,
        scalar_outputs_n=1,
        n_modes_x=8,
        n_modes_y=4,
        hidden_channels=16,
        n_layers=2,
        lifting_channels=16,
        projection_channels=16,
    )

    field_inputs = [torch.randn(batch_size, height, width)]
    scalar_inputs = [torch.randn(batch_size), torch.randn(batch_size)]

    field_out, scalar_out = model((field_inputs, scalar_inputs))

    assert field_out is not None
    assert field_out.shape == (batch_size, 2, height, width)

    assert scalar_out is not None
    assert scalar_out.shape == (batch_size, 1)


def test_fno_net_field_only_outputs():
    """Verify FNOnet with zero scalar outputs returns None for scalar_out."""
    batch_size = 2
    height, width = 11, 64

    model = FNOnet(
        field_inputs_n=1,
        scalar_inputs_n=0,
        field_outputs_n=1,
        scalar_outputs_n=0,
        n_modes_x=4,
        n_modes_y=4,
        hidden_channels=8,
        n_layers=1,
        lifting_channels=8,
        projection_channels=8,
    )

    field_inputs = [torch.randn(batch_size, height, width)]
    scalar_inputs = []

    field_out, scalar_out = model((field_inputs, scalar_inputs))

    assert field_out is not None
    assert field_out.shape == (batch_size, 1, height, width)
    assert scalar_out is None


def test_hdf5_dataset_loading_and_shapes(base_config):
    """Verify HDF5Dataset loads synthetic test data with correct tensor formatting and statistics."""
    test_data_path = Path(__file__).resolve().parents[2] / "data" / "test_data.hdf5"
    cfg = base_config.model_copy(deep=True)
    cfg.data.file_path = str(test_data_path)

    dataset = HDF5Dataset.from_config(cfg, file_path=str(test_data_path))
    assert len(dataset) > 0

    # Normalization requires statistics to be populated
    dataset.compute_statistics()
    assert hasattr(dataset, "stats")
    assert dataset.stats is not None

    inputs, targets, metadata = dataset[0]
    field_inputs, scalar_inputs = inputs
    field_targets, scalar_targets = targets

    assert isinstance(field_inputs, list)
    assert isinstance(scalar_inputs, list)
    assert isinstance(field_targets, list)
    assert isinstance(scalar_targets, list)
    assert isinstance(metadata, dict)

    # First field input has (height, width) matching mesh config
    if field_inputs:
        assert field_inputs[0].shape == (cfg.mesh.num_points_y, cfg.mesh.num_points_x)
