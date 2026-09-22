from unittest.mock import patch

import torch

from ML.modules.data import HDF5Dataset
from ML.modules.loss import PhysicsInformedLoss
from nconfig import get_config


def test_spatial_derivative_axis_unpacking():
    """
    Manufactured Solution Test:
    Ensures torch.gradient unpacking never swaps x and y axes.
    We define u(y, x) = 2.0 * x* + 7.0 * y*.
    The true derivatives are:
        du/dx* = 2.0
        du/dy* = 7.0
    If axes are swapped, du/dx* would be 7.0 and du/dy* would be 2.0.
    """
    ny, nx = 40, 60
    # Create non-square normalized grid in [0, 1]
    y = torch.linspace(0, 1, ny).view(1, ny, 1).repeat(1, 1, nx)
    x = torch.linspace(0, 1, nx).view(1, 1, nx).repeat(1, ny, 1)

    spacing = (1.0 / (ny - 1), 1.0 / (nx - 1))
    u = 2.0 * x + 7.0 * y

    # Batched gradient matching loss.py
    stacked = torch.stack([u], dim=1)
    d_dy, d_dx = torch.gradient(stacked, spacing=spacing, dim=(2, 3))
    du_dy = d_dy[:, 0]
    du_dx = d_dx[:, 0]

    # Interior points should match exact slopes
    interior_x = du_dx[:, 1:-1, 1:-1]
    interior_y = du_dy[:, 1:-1, 1:-1]

    assert torch.allclose(interior_x, torch.tensor(2.0), atol=1e-3), (
        f"Axis error: du/dx expected 2.0, got {interior_x.mean().item():.4f}"
    )
    assert torch.allclose(interior_y, torch.tensor(7.0), atol=1e-3), (
        f"Axis error: du/dy expected 7.0, got {interior_y.mean().item():.4f}"
    )


def test_data_augmentation_physical_symmetry():
    """
    Physical Symmetry Test:
    Flipping across the channel width (y-axis, dim 0) must negate transverse
    velocity V and V* (-v) to preserve fluid symmetry, while scalars and
    streamwise velocity remain unnegated.
    """
    ny, nx = 6, 8
    v_field = torch.arange(ny * nx, dtype=torch.float32).reshape(ny, nx) + 1.0
    h_field = torch.ones((ny, nx), dtype=torch.float32) * 3.0

    dataset = HDF5Dataset.__new__(HDF5Dataset)
    dataset.augment = True
    dataset.input_vars = ["H", "V"]
    dataset.output_vars = ["V"]
    dataset.SCALARS = ["H0"]
    dataset.FIELDS = ["H", "V"]
    dataset.non_scalar_input_indices = ["H", "V"]
    dataset.non_scalar_output_indices = ["V"]
    dataset.scalar_input_indices = []
    dataset.scalar_output_indices = []
    dataset.normalize_input = False
    dataset.normalize_output = False
    dataset.channel_length = 10.0
    dataset.channel_width = 5.0

    # Force random > 0.5 to trigger augmentation
    with (
        patch("torch.rand", return_value=torch.tensor([0.9])),
        patch.object(
            dataset,
            "_get_case",
            return_value={"H": h_field, "V": v_field, "L": 10.0, "W": 5.0},
        ),
        patch.object(
            dataset, "_process_variable", side_effect=lambda case, var, norm: case[var]
        ),
    ):
        (in_f, in_s), (out_f, out_s), _ = dataset[0]

    expected_h = torch.flip(h_field, [0])
    expected_v = -torch.flip(v_field, [0])

    assert torch.equal(in_f[0], expected_h), "H should be flipped without sign change"
    assert torch.equal(in_f[1], expected_v), "V must be negated upon y-flip"
    assert torch.equal(out_f[0], expected_v), "Output V must be negated upon y-flip"


def test_process_variable_returns_cpu_tensor():
    """
    Multi-worker Safety Test:
    _process_variable must return CPU tensors even when dataset device is CUDA,
    ensuring DataLoader with num_workers > 0 operates without subprocess crashes.
    """
    dataset = HDF5Dataset.__new__(HDF5Dataset)
    dataset.nx = 10
    dataset.ny = 5
    dataset.FIELDS = ["H"]
    dataset.SCALARS = []
    dataset.boxcox_transform = False
    dataset.boxcox_lambdas = {}
    dataset.device = "cuda"

    case = {"H": torch.ones((5, 10), dtype=torch.float32)}
    res = dataset._process_variable(case, "H", normalize=False)
    assert res.device.type == "cpu", f"Expected CPU tensor, got {res.device}"


def test_varied_flow_regimes_stability():
    """
    Flow Regimes & Stability Test:
    Evaluates loss over varied regimes (subcritical, supercritical, high Re, shallow depths)
    and verifies that outputs are finite with no NaNs or Infs.
    """
    config = get_config("config.yml")

    loss_fn = PhysicsInformedLoss.__new__(PhysicsInformedLoss)
    torch.nn.Module.__init__(loss_fn)
    loss_fn.config = config
    loss_fn.epsilon = 1e-5
    loss_fn.physics_loss_fn = torch.nn.MSELoss()

    ny, nx = 20, 40
    dx_norm = 1.0 / (nx - 1)
    dy_norm = 1.0 / (ny - 1)
    loss_fn.spacing = (dy_norm, dx_norm)

    batch_size = 4
    # Test cases: subcritical (Fr < 1), supercritical (Fr > 1), near-dry (small h)
    h = torch.tensor([0.001, 0.5, 1.0, 5.0]).view(batch_size, 1, 1).repeat(1, ny, nx)
    u = torch.tensor([0.2, 1.5, 3.0, 0.8]).view(batch_size, 1, 1).repeat(1, ny, nx)
    v = torch.tensor([0.01, -0.2, 0.5, 0.0]).view(batch_size, 1, 1).repeat(1, ny, nx)
    b = torch.zeros(batch_size, ny, nx)

    Fr = torch.tensor([0.2, 0.8, 1.5, 3.0]).view(batch_size, 1, 1)
    Re = torch.tensor([50.0, 500.0, 5000.0, 50000.0]).view(batch_size, 1, 1)
    Ar = torch.tensor([2.0, 5.0, 10.0, 20.0]).view(batch_size, 1, 1)
    Vr = Ar
    Hr = torch.tensor([0.1, 0.5, 1.0, 2.0]).view(batch_size, 1, 1)
    M = torch.tensor([0.01, 0.05, 0.1, 0.5]).view(batch_size, 1, 1)

    # Mock assign_variables
    variables = {
        "H*": h,
        "U*": u,
        "V*": v,
        "B*": b,
        "Fr": Fr,
        "Re": Re,
        "Ar": Ar,
        "Vr": Vr,
        "Hr": Hr,
        "M": M,
    }
    with patch.object(loss_fn, "assign_variables", return_value=(variables, [])):
        losses, missing = loss_fn.compute_adimensional_physics_loss(
            ([], []), (None, None)
        )

    assert missing == []
    assert len(losses) == 3
    for loss in losses:
        assert torch.isfinite(loss), f"Loss term contains NaN or Inf: {loss}"


def test_relobralo_eval_mode_no_mutation():
    """Verify that ReLoBRaLo loss weights and history do not mutate in eval mode."""
    config = get_config("config.yml")
    loss_fn = PhysicsInformedLoss.__new__(PhysicsInformedLoss)
    torch.nn.Module.__init__(loss_fn)
    loss_fn.config = config
    loss_fn.epsilon = 1e-5
    loss_fn.alpha = 0.999
    loss_fn.temperature = 0.1
    loss_fn.rho = 0.99
    loss_fn.call_count = 0
    loss_fn.data_loss = torch.nn.HuberLoss()
    loss_fn.physics_loss_fn = torch.nn.MSELoss()
    loss_fn.register_buffer("lambdas", torch.ones(4))
    loss_fn.register_buffer("last_losses", torch.ones(4))
    loss_fn.register_buffer("init_losses", torch.ones(4))
    loss_fn.input_vars = ["H", "U", "V"]
    loss_fn.output_vars = ["H", "U", "V"]
    loss_fn.use_physics_loss = False

    # Check that lambdas is a registered buffer
    assert "lambdas" in dict(loss_fn.named_buffers())

    loss_fn.eval()
    initial_lambdas = loss_fn.lambdas.clone()
    initial_call_count = loss_fn.call_count

    # Simulate forward call in eval mode
    dummy_pred = (torch.zeros(2, 3, 10, 10), None)
    dummy_target = ([torch.ones(2, 10, 10) for _ in range(3)], [])
    loss_fn(([], []), dummy_pred, dummy_target)

    assert loss_fn.call_count == initial_call_count
    assert torch.equal(loss_fn.lambdas, initial_lambdas)

    # In train mode with physics loss, it should mutate
    loss_fn.train()
    loss_fn.use_physics_loss = True
    dummy_physics = ([torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0)], [])
    compute_method = (
        "compute_adimensional_physics_loss"
        if config.data.is_adimensional
        else "compute_physics_loss"
    )
    with patch.object(loss_fn, compute_method, return_value=dummy_physics):
        loss_fn(([], []), dummy_pred, dummy_target)

    assert loss_fn.call_count == initial_call_count + 1

    # Verify reset() method restores initial state for fold isolation
    loss_fn.reset()
    assert loss_fn.call_count == 0
    assert torch.equal(loss_fn.lambdas, torch.ones(4))
    assert torch.equal(loss_fn.last_losses, torch.ones(4))
    assert torch.equal(loss_fn.init_losses, torch.ones(4))
