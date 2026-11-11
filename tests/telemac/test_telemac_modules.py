from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from modules.statistics import (
    calculate_statistics,
    combine_statistics,
    normalize_statistics,
)
from simulation_data_processor import (
    ProcessingConfig,
    get_output_files,
    process_simulation_results,
)
from telemac.modules.log_parser import LogParser, SimulationStatus

pytestmark = pytest.mark.unit


def test_log_parser_outcomes(tmp_path: Path):
    """Verify LogParser accurately classifies BALANCED, STALLED, and CRASHED logs."""
    # 1. Non-existent file returns UNKNOWN
    assert (
        LogParser.analyze(tmp_path / "does_not_exist.log") == SimulationStatus.UNKNOWN
    )

    # 2. Crashed file (python traceback)
    crashed_log = tmp_path / "crash.log"
    crashed_log.write_text(
        "Some Telemac header\nTraceback (most recent call last):\nFile 'foo.py', line 1, in <module>\n",
        encoding="latin1",
    )
    assert LogParser.analyze(crashed_log) == SimulationStatus.CRASHED

    # 3. Incomplete file without success message
    incomplete_log = tmp_path / "incomplete.log"
    incomplete_log.write_text("Starting run...\nRunning step 1\n", encoding="latin1")
    assert LogParser.analyze(incomplete_log) == SimulationStatus.CRASHED

    # 4. Stalled run (tiny time-step)
    stalled_log = tmp_path / "stalled.log"
    stalled_lines = [
        "Starting run...",
        "TIME-STEP:    1.0E-005",
        "TIME-STEP:    2.0E-005",
        "TIME-STEP:    1.5E-005",
        "TIME-STEP:    1.2E-005",
        "TIME-STEP:    1.1E-005",
    ]
    stalled_log.write_text("\n".join(stalled_lines), encoding="latin1")
    assert LogParser.analyze(stalled_log) == SimulationStatus.STALLED

    # 5. Successfully finished and balanced
    balanced_log = tmp_path / "balanced.log"
    balanced_content = """
    My Work is Done
    BALANCE OF WATER VOLUME
    FLUX BOUNDARY   1:   1.2345
    FLUX BOUNDARY   2:   -1.2346
    """
    balanced_log.write_text(balanced_content, encoding="latin1")
    assert LogParser.analyze(balanced_log) == SimulationStatus.BALANCED

    # 6. Successfully finished but unbalanced flux
    unbalanced_log = tmp_path / "unbalanced.log"
    unbalanced_content = """
    My Work is Done
    BALANCE OF WATER VOLUME
    FLUX BOUNDARY   1:   1.2345
    FLUX BOUNDARY   2:   -0.5000
    """
    unbalanced_log.write_text(unbalanced_content, encoding="latin1")
    assert LogParser.analyze(unbalanced_log) == SimulationStatus.UNBALANCED


def test_simulation_data_processor_get_output_files():
    """Verify get_output_files generates valid structured output mapping."""
    config = ProcessingConfig(
        base_dir=".",
        bottom_types=["TEST_BUMP", "TEST_SLOPE"],
        separate_critical_states=True,
    )
    output_files = get_output_files(config, set_type="val")

    assert "TEST_BUMP" in output_files
    assert "TEST_SLOPE" in output_files

    bump_files = output_files["TEST_BUMP"]
    assert "main" in bump_files
    assert "normalized" in bump_files
    assert "main_subcritical" in bump_files
    assert "main_supercritical" in bump_files
    assert "val_TEST_BUMP.hdf5" in bump_files["main"]
    assert "normalized_val_TEST_BUMP.hdf5" in bump_files["normalized"]


def test_statistics_calculation_and_combination():
    """Verify statistical aggregation and combination algorithms."""
    s1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    s2 = pd.Series([6.0, 7.0, 8.0, 9.0, 10.0])

    stats1 = calculate_statistics(s1)
    stats2 = calculate_statistics(s2)

    assert stats1["count"] == 5
    assert stats1["mean"] == pytest.approx(3.0)
    assert stats1["min"] == 1.0
    assert stats1["max"] == 5.0
    assert stats1["variance"] == pytest.approx(s1.var())

    combined = combine_statistics(stats1, stats2)
    assert combined["count"] == 10
    assert combined["mean"] == pytest.approx(5.5)
    assert combined["min"] == 1.0
    assert combined["max"] == 10.0

    full_series = pd.concat([s1, s2])
    assert combined["variance"] == pytest.approx(full_series.var(), rel=1e-5)

    # Test normalization function
    table = pd.DataFrame(
        [
            {"names": "var_a", "mean": 10.0, "variance": 4.0},
        ]
    )
    norm_val = normalize_statistics(12.0, "var_a", table)
    # (12 - 10) / sqrt(4) = 2 / 2 = 1.0
    assert norm_val == pytest.approx(1.0)


def test_simulation_data_processor_reads_nested_channel_config(tmp_path: Path):
    """Verify process_simulation_results reads nested channel width/length and bottom_types."""
    cfg = ProcessingConfig(
        base_dir=str(tmp_path),
        parameters_file=str(tmp_path / "parameters.csv"),
    )
    yaml_config = {
        "channel": {"width": 0.3, "length": 15.0},
        "simulation_params": {"bottom_types": ["BARS", "SLOPE"]},
        "paths": {"data_dir": str(tmp_path)},
    }
    params_df = pd.DataFrame(
        [
            {
                "id": "case_1",
                "H0": 0.2,
                "Q0": 0.01,
                "SLOPE": 0.001,
                "n": 0.02,
                "nut": 1e-4,
                "BOTTOM": "BARS",
                "Ar": 50.0,
                "Vr": 1.0,
                "Fr": 0.5,
                "Hr": 0.1,
                "Re": 1000.0,
                "M": 1.0,
            }
        ]
    ).set_index("id")

    with (
        patch(
            "simulation_data_processor.load_and_validate_data",
            return_value=(params_df, ["case_1.slf"], ["H", "U", "V"]),
        ),
        patch("simulation_data_processor.get_output_files", return_value={}),
        patch("simulation_data_processor.process_data") as mock_process_data,
    ):
        process_simulation_results(cfg, yaml_config)
        assert cfg.channel_width == pytest.approx(0.3)
        assert cfg.channel_length == pytest.approx(15.0)
        assert cfg.bottom_types == ["BARS", "SLOPE"]
        assert mock_process_data.called


def test_configured_mesh_and_boundary_files_exist():
    """Verify configured base mesh and .cli boundary files exist and match node counts."""
    from nconfig import PROJECT_ROOT, get_config

    cfg = get_config()
    telemac_dir = PROJECT_ROOT / cfg.paths.telemac_dir
    mesh_path = telemac_dir / "geo" / cfg.mesh.base_mesh_file
    riv_cli = telemac_dir / "bnd" / cfg.mesh.subcritical_cli
    tor_cli = telemac_dir / "bnd" / cfg.mesh.supercritical_cli

    assert mesh_path.is_file(), f"Missing base mesh: {mesh_path}"
    assert riv_cli.is_file(), f"Missing subcritical .cli: {riv_cli}"
    assert tor_cli.is_file(), f"Missing supercritical .cli: {tor_cli}"

    expected_bnd_nodes = 2 * cfg.mesh.num_points_x + 2 * cfg.mesh.num_points_y - 4
    assert len(riv_cli.read_text().splitlines()) == expected_bnd_nodes
    assert len(tor_cli.read_text().splitlines()) == expected_bnd_nodes
