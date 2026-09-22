from telemac.modules.flux_checker import is_flux_balanced
from telemac.modules.hydraulic_calculations import HydraulicCalculations
from telemac.modules.telemac_case import TelemacCase


def test_exploring_hydraulic_functions():
    """Verify numerical functions in telemac.modules.hydraulic_calculations execute and converge."""
    flow_rate = 0.01
    bottom_width = 0.3
    slope = 0.01
    roughness = 0.02

    yn = HydraulicCalculations.normal_depth(flow_rate, bottom_width, slope, roughness)
    assert yn > 0

    yc = HydraulicCalculations.critical_depth(flow_rate, bottom_width)
    assert yc > 0

    sc = HydraulicCalculations.critical_slope(flow_rate, bottom_width, roughness)
    assert sc > 0


def test_telemac_case_instantiation():
    """Verify TelemacCase initializes file paths correctly."""
    case_params = {
        "SLOPE": 0.01,
        "BOTTOM": "SLOPE",
        "Q0": 0.01,
        "H0": 0.15,
        "L": 12.0,
        "W": 0.3,
        "n": 0.02,
        "direction": "x",
        "num_points_x": 401,
        "num_points_y": 32,
    }

    case = TelemacCase(case_id=42, params=case_params)
    assert case.case_id == 42
    assert "42" in str(case.steering_file_path)
    assert "42" in str(case.geometry_file_path)
    assert "42" in str(case.results_file_path)


def test_flux_checker_tolerance():
    """Verify flux balance helper logic."""
    # Perfectly balanced
    assert is_flux_balanced(1.0, -1.0, tolerance=0.05) is True
    # Within 5% tolerance
    assert is_flux_balanced(1.0, -1.04, tolerance=0.05) is True
    # Outside tolerance
    assert is_flux_balanced(1.0, -1.15, tolerance=0.05) is False
    # Zero flux edge cases
    assert is_flux_balanced(0.0, 0.0, tolerance=0.05) is True
    assert is_flux_balanced(0.0, 0.5, tolerance=0.05) is False
