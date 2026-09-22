# tests/telemac/test_hydraulic_calculations.py


import pytest

from telemac.modules.hydraulic_calculations import HydraulicCalculations

# Define a tolerance for comparing floating-point numbers
TOLERANCE = 1e-4


def test_critical_depth_simple():
    """
    Tests the simplified critical depth calculation with a known value.
    Formula: yc = (q^2 / g)^(1/3), where q = Q/W
    """
    flow_rate = 0.01  # m^3/s
    width = 0.3  # m
    q = flow_rate / width  # m^2/s
    expected_yc = (q**2 / 9.81) ** (1 / 3)  # ~0.0483 m

    calculated_yc = HydraulicCalculations.critical_depth_simple(flow_rate, width)

    assert isinstance(calculated_yc, float)
    assert calculated_yc == pytest.approx(expected_yc, abs=TOLERANCE)


def test_normal_depth_simple():
    """
    Tests the simplified normal depth calculation with a known value.
    Formula: yn = (n*q / S^(1/2))^(3/5)
    """
    flow_rate = 0.015  # m^3/s
    width = 0.3  # m
    slope = 0.001  # 0.1%
    n = 0.025  # s/m^(1/3)
    q = flow_rate / width

    expected_yn = ((n * q) / (slope**0.5)) ** (3 / 5)  # ~0.103 m

    calculated_yn = HydraulicCalculations.normal_depth_simple(
        flow_rate, width, slope, n
    )

    assert isinstance(calculated_yn, float)
    assert calculated_yn == pytest.approx(expected_yn, abs=TOLERANCE)


@pytest.mark.parametrize(
    "flow_rate, width, slope, n, expected_subcritical",
    [
        (0.015, 0.3, 0.0001, 0.025, True),  # Mild slope, expect subcritical (yn > yc)
        (0.015, 0.3, 0.01, 0.012, False),  # Steep slope, expect supercritical (yn < yc)
        (0.01, 0.3, 0.0022, 0.015, True),  # A case closer to critical
    ],
)
def test_subcritical_determination(flow_rate, width, slope, n, expected_subcritical):
    """
    Tests if the relationship between normal and critical depth correctly
    determines the flow regime.
    """
    # We test this indirectly within ParameterManager, but here we test the core logic.
    yn = HydraulicCalculations.normal_depth_simple(flow_rate, width, slope, n)
    yc = HydraulicCalculations.critical_depth_simple(flow_rate, width)

    is_subcritical = yn > yc

    assert is_subcritical == expected_subcritical


def test_froude_number_actual_value():
    """Verify that froude_number returns actual Fr, not Fr - 1."""
    # Depth = 1.0 m, Q = 3.0 m^3/s, Width = 3.0 m -> Velocity = 1.0 m/s
    # Fr = V / sqrt(g * D) = 1.0 / sqrt(9.81 * 1.0) = 1 / 3.13209 ~ 0.31928
    fr = HydraulicCalculations.froude_number(1.0, 3.0, 3.0)
    assert isinstance(fr, float)
    assert fr == pytest.approx(0.31928, abs=1e-4)


def test_critical_depth_numerical_matches_simple():
    """Verify that critical_depth (numerical fsolve) matches critical_depth_simple."""
    flow_rate = 0.02
    width = 0.5
    yc_numerical = float(HydraulicCalculations.critical_depth(flow_rate, width))
    yc_simple = HydraulicCalculations.critical_depth_simple(flow_rate, width)

    # In a rectangular channel, both formulas describe the exact same critical state
    assert yc_numerical == pytest.approx(yc_simple, rel=1e-3)


def test_update_duration_robustness(tmp_path):
    """Verify update_duration handles missing DURATION lines and floats without crashing."""
    from telemac.modules.file_handler import update_duration

    # File without DURATION line
    file1 = tmp_path / "cas1.txt"
    file1.write_text("TITLE = 'Test'\nTIME STEP = 0.1\n")
    update_duration(str(file1))  # Should not raise UnboundLocalError

    # File with float DURATION
    file2 = tmp_path / "cas2.txt"
    file2.write_text("DURATION = 45.5\n")
    update_duration(str(file2), increment=15)
    assert "DURATION = 60\n" in file2.read_text()


def test_flux_checker_scientific_notation(tmp_path):
    """Verify flux_checker parses scientific notation with positive exponents (e.g., E+01)."""
    from telemac.modules.flux_checker import check_flux_boundaries, is_flux_balanced

    log_file = tmp_path / "log.txt"
    log_content = """
    BALANCE OF WATER VOLUME
    FLUX BOUNDARY 1:  1.25000000000000E+01
    FLUX BOUNDARY 2: -1.25000000000000E+01

    """
    log_file.write_text(log_content)
    f1, f2 = check_flux_boundaries(str(log_file))
    assert f1 == pytest.approx(12.5)
    assert f2 == pytest.approx(-12.5)
    assert is_flux_balanced(f1, f2)
