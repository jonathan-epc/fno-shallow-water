import subprocess
from unittest.mock import MagicMock, patch

import pytest

from telemac.modules.telemac_runner import run_telemac2d


@patch("subprocess.run")
@patch("builtins.open", new_callable=MagicMock)
def test_run_telemac2d_linux(mock_open, mock_run):
    """Verify command generation on Linux."""
    with patch("platform.system", return_value="Linux"):
        filename = "test_case.cas"
        output_dir = "results"

        run_telemac2d(filename, output_dir)

        expected_command = ["telemac2d.py", "--ncsize=1", filename]

        args, kwargs = mock_run.call_args
        assert args[0] == expected_command
        assert kwargs["check"] is True
        assert kwargs["stderr"] == subprocess.STDOUT


@patch("subprocess.run")
@patch("builtins.open", new_callable=MagicMock)
def test_run_telemac2d_windows(mock_open, mock_run):
    """Verify command generation on Windows."""
    with patch("platform.system", return_value="Windows"):
        filename = "test_case.cas"
        output_dir = "results"

        run_telemac2d(filename, output_dir)

        expected_command = ["python", "-m", "telemac2d", "--ncsize=1", filename]

        args, kwargs = mock_run.call_args
        assert args[0] == expected_command
        assert kwargs["check"] is True
        assert kwargs["stderr"] == subprocess.STDOUT


@patch("subprocess.run")
@patch("builtins.open", new_callable=MagicMock)
def test_run_telemac2d_error_handling(mock_open, mock_run):
    """Verify that subprocess errors are raised."""
    mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")

    with pytest.raises(subprocess.CalledProcessError):
        run_telemac2d("fail.cas", "results")
