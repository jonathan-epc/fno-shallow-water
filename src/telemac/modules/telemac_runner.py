import os
import platform
import subprocess

from loguru import logger


def run_telemac2d(
    filename: str, output_dir: str, cwd: str = ".", timeout: float = 600
) -> None:
    """
    Runs a Telemac2D simulation and writes the output to a specified directory.

    Parameters
    ----------
    filename : str
        The name of the input file for the Telemac2D simulation.
    output_dir : str
        The directory where the output file will be saved.
    cwd : str, optional
        The working directory to run the command in. Defaults to current directory.
    timeout : float, optional
        Maximum execution time in seconds. Defaults to 1200 (20 minutes).

    Raises
    ------
    subprocess.CalledProcessError
        If the Telemac2D simulation fails to run (non-zero exit code).
    subprocess.TimeoutExpired
        If the simulation exceeds the timeout.
    """
    command = (
        ["telemac2d.py"]
        if platform.system() == "Linux"
        else ["python", "-m", "telemac2d"]
    )
    name, ext = os.path.splitext(filename)
    output_file = os.path.join(output_dir, f"{name}.txt")

    try:
        with open(output_file, "w") as output_fh:
            subprocess.run(
                command + ["--ncsize=1", filename],
                check=True,
                stdout=output_fh,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                timeout=timeout,
            )
        logger.info(f"Completed Telemac2D simulation for {filename}")
    except subprocess.TimeoutExpired:
        logger.warning(f"Simulation timed out after {timeout}s: {filename}")
        # Ensure we write a note to the log file so parser can see it
        with open(output_file, "a") as output_fh:
            output_fh.write(
                f"\n\n[ANTIGRAVITY] SIMULATION TIMED OUT AFTER {timeout}s\n"
            )
        raise  # Re-raise to be handled by caller
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running Telemac2D simulation for {filename}: {e}")
        raise
