import re
from enum import Enum, auto
from pathlib import Path

from loguru import logger


class SimulationStatus(Enum):
    BALANCED = auto()
    UNBALANCED = auto()
    STALLED = auto()
    CRASHED = auto()
    UNKNOWN = auto()


class LogParser:
    """
    Parses Telemac2D log files to determine the outcome of a simulation.
    """

    # Threshold for considering a time step "stalled"
    STALL_DT_THRESHOLD = 1e-4

    @staticmethod
    def analyze(log_file: str | Path, flux_threshold: float = 1e-3) -> SimulationStatus:
        """
        Analyzes a log file to determine the simulation status.

        Args:
            log_file: Path to the log file.
            flux_threshold: Maximum allowable flux error for BALANCED status.

        Returns:
            SimulationStatus enum indicating the result.
        """
        log_path = Path(log_file)
        if not log_path.exists():
            return SimulationStatus.UNKNOWN

        try:
            with open(log_path, encoding="latin1") as f:
                content = f.read()
                lines = content.splitlines()

            # 1. Check for Crash/Interruption
            # Common crash indicators in Telemac/Python
            if "Traceback (most recent call last):" in content:
                return SimulationStatus.CRASHED
            content_upper = content.upper()
            if (
                "MY WORK IS DONE" not in content_upper
                and "CORRECT END OF RUN" not in content_upper
            ):
                # If it didn't finish cleanly and isn't obviously stalled, assume crashed/interrupted
                # But check for stalling first as a specific cause of "not finishing"
                if LogParser._check_stalled(lines):
                    return SimulationStatus.STALLED
                return SimulationStatus.CRASHED

            # 2. Check for Stalling (even if it somehow finished, though unlikely)
            if LogParser._check_stalled(lines):
                return SimulationStatus.STALLED

            # 3. Check Flux Balance
            # Parse the *last* balance block that actually contains boundary
            # fluxes. Scanning a fixed number of trailing lines can mix several
            # balance reports (and the final volume summary), producing a false
            # UNBALANCED result.
            balance_blocks = re.findall(
                r"BALANCE OF WATER VOLUME(.*?)(?=\n[ \t]*\n|\Z)",
                content,
                re.DOTALL,
            )
            if not balance_blocks:
                # If simulation finished without a reported volume balance, treat as non-converged/incomplete.
                return SimulationStatus.CRASHED

            flux_pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
            fluxes = []
            for block in reversed(balance_blocks):
                found = re.findall(rf"FLUX BOUNDARY\s+\d+:\s+({flux_pattern})", block)
                if len(found) >= 2:
                    fluxes = found
                    break

            if len(fluxes) >= 2:
                try:
                    total_flux = sum(float(f) for f in fluxes)
                    if abs(total_flux) < flux_threshold:
                        return SimulationStatus.BALANCED
                    else:
                        return SimulationStatus.UNBALANCED
                except ValueError:
                    return SimulationStatus.UNBALANCED

            # If we reached here, it finished but we couldn't parse fluxes
            return SimulationStatus.UNBALANCED

        except Exception as e:
            logger.error(f"Error parsing log {log_file}: {e}")
            return SimulationStatus.UNKNOWN

    @staticmethod
    def _check_stalled(lines: list[str], verify_last_n: int = 5) -> bool:
        """
        Checks if the last few time steps were consistently below the stall threshold.
        """
        # Extract all time steps
        # Regex: TIME-STEP:    1.3011121063535024E-005
        time_steps = []
        for line in reversed(lines):
            match = re.search(r"TIME-STEP:\s+([-\d.E]+)", line)
            if match:
                try:
                    time_steps.append(float(match.group(1)))
                except ValueError:
                    continue
            if len(time_steps) >= verify_last_n:
                break

        if not time_steps:
            return False

        # If the *average* of the last N steps is tiny, it's stalled
        avg_dt = sum(time_steps) / len(time_steps)
        return avg_dt < LogParser.STALL_DT_THRESHOLD
