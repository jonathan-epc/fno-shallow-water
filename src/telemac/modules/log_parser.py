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
            if "My Work is Done" not in content and "Correct end of run" not in content:
                # If it didn't finish cleanly and isn't obviously stalled, assume crashed/interrupted
                # But check for stalling first as a specific cause of "not finishing"
                if LogParser._check_stalled(lines):
                    return SimulationStatus.STALLED
                return SimulationStatus.CRASHED

            # 2. Check for Stalling (even if it somehow finished, though unlikely)
            if LogParser._check_stalled(lines):
                return SimulationStatus.STALLED

            # 3. Check Flux Balance
            # Reuse regex logic but robustly
            last_balance_match = re.search(
                r"BALANCE OF WATER VOLUME", content, re.DOTALL
            )
            if not last_balance_match:
                # Finished but no balance? Weird, but assume unbalanced or just incomplete log
                return SimulationStatus.CRASHED

            # Find all flux boundaries in the *last* balance section
            # We explicitly look at the end of the file
            last_lines = "\n".join(lines[-100:])  # Optimization: Only scan end
            fluxes = re.findall(r"FLUX BOUNDARY\s+\d+:\s+([-\d.E]+)", last_lines)

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
