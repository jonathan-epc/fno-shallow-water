import math


class BoundaryConditions:
    """
    A class to handle boundary conditions for a simulation.
    Methods
    -------
    get_boundary_and_elevations(direction, h0, bottom, borders)
        Determines the boundary file and elevations based on the given parameters.
    """

    @staticmethod
    def get_boundary_and_elevations(
        direction: str,  # Loosened type hint
        h0: float,
        bottom: str,
        borders: list[float],
        telemac_dir: str = "telemac",
        initial_depth: float = 0.3,  # Added to receive yn for Supercritical Inlet
        subcritical_cli: str = "3x3_riv.cli",
        supercritical_cli: str = "3x3_tor.cli",
    ) -> tuple[str, tuple[float, float]]:
        z_left, z_right = borders
        from pathlib import Path

        telemac_path = Path(telemac_dir)

        if direction == "subcritical" or direction == "Right to left":
            # Subcritical Flow (L->R or Legacy R->L).
            # Use riv.cli (River/Fluvial).
            # Outlet (Right) prescribed Head, Inlet (Left) prescribed Q.
            boundary_file = str(telemac_path / f"bnd/{subcritical_cli}")
            elevations = (z_right + h0, 0.0)

        else:
            # Supercritical Flow (L->R).
            # Use tor.cli (Torrential).
            # Inlet (Left) prescribed H+Q. Outlet (Right) prescribed H (for Jump).
            boundary_file = str(telemac_path / f"bnd/{supercritical_cli}")
            elevations = (z_right + h0, z_left + initial_depth)

        # Check for NaN values
        if any(math.isnan(x) for x in elevations):
            raise ValueError(
                f"NaN value encountered in elevation calculations. z={z_left, z_right} h0={h0}"
            )

        return boundary_file, elevations
