from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from .hydraulic_calculations import HydraulicCalculations
from .parameter_generation_strategies import ParameterGenerationStrategies
from .parameter_io import ParameterIO


class ParameterManager:
    """
    Manages the lifecycle of simulation parameters.
    Orchestrates Loading, Generation, and Updating, delegating heavy logic to:
    - ParameterIO (File ops)
    - ParameterGenerationStrategies (Algorithms)
    - HydraulicCalculations (Physics)
    """

    def __init__(self, config, sample_size: int = None):
        self.config = config
        self.output_dir = Path(config.paths.data_dir)
        self.csv_path = self.output_dir / "parameters.csv"

        # Initialize sub-modules
        self.io = ParameterIO(str(self.csv_path))

        # Cache for parameters
        self.parameters = None

        # Determine sample size (runtime override > config)
        if sample_size is not None:
            self.sample_size = sample_size
        else:
            # Try to get from config, handle missing field gracefully if needed
            self.sample_size = getattr(config.simulation_params, "sample_size", 100)

        # Extract specific ranges for easier access

        # Extract and parse parameter ranges
        self.param_ranges = self._parse_ranges(
            config.simulation_params.parameter_ranges
        )

    def _parse_ranges(
        self, flat_ranges: dict[str, float]
    ) -> dict[str, tuple[float, float]]:
        """
        Parses the flat dictionary of ranges (e.g., 'slope_min', 'slope_max')
        into structured tuples expected by SampleGenerator.
        Also normalizes keys to canonical names (e.g., 'slope' -> 'SLOPE').
        """
        parsed = {}
        # Buffer to store min/max values found
        # Key: base_name -> {min: val, max: val}
        buffers = {}

        # Canonical Name Mapping (config key base -> system key)
        NAME_MAPPING = {
            "slope": "SLOPE",
            "q0": "Q0",
            "h0": "H0",
            "n": "n",
            "ar": "Ar",
            "hr": "Hr",
            "fr": "Fr",
            "m": "M",
            "re": "Re",
        }

        for key, value in flat_ranges.items():
            key_lower = key.lower()
            if key_lower.endswith("_min") or key_lower.endswith("_max"):
                # Determine suffix and base name
                if key_lower.endswith("_min"):
                    suffix = "min"
                    base = key[:-4]  # preserve original case for now
                else:
                    suffix = "max"
                    base = key[:-4]

                # Normalize base name
                base_lower = base.lower()
                canonical_name = NAME_MAPPING.get(base_lower, base)

                if canonical_name not in buffers:
                    buffers[canonical_name] = {}

                buffers[canonical_name][suffix] = value

        # Construct tuples
        for name, bounds in buffers.items():
            if "min" in bounds and "max" in bounds:
                parsed[name] = (bounds["min"], bounds["max"])
            else:
                logger.warning(f"Incomplete range for parameter '{name}': {bounds}")

        return parsed

    def load_or_generate_parameters(self, mode: str = "add", **kwargs) -> pd.DataFrame:
        """
        Main entry point. Loads existing parameters and/or generates new ones.
        """
        existing_params = None

        if mode in ["read", "add"]:
            existing_params = self.io.load()

        if mode == "read":
            if existing_params is None:
                raise FileNotFoundError("Mode is 'read' but parameters.csv not found.")
            self.parameters = existing_params
            return existing_params

        # "add" mode logic
        if existing_params is not None:
            logger.info(f"Existing parameters found: {len(existing_params)} samples.")
        else:
            existing_params = pd.DataFrame()

        # Check if we actually need to generate
        logger.info(
            f"Generating new parameters for mode '{mode}' with args: {kwargs}. Target Sample Size: {self.sample_size}"
        )

        try:
            new_params = self._generate_new_parameters(**kwargs)
            logger.info(f"Generated {len(new_params)} new samples.")
        except Exception:
            logger.exception("Error in _generate_new_parameters")
            raise

        # Process andCombine
        final_params = self._combine_and_process(existing_params, new_params)

        # Validate critical columns (Backfill if needed)
        final_params = self._ensure_bottom_types(final_params)

        # Save
        self.io.save(final_params)
        self.parameters = final_params
        return final_params

    def _generate_new_parameters(self, **kwargs) -> pd.DataFrame:
        """
        Orchestrates the generation of new parameter sets.
        Accepts kwargs to assign metadata columns (e.g., set_type, sampling_method).
        """
        logger.info(
            f"Generating new parameters. Target Sample Size: {self.sample_size}"
        )

        if self.config.simulation_params.adimensional_generation:
            logger.warning(
                "Adimensional generation requested but not fully refactored yet. Using defaults."
            )
            raise NotImplementedError(
                "Adimensional generation is currently disabled in this refactor."
            )
        else:
            # Standard "Physics-Based" Generation with Dynamic Loop
            sampling_method = kwargs.get("sampling_method", "lhs")
            seed = getattr(self.config, "seed", 42)

            base_params = ParameterGenerationStrategies.generate_balanced_parameters(
                target_size=self.sample_size,
                param_ranges=self.param_ranges,
                channel_config=self.config.channel,
                gravity=HydraulicCalculations.GRAVITY,
                sampling_method=sampling_method,
                seed=seed,
            )

        # Derive Hydraulic Properties (Physics)
        if "W" not in base_params.columns:
            base_params["W"] = self.config.channel.width

        # Handle Channel Length (L) and Aspect Ratio (Ar)
        if "Ar" in base_params.columns:
            logger.info("Using LHS-sampled 'Ar' to derive 'L'.")
            base_params["L"] = base_params["Ar"] * base_params["W"]
        elif "Ar" in self.param_ranges:
            ar_min, ar_max = self.param_ranges["Ar"]
            base_params["Ar"] = np.random.uniform(ar_min, ar_max, size=len(base_params))
            base_params["L"] = base_params["Ar"] * base_params["W"]
            logger.info(
                f"Sampled variable 'Ar' from [{ar_min}, {ar_max}] and derived 'L'."
            )
        elif "L" not in base_params.columns:
            base_params["L"] = self.config.channel.length

        base_params = HydraulicCalculations.estimate_turbulent_viscosity(
            base_params, HydraulicCalculations.GRAVITY
        )
        base_params = HydraulicCalculations.calculate_adimensionals(
            base_params, self.config.channel, HydraulicCalculations.GRAVITY
        )

        # 3. Create Cartesian Product with Bottom Types
        bottom_types = self.config.simulation_params.bottom_types or [
            "BARS",
            "SLOPE",
            "NOISE",
        ]

        expanded_params = []
        for b_type in bottom_types:
            subset = base_params.copy()
            subset["BOTTOM"] = b_type
            expanded_params.append(subset)

        # Combine into final dataset
        new_params = pd.concat(expanded_params, ignore_index=True)

        # 4. Add Metadata & Derived Columns
        # Apply kwargs (set_type, sampling_method)
        for key, value in kwargs.items():
            new_params[key] = value

        # Derived Metrics
        # Subcritical: defined by initial condition H0 relative to Critical Depth yc
        # If H0 > yc -> Subcritical boundary
        # If H0 < yc -> Supercritical boundary
        new_params["subcritical"] = new_params["H0"] > new_params["yc"]

        # Depth Ratio: H0 / yc (Standard for characterizing proximity to critical flow)
        new_params["depth_ratio"] = new_params["H0"] / new_params["yc"]

        # Direction / Regime Description
        # regime column already exists from generator (e.g., "M1", "S1")
        # We can map it to semantic names if needed, or just copy it to 'direction' if that was the old name
        if "regime" in new_params.columns:
            new_params["direction"] = new_params["regime"]  # Historical alias

        logger.info(
            f"Generated {len(base_params)} physics samples. Expanded to {len(new_params)} cases ({len(base_params)} x {len(bottom_types)} bottoms)."
        )

        return new_params

    def _ensure_bottom_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures the BOTTOM column exists and has valid values."""
        if df.empty:
            return df

        if "BOTTOM" not in df.columns:
            df["BOTTOM"] = np.nan

        # Identify rows with missing BOTTOM (NaN or None)
        missing_mask = df["BOTTOM"].isna()

        if missing_mask.any():
            logger.warning(
                f"Found {missing_mask.sum()} samples with missing BOTTOM type. Backfilling randomly."
            )
            bottom_types = self.config.simulation_params.bottom_types
            if not bottom_types:
                bottom_types = [
                    "BARS",
                    "SLOPE",
                    "NOISE",
                ]  # Fallback hardcoded if config missing

            # Fill missing values
            df.loc[missing_mask, "BOTTOM"] = np.random.choice(
                bottom_types, size=missing_mask.sum()
            )

        return df

    def update_status(
        self, param_id: int, status: str, result_path: str = None
    ) -> None:
        """Updates the status of a simulation."""
        if self.parameters is None:
            self.parameters = self.io.load()

        # If still None, we can't update
        if self.parameters is None:
            logger.error("Cannot update status: No parameters loaded.")
            return

        if "status" not in self.parameters.columns:
            self.parameters["status"] = "pending"

        if param_id in self.parameters.index:
            self.parameters.loc[param_id, "status"] = status
            if result_path:
                self.parameters.loc[param_id, "result_path"] = result_path
            self.io.save(self.parameters)
        else:
            logger.warning(f"Parameter ID {param_id} not found.")

    def _combine_and_process(
        self, existing: pd.DataFrame, new_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Combines existing and new parameters, handling IDs and duplicates."""
        if new_df.empty:
            return existing

        # Assign IDs
        start_id = 0
        if not existing.empty:
            start_id = existing.index.max() + 1

        new_df.index = range(start_id, start_id + len(new_df))
        new_df.index.name = "id"

        # Initialize status columns
        new_df["status"] = "pending"
        new_df["result_path"] = ""

        combined = pd.concat([existing, new_df])
        return combined

    def create_cases(self) -> list:
        """
        Converts the current parameters DataFrame into a list of TelemacCase objects.
        Returns only cases with 'pending' status by default.
        """
        from .telemac_case import TelemacCase

        if self.parameters is None:
            self.load_or_generate_parameters(mode="read")

        cases = []
        if self.parameters is None or self.parameters.empty:
            return cases

        for idx, row in self.parameters.iterrows():
            if row.get("status") == "pending":
                case_params = row.to_dict()
                # TelemacCase expects 'telemac_dir' in params to resolve paths
                if "telemac_dir" not in case_params:
                    case_params["telemac_dir"] = str(self.config.paths.telemac_dir)

                # Inject Mesh Configuration
                case_params["num_points_x"] = self.config.mesh.num_points_x
                case_params["num_points_y"] = self.config.mesh.num_points_y
                case_params["subcritical_cli"] = getattr(
                    self.config.mesh, "subcritical_cli", "3x3_riv.cli"
                )
                case_params["supercritical_cli"] = getattr(
                    self.config.mesh, "supercritical_cli", "3x3_tor.cli"
                )

                cases.append(TelemacCase(idx, case_params))

        logger.info(f"Created {len(cases)} TelemacCase objects for processing.")
        return cases
