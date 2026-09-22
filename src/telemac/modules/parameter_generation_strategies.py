import numpy as np
import pandas as pd
from loguru import logger

from .hydraulic_calculations import HydraulicCalculations
from .sample_generator import SampleGenerator


class ParameterGenerationStrategies:
    """
    Encapsulates complex logic for generating and balancing simulation parameters.
    """

    @staticmethod
    def balance_slope_classes(params: pd.DataFrame, target_size: int) -> pd.DataFrame:
        """
        Balances the dataset between Mild (M) and Steep (S) slope classes.
        """
        mild = params[params["slope_class"] == "M"]
        steep = params[params["slope_class"] == "S"]

        n_mild = len(mild)
        n_steep = len(steep)

        logger.info(
            f"Slope Class Balance (Pre-Adjustment): Mild={n_mild}, Steep={n_steep}"
        )

        # Determine target per class (approx 50/50)
        target_per_class = target_size // 2

        # Sample or trim to target
        if n_mild > target_per_class:
            mild = mild.sample(n=target_per_class, random_state=42)
        if n_steep > target_per_class:
            steep = steep.sample(n=target_per_class, random_state=42)

        balanced = (
            pd.concat([mild, steep])
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )

        logger.info(
            f"Slope Class Balance (Post-Adjustment): Mild={len(mild)}, Steep={len(steep)}. Total: {len(balanced)}"
        )
        return balanced

    @staticmethod
    def assign_gvf_regimes(
        params: pd.DataFrame, yn: np.ndarray, yc: np.ndarray
    ) -> pd.DataFrame:
        """
        Assigns GVF flow regimes (M1, M2, M3, S1, S2, S3) based on normal depth (yn),
        critical depth (yc), and actual depth (H0 or h).
        """
        # Define tolerances

        # Classification Logic
        slope_class = np.where(yn > yc, "M", "S")  # M if yn > yc, S if yn < yc

        # Regime logic
        # For Mild (M):
        # Zone 1 (M1): h > yn > yc
        # Zone 2 (M2): yn > h > yc
        # Zone 3 (M3): yn > yc > h

        # For Steep (S):
        # Zone 1 (S1): h > yc > yn
        # Zone 2 (S2): yc > h > yn
        # Zone 3 (S3): yc > yn > h

        # The key is comparing 'h' (which is params['H0']) to yn and yc.
        h = params["H0"].values

        regimes = []
        for i in range(len(params)):
            curr_class = slope_class[i]
            curr_h = h[i]
            curr_yn = yn[i]
            curr_yc = yc[i]

            if curr_class == "M":
                if curr_h > curr_yn:
                    regimes.append("M1")
                elif curr_h > curr_yc:
                    regimes.append("M2")
                else:
                    regimes.append("M3")
            else:  # S
                if curr_h > curr_yc:
                    regimes.append("S1")
                elif curr_h > curr_yn:
                    regimes.append("S2")
                else:
                    regimes.append("S3")

        params["slope_class"] = slope_class
        params["regime"] = regimes
        return params

    @staticmethod
    def generate_balanced_parameters(
        target_size: int,
        param_ranges: dict,
        channel_config,
        gravity: float = 9.81,
        max_iterations: int = 100,
        sampling_method: str = "lhs",
        seed: int = None,
    ) -> pd.DataFrame:
        """
        Generates parameters using a dynamic loop to ensure the target size is reached
        while respecting Orthogonal Array sampling constraints ($p^2$) and balancing classes.

        Args:
            target_size: The desired final number of samples.
            param_ranges: Dictionary of parameter ranges (min, max).
            SampleGenerator: Class complying with p^2 constraints.
        """
        valid_pool = []
        iteration = 0

        # Calculate appropriate batch size
        if sampling_method == "lhs":
            # For LHS (OA), we need p^2 and p > d-1
            d = len(param_ranges)
            batch_size = SampleGenerator.get_valid_size(target_size, d=d)
        else:
            batch_size = target_size

        logger.info(
            f"Starting dynamic generation loop. Target: {target_size}. Batch Size: {batch_size}. Method: {sampling_method}"
        )

        while len(valid_pool) < target_size and iteration < max_iterations:
            iteration += 1
            logger.debug(
                f"Generation Loop: Iteration {iteration}. Current Pool: {len(valid_pool)}"
            )

            # 1. Generate Batch
            batch = SampleGenerator.sample_combinations(
                batch_size, param_ranges, method=sampling_method, seed=seed
            )

            # 2. Assign Regime Properties
            # Add fixed channel width
            batch["W"] = channel_config.width

            # Calculate normal (yn) and critical (yc) depths for classification
            batch["yn"] = HydraulicCalculations.normal_depth(
                batch["Q0"].values,
                batch["W"].values,
                batch["SLOPE"].values,
                batch["n"].values,
            )
            batch["yc"] = HydraulicCalculations.critical_depth(
                batch["Q0"].values, batch["W"].values
            )

            batch = ParameterGenerationStrategies.assign_gvf_regimes(
                batch, batch["yn"].values, batch["yc"].values
            )

            # 3. Accumulate candidate batch
            valid_pool.append(batch)

            target_per_class = target_size // 2

            # Count current valid pool
            current_total_df = pd.concat(valid_pool, ignore_index=True)
            n_mild = len(current_total_df[current_total_df["slope_class"] == "M"])
            n_steep = len(current_total_df[current_total_df["slope_class"] == "S"])

            logger.debug(
                f"Accumulated Samples: Mild={n_mild}, Steep={n_steep}. Target per class: {target_per_class}"
            )

            if n_mild >= target_per_class and n_steep >= target_per_class:
                logger.info("Sufficient samples collected for both classes.")
                break

            # Safety break if we simply can't find them
            total_samples = len(current_total_df)
            if total_samples >= target_size * 5:
                logger.warning("Exceeded safety buffer (5x). Stopping generation loop.")
                break

        if not valid_pool:
            raise ValueError("Failed to generate any valid parameters.")

        # Concatenate all batches
        full_df = pd.concat(valid_pool, ignore_index=True)

        # 4. Balance
        balanced_df = ParameterGenerationStrategies.balance_slope_classes(
            full_df, target_size
        )

        # 5. Trim dataset to target size if necessary
        if len(balanced_df) > target_size:
            balanced_df = balanced_df.iloc[:target_size]

        return balanced_df.reset_index(drop=True)
