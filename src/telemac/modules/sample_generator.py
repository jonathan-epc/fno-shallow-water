import numpy as np
import pandas as pd
from scipy.stats import qmc


class SampleGenerator:
    """
    A class to generate samples of combinations for given parameter ranges.

    Methods
    -------
    sample_combinations(n, param_ranges)
        Generates a DataFrame of sampled combinations for the specified parameter ranges.
    """

    @staticmethod
    def get_valid_size(n: int, d: int = None) -> int:
        """
        Returns the smallest squared prime number >= n.
        Useful for Orthogonal Array LHS which requires n = p^2.
        Also safeguards against Scipy's limitation: n > (d-1)^2.
        """

        def is_prime(num):
            if num < 2:
                return False
            return all(num % i != 0 for i in range(2, int(num**0.5) + 1))

        # Minimum p based on dimensionality d
        min_p_from_d = 0
        if d is not None:
            min_p_from_d = d  # roughly p > d-1, so p >= d is safe

        # Find approximate root
        root = int(np.ceil(np.sqrt(n)))

        # Start search from max required root
        p = max(root, min_p_from_d)

        while not is_prime(p):
            p += 1

        return p * p

    @staticmethod
    def sample_combinations(
        n: int, param_ranges: dict[str, tuple], method: str = "lhs", seed: int = None
    ) -> pd.DataFrame:
        """
        Generates a DataFrame of sampled combinations for the specified parameter ranges.

        Parameters
        ----------
        n : int
            The number of samples to generate.
        param_ranges : Dict[str, tuple]
            A dictionary where keys are parameter names and values are tuples (min, max).
        method : str, optional
            The sampling method to use: 'lhs' (default) or 'random'.
        seed : int, optional
            Seed for reproduction.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the sampled combinations of the parameters.
        """
        param_names = list(param_ranges.keys())
        lower_bounds = np.array([param_ranges[p][0] for p in param_names])
        upper_bounds = np.array([param_ranges[p][1] for p in param_names])

        if method == "lhs":
            try:
                # strength=2 enables Orthogonal Array-based LHS
                # Requirement: n must be the square of a prime number (p^2)
                # Use provided seed or default to 43 if None
                lhs_seed = seed if seed is not None else 43
                sampler = qmc.LatinHypercube(
                    d=len(param_names), strength=2, seed=lhs_seed
                )
                sample = sampler.random(n=n)
            except ValueError as e:
                # Scipy raises ValueError if n is not a prime square
                raise ValueError(
                    f"Sample size {n} is invalid for Orthogonal Array LHS (strength=2). "
                    f"Size must be the square of a prime number (e.g., 49, 121, 169, 12769). "
                    f"Original error: {e}"
                ) from e

            sample_scaled = qmc.scale(sample, lower_bounds, upper_bounds)
        elif method == "random":
            # Uniform random sampling
            sample = np.random.rand(n, len(param_names))
            sample_scaled = lower_bounds + sample * (upper_bounds - lower_bounds)
        else:
            raise ValueError(
                f"Invalid sampling method: {method}. Choose 'lhs' or 'random'."
            )

        return pd.DataFrame(sample_scaled, columns=param_names)
