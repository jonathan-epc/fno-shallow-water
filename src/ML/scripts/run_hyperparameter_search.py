# ML/scripts/run_hyperparameter_search.py

import sys

from common.utils import set_seed, setup_logger
from ML.core.optimizer import HyperparameterOptimizer
from ML.core.suite import (
    ALL_CASE_KEYS,
    apply_runtime_overrides,
    build_suite_parser,
    execute_case_batch,
)
from nconfig import get_config


def main():
    """Main function to execute single or multi-case hyperparameter optimization."""
    parser = build_suite_parser(
        description="Run Optuna hyperparameter search for FNO surrogate model(s).",
        include_trials=True,
    )
    args, _ = parser.parse_known_args()

    logger = setup_logger()
    try:
        config = get_config()
        set_seed(config.seed)

        def run_single(cfg):
            logger.info(
                f"Starting hyperparameter search for study: {cfg.optuna.study_name}"
            )
            optimizer = HyperparameterOptimizer(cfg)
            results = optimizer.run_optimization()
            if results:
                logger.info(
                    f"Optimization completed. Best CV value: {results.best_value:.6f}"
                )
            else:
                logger.warning("Optimization process did not yield results.")
            return results

        selected_cases = ALL_CASE_KEYS if args.all else args.cases
        if selected_cases:
            _, failed = execute_case_batch(
                base_config=config,
                case_keys=selected_cases,
                runner_fn=run_single,
                logger=logger,
                epochs=args.epochs,
                trials=args.trials,
                device=args.device,
            )
            if failed:
                sys.exit(1)
        else:
            run_cfg = (
                apply_runtime_overrides(
                    config.model_copy(deep=True),
                    epochs=args.epochs,
                    trials=args.trials,
                    device=args.device,
                )
                if (
                    args.epochs is not None
                    or args.trials is not None
                    or args.device is not None
                )
                else config
            )
            run_single(run_cfg)

        logger.info("--- Hyperparameter Search Finished ---")

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration or setup error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred during the run: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
