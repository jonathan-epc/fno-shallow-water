# ML/scripts/train_model.py

import sys

from common.utils import set_seed, setup_logger
from ML.core.suite import (
    ALL_CASE_KEYS,
    apply_runtime_overrides,
    build_suite_parser,
    execute_case_batch,
)
from ML.core.trainer import ModelTrainer
from nconfig import get_config


def main():
    """Main function to execute single or multi-case model training."""
    parser = build_suite_parser(
        description="Train FNO surrogate model(s) from config.yml."
    )
    args, _ = parser.parse_known_args()

    logger = setup_logger()
    try:
        config = get_config()
        set_seed(config.seed)

        def run_single(cfg):
            logger.info(f"Starting training for model: {cfg.model.name}")
            logger.info(f"Dataset: {cfg.data.file_path}")
            trainer = ModelTrainer(cfg)
            avg_cv_loss = trainer.train()
            if avg_cv_loss is not None:
                logger.info(
                    f"Training run completed. Average CV loss: {avg_cv_loss:.6f}"
                )
            else:
                logger.error(f"Training run failed for {cfg.model.name}.")
            return avg_cv_loss

        selected_cases = ALL_CASE_KEYS if args.all else args.cases
        if selected_cases:
            _, failed = execute_case_batch(
                base_config=config,
                case_keys=selected_cases,
                runner_fn=run_single,
                logger=logger,
                epochs=args.epochs,
                device=args.device,
            )
            if failed:
                sys.exit(1)
        else:
            run_cfg = (
                apply_runtime_overrides(
                    config.model_copy(deep=True),
                    epochs=args.epochs,
                    device=args.device,
                )
                if (args.epochs is not None or args.device is not None)
                else config
            )
            run_single(run_cfg)

        logger.info("--- Training Run Finished ---")

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration or setup error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred during the run: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
