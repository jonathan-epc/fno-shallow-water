# telemac/input_generator.py

import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

from common.utils import setup_logger
from modules.environment_setup import EnvironmentSetup
from modules.parameter_manager import ParameterManager
from nconfig import get_config

# Configure logger to play nice with tqdm
logger.remove()
logger.add(
    lambda msg: tqdm.write(msg, end=""),
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)


def process_case(case, flat_mesh, overwrite=False):
    """
    Processes a single TelemacCase by generating its geometry and steering file.
    """
    if case.steering_file_path.exists() and not overwrite:
        logger.info(f"Skipping case {case.case_id}: steering file exists.")
        return

    try:
        logger.debug(f"Processing case {case.case_id}...")
        # 1. Generate geometry and get the border elevations
        borders = case.generate_geometry(flat_mesh)

        # 2. Generate the steering file using the border elevations
        case.generate_steering_file(borders)

        logger.debug(f"Successfully generated files for case {case.case_id}.")
    except Exception as e:
        logger.exception(f"Error processing case {case.case_id}: {e}")


def cleanup_old_files(config):
    """
    Removes existing simulation files (.cas, .slf, .cli) from the target directories.
    """
    logger.info("Cleaning up old simulation files...")

    # Define directories and extensions to clean
    # Note: 'cas' contains steering files, 'geo' contains geometry files, 'res' contains results.
    # We DO NOT clean 'bnd' because it contains static template .cli files.
    dirs_to_clean = {
        "cas": [".cas"],
        "geo": [".slf"],
        "res": [".slf"],
    }

    # Files to explicitly exclude from deletion (protect base mesh templates)
    mesh_file = getattr(config.mesh, "base_mesh_file", "mesh_3x3.slf")
    excluded_files = {"mesh_3x3.slf", mesh_file}

    # Using config paths
    telemac_dir = Path(config.paths.telemac_dir)

    count = 0
    for subdir, extensions in dirs_to_clean.items():
        target_dir = telemac_dir / subdir
        if not target_dir.exists():
            continue

        for file in target_dir.iterdir():
            if file.name in excluded_files:
                logger.debug(f"Skipping excluded file: {file}")
                continue

            if file.suffix in extensions:
                try:
                    file.unlink()
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {file}: {e}")

    # Also clean parameters.csv if it exists
    param_file = Path(config.paths.data_dir) / "parameters.csv"
    if param_file.exists():
        try:
            param_file.unlink()
            count += 1
            logger.info(f"Deleted {param_file}")
        except Exception as e:
            logger.warning(f"Failed to delete {param_file}: {e}")

    logger.success(f"Cleanup complete. Removed {count} files.")


def main(
    param_mode="add",
    sample_size=100,
    overwrite=False,
    sampling_mode="lhs",
    parameters_file="parameters.csv",
    set_type="train_val",
    pipeline=False,
    **kwargs,
):
    config = get_config("config.yml")

    env_setup = EnvironmentSetup(config)
    setup_data = env_setup.get_setup_data()

    if kwargs.get("clean", False):
        cleanup_old_files(config)

    # Seed for reproducibility
    np.random.seed(config.seed)
    random.seed(config.seed)

    # Phase 1: Parameter Generation
    logger.info("--- Phase 1: Parameter Generation ---")

    def generate_step(mode, size, smethod, stype):
        logger.info(f"Generating {stype} dataset ({smethod} sampling)...")
        pm = ParameterManager(config, sample_size=size)

        # Logic to map 'new' to 'add' for PM
        pm_mode = "add" if mode == "new" else mode

        pm.load_or_generate_parameters(
            mode=pm_mode, set_type=stype, sampling_method=smethod
        )
        return pm

    if pipeline:
        logger.info("Running Full Pipeline Generation (Train + Test)")
        # 1. Generate Training/Validation set
        generate_step(param_mode, sample_size, "lhs", "train_val")

        # 2. Generate Testing set
        # User requested 10% of training sample size
        test_size = max(1, int(sample_size * 0.10))
        generate_step("add", test_size, "random", "test")
    else:
        generate_step(param_mode, sample_size, sampling_mode, set_type)

    logger.info(
        "Parameter generation complete. parameters.csv should now contain all cases."
    )

    # Phase 2: Case Processing (Geometry & Steering)
    logger.info("--- Phase 2: Case Processing & File Generation ---")

    # Reload the FULL parameter set
    final_pm = ParameterManager(config)  # sample_size irrelevant for reading
    cases_to_process = final_pm.create_cases()

    if not cases_to_process:
        logger.warning("No cases found to process (or all are already done/filtered).")
        return

    logger.info(f"Starting processing of {len(cases_to_process)} total cases...")

    for case in tqdm(cases_to_process, desc="Generating Simulation Files"):
        process_case(case, setup_data["flat_mesh"], overwrite)

    logger.info("Input generation process completed.")


if __name__ == "__main__":
    script_name = Path(__file__).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{script_name}_{timestamp}"
    logger = setup_logger(log_name)

    parser = argparse.ArgumentParser(
        description="Generate TELEMAC-2D simulation input files."
    )
    parser.add_argument(
        "--mode",
        choices=["new", "read", "add"],
        default="add",
        help="Parameter file handling mode.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Target number of samples. Will be adjusted upwards to the next prime square ($p^2$) for Orthogonal Arrays.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing geometry and steering files.",
    )
    parser.add_argument(
        "--sampling_mode",
        choices=["lhs", "random"],
        default="lhs",
        help="Sampling method to use.",
    )
    parser.add_argument(
        "--parameters_file",
        type=str,
        default="parameters.csv",
        help="Target parameter file name.",
    )
    parser.add_argument(
        "--set_type",
        choices=["train_val", "test"],
        default="train_val",
        help="Dataset split type.",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Generate both train_val (LHS) and test (Random) datasets sequentially.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean old simulation files before starting.",
    )
    args = parser.parse_args()

    main(
        param_mode=args.mode,
        sample_size=args.sample_size,
        overwrite=args.overwrite,
        sampling_mode=args.sampling_mode,
        parameters_file=args.parameters_file,
        set_type=args.set_type,
        pipeline=args.pipeline,
        clean=args.clean,
    )
