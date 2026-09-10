import argparse
import os
import sys
from datetime import datetime

import pandas as pd
from run_configurations import OUTPUT_FOLDER, PARAMETERS_FILE, STEERING_FOLDER

from common.utils import setup_logger
from modules.flux_checker import check_flux_boundaries, is_flux_balanced
from modules.param_utils import load_parameters
from modules.simulation_runner import run_telemac2d_on_files


def main():
    args = parse_arguments()
    logger = setup_logging()

    # Perform cleanup if requested
    if args.clean:
        if args.dry_run:
            logger.info("Dry run: Would clean up previous results and temporary files.")
        else:
            perform_cleanup(args.output_dir, logger)

    try:
        parameters_file = (
            args.parameters_file if args.parameters_file else PARAMETERS_FILE
        )
        parameters = load_and_validate_parameters(parameters_file, logger)
        start_index, end_index = determine_file_range(args, STEERING_FOLDER)
        cas_files = get_cas_files(STEERING_FOLDER, args.adimensional)
        selected_files = cas_files[start_index:end_index]

        # Always filter selected files by available parameters to avoid KeyError
        # This handles cases where legacy files exist in the cas/ folder
        parameter_ids = set(parameters.index.astype(str))

        initial_count = len(selected_files)
        selected_files = [
            f for f in selected_files if os.path.splitext(f)[0].strip() in parameter_ids
        ]

        if len(selected_files) < initial_count:
            logger.info(
                f"Filtered out {initial_count - len(selected_files)} files not found in parameters index (legacy files?)."
            )

        if args.bottom:
            parameters = filter_parameters_by_bottom(parameters, args.bottom, logger)
            parameter_ids = set(parameters.index.astype(str))

            # Sub-filter by bottom type
            selected_files = [
                file
                for file in selected_files
                if os.path.splitext(file)[0].strip() in parameter_ids
            ]

            if not selected_files and cas_files:
                logger.warning(
                    f"Bottom filter '{args.bottom}' resulted in 0 files. "
                    f"Check if IDs in parameters.csv match filenames in cas/ folder."
                )
                logger.debug(f"Sample IDs in CSV: {list(parameter_ids)[:5]}")
                logger.debug(
                    f"Sample Files in cas/: {[os.path.splitext(f)[0] for f in cas_files[:5]]}"
                )

        unbalanced_files = filter_unbalanced_files(
            selected_files, args.output_dir, logger
        )
        if args.dry_run:
            logger.info(f"Dry run: would process files {unbalanced_files}")
            logger.info(f"Number of unbalanced files: {len(unbalanced_files)}")
        else:
            run_telemac2d_on_files(
                unbalanced_files, parameters, args.output_dir, STEERING_FOLDER
            )
            logger.info("Telemac2D simulations completed successfully")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)


def filter_unbalanced_files(files: list[str], output_dir: str, logger) -> list[str]:
    """
    Filter files to include only those with unbalanced flux boundaries.

    Parameters
    ----------
    files : List[str]
        List of steering file names.
    output_dir : str
        Directory where result files are stored.
    logger : Logger
        Logger for logging messages.

    Returns
    -------
    List[str]
        List of file names with unbalanced flux boundaries.
    """
    unbalanced_files = []
    logger.info(f"Checking flux boundaries of {len(files)} files . . .")
    for filename in files:
        result_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
        if os.path.exists(result_file):
            flux_1, flux_2 = check_flux_boundaries(result_file)
            if flux_1 is not None and flux_2 is not None:
                if is_flux_balanced(flux_1, flux_2):
                    logger.info(f"Skipping {filename}: flux boundaries are balanced.")
                    continue
                else:
                    logger.info(f"Unbalanced flux for {filename}.")
            else:
                logger.info(f"Cannot determine flux for {filename}. Including in run.")
        else:
            logger.info(f"No result file found for {filename}. Including in run.")
        unbalanced_files.append(filename)
    return unbalanced_files


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Telemac2D simulations.")
    parser.add_argument(
        "--start", type=int, default=0, help="Start index for simulation files"
    )
    parser.add_argument(
        "--end", type=int, default=0, help="End index for simulation files"
    )
    parser.add_argument(
        "--bottom",
        type=str,
        default=None,
        choices=["SLOPE", "NOISE", "BUMP", "BARS", "STEP"],
        help="Select a specific bottom value for simulation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_FOLDER,
        help="Output directory for simulation results",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without actually running simulations",
    )
    parser.add_argument(
        "--adimensional",
        action="store_true",
        help="Filter files to start with 'a' and end with '.cas'",
    )
    parser.add_argument(
        "--parameters_file",
        type=str,
        default=None,
        help="Path to the parameter file (overrides default from run_configurations)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean up previous results and temporary files before running",
    )
    return parser.parse_args()


def setup_logging():
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    process_id = os.getpid()

    log_name = f"{script_name}_{timestamp}_{process_id}"
    logger = setup_logger(log_name)

    return logger


def load_and_validate_parameters(file_path: str, logger) -> pd.DataFrame:
    try:
        parameters = load_parameters(file_path)
        if parameters.empty:
            raise ValueError("Parameters file is empty")

        # Ensure index is string to avoid KeyErrors with numeric filenames
        parameters.index = parameters.index.astype(str)

        required_columns = ["SLOPE", "n", "Q0", "H0", "BOTTOM"]
        missing_columns = [
            col for col in required_columns if col not in parameters.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in parameters file: {', '.join(missing_columns)}"
            )
        logger.info(f"Loaded and validated parameters from {file_path}")
        return parameters
    except pd.errors.EmptyDataError:
        raise ValueError(f"Parameters file {file_path} is empty") from None
    except FileNotFoundError:
        raise FileNotFoundError(f"Parameters file not found: {file_path}") from None


def determine_file_range(
    args: argparse.Namespace, steering_folder: str
) -> tuple[int, int]:
    if args.start == 0 and args.end == 0:
        cas_files = get_cas_files(steering_folder, args.adimensional)
        return 0, len(cas_files)
    elif args.start < 0 or args.end < args.start:
        raise ValueError("Invalid start or end index")
    else:
        return args.start, args.end


def get_cas_files(folder: str, adimensional: bool) -> list[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Steering folder not found: {folder}")

    if adimensional:
        cas_files = [
            f for f in os.listdir(folder) if f.startswith("a") and f.endswith(".cas")
        ]
    else:
        cas_files = [f for f in os.listdir(folder) if f.endswith(".cas")]

    if not cas_files:
        raise ValueError(
            f"No {'adimensional ' if adimensional else ''}.cas files found in {folder}"
        )

    return cas_files


def filter_parameters_by_bottom(
    parameters: pd.DataFrame, selected_bottom: float, logger
) -> pd.DataFrame:
    if "BOTTOM" not in parameters.columns:
        logger.warning(
            "'BOTTOM' column not found in parameters. Running all simulations."
        )
        return parameters

    filtered_params = parameters[parameters["BOTTOM"] == selected_bottom]

    if filtered_params.empty:
        logger.warning(
            f"No parameters found for bottom value {selected_bottom}. Running all simulations."
        )
        return parameters

    logger.info(
        f"Filtered parameters to include only bottom value {selected_bottom}. "
        f"Running {len(filtered_params)} simulations."
    )
    return filtered_params

    return filtered_params


def perform_cleanup(output_dir: str, logger) -> None:
    """
    Cleans up previous results and temporary directories.

    Args:
        output_dir: The directory containing result files (.slf, .txt)
        logger: Logger instance
    """
    import glob
    import shutil

    logger.info("Starting cleanup...")

    # 1. Clean output directory (res)
    if os.path.exists(output_dir):
        logger.info(f"Cleaning output directory: {output_dir}")
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}. Reason: {e}")
    else:
        logger.info(f"Output directory {output_dir} does not exist. Creating it.")
        os.makedirs(output_dir, exist_ok=True)

    # 2. Clean temporary run directories in root and src/telemac
    # Typical pattern: *.cas_YYYY-MM-DD-HHhMMmSSs

    logger.info("Cleaning temporary run directories...")

    # Define locations to search for temp dirs
    # Assuming script is run from project root or src/telemac
    target_dirs = [".", "src/telemac"]

    for target_dir in target_dirs:
        if os.path.exists(target_dir):
            pattern = os.path.join(target_dir, "*.cas_????-??-??-??h??m??s")
            for temp_dir in glob.glob(pattern):
                if os.path.isdir(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                        logger.info(f"Deleted temporary directory: {temp_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {temp_dir}. Reason: {e}")

    logger.info("Cleanup completed.")


if __name__ == "__main__":
    main()
