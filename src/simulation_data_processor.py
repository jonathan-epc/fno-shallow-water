import argparse
import os
from dataclasses import dataclass

import h5py
import pandas as pd
import yaml
from loguru import logger

from modules.file_processing import process_and_save, process_and_save_normalized


@dataclass
class ProcessingConfig:
    base_dir: str
    generate_normalized: bool = False
    separate_critical_states: bool = False
    channel_length: float = 12
    channel_width: float = 0.3
    bottom_types: list[str] | None = None
    parameters_file: str = "parameters.csv"
    parameter_names: list[str] | None = None
    variable_names: list[str] | None = None
    set_type: str | None = None


def load_yaml_config(file_path: str) -> dict:
    """Load configuration parameters from a YAML file.

    Args:
        file_path (str): The path to the YAML configuration file.

    Returns:
        dict: The parsed configuration as a dictionary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path) as yaml_file:
        return yaml.safe_load(yaml_file)


def get_output_files(
    config: ProcessingConfig, set_type: str | None = None
) -> dict[str, dict[str, str]]:
    """Generate expected output file paths based on configuration and set type.

    Args:
        config (ProcessingConfig): The processing configuration settings.
        set_type (str | None): Optional prefix for the output files (e.g., 'test').

    Returns:
        dict: Dictionary mapping bottom types to their respective output file paths.
    """
    # Processed data goes into the 'data' folder in the root
    base_path = "data"
    os.makedirs(base_path, exist_ok=True)

    bottom_types = config.bottom_types or ["NOISE", "SLOPE", "BUMP", "BARS"]

    # If set_type is provided, use it as prefix (e.g., 'test_SLOPE.hdf5')
    # If not, just use the bottom type (e.g., 'SLOPE.hdf5')
    prefix = f"{set_type}_" if set_type else ""

    files = {}
    for bottom_type in bottom_types:
        base_filename = f"{prefix}{bottom_type}.hdf5"
        files[bottom_type] = {
            "main": os.path.join(base_path, base_filename),
            "normalized": os.path.join(base_path, f"normalized_{base_filename}"),
        }

        if config.separate_critical_states:
            for data_type in ["main", "normalized"]:
                base_name = files[bottom_type][data_type]
                files[bottom_type][f"{data_type}_subcritical"] = base_name.replace(
                    ".hdf5", "_subcritical.hdf5"
                )
                files[bottom_type][f"{data_type}_supercritical"] = base_name.replace(
                    ".hdf5", "_supercritical.hdf5"
                )

    return files


def load_and_validate_data(
    config: ProcessingConfig,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load simulation parameters and validate the existence of result files.
    
    Computes any missing adimensional numbers required for processing.

    Args:
        config (ProcessingConfig): The processing configuration settings.

    Returns:
        tuple: A tuple containing the parameters DataFrame, a list of result file paths, 
               and a list of variable names to extract.
    """
    # Use the specified parameters file
    parameters_file = config.parameters_file
    # Simulation results are in the 'res' folder within telemac_dir
    results_dir = os.path.join(config.base_dir, "res")

    if not os.path.exists(parameters_file):
        raise FileNotFoundError(f"Parameters file not found: {parameters_file}")
    if not os.path.isdir(results_dir):
        raise NotADirectoryError(f"Results directory not found: {results_dir}")

    parameters = pd.read_csv(parameters_file, index_col=0)
    result_files = [f for f in os.listdir(results_dir) if f.endswith(".slf")]

    if not result_files:
        raise ValueError(f"No result files found in {results_dir}")

    # Compute adimensional numbers
    def compute_adimensional_numbers(row):
        g = 9.81
        xc, yc = config.channel_length, config.channel_width
        bc, hc = row["SLOPE"] * xc, row["H0"]
        uc = row["Q0"] / (hc * yc)
        nut = row["nut"]

        return {
            "Ar": xc / yc,
            "Vr": 1,
            "Fr": uc / (g * hc) ** 0.5,
            "Hr": bc / hc,
            "Re": (uc * xc) / nut,
            "M": g * row["n"] ** 2 * xc / (hc ** (4 / 3)),
        }

    # Only compute adimensional numbers if they don't already exist
    # (They're already in parameters.csv from the generation step)
    required_adim_cols = {"Ar", "Vr", "Fr", "Hr", "Re", "M"}
    missing_adim_cols = required_adim_cols - set(parameters.columns)

    if missing_adim_cols:
        logger.info(f"Computing missing adimensional parameters: {missing_adim_cols}")
        adim_params = parameters.apply(
            compute_adimensional_numbers, axis=1, result_type="expand"
        )
        # Only add the missing columns
        parameters = pd.concat(
            [parameters, adim_params[list(missing_adim_cols)]], axis=1
        )
    else:
        logger.info("All adimensional parameters already present in parameters file")

    return (
        parameters,
        result_files,
        config.variable_names or ["H", "U", "V"],
    )


def prepare_parameter_table(
    parameters: pd.DataFrame, parameter_names: list[str], config: ProcessingConfig
) -> pd.DataFrame:
    """Compute statistics (mean, variance, etc.) for all numeric parameters.
    
    This is used for normalization across datasets.

    Args:
        parameters (pd.DataFrame): The DataFrame containing simulation parameters.
        parameter_names (list[str]): The specific parameters to track.
        config (ProcessingConfig): The processing configuration settings.

    Returns:
        pd.DataFrame: A DataFrame containing the computed statistics.
    """
    # Compute stats for ALL numeric parameters (including derived ones like nut, Ar, Fr, etc.)
    # This ensures normalization stats are available for all model inputs/outputs

    # Remove duplicate columns (keep first occurrence)
    parameters = parameters.loc[:, ~parameters.columns.duplicated()]

    parameter_stats = {}
    logger.debug(f"Parameters columns after deduplication: {list(parameters.columns)}")
    for param in parameters.columns:
        # Include all numeric columns, skip non-numeric ones (like BOTTOM, direction, etc.)
        if pd.api.types.is_numeric_dtype(parameters[param]):
            parameter_stats[param] = parameters[param].describe()

    logger.info(
        f"Computed statistics for {len(parameter_stats)} numeric parameters: {list(parameter_stats.keys())}"
    )

    parameter_table = (
        pd.DataFrame(parameter_stats)
        .T.reset_index()
        .rename(columns={"index": "names", "std": "variance"})
    )
    parameter_table["variance"] = parameter_table["variance"] ** 2
    return parameter_table


def process_data(
    config: ProcessingConfig,
    files: dict[str, dict[str, str]],
    parameters: pd.DataFrame,
    result_files: list[str],
    variable_names: list[str],
    parameter_names: list[str],
):
    """Process the simulation results and save them to HDF5 format.

    Handles creation of both main and normalized datasets, and optionally
    separates subcritical and supercritical states.

    Args:
        config (ProcessingConfig): The processing configuration settings.
        files (dict): Dictionary of target output files.
        parameters (pd.DataFrame): The parameters for each simulation.
        result_files (list[str]): List of paths to the TELEMAC-2D result files.
        variable_names (list[str]): Variables to extract from the results.
        parameter_names (list[str]): Parameter names to track.
    """
    parameter_table = prepare_parameter_table(parameters, parameter_names, config)
    bottom_types = config.bottom_types or ["NOISE", "SLOPE", "BUMP", "BARS"]

    for bottom_type in bottom_types:
        if config.separate_critical_states:
            for critical_state in ["subcritical", "supercritical"]:
                mask = (
                    parameters["subcritical"]
                    if critical_state == "subcritical"
                    else ~parameters["subcritical"]
                )
                filtered_parameters = parameters[mask]
                filtered_result_files = [
                    f for f, m in zip(result_files, mask, strict=False) if m
                ]

                with h5py.File(
                    files[bottom_type][f"main_{critical_state}"], "w"
                ) as hdf5_file:
                    process_and_save(
                        hdf5_file,
                        bottom_type.upper(),
                        config.base_dir,
                        config.channel_width,
                        config.channel_length,
                        filtered_parameters,
                        parameter_table,
                        parameter_names,
                        variable_names,
                        filtered_result_files,
                    )

                if config.generate_normalized:
                    with h5py.File(
                        files[bottom_type][f"normalized_{critical_state}"], "w"
                    ) as hdf5_file_norm:
                        process_and_save_normalized(
                            hdf5_file_norm,
                            bottom_type.upper(),
                            None,
                            config.base_dir,
                            config.channel_width,
                            config.channel_length,
                            filtered_parameters,
                            parameter_names,
                            variable_names,
                            filtered_result_files,
                        )
        else:
            with h5py.File(files[bottom_type]["main"], "w") as hdf5_file:
                table = process_and_save(
                    hdf5_file,
                    bottom_type.upper(),
                    config.base_dir,
                    config.channel_width,
                    config.channel_length,
                    parameters,
                    parameter_table,
                    parameter_names,
                    variable_names,
                    result_files,
                )

            if config.generate_normalized:
                with h5py.File(files[bottom_type]["normalized"], "w") as hdf5_file_norm:
                    process_and_save_normalized(
                        hdf5_file_norm,
                        bottom_type.upper(),
                        table,
                        config.base_dir,
                        config.channel_width,
                        config.channel_length,
                        parameters,
                        parameter_names,
                        variable_names,
                        result_files,
                    )


def process_simulation_results(config: ProcessingConfig, yaml_config: dict) -> None:
    """Main entry point for processing TELEMAC-2D simulation results.

    Orchestrates the loading, validation, and processing of simulation data
    into HDF5 format, handling different set types (e.g., train, test) and
    configuration settings.

    Args:
        config (ProcessingConfig): The parsed command-line configuration.
        yaml_config (dict): The full YAML configuration.
    """
    logger.info("Processing started")

    try:
        config.bottom_types = yaml_config.get(
            "bottom_types", ["NOISE", "SLOPE", "BUMP", "BARS"]
        )
        config.parameter_names = yaml_config.get(
            "parameter_names", ["H0", "Q0", "SLOPE", "n"]
        )
        config.variable_names = yaml_config.get("variable_names", ["H", "U", "V"])

        config.channel_length = yaml_config.get("channel_length", 12)

        config.channel_width = yaml_config.get("channel_width", 12)

        config.channel_width = yaml_config.get("channel_width", 12)

        # Resolve parameters file path using data_dir if using default
        if config.parameters_file == "parameters.csv":
            data_dir = yaml_config.get("paths", {}).get("data_dir", "data")
            # Avoid double joining if data_dir is already in path (unlikely for "parameters.csv" but safe)
            if not config.parameters_file.startswith(data_dir):
                config.parameters_file = os.path.join(data_dir, config.parameters_file)
            logger.info(f"Resolved parameters file to: {config.parameters_file}")

        # Load all data first
        all_parameters, all_result_files, variable_names = load_and_validate_data(
            config
        )

        # Determine which set_types to process
        if config.set_type:
            # If CLI argument provided, process only that set type
            set_types = [config.set_type]
            logger.info(f"Processing only set_type: {config.set_type}")
        elif "set_type" in all_parameters.columns:
            # If column exists, process all unique set types found
            set_types = all_parameters["set_type"].dropna().unique().tolist()
            logger.info(f"Found set_types in parameters: {set_types}")
        else:
            # Legacy/Single set behavior
            set_types = [None]
            logger.info(
                "No set_type column or argument found. Processing as single set."
            )

        for current_set_type in set_types:
            logger.info(
                f"Processing set: {current_set_type if current_set_type else 'All data'}"
            )

            # Filter parameters and result files
            if current_set_type:
                parameters = all_parameters[
                    all_parameters["set_type"] == current_set_type
                ]
            else:
                parameters = all_parameters

            if parameters.empty:
                logger.warning(
                    f"No parameters found for set_type: {current_set_type}. Skipping."
                )
                continue

            valid_ids = set(parameters.index.astype(str))
            result_files = [
                f for f in all_result_files if os.path.splitext(f)[0] in valid_ids
            ]

            if not result_files:
                logger.warning(
                    f"No result files matched for set_type: {current_set_type}. Skipping."
                )
                continue

            # Get output filenames with prefix (if set_type is not None)
            files = get_output_files(config, set_type=current_set_type)

            process_data(
                config,
                files,
                parameters,
                result_files,
                variable_names,
                config.parameter_names,
            )

        logger.success("Processing completed successfully")
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")
        # Log stack trace for better debugging
        logger.exception(e)


def parse_args() -> tuple[ProcessingConfig, str]:
    """Parse command-line arguments for the simulation data processor.

    Returns:
        tuple: A tuple containing the ProcessingConfig object and the path 
               to the configuration file.
    """
    parser = argparse.ArgumentParser(description="Process simulation data.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="src/telemac",
        help="Base directory for the simulation data",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--generate_normalized",
        default=False,
        action="store_true",
        help="Flag to generate normalized data",
    )
    parser.add_argument(
        "--separate_critical_states",
        default=False,
        action="store_true",
        help="Flag to separate subcritical and supercritical states",
    )
    parser.add_argument(
        "--parameters_file",
        type=str,
        default="parameters.csv",
        help="Path to the parameter CSV file",
    )
    parser.add_argument(
        "--set_type",
        type=str,
        default=None,
        help="Filter by set type (e.g., 'train_val', 'test')",
    )

    args = parser.parse_args()

    return (
        ProcessingConfig(
            base_dir=args.base_dir,
            generate_normalized=args.generate_normalized,
            separate_critical_states=args.separate_critical_states,
            parameters_file=args.parameters_file,
            set_type=args.set_type,
        ),
        args.config_file,
    )


if __name__ == "__main__":
    logger.remove()
    logger.add(
        "simulation_data_processor.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {function} | {line} | {message}",
        level="DEBUG",
        rotation="00:00",  # Rotate the log file at midnight
        retention=1,  # Keep only the most recent log file
    )

    config, config_file = parse_args()
    yaml_config = load_yaml_config(config_file)
    process_simulation_results(config, yaml_config)
