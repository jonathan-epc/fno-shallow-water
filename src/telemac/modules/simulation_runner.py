import argparse
import os
import subprocess

from loguru import logger
from tqdm import tqdm

from .file_handler import prepare_steering_file
from .file_utils import move_file, setup_output_dir
from .log_parser import LogParser, SimulationStatus
from .telemac_runner import run_telemac2d

# Configure Loguru to play nice with TQDM
logger.remove()
logger.add(
    lambda msg: tqdm.write(msg, end=""),
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)


def run_single_simulation(i, filename, case_parameters, output_dir, steering_folder):
    """
    Run a single Telemac2D simulation with robust status handling.
    """
    name, ext = os.path.splitext(filename)
    result_file = os.path.join(output_dir, f"{name}.txt")
    result_slf = os.path.join(output_dir, f"{name}.slf")
    stalled_file = os.path.join(output_dir, f"{name}_stalled.txt")

    if os.path.exists(stalled_file):
        logger.warning(f"Skipping {filename}: already quarantined as stalled.")
        return False

    # 1. Smart Status Check
    status = LogParser.analyze(result_file)

    should_run = False
    continue_simulation = False

    if status == SimulationStatus.BALANCED:
        logger.info(f"Skipping {filename}: Flux boundaries balanced.")
        return False

    elif status == SimulationStatus.UNBALANCED:
        if os.path.exists(result_slf):
            logger.info(f"Hot Start {filename}: Unbalanced. Extending duration.")
            continue_simulation = True
        else:
            logger.info(
                f"Cold Restart {filename}: Unbalanced log but no .slf available."
            )
        should_run = True

    elif status == SimulationStatus.STALLED:
        logger.warning(
            f"Skipping {filename}: Simulation previously stalled (small dt)."
        )
        # Quarantine incomplete/failed result files to prevent downstream ingestion
        if os.path.exists(result_file):
            move_file(result_file, stalled_file)
        return False

    elif status == SimulationStatus.CRASHED:
        logger.info(
            f"Cold Restart {filename}: Previous run crashed or was interrupted."
        )
        should_run = True
        continue_simulation = False

    elif status == SimulationStatus.UNKNOWN:
        # No log file or empty, treat as new run
        should_run = True
        continue_simulation = False

    if not should_run:
        return False

    # 2. Run Setup
    telemac_root = os.path.dirname(steering_folder)
    tmp_runs_dir = os.path.join(telemac_root, "tmp_runs")
    os.makedirs(tmp_runs_dir, exist_ok=True)

    run_dir = os.path.join(tmp_runs_dir, f"{name}_run")
    os.makedirs(run_dir, exist_ok=True)

    try:
        src_file = os.path.join(steering_folder, filename)
        dst_file = os.path.join(run_dir, filename)

        # For Hot Start, use a path relative to the run directory. TELEMAC
        # rejects long absolute paths here (the closing quote falls outside its
        # steering-file line limit).
        previous_result_path = os.path.relpath(
            os.path.abspath(result_slf), os.path.abspath(run_dir)
        ).replace("\\", "/")

        prepare_steering_file(
            src_file, dst_file, result_file, previous_result_path, continue_simulation
        )

        # Patch relative paths in the CAS file
        try:
            with open(dst_file) as f:
                content = f.read()

            abs_telemac_root = os.path.abspath(telemac_root)
            abs_run_dir = os.path.abspath(run_dir)
            rel_path_root = os.path.relpath(abs_telemac_root, abs_run_dir).replace(
                "\\", "/"
            )

            content = content.replace("\\", "/")
            import re

            content = re.sub(r"\s+=\s+", " = ", content)

            # Replace 'src/telemac/' with relative path
            # Depending on how paths are defined in the template, this might need adjustment
            # Assuming standard 'src/telemac/geo/...' structure in template
            patched_content = content.replace("src/telemac/", f"{rel_path_root}/")

            with open(dst_file, "w") as f:
                f.write(patched_content)

        except Exception as e:
            logger.warning(f"Failed to patch CAS file paths: {e}")

        # 3. Run Execution with Timeout
        timeout = float(os.environ.get("TELEMAC_TIMEOUT", "2400"))
        run_telemac2d(filename, output_dir, cwd=run_dir, timeout=timeout)

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout: {filename} stalled/hung. Marking as failed.")
        pass

    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")

    finally:
        # Cleanup temporary run directory
        import shutil

        if os.path.exists(run_dir):
            try:
                shutil.rmtree(run_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup run dir {run_dir}: {e}")

    return True


def run_telemac2d_on_files(
    file_list: list[str], parameters, output_dir, steering_folder
):
    """
    Run Telemac2D simulations on a list of files.

    Parameters
    ----------
    file_list : List[str]
        List of steering file names to process.
    parameters : pandas.DataFrame
        A DataFrame containing the parameters for each simulation case.
    output_dir : str
        The directory where the output files will be saved.
    steering_folder : str
        The folder containing the steering files.

    Returns
    -------
    None
    """
    setup_output_dir(output_dir)

    total_files = len(file_list)

    with tqdm(total=total_files, unit="case", dynamic_ncols=True) as pbar:
        for i, filename in enumerate(file_list):
            case_id = os.path.splitext(filename)[0]  # Keep it as a string
            case_parameters = parameters.loc[case_id]  # Use string-based index

            pbar.set_description(f"Processing {filename}")
            simulation_run = run_single_simulation(
                i, filename, case_parameters, output_dir, steering_folder
            )

            if simulation_run:
                pbar.update(1)
            else:
                pbar.total -= 1
                pbar.refresh()

    logger.info("All Telemac2D simulations completed")


def main() -> int:
    """CLI entrypoint for running a single TELEMAC-2D case via status-aware runner."""
    parser = argparse.ArgumentParser(description="Run one TELEMAC-2D case.")
    parser.add_argument("--cas", required=True, help="Path to the case .cas file")
    args = parser.parse_args()

    cas_path = os.path.abspath(args.cas)
    if not os.path.isfile(cas_path):
        print(f"ERROR: case file not found: {cas_path}")
        return 2

    filename = os.path.basename(cas_path)
    steering_folder = os.path.dirname(cas_path)
    telemac_root = os.path.dirname(steering_folder)
    output_dir = os.path.join(telemac_root, "res")
    os.makedirs(output_dir, exist_ok=True)

    ran = run_single_simulation(0, filename, None, output_dir, steering_folder)
    case_id = os.path.splitext(filename)[0]
    status = LogParser.analyze(os.path.join(output_dir, f"{case_id}.txt"))
    print(f"STATUS: {case_id} ran={ran} status={status.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
