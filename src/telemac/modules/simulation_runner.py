import os
import subprocess

from loguru import logger
from tqdm import tqdm

from modules.file_handler import prepare_steering_file
from modules.file_utils import move_file, setup_output_dir
from modules.log_parser import LogParser, SimulationStatus
from modules.telemac_runner import run_telemac2d

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

    # 1. Smart Status Check
    status = LogParser.analyze(result_file)

    should_run = False
    continue_simulation = False

    if status == SimulationStatus.BALANCED:
        logger.info(f"Skipping {filename}: Flux boundaries balanced.")
        return False

    elif status == SimulationStatus.UNBALANCED:
        logger.info(f"Hot Start {filename}: Unbalanced. Extending duration.")
        should_run = True
        continue_simulation = True

    elif status == SimulationStatus.STALLED:
        logger.warning(
            f"Skipping {filename}: Simulation previously stalled (small dt)."
        )
        # Rename the bad result so we know it failed
        fail_file = os.path.join(output_dir, f"{name}_stalled.txt")
        if os.path.exists(result_file):
            move_file(result_file, fail_file)
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

        # For Hot Start, we need the ABSOLUTE path to the *previous* result file
        # The result file is in output_dir
        abs_result_file = os.path.abspath(result_file)

        prepare_steering_file(
            src_file, dst_file, result_file, abs_result_file, continue_simulation
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
        run_telemac2d(
            filename, output_dir, cwd=run_dir, timeout=1200
        )  # 20 mins timeout

        # 4. Post-Run Analysis for Hot-Loop Optimization
        # If we just ran it, check if it stalled NOW to avoid re-queueing it needlessly?
        # The main loop re-scans everything, so next time it will see it as STALLED/UNBALANCED.

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout: {filename} stalled/hung. Marking as failed.")
        # The run_telemac2d already logged it, but we can do extra cleanup if needed
        pass

    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")

    finally:
        # Restore CAS file? No, prepare_steering_file copied it to tmp dir.
        # We don't touch the original CAS in steering_folder associated with 'src_file' except reading.
        # Wait, prepare_steering_file implementation in file_handler MIGHT move things?
        # Checking file_handler: 'move_file(src_file, dst_file)' was the original code!
        # I changed it to copy2. So src_file is safe.

        # Cleanup
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
