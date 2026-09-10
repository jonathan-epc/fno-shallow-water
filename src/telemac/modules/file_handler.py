import os

from loguru import logger


def add_continuation_lines(steering_file, previous_result_file):
    """
    Adds continuation lines to the steering file.

    Parameters
    ----------
    steering_file : str
        The path to the steering file.
    previous_result_file : str
        The absolute path to the previous computation file.
    """
    with open(steering_file) as f:
        content = f.readlines()

    # Use forward slashes for Telemac compatibility even on Windows
    prev_file_str = str(previous_result_file).replace("\\", "/")

    continuation_lines = [
        "\nCOMPUTATION CONTINUED = YES",
        f"\nPREVIOUS COMPUTATION FILE = '{prev_file_str}'\n",
    ]

    # Add lines after the last non-empty line
    for idx in range(len(content) - 1, -1, -1):
        if content[idx].strip():
            content.insert(idx + 1, "\n".join(continuation_lines))
            break

    with open(steering_file, "w") as f:
        f.writelines(content)
    logger.info(f"Continuation lines added to file {steering_file}")


def update_duration(steering_file):
    """
    Updates the duration in the steering file by adding 30 seconds.

    Parameters
    ----------
    steering_file : str
        The path to the steering file.

    Examples
    --------
    >>> update_duration('steering.txt')
    """
    with open(steering_file) as f:
        content = f.readlines()

    for idx, line in enumerate(content):
        if line.strip().startswith("DURATION"):
            current_duration = int(line.split("=")[1].strip())
            new_duration = current_duration + 30
            content[idx] = f"DURATION = {new_duration}\n"
            break

    with open(steering_file, "w") as f:
        f.writelines(content)
    logger.info(f"Duration updated to {new_duration} s in file {steering_file}")


def prepare_steering_file(
    src_file, dst_file, result_file, previous_result_file, continue_simulation
):
    """
    Prepares the steering file for the next simulation step.

    Parameters
    ----------
    src_file : str
        The path to the source steering file.
    dst_file : str
        The path to the destination steering file.
    result_file : str
        The path to the result file (used to check existence).
    previous_result_file : str
        The path to the previous result file (used for continuation).
    continue_simulation : bool
        Whether to continue the simulation from the previous result.
    """
    if os.path.exists(result_file):
        if continue_simulation:
            # We want to continue from the PREVIOUS result
            # But wait, if we are continuing, the 'result_file' IS the previous result
            # that we want to resume from.
            add_continuation_lines(src_file, previous_result_file)
        # We always move the file, but we only update duration if we are NOT continuing
        # Actually, if we continue, we generally WANT to extend duration too, right?
        # The original logic was:
        # if continue: add continuation (implies duration extension is handled elsewhere or implicitly?)
        # else: update duration (implies cold restart with longer time?)
        #
        # Let's align with the user request:
        # "checks if there are already logs... then it adds more time"
        #
        # For HOT START (Unbalanced): We need continuation + more time.
        # For COLD START (Crashed): We just want standard time (no update needed? or reset?)
        #
        # If we are reusing the *same* .cas file, it potentially already has modified duration/continuation?
        # No, we always copy from src_file (the template/original).

        if continue_simulation:
            update_duration(src_file)

    if src_file == dst_file:
        # In-place modification
        pass
    else:
        from shutil import copy2

        try:
            copy2(src_file, dst_file)
        except Exception as e:
            logger.error(f"Failed to copy {src_file} to {dst_file}: {e}")
