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
    prev_line = f"PREVIOUS COMPUTATION FILE = '{prev_file_str}'\n"

    # TELEMAC 9.0 enables continuation simply by defining PREVIOUS COMPUTATION
    # FILE; the old COMPUTATION CONTINUED boolean is rejected as an unknown
    # keyword.
    new_content = []
    prev_found = False
    for line in content:
        stripped = line.strip()
        if stripped.startswith("COMPUTATION CONTINUED"):
            continue
        if stripped.startswith("PREVIOUS COMPUTATION FILE"):
            if not prev_found:
                new_content.append(prev_line)
                prev_found = True
            continue
        new_content.append(line)

    if not prev_found:
        insert_at = len(new_content)
        for idx in range(len(new_content) - 1, -1, -1):
            if new_content[idx].strip():
                insert_at = idx + 1
                break
        new_content.insert(insert_at, prev_line)

    with open(steering_file, "w") as f:
        f.writelines(new_content)
    logger.info(f"Continuation lines set in file {steering_file}")


def update_duration(steering_file: str, increment: int = 30) -> None:
    """
    Updates the duration in the steering file by adding seconds.

    Parameters
    ----------
    steering_file : str
        The path to the steering file.
    increment : int, optional
        Duration increment in seconds. Defaults to 30.

    Examples
    --------
    >>> update_duration('steering.txt')
    """
    with open(steering_file) as f:
        content = f.readlines()

    duration_found = False
    new_duration = None

    for idx, line in enumerate(content):
        stripped = line.strip()
        if stripped.startswith("DURATION") and "=" in stripped:
            try:
                current_duration = float(stripped.split("=")[1].strip())
                new_duration = int(current_duration + increment)
                content[idx] = f"DURATION = {new_duration}\n"
                duration_found = True
                break
            except ValueError:
                logger.warning(
                    f"Could not parse DURATION line: {stripped} in {steering_file}"
                )

    if duration_found:
        with open(steering_file, "w") as f:
            f.writelines(content)
        logger.info(f"Duration updated to {new_duration} s in file {steering_file}")
    else:
        logger.warning(
            f"No active DURATION line found in {steering_file}; skipping duration update."
        )


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
    if src_file != dst_file:
        from shutil import copy2

        try:
            copy2(src_file, dst_file)
        except Exception as e:
            logger.error(f"Failed to copy {src_file} to {dst_file}: {e}")
            return

    target_file = dst_file

    if os.path.exists(result_file) and continue_simulation:
        # Resume simulation from previous result file
        add_continuation_lines(target_file, previous_result_file)
        update_duration(target_file)
