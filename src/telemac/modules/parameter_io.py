import os
import shutil
from datetime import datetime

import pandas as pd
from loguru import logger


class ParameterIO:
    """Handles file I/O operations for simulation parameters."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self) -> pd.DataFrame | None:
        """Load existing parameters from the CSV file."""
        if not os.path.exists(self.filepath):
            logger.info(f"No existing {self.filepath} found.")
            return None

        try:
            df = pd.read_csv(self.filepath, index_col="id")
            logger.info(
                f"Loaded existing parameters from {self.filepath}. Shape: {df.shape}"
            )
            return df
        except Exception as e:
            logger.error(f"Error loading {self.filepath}: {str(e)}")
            return None

    def save(self, df: pd.DataFrame) -> None:
        """Save parameters to the CSV file, creating a backup first."""
        self.backup()
        try:
            df.to_csv(self.filepath, index=True)
            logger.info(f"Wrote parameters to {self.filepath}. Shape: {df.shape}")
        except Exception as e:
            logger.error(f"Error saving {self.filepath}: {str(e)}")
            raise

    def backup(self) -> None:
        """Creates a timestamped backup of the current parameters file."""
        if not os.path.exists(self.filepath):
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(os.path.dirname(self.filepath), "backups")
        os.makedirs(backup_dir, exist_ok=True)

        backup_name = os.path.join(
            backup_dir, f"{os.path.basename(self.filepath)}.{timestamp}.bak"
        )
        try:
            shutil.copy2(self.filepath, backup_name)
            logger.info(f"Created backup of {self.filepath} in {backup_name}")
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
