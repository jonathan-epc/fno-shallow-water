# ML/core/repeater.py


import os

import optuna
import torch
from torch.utils.data import random_split

from common.utils import setup_logger
from ML.modules import models as model_zoo
from ML.modules.data import HDF5Dataset
from ML.modules.training import cross_validation_procedure
from nconfig import Config


class TrialRepeater:
    """
    Encapsulates the logic to load and re-run a specific Optuna trial.
    """

    def __init__(self, config: Config, trial_id: int):
        self.config = config
        self.trial_id = trial_id
        self.logger = setup_logger()

    def _get_model_class(self):
        """Gets the model class from the config file using its name."""
        model_name = self.config.model.class_name
        try:
            return getattr(model_zoo, model_name)
        except AttributeError:
            raise ValueError(
                f"Model class '{model_name}' not found in 'src/ML/modules/models.py'"
            ) from None

    def run(self) -> float | None:
        """
        Loads the study, retrieves the trial's hyperparameters, and re-runs
        the training and evaluation process.

        Returns:
            Optional[float]: The average cross-validation loss from the repeated run, or None on failure.
        """
        self.logger.info(
            f"Attempting to repeat Optuna trial {self.trial_id} from study '{self.config.optuna.study_name}'."
        )
        try:
            # --- 1. Load Study and Trial Hyperparameters ---
            study = optuna.load_study(
                study_name=self.config.optuna.study_name,
                storage=self.config.optuna_storage_url,
            )
            trial_to_repeat = study.trials[self.trial_id]

            if trial_to_repeat.state != optuna.trial.TrialState.COMPLETE:
                self.logger.warning(
                    f"Trial {self.trial_id} was not completed successfully (State: {trial_to_repeat.state})."
                )

            hparams = trial_to_repeat.params
            self.logger.info(f"Found trial {self.trial_id} with parameters: {hparams}")

            # --- 2. Load and Split Data ---
            self.logger.info("Loading and splitting data for the repeated trial...")
            if os.path.exists(self.config.data.parameters_path):
                train_val_dataset = HDF5Dataset.from_config(
                    self.config,
                    self.config.data.file_path,
                    filter_set_type="train_val",
                )
                full_dataset = train_val_dataset
            else:
                full_dataset = HDF5Dataset.from_config(
                    self.config, self.config.data.file_path
                )
                test_size = int(self.config.training.test_frac * len(full_dataset))
                train_val_size = len(full_dataset) - test_size
                g_split = torch.Generator().manual_seed(self.config.seed)
                train_val_dataset, _ = random_split(
                    full_dataset, [train_val_size, test_size], generator=g_split
                )

            # --- 3. Run Cross-Validation Procedure ---
            model_class = self._get_model_class()
            name = f"{self.config.optuna.study_name}_{self.config.model.architecture}_repeat_trial_{self.trial_id}"

            avg_cv_loss = cross_validation_procedure(
                name=name,
                model_class=model_class,
                kfolds=self.config.training.kfolds,
                hparams=hparams,
                config=self.config,
                train_val_dataset=train_val_dataset,
                full_dataset_for_stats=full_dataset,
                is_sweep=False,
                trial=None,
            )

            self.logger.info(
                f"Finished repeating trial {self.trial_id}. Average CV loss: {avg_cv_loss:.4f}"
            )
            self.logger.info(
                f"Original trial {self.trial_id} value was: {trial_to_repeat.value}"
            )

            # Note: Final test set evaluation is implicitly handled by the ModelTrainer/Optimizer scripts
            # after a full run. This repeater's primary job is to reproduce the training/validation result.

            return avg_cv_loss

        except IndexError:
            self.logger.error(
                f"Trial number {self.trial_id} is out of bounds for study '{self.config.optuna.study_name}'."
            )
            return None
        except Exception as e:
            self.logger.exception(
                f"Error during repetition of trial {self.trial_id}: {e}"
            )
            return None
