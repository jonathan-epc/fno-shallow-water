"""Helpers for configuring and running the 12-case experiment matrix."""

import argparse
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import torch

from ML.core.results import TRAINED_MODELS_INFO
from nconfig import PROJECT_ROOT, Config

ALL_CASE_KEYS: list[str] = list(TRAINED_MODELS_INFO.keys())

GEOM_LABELS: dict[str, str] = {
    "b": "BARS",
    "s": "SLOPE",
    "n": "NOISE",
}


def resolve_dataset_path(base_config: Config, geom_key: str) -> Path:
    """Resolve the HDF5 dataset path for a geometry key ('b', 's', or 'n')."""
    geom_label = GEOM_LABELS[geom_key]
    data_dir = Path(base_config.paths.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    candidates: list[Path] = [data_dir / f"train_val_{geom_label}.hdf5"]
    if base_config.datasets and geom_key in base_config.datasets:
        candidates.append(data_dir / base_config.datasets[geom_key].filename)
    candidates.append(data_dir / f"{geom_label}.hdf5")

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"No dataset found for geometry '{geom_label}'. Searched: {searched}"
    )


def resolve_test_dataset_path(file_path: str) -> str:
    """Resolve the corresponding test HDF5 path when given a split train_val_* file."""
    path = Path(file_path)
    if path.name.startswith("train_val_"):
        test_candidate = path.with_name(path.name.replace("train_val_", "test_", 1))
        if test_candidate.is_file():
            return str(test_candidate)
    return file_path


def apply_runtime_overrides(
    cfg: Config,
    epochs: int | None = None,
    trials: int | None = None,
    device: str | None = None,
) -> Config:
    """Apply optional runtime CLI overrides to a Config copy."""
    if epochs is not None:
        cfg.training.num_epochs = epochs
    if trials is not None:
        cfg.optuna.n_trials = trials
    if device is not None:
        cfg.device = (
            "cuda"
            if device in ("auto", "cuda") and torch.cuda.is_available()
            else ("cpu" if device in ("auto", "cuda") else device)
        )
    return cfg


def build_case_config(
    base_config: Config,
    case_key: str,
    epochs: int | None = None,
    trials: int | None = None,
    device: str | None = None,
) -> Config:
    """Build an isolated Config instance for one of the 12 canonical cases."""
    key = case_key.lower()
    if key not in TRAINED_MODELS_INFO:
        valid = ", ".join(ALL_CASE_KEYS)
        raise ValueError(f"Unknown case '{case_key}'. Valid cases: {valid}")

    spec = TRAINED_MODELS_INFO[key]
    geom_key = spec["source_geom"]
    geom_label = GEOM_LABELS[geom_key]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    cfg = base_config.model_copy(deep=True)
    cfg.data.file_path = str(resolve_dataset_path(base_config, geom_key))
    cfg.data.inputs = list(spec["inputs"])
    cfg.data.outputs = list(spec["outputs"])
    cfg.model.name = f"{key.upper()}-{geom_label}-{timestamp}"
    cfg.optuna.study_name = f"Study-{key.upper()}-{geom_label}"
    cfg.optuna.study_notes = f"Automated suite run for {key.upper()} ({geom_label})"

    return apply_runtime_overrides(cfg, epochs=epochs, trials=trials, device=device)


def build_suite_parser(
    description: str, include_trials: bool = False
) -> argparse.ArgumentParser:
    """Create a CLI parser supporting single-run, --cases, and --all execution."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 12 cases (3 geometries x direct/inverse x dim/adim).",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=ALL_CASE_KEYS,
        help="Run one or more specific case keys (e.g., --cases ddb idb).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.num_epochs from config.yml.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device from config.yml (e.g., cuda, cpu, auto).",
    )
    if include_trials:
        parser.add_argument(
            "--trials",
            type=int,
            default=None,
            help="Override optuna.n_trials from config.yml.",
        )
    return parser


def execute_case_batch(
    base_config: Config,
    case_keys: list[str],
    runner_fn: Callable[[Config], object],
    logger,
    epochs: int | None = None,
    trials: int | None = None,
    device: str | None = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Run runner_fn sequentially for each requested case key and collect outcomes."""
    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []

    for idx, case_key in enumerate(case_keys, start=1):
        logger.info(
            f"=== [{idx}/{len(case_keys)}] Starting case '{case_key.upper()}' ==="
        )
        try:
            case_cfg = build_case_config(
                base_config,
                case_key,
                epochs=epochs,
                trials=trials,
                device=device,
            )
            result = runner_fn(case_cfg)
            if result is None:
                failed.append((case_key, "Runner returned None"))
            else:
                succeeded.append(case_key)
        except Exception as exc:
            logger.exception(f"Case '{case_key}' failed: {exc}")
            failed.append((case_key, str(exc)))

    logger.info(
        f"Batch summary: {len(succeeded)}/{len(case_keys)} succeeded, {len(failed)} failed."
    )
    if failed:
        logger.error(f"Failed cases: {[k for k, _ in failed]}")

    return succeeded, failed
