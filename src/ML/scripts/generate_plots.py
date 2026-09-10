# src/ML/scripts/generate_plots.py

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Correct, absolute imports for the src-layout
from common.utils import setup_logger
from ML.core.results import (
    GEOM_NAMES,
    GEOMETRY_FILES,
    TRAINED_MODELS_INFO,
    ResultsLoader,
)
from ML.modules.data import HDF5Dataset
from ML.modules.metrics import evaluate_predictions
from ML.modules.plots import (
    PlotManager,
    plot_error_analysis,
    plot_field_comparison,
    plot_scatter_predictions,
)
from ML.plotting_style import set_plotting_style

# Define project root relative to this file's location. This is robust.
project_root = Path(__file__).resolve().parents[3]
logger = setup_logger()


# --- NEW: Helper function for filtering ---
def filter_variables_and_data(original_names, data_tensor, include_list, exclude_list):
    """
    Filters variable names and corresponding data tensors based on include/exclude lists.

    Returns:
        tuple: (filtered_names, filtered_data_tensor)
    """
    if not original_names or data_tensor is None:
        return original_names, data_tensor

    if include_list:
        indices_to_keep = []
        filtered_names = []
        for name in include_list:
            try:
                idx = original_names.index(name)
                indices_to_keep.append(idx)
                filtered_names.append(name)
            except ValueError:
                logger.warning(
                    f"Variable '{name}' in --include-vars not found in model outputs. Skipping."
                )

        if not indices_to_keep:
            return None, None  # Return None if no requested variables were found

        return filtered_names, data_tensor[:, indices_to_keep]

    if exclude_list:
        indices_to_keep = [
            i for i, name in enumerate(original_names) if name not in exclude_list
        ]
        filtered_names = [original_names[i] for i in indices_to_keep]
        return filtered_names, data_tensor[:, indices_to_keep]

    # If no filtering is specified, return original data
    return original_names, data_tensor


def run_single_job(
    model_key: str,
    data_key: str,
    lang: str,
    include_vars: list | None,
    exclude_vars: list | None,
    is_publication: bool,
    separate_plots: bool,
    dark_mode: bool,
    formats: list[str] | None,
):
    """
    This is the core "worker" function. It runs ONE evaluation and plotting job
    and then exits. It is designed to be called in a completely isolated process.
    """
    if include_vars and exclude_vars:
        logger.error(
            "Cannot use --include-vars and --exclude-vars simultaneously. Please choose one."
        )
        sys.exit(1)

    try:
        dataset_name = GEOM_NAMES[data_key]
        logger.info("=" * 80)
        logger.info(
            f"STARTING JOB: Model '{model_key.upper()}' on Data '{dataset_name}' (Lang: {lang})"
        )

        # 1. Load the model and configuration
        logger.info(
            "DEBUG: Initializing ResultsLoader skipping dataset load (load_dataset=False)"
        )
        results_loader = ResultsLoader(
            study_name=TRAINED_MODELS_INFO[model_key]["study_name"],
            trial_number=TRAINED_MODELS_INFO[model_key]["trial_number"],
            load_dataset=False,
        )
        logger.info("DEBUG: ResultsLoader initialized. Now patching path.")
        model = results_loader.model
        config = results_loader.config
        hparams = results_loader.hparams

        # Ensure correct training data path using project config
        # Instead of trusting the old config file path (which might be wrong/stale, e.g. case mismatch),
        # we deduce the correct file from the 'source_geom' metadata and the current project config.
        # This is 100% robust against "barsa.hdf5" vs "BARSa.hdf5" issues.
        source_geom = TRAINED_MODELS_INFO[model_key]["source_geom"]
        correct_filename = GEOMETRY_FILES[source_geom]

        data_dir = project_root / config.paths.data_dir
        corrected_path = data_dir / correct_filename

        config.data.file_path = str(corrected_path)

        logger.info(f"Resolved training data path: {corrected_path}")
        results_loader._setup_datasets()

        # Capture training stats to enforce consistency
        training_stats = results_loader.full_dataset.stats
        training_lambdas = results_loader.full_dataset.boxcox_lambdas
        training_shifts = results_loader.full_dataset.boxcox_shifts

        # 2. Prepare the evaluation dataset
        eval_config = config.model_copy(deep=True)
        data_dir = project_root / eval_config.paths.data_dir
        eval_config.data.file_path = str(data_dir / GEOMETRY_FILES[data_key])
        eval_config.data.preload_hdf5 = False  # Explicitly disable preloading

        eval_dataset = HDF5Dataset.from_config(
            eval_config, file_path=eval_config.data.file_path
        )

        # Overwrite evaluation stats with TRAINING stats
        logger.info(
            "Overwriting evaluation dataset statistics with TRAINING statistics."
        )
        eval_dataset.stats = training_stats
        eval_dataset.boxcox_lambdas = training_lambdas
        eval_dataset.boxcox_shifts = training_shifts

        # Smart Dataset Split Logic
        # If file starts with "test", assume it's a dedicated test set -> Use 100%
        # Otherwise, assume it's original training data -> Use random split test_frac (e.g. 20%)
        filename = Path(eval_config.data.file_path).name

        if filename.startswith("test"):
            logger.info(
                f"Detected dedicated test file '{filename}'. Using FULL dataset."
            )
            test_dataset = eval_dataset
        else:
            test_frac = eval_config.training.test_frac
            logger.info(
                f"Detected training file '{filename}'. Using random split ({test_frac:.0%})."
            )
            test_size = int(test_frac * len(eval_dataset))
            train_val_size = len(eval_dataset) - test_size
            _, test_dataset = torch.utils.data.random_split(
                eval_dataset,
                [train_val_size, test_size],
                generator=torch.Generator().manual_seed(eval_config.seed),
            )

        loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=hparams.get("batch_size", 32), shuffle=False
        )

        # 3. Run inference
        all_field_preds, all_scalar_preds = [], []
        all_field_targs, all_scalar_targs = [], []

        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc=f"Inference ({model_key.upper()})",
                file=sys.stdout,
                dynamic_ncols=True,
            ):
                inputs, targets_batch, metadata = batch  # Unpack metadata
                inputs_on_device = (
                    [t.to(config.device) for t in inputs[0]],
                    [t.to(config.device) for t in inputs[1]],
                )
                preds = model(inputs_on_device)

                if hparams.get("normalize_output", False):
                    from ML.modules.utils import denormalize_outputs_and_targets

                    preds, targets_batch = denormalize_outputs_and_targets(
                        preds, targets_batch, eval_dataset, eval_config, True
                    )

                if preds[0] is not None:
                    all_field_preds.append(preds[0].cpu())
                if preds[1] is not None:
                    all_scalar_preds.append(preds[1].cpu())
                if targets_batch[0]:
                    all_field_targs.append(torch.stack(targets_batch[0], dim=1).cpu())
                if targets_batch[1]:
                    all_scalar_targs.append(torch.stack(targets_batch[1], dim=1).cpu())

        predictions = (
            torch.cat(all_field_preds) if all_field_preds else None,
            torch.cat(all_scalar_preds) if all_scalar_preds else None,
        )
        targets = (
            torch.cat(all_field_targs) if all_field_targs else None,
            torch.cat(all_scalar_targs) if all_scalar_targs else None,
        )

        # --- NEW: Apply filtering based on command-line arguments ---
        output_fields, fields_tensor_pred = filter_variables_and_data(
            config.data.output_fields, predictions[0], include_vars, exclude_vars
        )
        _, fields_tensor_targ = filter_variables_and_data(
            config.data.output_fields, targets[0], include_vars, exclude_vars
        )

        output_scalars, scalars_tensor_pred = filter_variables_and_data(
            config.data.output_scalars, predictions[1], include_vars, exclude_vars
        )
        _, scalars_tensor_targ = filter_variables_and_data(
            config.data.output_scalars, targets[1], include_vars, exclude_vars
        )

        # Re-assemble the filtered predictions and targets tuples
        filtered_predictions = (fields_tensor_pred, scalars_tensor_pred)
        filtered_targets = (fields_tensor_targ, scalars_tensor_targ)

        logger.info(f"Original fields: {config.data.output_fields}")
        logger.info(f"Plotting for fields: {output_fields}")
        logger.info(f"Original scalars: {config.data.output_scalars}")
        logger.info(f"Plotting for scalars: {output_scalars}")

        # 4. Calculate Metrics & Setup Plotting (using filtered data)
        metrics, per_case_df = evaluate_predictions(
            filtered_predictions, filtered_targets, output_fields, output_scalars
        )
        output_dir = (
            project_root / "publication_figures" / lang / model_key / dataset_name
        )
        plot_manager = PlotManager(
            output_path=output_dir,
            publication_style=is_publication,
            dark_mode=dark_mode,
            formats=formats,
        )
        title_prefix = f"Model: {model_key.upper()} | Data: {dataset_name}"

        # 5. Generate Plots (using filtered data and names)
        plot_scatter_predictions(
            filtered_predictions,
            filtered_targets,
            output_fields,
            output_scalars,
            plot_manager,
            metrics,
            title_prefix,
            lang,
            publication=is_publication,
            separate_plots=separate_plots,
        )
        plot_error_analysis(
            filtered_predictions,
            filtered_targets,
            output_fields,
            plot_manager,
            title_prefix,
            lang,
            publication=is_publication,
        )

        if not per_case_df.empty and output_fields:
            for field_idx, field_name in enumerate(output_fields):
                field_df = per_case_df[per_case_df["variable"] == field_name]
                if not field_df.empty:
                    # Check for valid RMSE values to avoid crash
                    if field_df["rmse"].isna().all():
                        logger.warning(
                            f"All RMSE values are NaN for variable '{field_name}'. Skipping per-case plotting."
                        )
                        continue

                    cases_to_plot = {
                        "best": int(field_df.loc[field_df["rmse"].idxmin()]["case_id"]),
                        "worst": int(
                            field_df.loc[field_df["rmse"].idxmax()]["case_id"]
                        ),
                        "median": int(
                            field_df.loc[
                                (field_df["rmse"] - field_df["rmse"].median())
                                .abs()
                                .idxmin()
                            ]["case_id"]
                        ),
                    }
                    for case_type, case_id in cases_to_plot.items():
                        plot_field_comparison(
                            prediction=filtered_predictions[0][case_id, field_idx],
                            target=filtered_targets[0][case_id, field_idx],
                            variable_name=field_name,
                            plot_manager=plot_manager,
                            case_id=case_id,
                            title_prefix=title_prefix,
                            language=lang,
                            publication=is_publication,
                            case_type=case_type,  # Pass the new argument
                        )

        # 6. Save Metrics to JSON for aggregation
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": model_key,
                    "data": data_key,
                    "lang": lang,
                    "metrics": metrics,
                },
                f,
                indent=4,
            )

        logger.info(f"SUCCESS: Metrics and plots saved in: {output_dir}")

    except Exception as e:
        logger.exception(
            f"FATAL ERROR in worker for job ({model_key}, {data_key}, {lang}): {e}"
        )
        sys.exit(1)


def main():
    """
    Main entry point for generating evaluation plots.

    This script acts as both a manager and a worker process:
    - Manager: Parses arguments, determines which combinations of models, datasets,
      and languages need plotting, and spawns worker processes for each.
    - Worker: Executes a single, isolated plotting job to prevent memory leaks
      and ensure independent evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Generate comprehensive evaluation plots."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model key to plot (e.g., 'ddb', or 'all').",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="all",
        help="Dataset key to evaluate on (e.g., 'b', or 'all').",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="all",
        help="Language for plots (e.g., 'en', or 'all').",
    )
    parser.add_argument(
        "--include-vars",
        nargs="+",  # Accepts one or more arguments
        default=None,
        help="Space-separated list of variables to plot. Only these will be plotted.",
    )
    parser.add_argument(
        "--exclude-vars",
        nargs="+",
        default=None,
        help="Space-separated list of variables to exclude from plotting.",
    )
    parser.add_argument(
        "--publication",
        action="store_true",  # This makes it a flag, e.g., --publication
        help="Generate plots in publication mode (no titles, panel labels).",
    )
    parser.add_argument(
        "--separate-plots",
        action="store_true",
        help="Generate each variable's plot as a separate file, not in subplots.",
    )
    parser.add_argument(
        "--dark-mode",
        action="store_true",
        help="Use dark mode style (transparent background, white text).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf"],
        help="List of formats to save plots in (e.g., pdf png svg). Default: pdf png",
    )

    parser.add_argument(
        "--test-data",
        action="store_true",
        help="Use the dedicated test datasets (test_BARS.hdf5 etc). Defaults to training data if not set.",
    )

    args = parser.parse_args()

    # --- SCRIPT BEHAVIOR SWITCH ---
    is_worker = args.model != "all" and args.data != "all" and args.lang != "all"

    if is_worker:
        # --- WORKER LOGIC ---
        set_plotting_style()
        run_single_job(
            args.model,
            args.data,
            args.lang,
            args.include_vars,
            args.exclude_vars,
            args.publication,
            args.separate_plots,
            args.dark_mode,
            args.formats,
        )

    else:
        # --- MANAGER LOGIC ---
        logger.info(
            "--- Running as MANAGER: Spawning worker processes for each job ---"
        )

        models_to_plot = (
            list(TRAINED_MODELS_INFO.keys()) if args.model == "all" else [args.model]
        )

        # Dynamic Dataset Selection based on flag
        all_keys = list(GEOMETRY_FILES.keys())
        test_keys = [k for k in all_keys if k.startswith("test_")]
        train_keys = [k for k in all_keys if not k.startswith("test_")]

        if args.data == "all":
            datasets_to_eval = test_keys if args.test_data else train_keys
        else:
            # Smart logic: if user requests 'b' with --test-data, switch to 'test_b'
            if args.test_data and not args.data.startswith("test_"):
                datasets_to_eval = [f"test_{args.data}"]
            else:
                datasets_to_eval = [args.data]

        languages = ["en", "es"] if args.lang == "all" else [args.lang]

        all_jobs = []
        for model_key in models_to_plot:
            for data_key in datasets_to_eval:
                for lang in languages:
                    all_jobs.append((model_key, data_key, lang))

        success_count = 0
        with tqdm(
            total=len(all_jobs),
            desc="Overall Plotting Progress",
            file=sys.stdout,
            dynamic_ncols=True,
        ) as pbar:
            for model_key, data_key, lang in all_jobs:
                pbar.set_description(
                    f"Job: {model_key.upper()} on {GEOM_NAMES.get(data_key, data_key)} ({lang})"
                )

                command = [
                    sys.executable,
                    __file__,
                    "--model",
                    model_key,
                    "--data",
                    data_key,
                    "--lang",
                    lang,
                ]

                if args.publication:
                    command.append("--publication")
                if args.separate_plots:
                    command.append("--separate-plots")
                if args.dark_mode:
                    command.append("--dark-mode")
                if args.formats:
                    command.extend(["--formats"] + args.formats)
                if args.include_vars:
                    command.extend(["--include-vars"] + args.include_vars)
                if args.exclude_vars:
                    command.extend(["--exclude-vars"] + args.exclude_vars)

                # Use default output behavior to allow real-time logs and tqdm from worker
                result = subprocess.run(command)

                if result.returncode == 0:
                    success_count += 1
                else:
                    logger.error(
                        f"--- Worker FAILED for job ({model_key}, {data_key}, {lang}) ---"
                    )
                    # No stdout/stderr to print since we didn't capture it; it went to console already

                pbar.update(1)

        logger.info("--- Plot Generation Summary ---")
        logger.info(
            f"Successfully completed {success_count} out of {len(all_jobs)} plotting jobs."
        )

        # --- NEW: Aggregate Metrics into a Table ---
        if success_count > 0:
            logger.info("--- Aggregating Metrics ---")
            all_metrics_data = []
            for model_key, data_key, lang in all_jobs:
                dataset_name = GEOM_NAMES.get(data_key, data_key)
                metrics_file = (
                    project_root
                    / "publication_figures"
                    / lang
                    / model_key
                    / dataset_name
                    / "metrics.json"
                )
                if metrics_file.exists():
                    try:
                        with open(metrics_file, encoding="utf-8") as f:
                            data = json.load(f)
                            # Flatten the metrics dictionary
                            for var_name, var_metrics in data["metrics"].items():
                                row = {
                                    "Model": data["model"].upper(),
                                    "Dataset": data["data"],
                                    "Lang": data["lang"],
                                    "Variable": var_name,
                                }
                                row.update(var_metrics)
                                all_metrics_data.append(row)
                    except Exception as e:
                        logger.warning(
                            f"Failed to read metrics from {metrics_file}: {e}"
                        )

            if all_metrics_data:
                import pandas as pd

                df = pd.DataFrame(all_metrics_data)

                # Reorder columns for readability
                cols = [
                    "Model",
                    "Dataset",
                    "Lang",
                    "Variable",
                    "rmse",
                    "mae",
                    "r2",
                    "smape",
                ]
                available_cols = [c for c in cols if c in df.columns]
                df = df[available_cols]

                summary_dir = project_root / "publication_figures"
                summary_dir.mkdir(parents=True, exist_ok=True)

                csv_path = summary_dir / "metrics_summary.csv"
                md_path = summary_dir / "metrics_summary.md"

                df.to_csv(csv_path, index=False)
                # Generate a nice Markdown table
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write("# Metrics Summary\n\n")
                    f.write(df.to_markdown(index=False))

                logger.info("Consolidated metrics saved to:")
                logger.info(f"  - CSV: {csv_path}")
                logger.info(f"  - Markdown: {md_path}")
            else:
                logger.warning("No metrics data found to aggregate.")


if __name__ == "__main__":
    main()
