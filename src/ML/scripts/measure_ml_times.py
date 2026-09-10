# src/ML/scripts/measure_ml_times.py

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

# Correct, absolute imports for the src-layout
from common.utils import setup_logger
from ML.core.results import (
    GEOMETRY_FILES,
    TRAINED_MODELS_INFO,
    ResultsLoader,
)
from ML.modules.data import HDF5Dataset

# Define project root relative to this file's location
project_root = Path(__file__).resolve().parents[3]
logger = setup_logger()


def measure_inference_times(model_key: str, data_key: str, output_csv: str):
    if model_key not in TRAINED_MODELS_INFO:
        logger.error(f"Model key '{model_key}' not found.")
        sys.exit(1)

    if data_key not in GEOMETRY_FILES:
        logger.error(f"Data key '{data_key}' not found.")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info(
        f"Measuring ML Inference Times for Model '{model_key.upper()}' on Data '{data_key}'"
    )

    try:
        # Load the model and configuration
        results_loader = ResultsLoader(
            study_name=TRAINED_MODELS_INFO[model_key]["study_name"],
            trial_number=TRAINED_MODELS_INFO[model_key]["trial_number"],
            load_dataset=False,
        )
        model = results_loader.model
        config = results_loader.config

        # We put the model in evaluation mode since we are only inferencing
        model.eval()
        device_str = str(config.device)
        logger.info(f"Model loaded and moved to {device_str}")

        # Resolve correct training data path to grab stats, similar to generate_plots.py
        source_geom = TRAINED_MODELS_INFO[model_key]["source_geom"]
        correct_filename = GEOMETRY_FILES[source_geom]
        data_dir = project_root / config.paths.data_dir

        config.data.file_path = str(data_dir / correct_filename)
        results_loader._setup_datasets()

        training_stats = results_loader.full_dataset.stats
        training_lambdas = results_loader.full_dataset.boxcox_lambdas
        training_shifts = results_loader.full_dataset.boxcox_shifts

        # Prepare the evaluation dataset
        eval_config = config.model_copy(deep=True)
        eval_config.data.file_path = str(data_dir / GEOMETRY_FILES[data_key])
        eval_config.data.preload_hdf5 = False  # Explicitly disable preloading for raw speed testing, or enable if wanted

        logger.info(f"Loading dataset from: {eval_config.data.file_path}")
        eval_dataset = HDF5Dataset.from_config(
            eval_config, file_path=eval_config.data.file_path
        )

        # Overwrite evaluation stats with training stats for correct normalization during inference
        eval_dataset.stats = training_stats
        eval_dataset.boxcox_lambdas = training_lambdas
        eval_dataset.boxcox_shifts = training_shifts

        # We will iterate in batches to measure the fastest possible throughput
        batch_size = (
            eval_config.training.batch_size
            if getattr(eval_config, "training", None)
            else 32
        )

        # Use num_workers > 0 if not on CPU to hide IO/Preprocessing latency
        # Note: Windows can be finicky with workers and HDF5
        num_workers = 4 if device_str != "cpu" and sys.platform != "win32" else 0

        loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        num_cases = len(eval_dataset)
        logger.info(f"Total cases to evaluate: {num_cases}")
        logger.info(f"Batch Size: {batch_size} | Num Workers: {num_workers}")

        if num_cases == 0:
            logger.error("Dataset is empty.")
            sys.exit(1)

        # WARMUP RUNS (to avoid counting PyTorch/CUDA initialization overhead)
        logger.info("Running 1 warmup batch...")
        with torch.no_grad():
            for batch in loader:
                inputs_raw, _, _ = batch
                inputs = (
                    [t.to(config.device) for t in inputs_raw[0]],
                    [t.to(config.device) for t in inputs_raw[1]],
                )
                _ = model(inputs)
                if "cuda" in device_str:
                    torch.cuda.synchronize()
                break  # Just one batch is enough for warmup

        logger.info(
            "Warmup complete. Starting timed inferences for the entire dataset..."
        )

        total_inference_time = 0.0
        start_time_total = time.perf_counter()

        with torch.no_grad():
            for batch in tqdm(
                loader, desc="Measuring Batches", file=sys.stdout, dynamic_ncols=True
            ):
                # Unpack and move to device
                inputs_raw, _, _ = batch
                inputs = (
                    [t.to(config.device) for t in inputs_raw[0]],
                    [t.to(config.device) for t in inputs_raw[1]],
                )

                # ISOLATE PURE INFERENCE (Forward Pass)
                if "cuda" in device_str:
                    torch.cuda.synchronize()

                inf_start = time.perf_counter()

                _ = model(inputs)

                if "cuda" in device_str:
                    torch.cuda.synchronize()

                total_inference_time += time.perf_counter() - inf_start

        end_time_total = time.perf_counter()

        total_loop_duration = end_time_total - start_time_total

        # Calculations for Full Loop (End-to-End)
        avg_e2e_per_case = total_loop_duration / num_cases
        e2e_throughput = num_cases / total_loop_duration

        # Calculations for Pure Model Inference
        avg_model_per_case = total_inference_time / num_cases
        model_throughput = num_cases / total_inference_time

        # Write summary to CSV
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    "model",
                    "dataset",
                    "total_cases",
                    "batch_size",
                    "pure_inference_total_s",
                    "pure_inference_avg_ms",
                    "pure_inf_throughput_fps",
                    "total_loop_s",
                    "avg_e2e_ms",
                    "e2e_throughput_fps",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "model": model_key,
                    "dataset": data_key,
                    "total_cases": num_cases,
                    "batch_size": batch_size,
                    "pure_inference_total_s": round(total_inference_time, 4),
                    "pure_inference_avg_ms": round(avg_model_per_case * 1000, 2),
                    "pure_inf_throughput_fps": round(model_throughput, 2),
                    "total_loop_s": round(total_loop_duration, 4),
                    "avg_e2e_ms": round(avg_e2e_per_case * 1000, 2),
                    "e2e_throughput_fps": round(e2e_throughput, 2),
                }
            )

        logger.info("=" * 80)
        logger.info(f"PERFORMANCE SUMMARY for {model_key.upper()} on {data_key}:")
        logger.info("--- Pure Model Inference (Computation Only) ---")
        logger.info(f"  Total Time:   {total_inference_time:.4f} s")
        logger.info(f"  Avg per Case: {avg_model_per_case * 1000:.2f} ms")
        logger.info(f"  Throughput:   {model_throughput:.2f} cases/sec")
        logger.info("--- Full Loop (Includes IO, Preprocessing, GPU Transfer) ---")
        logger.info(f"  Total Time:   {total_loop_duration:.4f} s")
        logger.info(f"  Avg per Case: {avg_e2e_per_case * 1000:.2f} ms")
        logger.info(f"  Throughput:   {e2e_throughput:.2f} cases/sec")
        logger.info(f"Summary saved to {output_path}")

    except Exception as e:
        logger.exception(f"FATAL ERROR while measuring times: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Measure inference times for an ML model per case and save to CSV."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model key to evaluate (e.g., 'ddb').",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Dataset key to evaluate on (e.g., 'b' or 'test_b').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ml_inference_times.csv",
        help="Path to output CSV file (default: ml_inference_times.csv).",
    )

    args = parser.parse_args()

    measure_inference_times(args.model, args.data, args.output)


if __name__ == "__main__":
    main()
