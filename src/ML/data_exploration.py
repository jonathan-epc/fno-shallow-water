import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats

from modules.data import HDF5Dataset
from nconfig import get_config


def load_dataset(config, file_path):
    """Load the dataset based on the given configuration."""
    return HDF5Dataset(
        file_path=file_path,
        input_vars=config.data.inputs,
        output_vars=config.data.outputs,
        numpoints_x=config.mesh.num_points_x,
        numpoints_y=config.mesh.num_points_y,
        channel_length=config.channel.length,
        channel_width=config.channel.width,
        normalize_input=False,
        normalize_output=False,
        device="cpu",
        preload=True,
    )


def discover_dataset_variables(file_path):
    """Discover all field and scalar variables in the HDF5 dataset."""
    with h5py.File(file_path, "r") as f:
        # Avoid the 'statistics' group if it exists
        keys = [k for k in f if k != "statistics"]
        if not keys:
            return [], []

        first_case = f[keys[0]]
        # Fields are groups/datasets within the case
        fields = [k for k in first_case if isinstance(first_case[k], h5py.Dataset)]
        # Scalars are attributes of the case
        scalars = list(first_case.attrs.keys())

        # Filter out common metadata that shouldn't be plotted as data distributions
        # (e.g., 'id', 'L', 'W' if they are just identifiers/dimensions)
        metadata_to_exclude = {"id", "L", "W", "BOTTOM", "direction", "subcritical"}
        fields = [f for f in fields if f not in metadata_to_exclude]
        scalars = [s for s in scalars if s not in metadata_to_exclude]

        return fields, scalars


def extract_data(dataset, config):
    """Extract input and output data from the dataset."""
    input_data = {key: [] for key in config.data.inputs}
    output_data = {key: [] for key in config.data.outputs}

    for data in dataset:
        inputs, outputs, _ = data
        field_inputs, scalar_inputs = inputs
        field_outputs, scalar_outputs = outputs

        for key, value in zip(
            config.data.inputs, field_inputs + scalar_inputs, strict=False
        ):
            input_data[key].append(value)
        for key, value in zip(
            config.data.outputs, field_outputs + scalar_outputs, strict=False
        ):
            output_data[key].append(value)

    # Convert lists to tensors
    input_data = {key: torch.stack(values) for key, values in input_data.items()}
    output_data = {key: torch.stack(values) for key, values in output_data.items()}
    return input_data, output_data


def compute_statistics(data):
    """Compute statistical measures for the given data."""
    numpy_data = data.cpu().numpy().flatten()
    return {
        "mean": np.mean(numpy_data),
        "std": np.std(numpy_data),
        "min": np.min(numpy_data),
        "max": np.max(numpy_data),
        "skewness": stats.skew(numpy_data),
        "kurtosis": stats.kurtosis(numpy_data),
        "percentile_25": np.percentile(numpy_data, 25),
        "median": np.median(numpy_data),
        "percentile_75": np.percentile(numpy_data, 75),
    }


def check_missing_values(input_data, output_data):
    """Check for missing or NaN values in input and output data."""
    return {
        key: (data.isnan().sum().item(), data.numel())
        for key, data in {**input_data, **output_data}.items()
    }


def create_summary_plot(data_dict, title_prefix, output_file, plot_type="histogram"):
    """Create and save a summary plot for all variables."""
    num_vars = len(data_dict)
    cols = 3
    rows = (num_vars + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for i, (var, data) in enumerate(data_dict.items()):
        numpy_data = data.cpu().numpy().flatten()
        variable_stats = compute_statistics(data)

        ax = axes[i]
        if plot_type == "histogram":
            sns.histplot(numpy_data, bins=50, kde=True, ax=ax)
        elif plot_type == "violin":
            sns.violinplot(data=numpy_data, orient="h", ax=ax)

        ax.set_title(f"{title_prefix} {var}")
        ax.set_xlabel(var)
        ax.set_ylabel("Density")

        # Add statistics to the plot
        stat_text = (
            f"Mean: {variable_stats['mean']:.2f}\n"
            f"Std: {variable_stats['std']:.2f}\n"
            f"Min: {variable_stats['min']:.2f}\n"
            f"Max: {variable_stats['max']:.2f}\n"
            f"Skew: {variable_stats['skewness']:.2f}\n"
            f"Kurt: {variable_stats['kurtosis']:.2f}"
        )
        ax.text(
            0.95,
            0.95,
            stat_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"),
        )

    # Hide unused subplots
    for ax in axes[num_vars:]:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def save_individual_plots(data_dict, title_prefix, output_dir, plot_type="histogram"):
    """Save a distribution plot for each variable in a separate image."""
    os.makedirs(output_dir, exist_ok=True)

    for var, data in data_dict.items():
        numpy_data = data.cpu().numpy().flatten()
        variable_stats = compute_statistics(data)

        plt.figure(figsize=(10, 6))
        if plot_type == "histogram":
            sns.histplot(numpy_data, bins=50, kde=True)
        elif plot_type == "violin":
            sns.violinplot(data=numpy_data, orient="h")

        plt.title(f"{title_prefix} {var}")
        plt.xlabel(var)
        plt.ylabel("Density")

        # Add statistics to the plot
        stat_text = (
            f"Mean: {variable_stats['mean']:.2f}\n"
            f"Std: {variable_stats['std']:.2f}\n"
            f"Min: {variable_stats['min']:.2f}\n"
            f"Max: {variable_stats['max']:.2f}\n"
            f"Skew: {variable_stats['skewness']:.2f}\n"
            f"Kurt: {variable_stats['kurtosis']:.2f}"
        )
        plt.text(
            0.95,
            0.95,
            stat_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"),
        )

        output_file = os.path.join(output_dir, f"{var.replace('*', '_adim')}_dist.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Explore HDF5 hydraulic datasets.")
    parser.add_argument(
        "--config", type=str, default="config.yml", help="Path to config file"
    )
    parser.add_argument(
        "--data", type=str, default="data/barsa.hdf5", help="Path to HDF5 data file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="img/exploration",
        help="Root output directory",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Plot all variables found in the dataset, ignoring config lists",
    )
    args = parser.parse_args()

    config = get_config(args.config)

    if args.all:
        print(f"Discovering all variables in {args.data}...")
        discovered_fields, discovered_scalars = discover_dataset_variables(args.data)
        print(f"Found fields: {discovered_fields}")
        print(f"Found scalars: {discovered_scalars}")

        # Override config inputs/outputs for the rest of the script
        # We group them for the logic below
        config.data.inputs = discovered_scalars
        config.data.outputs = discovered_fields
        title_in = "Scalar Parameter:"
        title_out = "Field Variable:"
    else:
        title_in = "Input Parameter:"
        title_out = "Output Variable:"

    dataset = load_dataset(config, args.data)
    print(f"Loaded dataset: {args.data}")
    print(dataset)

    # Extract data
    input_data, output_data = extract_data(dataset, config)

    # Set plot style
    plt.style.use("default")

    # Compute statistics
    stats_dict = {}
    for key, data in input_data.items():
        stats_dict[f"input_{key}"] = compute_statistics(data)
    for key, data in output_data.items():
        stats_dict[f"output_{key}"] = compute_statistics(data)

    # Check for missing values
    missing_values = check_missing_values(input_data, output_data)
    print("\nMissing Values:")
    for var, (missing, total) in missing_values.items():
        print(f"{var}: {missing}/{total} missing ({100 * missing / total:.2f}%)")

    # Save summary plots
    create_summary_plot(
        input_data,
        title_in,
        os.path.join(args.output_dir, "adim_input_summary.png"),
        plot_type="histogram",
    )
    create_summary_plot(
        output_data,
        title_out,
        os.path.join(args.output_dir, "adim_output_summary.png"),
        plot_type="histogram",
    )

    # Save individual plots
    print(f"\nSaving individual plots to {args.output_dir}/individual/...")
    save_individual_plots(
        input_data,
        title_in,
        os.path.join(args.output_dir, "individual", "inputs"),
    )
    save_individual_plots(
        output_data,
        title_out,
        os.path.join(args.output_dir, "individual", "outputs"),
    )

    # Convert statistics to a DataFrame and print
    stats_df = pd.DataFrame(stats_dict).T
    print("\nStatistics Table:")
    print(stats_df)


if __name__ == "__main__":
    main()
