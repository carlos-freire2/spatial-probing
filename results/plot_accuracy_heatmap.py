import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import glob

def find_results_csv(default_path="results/results.csv"):
    if os.path.exists(default_path):
        print(f"CSV at default location: {default_path}")
        return default_path

    matches = glob.glob("**/results.csv", recursive=True)
    if matches:
        print(f"CSV automatically at: {matches[0]}")
        return matches[0]

    raise FileNotFoundError(
        "Could not find results.csv.\n"
        "Run:\n"
        "    python utils/get_results_csv.py --data_dir experiments/ --save_dir results/"
    )


def plot_accuracy_heatmap(df, save_path=None):
    sns.set_theme(style="white")
    plt.rcParams.update({"figure.dpi": 150})

    df["layer_num"] = df["layer"].apply(lambda x: int(x.split("_")[1]))

    models = ["CLIP", "DINOv3"]
    probes = ["linear", "MLP"]

    fig, axes = plt.subplots(len(models), len(probes), figsize=(14, 10))

    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    for i, model in enumerate(models):
        for j, probe in enumerate(probes):
            ax = axes[i, j]

            sub_df = df[(df["model"] == model) & (df["probe"] == probe)]
            pivot_df = sub_df.pivot_table(
                values="acc",
                index="layer_num",
                columns="task",
                aggfunc="mean"
            ).sort_index()

            sns.heatmap(
                pivot_df,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                vmin=0,
                vmax=1,
                cbar=(j == len(probes) - 1),
                square=True,
                ax=ax
            )

            ax.set_title(f"{model} — {probe.capitalize()} Probe", fontweight="bold")
            ax.set_xlabel("Task")
            ax.set_ylabel("Layer")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved heatmap to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, help="Optional path to results.csv")
    parser.add_argument("--save_path", type=str, default="results/accuracy_heatmap.png")
    args = parser.parse_args()

    csv_path = args.csv if args.csv else find_results_csv()
    df = pd.read_csv(csv_path)

    plot_accuracy_heatmap(df, args.save_path)


if __name__ == "__main__":
    main()
