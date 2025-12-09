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


def plot_model_comparison(df, save_path=None):
    sns.set(style="whitegrid", font_scale=1.3)

    df["layer_num"] = df["layer"].apply(lambda x: int(x.split("_")[1]))

    tasks = df["task"].unique()
    models = ["CLIP", "DINOv3"]

    palette = {
        ("CLIP", "linear"): "blue",
        ("CLIP", "MLP"): "dodgerblue",
        ("DINOv3", "linear"): "darkorange",
        ("DINOv3", "MLP"): "orange",
    }

    fig, axes = plt.subplots(1, len(tasks), figsize=(20, 5), sharey=True)

    if len(tasks) == 1:
        axes = [axes]

    for i, task in enumerate(tasks):
        ax = axes[i]
        task_df = df[df["task"] == task]

        for model in models:
            for probe in ["linear", "MLP"]:
                sub = task_df[(task_df["model"] == model) & (task_df["probe"] == probe)]
                if sub.empty:
                    continue

                sns.lineplot(
                    data=sub,
                    x="layer_num",
                    y="acc",
                    marker="o",
                    ax=ax,
                    label=f"{model} — {probe}",
                    color=palette.get((model, probe))
                )

        ax.set_title(f"Task: {task}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(sorted(df["layer_num"].unique()))
        ax.set_ylim(0, 1)
        ax.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved model comparison plot to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, help="Optional path to results.csv")
    parser.add_argument("--save_path", type=str, default="results/model_comparison.png")
    args = parser.parse_args()

    csv_path = args.csv if args.csv else find_results_csv()
    df = pd.read_csv(csv_path)

    plot_model_comparison(df, args.save_path)


if __name__ == "__main__":
    main()
