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
    if len(matches) > 0:
        print(f"CSV automatically at: {matches[0]}")
        return matches[0]

    raise FileNotFoundError(
        "Could not find results.csv.\n"
        "Run:\n"
        "    python utils/get_results_csv.py --data_dir experiments/ --save_dir results/"
    )


def plot_accuracy(df, save_path=None):
    sns.set(style="whitegrid", font_scale=1.3)

    df["layer_num"] = df["layer"].apply(lambda x: int(x.split("_")[1]))

    tasks = df["task"].unique()
    models = df["model"].unique()

    fig, axes = plt.subplots(len(models), len(tasks), figsize=(18, 10), sharey=True)

    if len(models) == 1:
        axes = [axes]
    if len(tasks) == 1:
        axes = [[ax] for ax in axes]

    for i, model in enumerate(models):
        model_df = df[df["model"] == model]

        for j, task in enumerate(tasks):
            task_df = model_df[model_df["task"] == task]
            ax = axes[i][j]

            sns.lineplot(
                data=task_df,
                x="layer_num",
                y="acc",
                hue="probe",
                marker="o",
                ax=ax
            )

            ax.set_title(f"{model} — {task}")
            ax.set_xlabel("Layer")
            ax.set_ylabel("Accuracy")
            ax.set_xticks(sorted(df["layer_num"].unique()))
            ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, help="Optional path to results.csv")
    parser.add_argument("--save_path", type=str, default="results/accuracy_by_layer.png")
    args = parser.parse_args()

    csv_path = args.csv if args.csv else find_results_csv()
    df = pd.read_csv(csv_path)

    plot_accuracy(df, args.save_path)


if __name__ == "__main__":
    main()
