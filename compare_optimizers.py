from pathlib import Path
import csv

import matplotlib.pyplot as plt


def main():
    results_dir = Path(__file__).resolve().parent / "results"

    runs = {
        "model_a_adam_01": "Adam (lr=0.001)",
        "model_a_sgd_01": "SGD (lr=0.001)",
        "model_a_sgd_lr001": "SGD (lr=0.01)",
        "model_a_sgd_momentum_lr001": "SGD + momentum 0.9 (lr=0.01)",
    }

    # Read all histories before creating outputs.
    histories = {}
    for folder, label in runs.items():
        path = results_dir / folder / "history.csv"
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        if not rows:
            raise ValueError(f"No records in {path}")

        histories[label] = rows

    output_dir = results_dir / "optimizer_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    summaries = []

    for label, rows in histories.items():
        epochs = [int(row["epoch"]) for row in rows]

        metrics = [
            ("train_loss", axes[0, 0], 1),
            ("val_loss", axes[0, 1], 1),
            ("train_accuracy", axes[1, 0], 100),
            ("val_accuracy", axes[1, 1], 100),
        ]

        for key, ax, scale in metrics:
            values = [float(row[key]) * scale for row in rows]
            ax.plot(epochs, values, label=label, linewidth=2)

        # Use the same selection rule as the training script.
        best = min(rows, key=lambda row: float(row["val_loss"]))
        times = [float(row["train_seconds"]) for row in rows]

        summaries.append({
            "configuration": label,
            "selected_epoch": int(best["epoch"]),
            "validation_loss": float(best["val_loss"]),
            "validation_accuracy_percent": (
                100 * float(best["val_accuracy"])
            ),
            "mean_train_seconds_per_epoch": sum(times) / len(times),
            "total_train_seconds": sum(times),
        })

    titles = [
        "Training loss",
        "Validation loss",
        "Training accuracy",
        "Validation accuracy",
    ]

    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)" if "accuracy" in title else "Loss")
        ax.grid(True, alpha=0.3)
        if "accuracy" in title:
            ax.set_ylim(0, 100)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle("Model A: optimizer comparison — seed 42, 20 epochs")
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))

    fig.savefig(
        output_dir / "optimizer_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "optimizer_comparison.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    print("\nResults selected by lowest validation loss:")
    for row in summaries:
        print(
            f"{row['configuration']}: "
            f"epoch {row['selected_epoch']}, "
            f"val loss {row['validation_loss']:.4f}, "
            f"val accuracy {row['validation_accuracy_percent']:.2f}%"
        )

    print(f"\nSaved comparison plots and summary to: {output_dir}")


if __name__ == "__main__":
    main()