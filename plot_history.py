from pathlib import Path
import argparse
import csv

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        default="model_a_adam_01",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    results_dir = project_dir / "results" / args.run_name
    history_path = results_dir / "history.csv"

    with history_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"No training records found in {history_path}")

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]

    # The CSV stores accuracy between 0 and 1.
    # Multiply by 100 to display percentages.
    train_acc = [100 * float(row["train_accuracy"]) for row in rows]
    val_acc = [100 * float(row["val_accuracy"]) for row in rows]

    # Match the checkpoint selection rule used during training.
    best_index = min(range(len(rows)), key=lambda i: val_loss[i])
    best_epoch = epochs[best_index]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, train_loss, label="Training", marker="o")
    axes[0].plot(epochs, val_loss, label="Validation", marker="o")
    axes[0].set_title("Cross-entropy loss")
    axes[0].set_ylabel("Loss")

    axes[1].plot(epochs, train_acc, label="Training", marker="o")
    axes[1].plot(epochs, val_acc, label="Validation", marker="o")
    axes[1].set_title("Classification accuracy")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 100)

    for ax in axes:
        ax.axvline(
            best_epoch,
            color="gray",
            linestyle="--",
            label=f"Best validation loss: epoch {best_epoch}",
        )
        ax.set_xlabel("Epoch")
        ax.set_xticks(epochs[::2])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"Model A — {args.run_name}")
    fig.tight_layout()

    output_path = results_dir / "learning_curves.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"Saved plot: {output_path}")
    print(f"Selected epoch: {best_epoch}")
    print(f"Validation accuracy at that epoch: {val_acc[best_index]:.2f}%")

    plt.show()


if __name__ == "__main__":
    main()