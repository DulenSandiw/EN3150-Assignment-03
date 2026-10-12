
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Checkpoints containing training histories
checkpoints = {
    "Adam": "checkpoints/model_b_adam_latest.pth",
    "Plain SGD": "checkpoints/model_b_sgd_latest.pth",
    "SGD with Momentum": "checkpoints/model_b_momentum_latest.pth",
}

# Folder for saving graphs
output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

histories = {}

for name, path in checkpoints.items():
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    histories[name] = checkpoint
    print(f"Loaded history: {name}")

# These are the metrics we want to compare
metrics = {
    "train_losses": ("Training Loss", "Loss", "training_loss.png"),
    "val_losses": ("Validation Loss", "Loss", "validation_loss.png"),
    "train_accuracies": (
        "Training Accuracy", "Accuracy (%)", "training_accuracy.png"
    ),
    "val_accuracies": (
        "Validation Accuracy", "Accuracy (%)", "validation_accuracy.png"
    ),
}

for key, (title, ylabel, filename) in metrics.items():
    plt.figure(figsize=(9, 5))

    for name, checkpoint in histories.items():
        values = checkpoint[key]
        epochs = range(1, len(values) + 1)

        plt.plot(
            epochs,
            values,
            marker="o",
            markersize=3,
            label=name,
        )

    plt.title(f"Model B - {title}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.xticks(range(1, 21))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")

print("\nAll comparison graphs generated successfully!")
