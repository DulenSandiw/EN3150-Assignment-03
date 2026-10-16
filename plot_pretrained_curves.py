
import json
from pathlib import Path

import matplotlib.pyplot as plt


MODELS = {
    "mobilenet_v2": "MobileNetV2",
    "efficientnet_b0": "EfficientNet-B0",
}


for model_folder, model_title in MODELS.items():
    history_path = (
        Path("outputs")
        / model_folder
        / "fine_tuning"
        / "history.json"
    )

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = [record["epoch"] for record in history]

    train_accuracy = [
        record["train_accuracy"] * 100
        for record in history
    ]
    val_accuracy = [
        record["val_accuracy"] * 100
        for record in history
    ]

    train_loss = [
        record["train_loss"]
        for record in history
    ]
    val_loss = [
        record["val_loss"]
        for record in history
    ]

    output_dir = Path("outputs") / model_folder / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(
        epochs, train_accuracy, marker="o",
        label="Training accuracy"
    )
    axes[0].plot(
        epochs, val_accuracy, marker="s",
        label="Validation accuracy"
    )
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Fine-tuning epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xticks(epochs)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs, train_loss, marker="o",
        label="Training loss"
    )
    axes[1].plot(
        epochs, val_loss, marker="s",
        label="Validation loss"
    )
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Fine-tuning epoch")
    axes[1].set_ylabel("Cross-entropy loss")
    axes[1].set_xticks(epochs)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"{model_title} — Fine-tuning Learning Curves")
    fig.tight_layout()

    output_path = output_dir / "learning_curves.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")

print("\nAll learning curves generated successfully.")
