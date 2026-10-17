from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from dataset import create_loaders
from models.model_a import StandardCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--run-name",
        default="model_a_sgd_momentum_lr001",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    run_dir = project_dir / "results" / args.run_name
    checkpoint_path = run_dir / "best_model.pt"
    csv_path = project_dir / "splits.csv"

    # Read class names in the same order as their numeric labels.
    table = pd.read_csv(csv_path)
    mapping = (
        table[["label_id", "label"]]
        .drop_duplicates()
        .sort_values("label_id")
    )
    class_ids = mapping["label_id"].astype(int).tolist()
    class_names = mapping["label"].tolist()

    if class_ids != list(range(10)):
        raise ValueError("Expected class IDs 0 through 9.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = StandardCNN(num_classes=10).to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loaders = create_loaders(
        data_dir=args.data_dir,
        csv_path=csv_path,
        batch_size=32,
    )
    test_loader = loaders["test"]

    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    targets = []
    predictions = []

    print("Device:", device)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Evaluating test images...")

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            total_loss += criterion(outputs, labels).item()

            targets.extend(labels.cpu().tolist())
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())

    targets = np.asarray(targets)
    predictions = np.asarray(predictions)

    report = classification_report(
        targets,
        predictions,
        labels=class_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        targets,
        predictions,
        labels=class_ids,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    accuracy = float((targets == predictions).mean())
    metrics = {
        "run_name": args.run_name,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "test_images": len(targets),
        "test_loss": total_loss / len(targets),
        "test_accuracy": accuracy,
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
    }

    output_dir = run_dir / "test_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    (output_dir / "classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )
    pd.DataFrame({
        name: report[name] for name in class_names
    }).T.to_csv(output_dir / "per_class_metrics.csv")

    cm = confusion_matrix(
        targets, predictions, labels=class_ids
    )
    pd.DataFrame(
        cm, index=class_names, columns=class_names
    ).to_csv(output_dir / "confusion_matrix.csv", index_label="true_label")

    fig, ax = plt.subplots(figsize=(12, 10))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )
    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        xticks_rotation=45,
        colorbar=False,
    )
    ax.set_title(f"Model A — test accuracy: {accuracy:.2%}")
    fig.tight_layout()
    fig.savefig(
        output_dir / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"\nTest loss: {metrics['test_loss']:.4f}")
    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("\n" + report_text)
    print("Saved evaluation to:", output_dir)


if __name__ == "__main__":
    main()