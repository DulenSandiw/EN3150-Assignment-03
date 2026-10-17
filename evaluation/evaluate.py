"""
EN3150 Assignment 03 - Member 4
evaluation/evaluate.py

A single, shared PyTorch test-set evaluation pipeline for all four models
(Model A, Model B, MobileNetV2, EfficientNet-B0). Every model is scored on
the exact same held-out test split from splits.csv, using the class
ordering defined there, so the resulting metrics are directly comparable.

Usage:
    python evaluation/evaluate.py --model all --data-dir /path/to/EuroSAT
    python evaluation/evaluate.py --model model_b --variant sgd \
        --data-dir /path/to/EuroSAT
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dataset import EuroSATDataset  # noqa: E402
from evaluation.registry import (  # noqa: E402
    CLASS_NAMES,
    MODEL_REGISTRY,
    build_model,
    resolve_checkpoint_path,
)

DEFAULT_RESULTS_DIR = PROJECT_DIR / "evaluation" / "results"


def get_test_loader(model_name, data_dir, batch_size=32):
    entry = MODEL_REGISTRY[model_name]
    dataset = EuroSATDataset(
        data_dir=data_dir,
        csv_path=PROJECT_DIR / "splits.csv",
        split="test",
        transform=entry["eval_transform"],
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def evaluate_model(model_name, checkpoint_path, data_dir, device, output_dir, batch_size=32):
    model = build_model(model_name, checkpoint_path=checkpoint_path, device=device)
    model.eval()

    loader = get_test_loader(model_name, data_dir, batch_size=batch_size)

    all_true = []
    all_pred = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            all_pred.extend(predictions.tolist())
            all_true.extend(labels.numpy().tolist())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    accuracy = accuracy_score(all_true, all_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true, all_pred, average="macro", zero_division=0,
    )
    report = classification_report(
        all_true, all_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(all_true, all_pred, labels=list(range(len(CLASS_NAMES))))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=CLASS_NAMES
    ).plot(ax=ax, xticks_rotation=45, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"{MODEL_REGISTRY[model_name]['display_name']} - Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    metrics = {
        "model": model_name,
        "display_name": MODEL_REGISTRY[model_name]["display_name"],
        "checkpoint": str(checkpoint_path),
        "num_test_images": int(len(all_true)),
        "test_accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(
        f"[{model_name}] test_accuracy={accuracy:.4f} "
        f"macro_precision={precision:.4f} macro_recall={recall:.4f} "
        f"macro_f1={f1:.4f}"
    )
    print(f"Saved metrics + confusion matrix to {output_dir}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Shared test-set evaluation for Model A, Model B, "
                     "MobileNetV2 and EfficientNet-B0."
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=list(MODEL_REGISTRY) + ["all"],
    )
    parser.add_argument(
        "--variant",
        default=None,
        choices=["adam", "sgd", "momentum"],
        help="Only used for --model model_b: which optimizer checkpoint to evaluate.",
    )
    parser.add_argument("--data-dir", required=True, help="Local EuroSAT RGB root folder.")
    parser.add_argument("--checkpoint", default=None, help="Override the default checkpoint path.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write metrics.json/confusion_matrix.png "
             "(default: evaluation/results/<model>/).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    models_to_run = list(MODEL_REGISTRY) if args.model == "all" else [args.model]

    for model_name in models_to_run:
        checkpoint_path = resolve_checkpoint_path(
            model_name, variant=args.variant, override=args.checkpoint,
        )

        if not checkpoint_path.exists():
            print(f"[{model_name}] SKIPPED - checkpoint not found: {checkpoint_path}")
            continue

        output_dir = (
            Path(args.output_dir) if args.output_dir
            else DEFAULT_RESULTS_DIR / model_name
        )

        evaluate_model(
            model_name, checkpoint_path, args.data_dir, device, output_dir, args.batch_size,
        )


if __name__ == "__main__":
    main()
