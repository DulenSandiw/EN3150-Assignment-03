
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from models.pretrained import create_model
from pretrained_data import create_pretrained_loaders


CHECKPOINTS = {
    "mobilenet_v2": Path(
        "outputs/mobilenet_v2/fine_tuning/best_model.pth"
    ),
    "efficientnet_b0": Path(
        "outputs/efficientnet_b0/fine_tuning/best_model.pth"
    ),
}


def evaluate(model_name, checkpoint_path, loader, device):
    print(f"\nEvaluating {model_name}...")

    model = create_model(
        model_name,
        num_classes=10,
        pretrained=False,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

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
        all_true,
        all_pred,
        average="macro",
        zero_division=0,
    )


    class_names = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
    ]


    report = classification_report(
        all_true,
        all_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    matrix = confusion_matrix(
        all_true,
        all_pred,
        labels=list(range(10)),
    )

    total_parameters = sum(
        p.numel() for p in model.parameters()
    )

    checkpoint_mb = checkpoint_path.stat().st_size / (1024 ** 2)

    output_dir = Path("outputs") / model_name / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 9))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )
    display.plot(
        ax=ax,
        xticks_rotation=45,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )
    ax.set_title(f"{model_name} - Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(
        output_dir / "confusion_matrix.png",
        dpi=200,
    )
    plt.close(fig)

    metrics = {
        "model": model_name,
        "test_accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "total_parameters": total_parameters,
        "checkpoint_size_mb": checkpoint_mb,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall: {recall:.4f}")
    print(f"Macro F1-score: {f1:.4f}")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Checkpoint size: {checkpoint_mb:.2f} MB")
    print("Saved evaluation results to:", output_dir)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    loaders = create_pretrained_loaders(batch_size=32)
    test_loader = loaders["test"]

    for model_name, checkpoint_path in CHECKPOINTS.items():
        evaluate(
            model_name,
            checkpoint_path,
            test_loader,
            device,
        )


if __name__ == "__main__":
    main()
