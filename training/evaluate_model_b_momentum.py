
import torch
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from models.model_b import LightweightCNN
from training.data_adapter import load_eurosat_data


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load the shared EuroSAT dataset splits
    _, _, test_loader = load_eurosat_data(batch_size=32)

    # Create Model B
    model = LightweightCNN(num_classes=10).to(device)

    # Load the best checkpoint selected by validation accuracy
    checkpoint_path = Path("checkpoints/model_b_momentum_best.pth")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded checkpoint:", checkpoint_path)
    print("Best epoch:", checkpoint["epoch"])

    all_labels = []
    all_predictions = []

    # Evaluate without updating model weights
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.cpu().tolist())

    # Calculate test metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(
        all_labels, all_predictions,
        average="macro", zero_division=0
    )
    recall = recall_score(
        all_labels, all_predictions,
        average="macro", zero_division=0
    )
    cm = confusion_matrix(
        all_labels, all_predictions, labels=list(range(10))
    )

    print("\n===== MODEL B TEST RESULTS =====")
    print(f"Test Accuracy:  {accuracy * 100:.2f}%")
    print(f"Macro Precision: {precision * 100:.2f}%")
    print(f"Macro Recall:    {recall * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)

    
    # Save evaluation results for the report
    import json

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model": "Model B",
        "optimizer": "SGD with Momentum",
        "best_epoch": checkpoint["epoch"],
        "test_accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "confusion_matrix": cm.tolist(),
    }

    output_path = results_dir / "model_b_momentum_metrics.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {output_path}")



if __name__ == "__main__":
    main()
