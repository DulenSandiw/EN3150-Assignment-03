
import argparse
import json
from pathlib import Path

import torch
from torch import nn

from models.pretrained import create_model
from pretrained_data import create_pretrained_loaders
from train_pretrained import run_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    args = parser.parse_args()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    model = create_model(
        "mobilenet_v2",
        num_classes=10,
        pretrained=False,
    )

    checkpoint_path = Path(
        "outputs/mobilenet_v2/head_training/best_model.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(
        "Loaded head-training checkpoint from epoch:",
        checkpoint["epoch"]
    )

    # Freeze the entire network first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Fine-tune the final two MobileNetV2 feature blocks.
    for block in model.features[-2:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    # Keep the classification head trainable.
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    model = model.to(device)

    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    print("Trainable parameters:", trainable)

    loaders = create_pretrained_loaders(
        batch_size=args.batch_size
    )

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
    )

    criterion = nn.CrossEntropyLoss()

    output_dir = Path("outputs/mobilenet_v2/fine_tuning")
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_accuracy = -1.0

    for epoch in range(args.epochs):
        print(f"\nFine-tuning epoch {epoch + 1}/{args.epochs}")

        train_metrics = run_epoch(
            model, loaders["train"], criterion,
            device, optimizer
        )

        val_metrics = run_epoch(
            model, loaders["validation"], criterion,
            device
        )

        record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(record)

        print(
            f"Train loss: {record['train_loss']:.4f} | "
            f"Train accuracy: {record['train_accuracy']:.4f}"
        )
        print(
            f"Val loss: {record['val_loss']:.4f} | "
            f"Val accuracy: {record['val_accuracy']:.4f}"
        )

        if record["val_accuracy"] > best_val_accuracy:
            best_val_accuracy = record["val_accuracy"]

            torch.save(
                {
                    "model_name": "mobilenet_v2",
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_accuracy": best_val_accuracy,
                },
                output_dir / "best_model.pth",
            )

            print("Fine-tuning checkpoint saved.")

        with open(
            output_dir / "history.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(history, file, indent=2)

    print("\nFine-tuning complete.")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print("Results saved in:", output_dir)


if __name__ == "__main__":
    main()
