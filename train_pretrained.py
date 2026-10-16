
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from models.pretrained import create_model
from pretrained_data import create_pretrained_loaders


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["mobilenet_v2", "efficientnet_b0"],
        default="mobilenet_v2",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    loaders = create_pretrained_loaders(
        batch_size=args.batch_size
    )

    model = create_model(
        args.model,
        num_classes=10,
        pretrained=True,
    ).to(device)

    # Freeze backbone; train the classification head.
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    criterion = nn.CrossEntropyLoss()

    output_dir = Path("outputs") / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_accuracy = -1.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

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
                    "model_name": args.model,
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_accuracy": best_val_accuracy,
                },
                output_dir / "best_model.pth",
            )
            print("Best checkpoint saved.")

        with open(
            output_dir / "history.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(history, file, indent=2)

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print("Results saved in:", output_dir)


if __name__ == "__main__":
    main()
