from pathlib import Path
import argparse
import csv
import random
import time

import numpy as np
import torch
from torch import nn

from dataset import create_loaders
from models.model_a import StandardCNN


def run_epoch(model, loader, criterion, device, optimizer=None):
    # An optimizer is provided only during training.
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    # Validation does not need gradient calculations.
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad()

            # Forward pass and loss calculation.
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)

            # Weight each batch's mean loss by its image count.
            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_images += batch_size

    return (
        total_loss / total_images,
        total_correct / total_images,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--run-name", default="model_a_adam_01")
    parser.add_argument(
        "--optimizer",
        choices=["adam", "sgd", "sgd_momentum"],
        default="adam",
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    args = parser.parse_args()

    # Fix random seeds to make experiments more reproducible.
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "results" / args.run_name

    # Protect previous experiment results from being overwritten.
    output_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    loaders = create_loaders(
        data_dir=args.data_dir,
        csv_path=project_dir / "splits.csv",
        batch_size=args.batch_size,
    )
    model = StandardCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
        )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
        )

    print("Optimizer:", args.optimizer)
    print("Learning rate:", args.lr)

    if args.optimizer == "sgd_momentum":
        print("Momentum:", args.momentum)

    best_val_loss = float("inf")

    print("Device:", device)
    print("Results folder:", output_dir)

    history_path = output_dir / "history.csv"

    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "train_accuracy",
                "val_accuracy",
                "train_seconds",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            # Synchronize for meaningful GPU timing.
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            train_loss, train_accuracy = run_epoch(
                model, loaders["train"], criterion, device, optimizer
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            train_seconds = time.perf_counter() - start

            val_loss, val_accuracy = run_epoch(
                model, loaders["validation"], criterion, device
            )

            writer.writerow({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "train_seconds": train_seconds,
            })
            file.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy,
                        "seed": seed,
                        "learning_rate": args.lr,
                        "batch_size": args.batch_size,
                        "optimizer_name": args.optimizer,
                        "momentum":(
                            args.momentum
                            if args.optimizer == "sgd_momentum"
                            else 0.0
                        ),
                    },
                    output_dir / "best_model.pt",
                )

            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"Train acc: {train_accuracy:.2%} | "
                f"Val acc: {val_accuracy:.2%} | "
                f"Train time: {train_seconds:.1f}s"
            )

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Saved checkpoint: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()