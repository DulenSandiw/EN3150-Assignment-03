
import os
from pathlib import Path

from dotenv import load_dotenv
from dataset import create_loaders


def load_eurosat_data(batch_size=32):
    load_dotenv()

    data_dir = os.getenv("EUROSAT_ROOT")

    if not data_dir:
        raise ValueError(
            "EUROSAT_ROOT is missing from your .env file"
        )

    data_dir = Path(data_dir)

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}"
        )

    csv_path = Path(__file__).resolve().parents[1] / "splits.csv"

    loaders = create_loaders(
        data_dir=data_dir,
        csv_path=csv_path,
        batch_size=batch_size,
    )

    return (
        loaders["train"],
        loaders["validation"],
        loaders["test"],
    )


if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_eurosat_data()

    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))
    print("Testing images:", len(test_loader.dataset))

    images, labels = next(iter(train_loader))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
