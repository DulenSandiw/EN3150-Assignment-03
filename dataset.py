'''reads shared CSV, loads the corresponding images, and groups them into batches(32 images for each) for a CNN.
 It does not train a model or create a new split.'''

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class EuroSATDataset(Dataset):
    """Load one split from the group's shared CSV."""

    def __init__(self, data_dir, csv_path, split, transform=None):
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")

        self.data_dir = Path(data_dir)
        self.transform = transform

        # Read the existing split; do not generate a new one.
        table = pd.read_csv(csv_path)
        self.samples = table[table["split"] == split].reset_index(drop=True)

        if self.samples.empty:
            raise ValueError(f"No images found for split: {split}")

    def __len__(self):
        """Number of images in this split."""
        return len(self.samples)

    def __getitem__(self, index):
        """Load one image and its numerical class label."""
        row = self.samples.iloc[index]
        image_path = self.data_dir / row["filepath"]

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        if image.size != (64, 64):
            raise ValueError(f"Unexpected image size at {image_path}")

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label_id"])
        return image, label


def create_loaders(data_dir, csv_path, batch_size=32):
    # Initially use only conversion to tensors.
    # We can add training augmentation and normalization later.
    transform = transforms.ToTensor()

    loaders = {}

    for split in ["train", "validation", "test"]:
        dataset = EuroSATDataset(
            data_dir=data_dir,
            csv_path=csv_path,
            split=split,
            transform=transform,
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    csv_path = Path(__file__).resolve().parent / "splits.csv"

    loaders = create_loaders(args.data_dir, csv_path)

    for split, loader in loaders.items():
        print(f"{split}: {len(loader.dataset)} images")

    images, labels = next(iter(loaders["train"]))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Pixel range:", images.min().item(), images.max().item())