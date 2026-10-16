
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import EuroSATDataset


DATA_DIR = Path(
    r"D:\ACA\Sem 5\PR\Assignment 3\data\EuroSAT"
)
CSV_PATH = Path("splits.csv")

# Normalization used by ImageNet pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def create_pretrained_loaders(batch_size=32):
    train_dataset = EuroSATDataset(
        DATA_DIR, CSV_PATH, "train",
        transform=train_transform
    )

    val_dataset = EuroSATDataset(
        DATA_DIR, CSV_PATH, "validation",
        transform=eval_transform
    )

    test_dataset = EuroSATDataset(
        DATA_DIR, CSV_PATH, "test",
        transform=eval_transform
    )

    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        ),
        "validation": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        ),
    }


if __name__ == "__main__":
    loaders = create_pretrained_loaders()

    for name, loader in loaders.items():
        print(f"{name}: {len(loader.dataset)} images")

    images, labels = next(iter(loaders["train"]))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Normalized pixel range:",
          images.min().item(), images.max().item())
