from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Folder directly containing the 10 class folders.
DATA_DIR = Path(
    r"C:\my files\5th sem\pattern recognision\project\EuroSAT_RGB\EuroSAT_RGB"
)

# Save the CSV beside this Python script.
OUTPUT_FILE = Path(__file__).resolve().parent / "splits.csv"
SEED = 42

EXPECTED_CLASSES = [
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

# Check that we are pointing to the correct folder.
missing = [
    name for name in EXPECTED_CLASSES
    if not (DATA_DIR / name).is_dir()
]

if missing:
    raise ValueError(
        f"Class folders not found: {missing}\n"
        "DATA_DIR must directly contain the 10 class folders. "
        "Check whether they are inside another folder such as '2750'."
    )

# Build a table of image paths and labels.
records = []

for label_id, class_name in enumerate(EXPECTED_CLASSES):
    class_dir = DATA_DIR / class_name

    for image_path in sorted(class_dir.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in {
            ".jpg", ".jpeg", ".png"
        }:
            records.append({
                # Relative paths work on everyone's computer.
                "filepath": image_path.relative_to(DATA_DIR).as_posix(),
                "label": class_name,
                "label_id": label_id,
            })

data = pd.DataFrame(
    records, columns=["filepath", "label", "label_id"]
)

if len(data) != 27000:
    raise ValueError(
        f"Found {len(data)} images; expected 27,000. "
        "Check that the RGB dataset was fully extracted."
    )

# First split: 70% training and 30% remaining.
train, remaining = train_test_split(
    data,
    test_size=0.30,
    random_state=SEED,
    stratify=data["label"],
)

# Split the remaining 30% equally: 15% validation, 15% testing.
validation, test = train_test_split(
    remaining,
    test_size=0.50,
    random_state=SEED,
    stratify=remaining["label"],
)

splits = pd.concat(
    [
        train.assign(split="train"),
        validation.assign(split="validation"),
        test.assign(split="test"),
    ],
    ignore_index=True,
)

# Confirm that every image appears exactly once.
assert len(splits) == len(data)
assert splits["filepath"].is_unique
assert set(splits["filepath"]) == set(data["filepath"])

splits = splits.sort_values(
    ["split", "label", "filepath"]
).reset_index(drop=True)

splits.to_csv(OUTPUT_FILE, index=False)

# Display the number of images in each class and split.
summary = pd.crosstab(splits["label"], splits["split"])
summary = summary.reindex(columns=["train", "validation", "test"])

print(summary)
print("\nTotal images per split:")
print(summary.sum())
print(f"\nSaved to: {OUTPUT_FILE}")