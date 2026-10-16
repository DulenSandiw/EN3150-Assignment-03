
from pathlib import Path
from datasets import load_dataset
import pandas as pd

# Dataset destination outside the Git repository
OUTPUT_DIR = Path(
    r"D:\ACA\Sem 5\PR\Assignment 3\data\EuroSAT"
)

# Read the group's original CSV
csv = pd.read_csv("splits.csv")

# Map filename stems to expected relative paths
expected_paths = {
    Path(row.filepath).stem: row.filepath
    for row in csv.itertuples()
}

print("Loading EuroSAT from Hugging Face...")
dataset = load_dataset("blanchon/EuroSAT_RGB")

exported = set()

for split_name, split_data in dataset.items():
    print(f"Processing Hugging Face split: {split_name}")

    for row in split_data:
        filename_stem = Path(row["filename"]).stem

        if filename_stem not in expected_paths:
            raise ValueError(
                f"Unexpected image: {filename_stem}"
            )

        relative_path = Path(
            expected_paths[filename_stem]
        )

        output_path = OUTPUT_DIR / relative_path
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Convert image into RGB JPEG format
        image = row["image"].convert("RGB")
        image.save(
            output_path,
            format="JPEG",
            quality=95
        )

        exported.add(filename_stem)

print("\nExport completed!")
print("Exported images:", len(exported))
print("Expected images:", len(expected_paths))

if exported != set(expected_paths):
    raise RuntimeError(
        "Exported dataset does not match splits.csv"
    )

print("All expected filenames exported.")
