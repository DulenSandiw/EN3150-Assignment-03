"""It checks for missing or unreadable images, incorrect dimensions or colour mode,
invalid CSV entries, and exact duplicate image content, including duplicates across splits."""

from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import json

import pandas as pd
from PIL import Image


CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation",
    "Highway", "Industrial", "Pasture", "PermanentCrop",
    "Residential", "River", "SeaLake",
]

EXPECTED_TOTALS = {
    "train": 18900,
    "validation": 4050,
    "test": 4050,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).resolve()
    csv_path = project_dir / "splits.csv"

    if not data_dir.is_dir():
        raise SystemExit(f"Dataset folder does not exist: {data_dir}")

    table = pd.read_csv(csv_path)
    required = {"filepath", "label", "label_id", "split"}

    if not required.issubset(table.columns):
        raise SystemExit(f"CSV must contain these columns: {required}")

    if table[list(required)].isna().any().any():
        raise SystemExit("CSV contains missing values. Check it before continuing.")

    issues = []
    hashes = defaultdict(list)
    sizes = Counter()
    modes = Counter()
    decoded_count = 0
    listed_paths = set()

    def record_issue(filepath, problem):
        issues.append({"filepath": str(filepath), "problem": problem})

    # Check CSV totals and repeated paths.
    actual_totals = {
        str(name): int(count)
        for name, count in table["split"].value_counts().items()
    }

    if actual_totals != EXPECTED_TOTALS:
        record_issue(
            "[CSV]",
            f"Unexpected split totals: {actual_totals}"
        )

    repeated = table.loc[
        table["filepath"].duplicated(keep=False), "filepath"
    ].unique()

    for filepath in repeated:
        record_issue(filepath, "Path appears more than once in CSV")

    # Inspect every image listed in the CSV.
    for number, row in enumerate(table.itertuples(index=False), start=1):
        relative_path = Path(str(row.filepath))
        image_path = (data_dir / relative_path).resolve()

        # Ensure CSV paths stay inside the chosen dataset folder.
        if relative_path.is_absolute() or not image_path.is_relative_to(data_dir):
            record_issue(row.filepath, "Invalid path outside dataset root")
            continue

        listed_paths.add(image_path)

        if row.split not in EXPECTED_TOTALS:
            record_issue(row.filepath, f"Invalid split: {row.split}")

        if row.label not in CLASSES:
            record_issue(row.filepath, f"Unknown class: {row.label}")
        elif row.label_id != CLASSES.index(row.label):
            record_issue(row.filepath, "Label ID does not match class name")

        if relative_path.parent.as_posix() != row.label:
            record_issue(row.filepath, "Class folder does not match CSV label")

        try:
            with Image.open(image_path) as image:
                # Fully decode the image to catch loading errors.
                image.load()
                decoded_count += 1

                sizes[str(image.size)] += 1
                modes[image.mode] += 1

                if image.size != (64, 64):
                    record_issue(
                        row.filepath,
                        f"Expected 64x64; found {image.size}"
                    )

                if image.mode != "RGB":
                    record_issue(
                        row.filepath,
                        f"Expected RGB; found {image.mode}"
                    )

                # Hash decoded RGB pixels, not JPEG file metadata.
                rgb = image.convert("RGB")
                digest = hashlib.sha256(
                    str(rgb.size).encode("utf-8") + rgb.tobytes()
                ).hexdigest()

                hashes[digest].append({
                    "filepath": str(row.filepath),
                    "label": str(row.label),
                    "split": str(row.split),
                })

        except Exception as error:
            record_issue(
                row.filepath,
                f"Cannot read image: {type(error).__name__}: {error}"
            )

        if number % 3000 == 0:
            print(f"Checked {number:,}/{len(table):,} CSV rows")

    # Find image files on disk that are not represented in the CSV.
    disk_paths = {
        path.resolve()
        for path in data_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }

    for path in sorted(disk_paths - listed_paths):
        record_issue(
            path.relative_to(data_dir).as_posix(),
            "Image exists on disk but is not listed in CSV"
        )

    # Group identical decoded images.
    duplicate_groups = []

    for entries in hashes.values():
        distinct_paths = {entry["filepath"] for entry in entries}

        if len(distinct_paths) > 1:
            duplicate_groups.append({
                "cross_split": len({e["split"] for e in entries}) > 1,
                "conflicting_labels": len({e["label"] for e in entries}) > 1,
                "images": entries,
            })

    cross_split_count = sum(
        group["cross_split"] for group in duplicate_groups
    )

    summary = {
        "csv_rows": len(table),
        "images_on_disk": len(disk_paths),
        "successfully_decoded_rows": decoded_count,
        "split_counts": actual_totals,
        "image_sizes": dict(sizes),
        "colour_modes": dict(modes),
        "issue_count": len(issues),
        "exact_duplicate_groups": len(duplicate_groups),
        "cross_split_duplicate_groups": cross_split_count,
        "status": (
            "PASS"
            if not issues and not duplicate_groups
            else "REVIEW NEEDED"
        ),
    }

    # Save diagnostic reports, leaving the original split unchanged.
    report_dir = project_dir / "data_checks"
    report_dir.mkdir(exist_ok=True)

    pd.DataFrame(
        issues, columns=["filepath", "problem"]
    ).to_csv(report_dir / "issues.csv", index=False)

    with open(report_dir / "duplicates.json", "w", encoding="utf-8") as file:
        json.dump(duplicate_groups, file, indent=2)

    with open(report_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nClass distribution:")
    print(
        pd.crosstab(table["label"], table["split"])
        .reindex(columns=["train", "validation", "test"], fill_value=0)
    )

    print("\nCheck summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nReports saved in: {report_dir}")


if __name__ == "__main__":
    main()