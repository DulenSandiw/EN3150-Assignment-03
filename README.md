# EN3150 Assignment 03
Resource-Constrained CNNs for EuroSAT Image Classification

## Project Overview
We compare:
- Model A: a custom standard CNN.
- Model B: a custom CNN using depthwise separable convolutions,
  with at most 100,000 trainable parameters.
- Two pretrained lightweight image-classification models.

## Dataset
We use EuroSAT RGB: 27,000 RGB images, 10 classes, and
64 × 64 pixels per image.

Download the RGB dataset from:
https://github.com/phelber/EuroSAT

Extract it locally. The dataset root must directly contain:
AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial,
Pasture, PermanentCrop, Residential, River, and SeaLake.

Images are not included in this repository.
Each member must download and extract the same dataset.

## Shared Data Split
The committed splits.csv defines the split for all experiments:

| Split | Images | Percentage |
|---|---:|---:|
| Training | 18,900 | 70% |
| Validation | 4,050 | 15% |
| Testing | 4,050 | 15% |

The split is stratified by class and was generated using seed 42.

All members must use the committed splits.csv.
Do not generate a different split or overwrite it during experiments.

The CSV contains:
- filepath: image path relative to the dataset root.
- label: class name.
- label_id: numerical class label.
- split: train, validation, or test.

The split does not create separate image folders or move images.

## Files
- prepare_data.py: script used to generate the shared split.
- splits.csv: shared image paths, labels, and split assignments.
- .gitignore: excludes local environments and other unnecessary files.

## Local Dataset Paths
Each member's dataset root may be different.

For example, if the CSV contains:
Forest/Forest_1.jpg

The image is located at:
<your dataset root>/Forest/Forest_1.jpg

Use your own dataset root when loading images, while preserving
the relative paths and label IDs from the CSV.

prepare_data.py currently contains the original author's local path.
It only needs changing if deliberately reproducing the split;
normal training should read the existing CSV.

## Current Setup
The data-preparation script requires:

python -m pip install pandas scikit-learn

Training dependencies and run instructions will be added when
the group agrees on the framework and implements the models.

## Team Workflow
1. Pull the latest changes before starting work: git pull
2. Work on your assigned files.
3. Review changes with git status.
4. Add and commit relevant files with a descriptive message.
5. Push changes so other members can access them.

Coordinate before editing shared files.
Do not upload the dataset or your .venv folder.

## Evaluation Rules
- Use training data to learn model parameters.
- Use validation data for tuning and checkpoint selection.
- Reserve test data for final evaluation.
- Apply random data augmentation only to training images.
- Fit any dataset-specific normalization statistics on training data only.

## Current Progress
- [x] Dataset selected and downloaded.
- [x] Shared stratified split generated.
- [x] Preparation script and split uploaded.
- [ ] Image integrity and format checks.
- [ ] Shared dataset loader.
- [ ] Custom CNNs and optimizer experiments.
- [ ] Pretrained model experiments.
- [ ] Final evaluation and report.