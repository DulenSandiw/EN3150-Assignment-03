# EN3150 Assignment 03
Resource-Constrained CNNs for EuroSAT Image Classification

## Project Overview
This project benchmarks resource-constrained deep learning models on the EuroSAT land use and land cover classification dataset. We evaluate and compare:
- **Model A**: A custom standard CNN.
- **Model B**: A custom CNN using depthwise separable convolutions, constrained to at most 100,000 trainable parameters.
- **Pretrained Lightweight Models**: Pretrained **MobileNetV2** and **EfficientNet-B0** adapted and fine-tuned for EuroSAT.

---

## Dataset
We use the **EuroSAT RGB** dataset consisting of 27,000 RGB images across 10 classes at 64 × 64 resolution.

- Download the RGB dataset from: https://github.com/phelber/EuroSAT
- Extract the archive locally. The dataset root must directly contain the 10 class directories:
  `AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, and `SeaLake`.

> **Note**: Raw images are not committed to this repository. Each group member must extract the dataset locally.

---

## Shared Data Split
The committed `splits.csv` defines the exact split for all team experiments (generated with seed 42):

| Split | Images | Percentage |
|---|---:|---:|
| Training | 18,900 | 70% |
| Validation | 4,050 | 15% |
| Testing | 4,050 | 15% |

All members must use the committed `splits.csv`. Do not regenerate or overwrite the split file.

The CSV specifies:
- `filepath`: Image path relative to the dataset root (e.g., `Forest/Forest_1.jpg`).
- `label`: Class name.
- `label_id`: Numerical class index (0–9).
- `split`: Partition assignment (`train`, `validation`, or `test`).

---

## Local Dataset Configuration
Because dataset paths differ across machines, keep `splits.csv` relative.

When configuring scripts:
- In `dataset.py`, pass `--data-dir <path_to_eurosat_root>` via command line.
- In `pretrained_data.py`, set `DATA_DIR` to your local EuroSAT root directory.
- In `check_data.py`, provide `--data-dir <path_to_eurosat_root>` to verify your local dataset folder.

---

## Repository Structure

```text
├── data_checks/
│   ├── duplicates.json          # Audit report of duplicate image hashes
│   ├── issues.csv               # Recorded dataset issues (0 found)
│   └── summary.json             # Verification summary (Status: PASS)
├── models/
│   ├── __init__.py
│   └── pretrained.py            # MobileNetV2 & EfficientNet-B0 definitions & helpers
├── outputs/
│   ├── efficientnet_b0/
│   │   ├── best_model.pth       # Top checkpoint
│   │   ├── evaluation/          # Confusion matrix, metrics.json, learning curves
│   │   ├── fine_tuning/         # 5-epoch fine-tuning history & checkpoints
│   │   └── head_training/       # Frozen-backbone classifier training output
│   ├── mobilenet_v2/
│   │   ├── best_model.pth       # Top checkpoint
│   │   ├── evaluation/          # Confusion matrix, metrics.json, learning curves
│   │   ├── fine_tuning/         # 5-epoch fine-tuning history & checkpoints
│   │   └── head_training/       # Frozen-backbone classifier training output
│   └── pretrained_model_comparison.png  # Comparison chart (accuracy, precision, recall, F1)
├── check_data.py                # Dataset audit and integrity verification script
├── dataset.py                   # PyTorch EuroSATDataset and DataLoader factory
├── evaluate_pretrained.py       # Test set evaluation and confusion matrix generator
├── finetune_efficientnet.py     # EfficientNet-B0 2-stage fine-tuning script
├── finetune_pretrained.py       # MobileNetV2 2-stage fine-tuning script
├── plot_model_comparison.py     # Generates benchmark comparison plot across models
├── plot_pretrained_curves.py    # Generates fine-tuning learning curves (loss & accuracy)
├── prepare_data.py              # Script that generated the shared stratified split
├── pretrained_data.py           # Pretrained-specific transforms, augmentations, and loaders
├── splits.csv                   # Master split metadata (27,000 entries)
├── train_pretrained.py          # Stage 1: Classification head training script
└── README.md
```

---

## Environment & Dependencies

Install required dependencies using pip or activate your project virtual environment:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # or standard torch
python -m pip install pandas scikit-learn matplotlib numpy pillow tqdm
```

---

## Workflow & Reproduction Instructions

### 1. Verify Dataset Integrity
Verify all 27,000 images on disk against `splits.csv` (dimensions, RGB mode, unreadable files, duplicates):
```bash
python check_data.py --data-dir "path/to/EuroSAT"
```
*Current result: 27,000 / 27,000 images verified, 0 unreadable, 0 cross-split duplicates (`PASS`).*

### 2. Test Dataset Loader
Verify that batches and tensors load correctly:
```bash
python dataset.py --data-dir "path/to/EuroSAT"
```

### 3. Train Pretrained Models (Two-Stage Transfer Learning)
Pretrained experiments follow a robust two-stage training scheme:
1. **Stage 1 (Head Training)**: Feature extractor backbone is frozen; only the new 10-class linear classification head is trained using Adam ($lr = 10^{-3}$).
2. **Stage 2 (Fine-Tuning)**: The last two convolutional/inverted residual feature blocks are unfrozen along with the classifier head, and fine-tuned with a lower learning rate ($lr = 10^{-4}$, Adam, seed 42) for 5 epochs.

To train/fine-tune:
```bash
# Stage 1: Classifier head training
python train_pretrained.py --model mobilenet_v2 --epochs 5 --lr 0.001
python train_pretrained.py --model efficientnet_b0 --epochs 5 --lr 0.001

# Stage 2: Feature fine-tuning (last 2 blocks)
python finetune_pretrained.py --epochs 5 --lr 0.0001
python finetune_efficientnet.py --epochs 5 --lr 0.0001
```

### 4. Evaluate on Test Split
Compute test metrics (Accuracy, Macro Precision, Macro Recall, Macro F1, Checkpoint size, Parameter count) and generate confusion matrices:
```bash
python evaluate_pretrained.py
```

### 5. Generate Visualizations
Generate fine-tuning learning curves and model comparison charts:
```bash
python plot_pretrained_curves.py
python plot_model_comparison.py
```

---

## Experimental Results (Pretrained Models)

Evaluation was performed on the held-out test split (4,050 images) using the best fine-tuned checkpoints:

| Model | Total Parameters | Trainable Parameters (Fine-tuning) | Checkpoint Size | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| **MobileNetV2** | **2,236,682** (~2.24M) | 425,098 | **8.77 MB** | **91.41%** | **91.26%** | **91.05%** | **91.10%** |
| **EfficientNet-B0** | 4,020,358 (~4.02M) | 832,234 | 15.62 MB | 89.88% | 89.57% | 89.65% | 89.57% |

### Key Observations:
- **Efficiency vs. Performance**: MobileNetV2 achieved superior classification accuracy (+1.53%) and macro F1-score (+1.53%) compared to EfficientNet-B0, while requiring **44.4% fewer parameters** (2.24M vs. 4.02M) and nearly half the storage size (8.77 MB vs. 15.62 MB).
- **Class Highlights**: Both models achieved very high accuracy on distinctive textures such as `Residential` (F1: 96.2% on MobileNetV2), `SeaLake` (F1: 94.5% on MobileNetV2, 95.2% on EfficientNet-B0), and `Forest` (F1: 94.4%). Inter-class confusion occurred primarily between structurally similar classes such as `Highway` vs. `River` and `PermanentCrop` vs. `HerbaceousVegetation`/`AnnualCrop`.
- **Learning Convergence**: Both models demonstrated smooth convergence during fine-tuning with validation accuracy improving steadily without severe overfitting over 5 epochs.

---

## Evaluation Rules
- Use training data to learn model parameters.
- Use validation data for tuning, early stopping, and checkpoint selection.
- Reserve test data strictly for final evaluation.
- Apply random data augmentations (flips) only to training images.
- Fit normalization statistics strictly on training data / ImageNet canonical statistics for pretrained models.

---

## Team Workflow
1. Pull the latest changes before starting work: `git pull`
2. Work on your assigned files / branch.
3. Review changes with `git status` and `git diff`.
4. Commit relevant files with clear descriptive messages.
5. Push changes and open PRs for review.
6. Coordinate before modifying shared files (`splits.csv`, `dataset.py`).
7. **Do not commit raw dataset folders, local checkpoints (.pth files over Git LFS limit), or `.venv` environments.**

---

## Current Progress

- [x] **Dataset Preparation**: EuroSAT RGB dataset acquired and extracted.
- [x] **Shared Split Generation**: 70/15/15 stratified train/val/test split generated with seed 42 (`splits.csv`).
- [x] **Data Integrity Verification**: Validated 27,000 images for 64×64 size, RGB format, CSV alignment, and zero duplicate leakages (`check_data.py`, `PASS`).
- [x] **Shared Dataset Loader**: Built PyTorch `EuroSATDataset` and `DataLoader` pipeline (`dataset.py`, `pretrained_data.py`).
- [x] **Pretrained Model Implementations**: Configured MobileNetV2 and EfficientNet-B0 architectures for 10-class EuroSAT classification (`models/pretrained.py`).
- [x] **Pretrained Transfer Learning Pipeline**:
  - [x] Stage 1: Frozen-backbone classifier head training (`train_pretrained.py`).
  - [x] Stage 2: Convolutional block fine-tuning (`finetune_pretrained.py`, `finetune_efficientnet.py`).
- [x] **Pretrained Evaluation & Visualizations**:
  - [x] Test set evaluation with full classification reports and confusion matrices (`evaluate_pretrained.py`).
  - [x] Fine-tuning learning curves (`plot_pretrained_curves.py`).
  - [x] Pretrained model comparative performance chart (`plot_model_comparison.py`).
- [ ] **Custom CNN Model A**: Standard custom CNN architecture implementation and training.
- [ ] **Custom CNN Model B**: Lightweight custom CNN using depthwise separable convolutions (param count $\le$ 100,000).
- [ ] **Optimization Experiments**: Optimizer benchmarking (e.g., Adam vs. SGD / learning rate schedules) on custom CNNs.
- [ ] **Final Consolidated Evaluation & Report**: Comparative analysis across Custom Model A, Custom Model B, MobileNetV2, and EfficientNet-B0.