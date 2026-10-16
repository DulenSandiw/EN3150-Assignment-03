
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# JSON files created during model evaluation
result_files = {
    "Adam": "results/model_b_adam_metrics.json",
    "Plain SGD": "results/model_b_sgd_metrics.json",
    "SGD with Momentum": "results/model_b_momentum_metrics.json",
}

# EuroSAT class names in alphabetical order
class_names = [
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

output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

for optimizer, file_path in result_files.items():

    with open(file_path, "r") as f:
        results = json.load(f)

    cm = np.array(results["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(11, 9))

    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Number of Images")

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title(f"Model B - {optimizer} Confusion Matrix")

    # Display counts inside each cell
    threshold = cm.max() / 2

    for i in range(10):
        for j in range(10):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()

    filename = optimizer.lower().replace(" ", "_") + "_confusion_matrix.png"
    output_path = output_dir / filename

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")

print("\nAll confusion matrix figures generated successfully!")
