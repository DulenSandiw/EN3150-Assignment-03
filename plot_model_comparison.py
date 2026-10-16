
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


models = {
    "MobileNetV2": "mobilenet_v2",
    "EfficientNet-B0": "efficientnet_b0",
}

metrics = {}

for name, folder in models.items():
    path = Path("outputs") / folder / "evaluation" / "metrics.json"
    with open(path, "r", encoding="utf-8") as file:
        metrics[name] = json.load(file)

metric_keys = [
    "test_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
]

metric_labels = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
]

x = np.arange(len(metric_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

for index, (name, values) in enumerate(metrics.items()):
    scores = [
        values[key] * 100
        for key in metric_keys
    ]

    offset = (index - 0.5) * width

    bars = ax.bar(
        x + offset,
        scores,
        width,
        label=name,
    )

    ax.bar_label(
        bars,
        fmt="%.2f",
        padding=3,
        fontsize=9,
    )

ax.set_ylabel("Test performance (%)")
ax.set_title("MobileNetV2 vs EfficientNet-B0 — EuroSAT Test Performance")
ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 105)
ax.legend()
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()

output_path = Path("outputs/pretrained_model_comparison.png")
fig.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print("Saved:", output_path)
