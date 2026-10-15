
from datasets import load_dataset

print("Loading EuroSAT RGB...")

dataset = load_dataset("blanchon/EuroSAT_RGB")

print("\nDataset information:")
print(dataset)

split_name = list(dataset.keys())[0]
sample = dataset[split_name][0]

print("\nSample fields:")
print(sample.keys())

print("\nFeature information:")
print(dataset[split_name].features)

print("\nFirst sample:")
for key, value in sample.items():
    if key == "image":
        print("image:", type(value), getattr(value, "size", None))
    else:
        print(key, ":", str(value)[:200])
