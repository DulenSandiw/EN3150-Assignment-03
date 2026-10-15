
from datasets import load_dataset
import pandas as pd

# Load downloaded Hugging Face dataset
dataset = load_dataset("blanchon/EuroSAT_RGB")

# Read our group's existing split assignments
csv = pd.read_csv("splits.csv")

print("=== GROUP DATASET ===")
print(csv.head(10))

print("\n=== HUGGING FACE FILENAMES ===")
for split in dataset:
    print(f"\n{split}:")
    for row in dataset[split].select(
        range(min(5, len(dataset[split])))
    ):
        print(row["filename"])

# Compare filenames without extensions
group_names = {
    str(path).replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]
    for path in csv["filepath"]
}

hf_names = {
    str(name).replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]
    for split in dataset
    for name in dataset[split]["filename"]
}

matches = group_names & hf_names

print("\n=== COMPATIBILITY CHECK ===")
print("Group images:", len(group_names))
print("Hugging Face images:", len(hf_names))
print("Matching filename stems:", len(matches))
print("Unmatched group images:", len(group_names - hf_names))
