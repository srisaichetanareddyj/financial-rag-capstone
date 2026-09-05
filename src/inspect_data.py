import json
from pathlib import Path


# Location of the FinQA training dataset
dataset_path = Path("FinQA/dataset/train.json")


# Open and read the JSON file
with open(dataset_path, "r", encoding="utf-8") as file:
    data = json.load(file)


# Take the first example
example = data[0]


# Get its question and correct answer
question = example["qa"]["question"]
answer = example["qa"]["answer"]


print("FINQA EXAMPLE")
print("=" * 50)

print("\nQuestion:")
print(question)

print("\nExpected Answer:")
print(answer)