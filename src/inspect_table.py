import json
from pathlib import Path


dataset_path = Path("FinQA/dataset/train.json")

with open(dataset_path, "r", encoding="utf-8") as file:
    data = json.load(file)


example = data[0]
qa = example["qa"]

print("QUESTION:")
print(qa["question"])

print("\nEXPECTED ANSWER:")
print(qa["answer"])

print("\nANNOTATED TEXT ROWS:")
print(qa.get("ann_text_rows"))

print("\nANNOTATED TABLE ROWS:")
print(qa.get("ann_table_rows"))

print("\nREASONING STEPS:")

for step in qa.get("steps", []):
    print(step)

print("\nPRE_TEXT WITH ROW NUMBERS:")

for index, paragraph in enumerate(example["pre_text"]):
    print(f"\n[{index}] {paragraph}")