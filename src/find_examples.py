import json
from pathlib import Path


dataset_path = Path("FinQA/dataset/train.json")

with open(dataset_path, "r", encoding="utf-8") as file:
    data = json.load(file)


print("SAMPLE FINQA QUESTIONS")
print("=" * 80)

for index, example in enumerate(data[:30]):
    qa = example["qa"]

    question = qa.get("question", "")
    answer = qa.get("answer", "")
    steps = qa.get("steps", [])

    print(f"\nINDEX: {index}")
    print(f"QUESTION: {question}")
    print(f"ANSWER: {answer}")
    print(f"REASONING STEPS: {len(steps)}")
    print("-" * 80)