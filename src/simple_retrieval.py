import json
import re
from pathlib import Path


dataset_path = Path("FinQA/dataset/train.json")

with open(dataset_path, "r", encoding="utf-8") as file:
    data = json.load(file)


# Use the first FinQA example
example = data[0]

question = example["qa"]["question"]
expected_answer = example["qa"]["answer"]

# Financial report paragraphs
paragraphs = example["pre_text"] + example["post_text"]


def tokenize(text):
    """Convert text into simple searchable words."""
    return set(re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()))


question_words = tokenize(question)

results = []

# Compare question words with each paragraph
for paragraph in paragraphs:
    paragraph_words = tokenize(paragraph)

    common_words = question_words.intersection(paragraph_words)

    score = len(common_words)

    results.append((score, paragraph))


# Highest score first
results.sort(reverse=True, key=lambda x: x[0])

# Retrieve top 3 paragraphs
top_results = results[:3]


print("FINANCIAL RAG - SIMPLE RETRIEVAL")
print("=" * 60)

print("\nQUESTION:")
print(question)

print("\nEXPECTED ANSWER:")
print(expected_answer)

print("\nTOP RETRIEVED PARAGRAPHS:")

for rank, (score, paragraph) in enumerate(top_results, start=1):
    print(f"\n--- Result {rank} | Score: {score} ---")
    print(paragraph)