import json
from pathlib import Path

from sentence_transformers import SentenceTransformer, util


# -----------------------------
# 1. Load FinQA dataset
# -----------------------------

dataset_path = Path("FinQA/dataset/train.json")

with open(dataset_path, "r", encoding="utf-8") as file:
    data = json.load(file)


# -----------------------------
# 2. Use first FinQA example
# -----------------------------

example = data[0]

question = example["qa"]["question"]
expected_answer = example["qa"]["answer"]

paragraphs = example["pre_text"] + example["post_text"]


# -----------------------------
# 3. Load embedding model
# -----------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 4. Create embeddings
# -----------------------------

paragraph_embeddings = model.encode(
    paragraphs,
    convert_to_tensor=True
)

question_embedding = model.encode(
    question,
    convert_to_tensor=True
)


# -----------------------------
# 5. Compare question to paragraphs
# -----------------------------

scores = util.cos_sim(
    question_embedding,
    paragraph_embeddings
)[0]


# -----------------------------
# 6. Retrieve top 3 paragraphs
# -----------------------------

top_results = scores.topk(k=3)


# -----------------------------
# 7. Display results
# -----------------------------

print("\nFINANCIAL RAG - EMBEDDING RETRIEVAL")
print("=" * 60)

print("\nQUESTION:")
print(question)

print("\nEXPECTED ANSWER:")
print(expected_answer)

print("\nTOP RETRIEVED PARAGRAPHS:")

for rank, index in enumerate(top_results.indices, start=1):
    index = index.item()
    score = scores[index].item()

    print(f"\n--- Result {rank} | Similarity: {score:.4f} ---")
    print(paragraphs[index])