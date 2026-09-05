import argparse
import json
import os
from pathlib import Path

from google import genai
from google.genai import errors
from sentence_transformers import SentenceTransformer, util


parser = argparse.ArgumentParser(
    description="Run the Financial RAG baseline on a FinQA example."
)

parser.add_argument(
    "--index",
    type=int,
    default=0,
    help="Index of the FinQA example to test."
)

args = parser.parse_args()

# --------------------------------------------------
# 1. Load FinQA dataset
# --------------------------------------------------

dataset_path = Path("FinQA/dataset/train.json")

with open(dataset_path, "r", encoding="utf-8") as file:
    data = json.load(file)


# --------------------------------------------------
# 2. Select one FinQA example
# --------------------------------------------------

if args.index < 0 or args.index >= len(data):
    raise ValueError(
        f"Index must be between 0 and {len(data) - 1}."
    )

example = data[args.index]

question = example["qa"]["question"]
expected_answer = example["qa"]["answer"]

# Financial report text
paragraphs = example["pre_text"] + example["post_text"]

# Convert each table row into searchable text
table_rows = []

for row in example["table"]:
    row_text = " | ".join(row)
    table_rows.append(row_text)

# Search both paragraphs and table rows
documents = paragraphs + table_rows


# --------------------------------------------------
# 3. Load embedding model
# --------------------------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# 4. Create embeddings
# --------------------------------------------------

document_embeddings = embedding_model.encode(
    documents,
    convert_to_tensor=True
)

question_embedding = embedding_model.encode(
    question,
    convert_to_tensor=True
)


# --------------------------------------------------
# 5. Retrieve top 3 relevant documents
# --------------------------------------------------

scores = util.cos_sim(
    question_embedding,
    document_embeddings
)[0]

top_results = scores.topk(k=3)

retrieved_documents = []

for index in top_results.indices:
    retrieved_documents.append(
        documents[index.item()]
    )

context = "\n\n".join(retrieved_documents)


# --------------------------------------------------
# 6. Create Gemini client
# --------------------------------------------------

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY environment variable was not found."
    )

client = genai.Client(api_key=api_key)


# --------------------------------------------------
# 7. Build prompt
# --------------------------------------------------

prompt = f"""
You are a financial question-answering assistant.

Answer the question using only the financial evidence
provided below.

If a calculation is required, show the calculation briefly.

Do not use outside information.

QUESTION:
{question}

FINANCIAL EVIDENCE:
{context}

Provide:
1. A short explanation
2. The final numerical answer
"""


# --------------------------------------------------
# 8. Ask Gemini
# --------------------------------------------------

print("Asking Gemini...")

try:
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    generated_answer = response.text

except errors.ServerError as error:
    print("\nGemini is temporarily unavailable.")
    print("Please try running the baseline again later.")
    print(f"API error: {error}")
    raise SystemExit(1)


# --------------------------------------------------
# 9. Display results
# --------------------------------------------------

print("\n" + "=" * 70)
print("FINANCIAL RAG BASELINE")
print("=" * 70)

print("\nQUESTION:")
print(question)

print("\nRETRIEVED EVIDENCE:")

for rank, document in enumerate(
    retrieved_documents,
    start=1
):
    print(f"\n[{rank}] {document}")

print("\nGEMINI ANSWER:")
print(generated_answer)

print("\nFINQA EXPECTED ANSWER:")
print(expected_answer)

print("\n" + "=" * 70)