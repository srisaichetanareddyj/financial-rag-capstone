# Financial RAG Capstone

## Overview

This project implements a simple Retrieval-Augmented Generation (RAG) baseline for financial question answering using the FinQA dataset.

Given a financial question, the system retrieves relevant text passages and table rows from a financial report using semantic embeddings. The retrieved evidence is then provided to Google's Gemini model to generate an answer.

This repository contains the baseline implementation for a capstone project investigating whether a future agentic financial QA system can improve upon standard RAG, particularly for questions requiring numerical reasoning.

## Baseline Architecture

The baseline follows this pipeline:

FinQA question  
→ Financial report text and tables  
→ Sentence Transformer embeddings  
→ Semantic retrieval of top 3 pieces of evidence  
→ Gemini  
→ Generated answer  
→ Comparison with FinQA ground-truth answer

The baseline performs only one retrieval step. It does not independently decide to search again, select tools, or verify calculations.

## Dataset

This project uses the public FinQA dataset:

https://github.com/czyssrs/FinQA

FinQA contains financial-report text, tables, questions, answers, and numerical reasoning annotations.

The FinQA repository is not included directly in this repository.

## Requirements

- Python 3.10+
- Git
- Internet connection for initial model download and Gemini API access
- Gemini API key

## Setup

### 1. Clone this repository

```bash
git clone https://github.com/srisaichetanareddyj/financial-rag-capstone.git
cd financial-rag-capstone
```

### 2. Create a virtual environment

Windows:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation, temporarily allow scripts for the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download FinQA

From inside the `financial-rag-capstone` directory:

```bash
git clone https://github.com/czyssrs/FinQA.git
```

The expected location is:

```text
financial-rag-capstone/
└── FinQA/
    └── dataset/
        └── train.json
```

### 5. Configure Gemini API key

Create a Gemini API key using Google AI Studio.

Set the environment variable:

```text
GEMINI_API_KEY
```

On Windows PowerShell, for example:

```powershell
[Environment]::SetEnvironmentVariable(
    "GEMINI_API_KEY",
    "YOUR_API_KEY",
    "User"
)
```

Restart the terminal after setting the environment variable.

Do not commit API keys to GitHub.

## Run the Baseline

The baseline accepts a FinQA training-example index.

For the included demonstration:

```bash
python src/run_baseline.py --index 2
```

The program:

1. Loads FinQA example 2.
2. Converts report text and financial table rows into searchable documents.
3. Generates embeddings using `all-MiniLM-L6-v2`.
4. Retrieves the three most semantically similar pieces of evidence.
5. Sends the question and retrieved evidence to Gemini.
6. Displays the generated answer and FinQA ground-truth answer.

## Example Test Case

**Input question:**

> What was the total operating expenses in 2018 in millions?

**Retrieved evidence includes:**

```text
2018 | 4447 | $ 2.23 | $ 9896 | 23.6% ( 23.6 % )
```

The retrieved evidence indicates that aircraft fuel expense was $9,896 million and represented 23.6% of total operating expenses.

**Expected FinQA answer:**

```text
41932
```

**Baseline output:**

Gemini calculated approximately:

```text
$41,932 million
```

This matches the FinQA ground-truth answer after rounding.

## Known Limitations

The baseline performs a single retrieval operation and sends the retrieved evidence directly to the language model.

It currently:

- does not perform iterative retrieval,
- does not independently verify calculations,
- does not use a calculator or other reasoning tools,
- may fail when the initially retrieved evidence is incomplete,
- depends on availability of the Gemini API,
- may produce different wording across runs because an LLM is used.

A future agentic version could determine whether additional retrieval is required, select numerical reasoning tools, verify calculations, and revise its answer before returning a final response.

## Project Files

```text
src/run_baseline.py
```

Main end-to-end RAG baseline.

```text
src/embedding_retrieval.py
```

Demonstrates semantic embedding retrieval.

```text
src/simple_retrieval.py
```

Demonstrates the earlier keyword-based retrieval baseline.

```text
src/find_examples.py
```

Displays sample FinQA questions for testing.

```text
.env.example
```

Documents the required Gemini environment variable.

## Reproducibility Notes

The embedding model is downloaded automatically from Hugging Face during the first run.

Gemini requires an internet connection and a valid `GEMINI_API_KEY`.

Temporary Gemini service-capacity errors may occur. If the API returns a temporary `503 UNAVAILABLE` response, rerun the command later.