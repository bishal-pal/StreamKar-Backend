# StreamKar FAQ Bot (FastAPI + Qdrant + Local LLM)

This project is a Retrieval-Augmented Generation (RAG) chatbot for StreamKar FAQs.

It lets you:
- Add single FAQ entries.
- Bulk-ingest many FAQ entries.
- Ask user questions and get context-grounded answers.
- Return source nodes and similarity scores for transparency.

The stack is fully local/self-hosted at runtime:
- API: FastAPI
- Vector DB: Qdrant
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Reasoning model: `Qwen/Qwen2.5-1.5B-Instruct`

## Project Structure

```
StreamKarApp/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schema.py
│   ├── settings.py
│   ├── helpers.py
│   ├── vector_store.py
│   └── llm_setup.py
├── sample_data.json
├── pyproject.toml
├── uv.lock
└── README.md
```

## How The App Works (End-to-End)

1. A user calls `/query` with a natural-language question.
2. The question is transformed to an embedding-friendly format (`format_query`).
3. Qdrant retrieves the top matching FAQ nodes with relevance scores.
4. If no nodes are found, the API returns a fallback message.
5. If the best score is below threshold (`0.7`), the API returns:
	 `I do not have context for this information`.
6. Retrieved nodes are converted into a prompt context block.
7. The local LLM generates an answer using strict context-only instructions.
8. The response is cleaned (`clean_answer`) and returned with `sources` + scores.

## File-by-File Explanation

### `app/main.py`

This is the FastAPI entrypoint and orchestration layer.

It defines 3 endpoints:

1. `POST /add_faq`
	 - Input: one FAQ item (`question`, `answer`).
	 - Stores a LangChain `Document` in Qdrant:
		 - `page_content = question`
		 - `metadata["answer"] = answer`

2. `POST /bulk_add`
	 - Input: list of FAQ items.
	 - Converts all FAQs to documents.
	 - Inserts in batches (`BATCH_SIZE` from settings).

3. `POST /query`
	 - Input: `query` string.
	 - Uses scored retrieval (`retrieve_scored_nodes`) from vector store.
	 - Enforces relevance gate (`top_score < 0.7` => reject with fallback).
	 - Builds LLM prompt with strict grounding rules.
	 - Returns:
		 - final answer
		 - source nodes with question, answer, and score

### `app/vector_store.py`

This file handles embeddings + Qdrant setup.

Responsibilities:
- Create embedding model (`all-MiniLM-L6-v2`).
- Detect embedding vector size dynamically.
- Create Qdrant client and collection (if missing).
- Initialize Qdrant client as `client`.
- Expose `qdrant` store object for writes (`add_documents`).
- Expose `retrieve_scored_nodes(query, k, score_threshold)` for scored retrieval.

Retrieval function used by API:
- `qdrant.similarity_search_with_relevance_scores(...)`
- Returns: list of `(Document, score)` tuples.

### `app/llm_setup.py`

This file initializes the local generation model and wraps it for LangChain.

Current model:
- `Qwen/Qwen2.5-1.5B-Instruct`

Device behavior:
- Uses `mps` on Apple Silicon when available.
- Falls back to CPU otherwise.

Generation settings include:
- `max_new_tokens = 150`
- `temperature = 0.1`
- `do_sample = False`
- `repetition_penalty = 1.1`

The final exported object is:
- `llm` (LangChain `HuggingFacePipeline`)

### `app/helpers.py`

Utility functions:

- `format_query(query: str) -> str`
	- Adds retrieval instruction prefix:
		`Represent this sentence for searching: ...`

- `clean_answer(text: str) -> str`
	- Removes prompt artifacts like `Answer:` and `Question:` if generated.

### `app/schema.py`

Pydantic request schemas used by FastAPI:

- `FAQ`
	- `question: str`
	- `answer: str`

- `BulkFAQ`
	- `faqs: List[FAQ]`

- `Query`
	- `query: str`

### `app/settings.py`

Central constants:
- `COLLECTION_NAME = "streamkar"`
- `BATCH_SIZE = 100`
- `DEVICE` auto-detected via `torch.backends.mps.is_available()`

## Data Model in Qdrant

Each FAQ is stored as one vectorized document:
- Vectorized text: question
- Metadata: answer

This design retrieves the most similar questions, then uses associated answers as context.

## API Contracts

### 1) Add Single FAQ

`POST /add_faq`

Request:
```json
{
	"question": "Where is the top-up option for beans?",
	"answer": "Go to your profile page and tap Top-up to purchase beans."
}
```

Response:
```json
{
	"message": "FAQ added successfully"
}
```

### 2) Bulk Add FAQs

`POST /bulk_add`

Request:
```json
{
	"faqs": [
		{
			"question": "Where is the top-up option for beans?",
			"answer": "Go to your profile page and tap Top-up to purchase beans."
		},
		{
			"question": "I received gifts, where can I exchange gems?",
			"answer": "Open Earnings from the Me page to exchange gems into beans."
		}
	]
}
```

Response:
```json
{
	"message": "Bulk FAQs added successfully",
	"total_added": 2
}
```

### 3) Query FAQ Bot

`POST /query`

Request:
```json
{
	"query": "What is StreamKar?"
}
```

Response (example):
```json
{
	"answer": "StreamKar is a live video and audio streaming social app where users can watch live videos, broadcast, and chat.",
	"sources": [
		{
			"question": "What is StreamKar?",
			"answer": "StreamKar is a live video and audio streaming social app where users can watch live videos, broadcast their life, video chat, and make new friends globally.",
			"score": 0.8969
		}
	]
}
```

Low-confidence response behavior:
- If no nodes: `No relevant answer found`
- If best score < 0.7: `I do not have context for this information`

## Setup and Run

## Prerequisites
- Python 3.12+
- Running Qdrant at `http://localhost:6333`
- `uv` installed

### Install dependencies
```bash
uv sync
```

### Run server
```bash
uv run uvicorn app.main:app --reload
```

### Open API docs
- Swagger UI: `http://127.0.0.1:8000/docs`

## Typical Development Flow

1. Start Qdrant.
2. Start API.
3. Ingest FAQ data (`/bulk_add` or `/add_faq`).
4. Query bot with `/query`.
5. Inspect `sources` and `score` for retrieval quality.

## Future Improvements

- Asynchronous Qdrant Client
- Add cross-encoders / Node post-processors
- Add auto-ingestion pipeline for dataset with cron-jobs and Redis queue
- Add multi-tenant chat support.
- Add Neo4J for cross-data relationships
- Add reranking for better top-context selection.
