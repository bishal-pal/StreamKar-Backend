from fastapi import FastAPI
from langchain_core.documents import Document

from .helpers import clean_answer, format_query
from .llm_setup import llm
from .schema import BulkFAQ, FAQ, Query
from .settings import BATCH_SIZE
from .vector_store import qdrant, retrieve_scored_nodes


app = FastAPI()

@app.post("/add_faq")
def add_faq(faq: FAQ):
    doc = Document(
        page_content=faq.question,
        metadata={"answer": faq.answer}
    )

    qdrant.add_documents([doc])

    return {"message": "FAQ added successfully"}

BATCH_SIZE = 100

@app.post("/bulk_add")
def bulk_add(data: BulkFAQ):
    docs = [
        Document(
            page_content=faq.question,
            metadata={"answer": faq.answer}
        )
        for faq in data.faqs
    ]

    total = len(docs)

    for i in range(0, total, BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        qdrant.add_documents(batch)

    return {
        "message": "Bulk FAQs added successfully",
        "total_added": total
    }


@app.post("/query")
def query_faq(q: Query):
    formatted_query = format_query(q.query)

    scored_results = retrieve_scored_nodes(formatted_query, k=3, score_threshold=0.7)
    docs = [doc for doc, _ in scored_results]

    if not docs:
        return {"answer": "No relevant answer found"}

    top_score = max(float(score) for _, score in scored_results)
    if top_score < 0.7:
        return {
            "answer": "I do not have context for this information",
        }

    context = "\n".join([
        f"Q: {d.page_content}\nA: {d.metadata['answer']}"
        for d in docs
    ])

    prompt = f"""
You are StreamKar Support Chatbot.

Objective: Give the most relevant answer to the user question using only the retrieved context.

Hard rules:
1. Use ONLY facts present inside <context>...</context>.
2. Do NOT use outside knowledge, assumptions, or guesses.
3. If context is insufficient or not directly relevant, reply exactly: I don't know.
4. Prefer the most relevant context item to the question. Do not merge unrelated items.
5. Answer only what the user asked. Do not add extra details.
6. Output a single short answer (1-2 sentences), no bullet points.
7. Do not repeat phrases or provide multiple alternative answers.
8. Do not include policy text, reasoning steps, or meta commentary.

<context>
{context}
</context>

User question:
{q.query}

Final answer:
"""

    response = llm.invoke(prompt)
    answer_text = clean_answer(response)

    return {
        "answer": answer_text or "No relevant answer found",
        "sources": [
            {
                "question": d.page_content,
                "answer": d.metadata["answer"],
                "score": round(float(score), 4),
            }
            for d, score in scored_results
        ]
    }
