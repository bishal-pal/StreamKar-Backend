def format_query(query: str) -> str:
    return f"Represent this sentence for searching: {query}"


def clean_answer(text: str) -> str:
    cleaned = text.strip()

    if "Answer:" in cleaned:
        cleaned = cleaned.split("Answer:", 1)[1].strip()

    if "Question:" in cleaned:
        cleaned = cleaned.split("Question:", 1)[0].strip()

    return cleaned
