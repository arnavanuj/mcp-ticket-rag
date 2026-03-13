"""Prompt templates placeholder for answer synthesis."""


def build_answer_prompt(question: str, context: str) -> str:
    """Build a minimal prompt stub for later LLM integration."""
    return f"Question: {question}\n\nContext:\n{context}"
