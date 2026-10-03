"""Formatting helpers for retrieved-document citations."""


def format_retrieved_citations(documents: list) -> list[str]:
    citations = []
    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        source = str(metadata.get("source", "local knowledge base"))
        if metadata.get("page") is not None:
            source += f" (page {metadata['page']})"
        if metadata.get("chunk") is not None:
            source += f" (chunk {metadata['chunk']})"
        if source not in citations:
            citations.append(source)
    return citations
