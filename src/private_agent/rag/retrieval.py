"""Hybrid vector and lexical document retrieval."""

import heapq
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rich.console import Console

from ..config import RAG_SIMILARITY_RESULTS

console = Console()


class HybridRAGRetriever:
    """Combine Chroma vector similarity and BM25 lexical retrieval."""

    def __init__(self, vectorstore: Chroma, documents: List[Document]):
        self.vectorstore = vectorstore
        self.documents = documents
        self.bm25 = None

        try:
            from rank_bm25 import BM25Okapi

            if documents:
                tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
                self.bm25 = BM25Okapi(tokenized_corpus)
        except ImportError:
            console.print(
                "[yellow][Warning] 'rank_bm25' package not found. Falling back "
                "to pure vector similarity search.[/yellow]"
            )

    def similarity_search(
        self, query: str, k: int = RAG_SIMILARITY_RESULTS
    ) -> List[Document]:
        vector_results = self.vectorstore.similarity_search(query, k=k)
        if not self.bm25 or not self.documents:
            return vector_results

        bm25_scores = self.bm25.get_scores(query.lower().split())
        top_scored_docs = heapq.nsmallest(
            k,
            enumerate(bm25_scores),
            key=lambda item: (-item[1], item[0]),
        )
        bm25_results = [
            self.documents[index]
            for index, score in top_scored_docs
            if score > 0
        ]

        seen = set()
        hybrid_results = []
        for document in vector_results + bm25_results:
            source = (
                document.metadata.get("source", ""),
                document.metadata.get("chunk", document.page_content[:30]),
            )
            if source not in seen:
                seen.add(source)
                hybrid_results.append(document)
            if len(hybrid_results) >= k:
                break
        return hybrid_results
