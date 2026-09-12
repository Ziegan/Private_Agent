import os
import pathlib
import json
from typing import Optional, List
from rich.console import Console
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from .config import EMBEDDING_MODEL

console = Console()

class HybridRAGRetriever:
    """Combines Chroma vector similarity search with BM25 lexical keyword search for high-precision hybrid retrieval."""
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
            console.print("[yellow][Warning] 'rank_bm25' package not found. Falling back to pure vector similarity search.[/yellow]")

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        vector_results = self.vectorstore.similarity_search(query, k=k)
        
        if not self.bm25 or not self.documents:
            return vector_results

        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        scored_docs = list(enumerate(bm25_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_bm25_indices = [idx for idx, score in scored_docs[:k] if score > 0]
        bm25_results = [self.documents[idx] for idx in top_bm25_indices]

        seen = set()
        hybrid_results = []
        for doc in vector_results + bm25_results:
            source = doc.metadata.get("source", doc.page_content[:30])
            if source not in seen:
                seen.add(source)
                hybrid_results.append(doc)
            if len(hybrid_results) >= k:
                break
        return hybrid_results

def initialize_knowledge_base(docs_path: Optional[str]):
    """Initializes the Chroma vector store exclusively when an explicit, valid knowledge base folder path is provided."""
    if not docs_path or not str(docs_path).strip():
        return None

    path = pathlib.Path(docs_path).expanduser().resolve()
    if not path.is_dir():
        console.print(f"[yellow][Warning] Provided KB path '{path}' is not a valid directory. Skipping knowledge base initialization.[/yellow]")
        return None

    console.print(f"[cyan][Info] Initializing local embedding model '{EMBEDDING_MODEL}' via Ollama...[/cyan]")
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    except Exception as e:
        console.print(f"[red][Error] Could not bind embedding model: {e}[/red]")
        return None

    chroma_persist_dir = os.path.abspath("./.local_ai_chroma_db")
    state_file_path = os.path.join(chroma_persist_dir, ".index_state.json")
    os.makedirs(chroma_persist_dir, exist_ok=True)

    console.print(f"[bold green][Chroma DB][/bold green] Initializing local project vector store at: [cyan]{chroma_persist_dir}[/cyan]")

    db_exists = os.path.exists(chroma_persist_dir) and any(f for f in os.listdir(chroma_persist_dir) if f != ".index_state.json")

    index_state = {}
    if os.path.exists(state_file_path):
        try:
            index_state = json.loads(pathlib.Path(state_file_path).read_text(encoding="utf-8"))
        except Exception:
            index_state = {}

    new_docs = []
    all_parsed_docs = []
    updated_state = {}
    supported_exts = {".pdf", ".txt", ".md", ".py", ".json", ".csv", ".rs", ".js", ".ts", ".html"}

    for file_path in path.glob("**/*.*"):
        ext = file_path.suffix.lower()
        if ext not in supported_exts:
            continue
        
        try:
            stat = file_path.stat()
            mtime = stat.st_mtime
            file_key = str(file_path.resolve())
            updated_state[file_key] = mtime

            content = ""
            if ext == ".pdf":
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(str(file_path))
                    content = "".join([page.extract_text() or "" for page in reader.pages])
                except ImportError:
                    console.print(f"[yellow][Warning] pypdf not installed, skipping PDF: {file_path.name}[/yellow]")
                    continue
            else:
                content = file_path.read_text(encoding="utf-8", errors="ignore")

            if content.strip():
                doc = Document(page_content=content, metadata={"source": file_key})
                all_parsed_docs.append(doc)
                if file_key not in index_state or index_state[file_key] != mtime:
                    new_docs.append(doc)
        except Exception as e:
            console.print(f"[yellow][Warning] Failed parsing {file_path.name}: {e}[/yellow]")

    try:
        if db_exists:
            vectorstore = Chroma(persist_directory=chroma_persist_dir, embedding_function=embeddings)
            if new_docs:
                console.print(f"[green][Incremental Indexing][/green] Embedding {len(new_docs)} new or modified document(s)...")
                vectorstore.add_documents(new_docs)
            else:
                console.print(f"[green][Incremental Indexing][/green] No document changes detected. Skipping re-embedding.")
        else:
            if new_docs or all_parsed_docs:
                docs_to_embed = new_docs if new_docs else all_parsed_docs
                console.print(f"[green][Initial Indexing][/green] Creating vector store with {len(docs_to_embed)} document(s)...")
                vectorstore = Chroma.from_documents(docs_to_embed, embeddings, persist_directory=chroma_persist_dir)
            else:
                vectorstore = Chroma(persist_directory=chroma_persist_dir, embedding_function=embeddings)

        pathlib.Path(state_file_path).write_text(json.dumps(updated_state, indent=4), encoding="utf-8")
        
        # Return Hybrid RAG retriever wrapper
        return HybridRAGRetriever(vectorstore, all_parsed_docs)
    except Exception as e:
        console.print(f"[red][Error] Vector DB initialization error: {e}[/red]")
        return None
