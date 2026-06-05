from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            self._client = chromadb.EphemeralClient()
            self._collection = self._client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        return {
            "id": doc.id,
            "content": doc.content,
            "embedding": self._embedding_fn(doc.content),
            "metadata": doc.metadata if doc.metadata is not None else {}
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_emb = self._embedding_fn(query)
        results = []
        for rec in records:
            score = _dot(query_emb, rec["embedding"])
            results.append({
                "id": rec["id"],
                "content": rec["content"],
                "metadata": rec["metadata"],
                "score": score
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if self._use_chroma:
            ids = [doc.id for doc in docs]
            documents = [doc.content for doc in docs]
            embeddings = [self._embedding_fn(doc.content) for doc in docs]
            metadatas = [doc.metadata if doc.metadata is not None else {} for doc in docs]
            self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        else:
            for doc in docs:
                self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma:
            query_emb = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_emb],
                n_results=top_k
            )
            formatted = []
            if results and results.get("ids"):
                ids = results["ids"][0]
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results.get("distances", [[]])[0]
                for i in range(len(ids)):
                    dist = distances[i] if i < len(distances) else 0.0
                    score = 1.0 - dist
                    formatted.append({
                        "id": ids[i],
                        "content": documents[i],
                        "metadata": metadatas[i] or {},
                        "score": score
                    })
            formatted.sort(key=lambda x: x["score"], reverse=True)
            return formatted
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if self._use_chroma:
            query_emb = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_emb],
                n_results=top_k,
                where=metadata_filter
            )
            formatted = []
            if results and results.get("ids"):
                ids = results["ids"][0]
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results.get("distances", [[]])[0]
                for i in range(len(ids)):
                    dist = distances[i] if i < len(distances) else 0.0
                    score = 1.0 - dist
                    formatted.append({
                        "id": ids[i],
                        "content": documents[i],
                        "metadata": metadatas[i] or {},
                        "score": score
                    })
            formatted.sort(key=lambda x: x["score"], reverse=True)
            return formatted
        else:
            records = self._store
            if metadata_filter:
                filtered_records = []
                for rec in records:
                    matches = True
                    for k, v in metadata_filter.items():
                        if rec["metadata"].get(k) != v:
                            matches = False
                            break
                    if matches:
                        filtered_records.append(rec)
                records = filtered_records
            return self._search_records(query, records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma:
            try:
                size_before = self._collection.count()
                self._collection.delete(ids=[doc_id])
                self._collection.delete(where={"doc_id": doc_id})
                size_after = self._collection.count()
                return size_after < size_before
            except Exception:
                return False
        else:
            initial_len = len(self._store)
            self._store = [rec for rec in self._store if rec["id"] != doc_id and rec["metadata"].get("doc_id") != doc_id]
            return len(self._store) < initial_len
