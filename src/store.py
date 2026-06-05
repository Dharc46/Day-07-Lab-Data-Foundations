from __future__ import annotations

import json
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

            self._chroma_client = chromadb.Client()
            self._collection = self._chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata,
            "embedding": self._embedding_fn(doc.content),
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_embedding = self._embedding_fn(query)
        scored = []
        for rec in records:
            score = _dot(query_embedding, rec["embedding"])
            scored.append({"content": rec["content"], "score": score, "metadata": rec["metadata"]})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _sanitize_metadata(metadata: dict) -> dict:
        sanitized = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            else:
                sanitized[k] = json.dumps(v, ensure_ascii=False)
        return sanitized

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
            metadatas = [self._sanitize_metadata(doc.metadata) for doc in docs]
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        else:
            for doc in docs:
                self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For ChromaDB: use collection.query(query_embeddings=[...], n_results=top_k)
        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma:
            query_embedding = self._embedding_fn(query)
            n = min(top_k, self._collection.count())
            if n == 0:
                return []
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
            output = []
            for i in range(len(results["ids"][0])):
                score = 1.0 - results["distances"][0][i]
                output.append({
                    "content": results["documents"][0][i],
                    "score": score,
                    "metadata": results["metadatas"][0][i] or {},
                })
            output.sort(key=lambda x: x["score"], reverse=True)
            return output
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        For ChromaDB: use where= parameter for metadata filtering.
        For in-memory: filter stored chunks by metadata_filter, then run similarity search.
        """
        if self._use_chroma:
            query_embedding = self._embedding_fn(query)
            n = min(top_k, self._collection.count())
            if n == 0:
                return []
            where = None
            if metadata_filter:
                conditions = [{k: {"$eq": v}} for k, v in metadata_filter.items()]
                where = {"$and": conditions} if len(conditions) > 1 else conditions[0]
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            output = []
            for i in range(len(results["ids"][0])):
                score = 1.0 - results["distances"][0][i]
                output.append({
                    "content": results["documents"][0][i],
                    "score": score,
                    "metadata": results["metadatas"][0][i] or {},
                })
            output.sort(key=lambda x: x["score"], reverse=True)
            return output

        records = self._store
        if metadata_filter:
            records = [
                r for r in records
                if all(r["metadata"].get(k) == v for k, v in metadata_filter.items())
            ]
        return self._search_records(query, records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma:
            existing = self._collection.get(ids=[doc_id])
            if existing["ids"]:
                self._collection.delete(ids=[doc_id])
                return True
            return False

        before = len(self._store)
        self._store = [r for r in self._store if r["id"] != doc_id]
        return len(self._store) < before
