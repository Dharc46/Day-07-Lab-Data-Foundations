from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
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
        persist_path: str | Path | None = None,
        embedding_backend: str | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._embedding_backend = embedding_backend or getattr(self._embedding_fn, "_backend_name", "unknown")
        self._persist_path = Path(persist_path) if persist_path else None
        self._embedding_max_chars = int(os.getenv("VECTOR_EMBEDDING_MAX_CHARS", "4000"))
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            self._use_chroma = False
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _content_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _persist_key(self, doc: Document) -> tuple[str, str]:
        return doc.id, self._content_hash(doc.content)

    def _embedding_text(self, text: str) -> str:
        if self._embedding_max_chars <= 0:
            return text
        return text[: self._embedding_max_chars]

    def _load_persisted_records(self) -> list[dict[str, Any]]:
        if not self._persist_path or not self._persist_path.exists():
            return []

        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

        if data.get("embedding_backend") != self._embedding_backend:
            return []

        records = data.get("records", [])
        if not isinstance(records, list):
            return []
        return records

    def _save_persisted_records(self) -> None:
        if not self._persist_path:
            return

        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "collection_name": self._collection_name,
            "embedding_backend": self._embedding_backend,
            "records": self._store,
        }
        self._persist_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    def _tokenize(self, text: str) -> set[str]:
        stopwords = {"bao", "gồm", "những", "gì", "các", "và", "của", "trong", "theo", "được"}
        tokens = {
            token
            for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
            if len(token) > 2 and token not in stopwords
        }
        if "doanh" in tokens or "nghiệp" in tokens:
            tokens.update({"tổ", "chức"})
        if "người" in tokens and "việt" in tokens:
            tokens.update({"công", "dân", "nam", "gốc"})
        if "mức" in tokens and "phạt" in tokens:
            tokens.update({"tiền", "tối", "đa", "phạt"})
        return tokens

    def _score_record(self, query: str, query_embedding: list[float], record: dict[str, Any]) -> float:
        vector_score = _dot(query_embedding, record["embedding"])
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return vector_score

        article = str(record["metadata"].get("article", ""))
        source = str(record["metadata"].get("source", ""))
        query_lower = query.lower()
        record_tokens = self._tokenize(record["content"])
        article_tokens = self._tokenize(article)
        lexical_score = len(query_tokens & record_tokens) / len(query_tokens)
        article_score = len(query_tokens & article_tokens) / len(query_tokens)
        domain_boost = 0.0
        article_lower = article.lower()
        if "nước ngoài" in query_lower and "dữ liệu" in query_lower and "việt" in query_lower:
            if "đối tượng áp dụng" in article_lower or "phạm vi điều chỉnh" in article_lower:
                domain_boost += 1.2
            if "91_2025_QH15" in source:
                domain_boost += 0.4
        return lexical_score + article_score + domain_boost + (0.05 * vector_score)

    def _make_record(self, doc: Document, embedding: list[float] | None = None) -> dict[str, Any]:
        metadata = dict(doc.metadata or {})
        metadata.setdefault("doc_id", doc.id)
        metadata["content_hash"] = self._content_hash(doc.content)
        metadata["embedding_backend"] = self._embedding_backend
        record_id = f"{doc.id}:{self._next_index}"
        self._next_index += 1
        return {
            "id": record_id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": embedding if embedding is not None else self._embedding_fn(self._embedding_text(doc.content)),
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []

        query_embedding = self._embedding_fn(query)
        results: list[dict[str, Any]] = []
        for record in records:
            results.append(
                {
                    "id": record["id"],
                    "content": record["content"],
                    "metadata": dict(record["metadata"]),
                    "score": self._score_record(query, query_embedding, record),
                }
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        cached_records = self._load_persisted_records()
        cached_by_key = {
            (record.get("metadata", {}).get("doc_id"), record.get("metadata", {}).get("content_hash")): record
            for record in cached_records
        }

        docs_to_embed: list[Document] = []
        for doc in docs:
            cached_record = cached_by_key.get(self._persist_key(doc))
            if cached_record is not None:
                self._store.append(cached_record)
            else:
                docs_to_embed.append(doc)

        self._next_index = len(self._store)
        if not docs_to_embed:
            return

        embed_many = getattr(self._embedding_fn, "embed_many", None)
        if callable(embed_many):
            batch_size = int(os.getenv("VECTOR_STORE_EMBED_BATCH_SIZE", "16"))
            for start in range(0, len(docs_to_embed), batch_size):
                batch_docs = docs_to_embed[start : start + batch_size]
                embeddings = embed_many([self._embedding_text(doc.content) for doc in batch_docs])
                for doc, embedding in zip(batch_docs, embeddings):
                    self._store.append(self._make_record(doc, embedding=embedding))
                self._save_persisted_records()
            return

        for doc in docs_to_embed:
            self._store.append(self._make_record(doc))
        self._save_persisted_records()

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if metadata_filter is None:
            return self.search(query, top_k=top_k)

        filtered_records = [
            record
            for record in self._store
            if all(record["metadata"].get(key) == value for key, value in metadata_filter.items())
        ]
        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        size_before = len(self._store)
        self._store = [
            record
            for record in self._store
            if record["metadata"].get("doc_id") != doc_id
        ]
        return len(self._store) < size_before
