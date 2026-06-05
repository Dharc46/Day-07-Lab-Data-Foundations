from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        if not results:
            return "I could not find enough relevant context in the knowledge base to answer this question reliably."

        context_blocks = []
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            source = metadata.get("source") or metadata.get("doc_id") or result.get("doc_id") or result.get("id")
            score = result.get("score", 0.0)
            context_blocks.append(
                f"[{index}] source={source} score={score:.3f}\n{result.get('content', '')}"
            )

        prompt = (
            "You are a knowledge-base assistant. Answer only from the retrieved context below. "
            "If the context is not sufficient, say that there is not enough information in the knowledge base.\n\n"
            "Retrieved context:\n"
            + "\n\n".join(context_blocks)
            + "\n\nQuestion:\n"
            + question
            + "\n\nAnswer:"
        )
        answer = self.llm_fn(prompt)
        return str(answer).strip()
