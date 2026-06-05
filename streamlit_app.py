from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from main import (
    EMBEDDING_CACHE_DIR,
    SAMPLE_FILES,
    get_llm_fn,
    iter_supported_paths,
    load_documents_from_files,
)
from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.store import EmbeddingStore


APP_TITLE = "Legal RAG Assistant"
DEFAULT_QUESTION = "Dữ liệu cá nhân nhạy cảm bao gồm những gì?"


def make_cache_name(value: str) -> str:
    return re.sub(r"\W+", "_", value, flags=re.UNICODE).strip("_").lower()


def build_embedder() -> tuple[Any, str, str | None]:
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    warning = None

    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception as exc:
            embedder = _mock_embed
            warning = f"Local embedder failed; using mock fallback. Details: {exc}"
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception as exc:
            embedder = _mock_embed
            warning = f"OpenAI embedder failed; using mock fallback. Details: {exc}"
    else:
        embedder = _mock_embed

    backend_name = getattr(embedder, "_backend_name", embedder.__class__.__name__)
    return embedder, backend_name, warning


@st.cache_resource(show_spinner="Loading legal documents and vector store...")
def load_backend() -> dict[str, Any]:
    load_dotenv(override=False)

    source_paths = iter_supported_paths(SAMPLE_FILES)
    documents = load_documents_from_files(SAMPLE_FILES)
    embedder, embedding_backend, warning = build_embedder()

    cache_path = EMBEDDING_CACHE_DIR / f"manual_test_store_{make_cache_name(embedding_backend)}.json"
    store = EmbeddingStore(
        collection_name="manual_test_store",
        embedding_fn=embedder,
        persist_path=cache_path,
        embedding_backend=embedding_backend,
    )
    store.add_documents(documents)

    return {
        "agent": KnowledgeBaseAgent(store=store, llm_fn=get_llm_fn()),
        "cache_path": cache_path,
        "documents": documents,
        "embedding_backend": embedding_backend,
        "source_paths": source_paths,
        "store": store,
        "warning": warning,
    }


def source_label(result: dict[str, Any]) -> str:
    metadata = result.get("metadata", {})
    source = Path(str(metadata.get("source", "unknown source"))).name
    article = metadata.get("article")
    if article:
        return f"{source} - {article}"
    return source


def render_sources(results: list[dict[str, Any]], preview_chars: int) -> None:
    if not results:
        st.info("No retrieved sources were returned.")
        return

    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        score = result.get("score", 0.0)
        title = f"[{index}] {source_label(result)} - score {score:.3f}"

        with st.expander(title, expanded=index == 1):
            st.caption(f"Source: `{metadata.get('source', 'unknown')}`")
            if metadata.get("article"):
                st.caption(f"Article: `{metadata['article']}`")
            st.caption(f"Document id: `{metadata.get('doc_id', result.get('id', 'unknown'))}`")
            content = result.get("content", "")
            st.markdown(content[:preview_chars] + ("..." if len(content) > preview_chars else ""))


def render_message(message: dict[str, Any], preview_chars: int) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            st.markdown("**Citations used**")
            for index, result in enumerate(message["sources"], start=1):
                st.markdown(f"{index}. `{source_label(result)}` - score `{result.get('score', 0.0):.3f}`")
            render_sources(message["sources"], preview_chars=preview_chars)


def initialize_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ask a Vietnamese legal question. I will retrieve relevant legal chunks and answer from the cited context.",
                "sources": [],
            }
        ]


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_state()

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; max-width: 1180px; }
        [data-testid="stChatMessage"] { border-radius: 10px; }
        .small-muted { color: #64748b; font-size: 0.9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    backend = load_backend()
    store: EmbeddingStore = backend["store"]
    agent: KnowledgeBaseAgent = backend["agent"]

    with st.sidebar:
        st.title("Legal RAG")
        st.caption("Vietnamese law retrieval assistant")

        if backend["warning"]:
            st.warning(backend["warning"])

        st.subheader("System information")
        st.metric("Source files", len(backend["source_paths"]))
        st.metric("Loaded documents", len(backend["documents"]))
        st.metric("Chunk count", store.get_collection_size())

        st.subheader("Embedding")
        st.write(f"Provider: `{os.getenv(EMBEDDING_PROVIDER_ENV, 'mock')}`")
        st.write(f"Model/backend: `{backend['embedding_backend']}`")
        st.write(f"Cache: `{backend['cache_path']}`")

        st.subheader("Generation")
        st.write(f"LLM provider: `{os.getenv('LLM_PROVIDER', 'mock')}`")
        st.write(f"Gemini model: `{os.getenv('GEMINI_MODEL', 'not configured')}`")

        st.subheader("Retrieval settings")
        top_k = st.slider("Top-k retrieval results", min_value=1, max_value=10, value=3)
        preview_chars = st.slider("Retrieved chunk preview length", min_value=300, max_value=4000, value=1400, step=100)

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.title(APP_TITLE)
    st.caption("Chat with the legal knowledge base. Each answer shows retrieved chunks, source documents, scores, and citations.")

    for message in st.session_state.messages:
        render_message(message, preview_chars=preview_chars)

    prompt = st.chat_input("Ask a legal question...", key="legal_question")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving legal context and generating answer..."):
                try:
                    retrieved = store.search(prompt, top_k=top_k)
                    answer = agent.answer(prompt, top_k=top_k)
                except Exception as exc:
                    retrieved = []
                    answer = f"Sorry, I could not process this request. Details: `{exc}`"

            st.markdown(answer)
            if retrieved:
                st.markdown("**Citations used**")
                for index, result in enumerate(retrieved, start=1):
                    st.markdown(f"{index}. `{source_label(result)}` - score `{result.get('score', 0.0):.3f}`")
                render_sources(retrieved, preview_chars=preview_chars)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": retrieved,
            }
        )

    if len(st.session_state.messages) <= 1:
        st.markdown("#### Try a benchmark question")
        cols = st.columns(2)
        examples = [
            DEFAULT_QUESTION,
            "Chữ ký điện tử có giá trị pháp lý không?",
            "Mức phạt vi phạm bảo vệ dữ liệu cá nhân?",
            "Quyền của chủ thể dữ liệu cá nhân gồm những gì?",
        ]
        for index, example in enumerate(examples):
            with cols[index % 2]:
                st.code(example, language=None)


if __name__ == "__main__":
    main()
