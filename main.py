from __future__ import annotations

import os
import re
import sys
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

LEGAL_DATA_DIR = "data/luat_an_ninh_mang_ve_bao_ve_du_lieu_ca_nhan"
SAMPLE_FILES = [LEGAL_DATA_DIR]
EMBEDDING_CACHE_DIR = Path(".embedding-cache")

ARTICLE_HEADING_RE = re.compile(r"(?m)^####\s+(Điều\s+\d+\.?.*)$")


def iter_supported_paths(paths: list[str]) -> list[Path]:
    """Return supported files from a mix of explicit file and directory paths."""
    allowed_extensions = {".md", ".txt"}
    supported_paths: list[Path] = []

    for raw_path in paths:
        path = Path(raw_path)

        if path.is_dir():
            supported_paths.extend(
                child
                for child in sorted(path.rglob("*"))
                if child.is_file() and child.suffix.lower() in allowed_extensions
            )
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        supported_paths.append(path)

    return supported_paths


def split_markdown_articles(content: str) -> list[tuple[str | None, str]]:
    """Split Vietnamese legal Markdown into article-level sections when possible."""
    matches = list(ARTICLE_HEADING_RE.finditer(content))
    if not matches:
        return [(None, content.strip())] if content.strip() else []

    preamble = content[: matches[0].start()].strip()
    sections: list[tuple[str | None, str]] = []
    if preamble:
        sections.append(("Preamble", preamble))

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        article = match.group(1).strip()
        section = content[start:end].strip()
        if section:
            sections.append((article, section))

    return sections


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load files or folders for the manual demo, splitting legal Markdown by article."""
    documents: list[Document] = []

    for path in iter_supported_paths(file_paths):
        content = path.read_text(encoding="utf-8")
        sections = split_markdown_articles(content) if path.suffix.lower() == ".md" else [(None, content.strip())]
        for section_index, (article, section_content) in enumerate(sections, start=1):
            if not section_content:
                continue

            metadata = {
                "source": str(path),
                "extension": path.suffix.lower(),
                "language": "vi" if "luat_an_ninh_mang" in str(path) else "unknown",
            }
            if article:
                metadata["article"] = article

            document_id = path.stem
            if article:
                safe_article = re.sub(r"\W+", "_", article, flags=re.UNICODE).strip("_").lower()
                document_id = f"{path.stem}_{safe_article or section_index}"

            documents.append(
                Document(
                    id=document_id,
                    content=section_content,
                    metadata=metadata,
                )
            )

    return documents


def mock_llm(prompt: str) -> str:
    """Fallback mock LLM for manual RAG testing."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def openai_llm(prompt: str) -> str:
    """OpenAI-backed LLM for manual RAG testing."""
    from openai import OpenAI
    from openai import OpenAIError

    client = OpenAI()
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer using only the provided context. If the context is insufficient, say you do not know.",
                },
                {"role": "user", "content": prompt},
            ],
        )
    except OpenAIError as exc:
        return f"[OpenAI error] {exc}"
    return response.choices[0].message.content or ""


def ollama_llm(prompt: str) -> str:
    """Ollama-backed local LLM for free offline generation."""
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": "Answer using only the provided context. If the context is insufficient, say you do not know.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return f"[Ollama error] Could not reach Ollama at {host}. Install Ollama, run `ollama pull {model}`, then try again. Details: {exc}"
    return data.get("message", {}).get("content", "")


def gemini_llm(prompt: str) -> str:
    """Gemini-backed LLM for manual RAG testing."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "[Gemini error] GEMINI_API_KEY is not set."

    configured_models = [
        model.strip()
        for model in os.getenv("GEMINI_MODEL", "gemini-2.5-flash").split(",")
        if model.strip()
    ]
    retry_count = int(os.getenv("GEMINI_RETRY_COUNT", "3"))
    payload = {
        "systemInstruction": {
            "parts": [
                {
                    "text": "Answer using only the provided context. If the context is insufficient, say you do not know."
                }
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
    }

    last_error = ""
    for model in configured_models:
        for attempt in range(retry_count + 1):
            request = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                last_error = f"[Gemini error] model={model} HTTP {exc.code}: {details}"
                if exc.code not in {429, 500, 502, 503, 504} or attempt >= retry_count:
                    break
                time.sleep(min(2**attempt, 8))
            except urllib.error.URLError as exc:
                last_error = f"[Gemini error] model={model} could not reach Gemini API. Details: {exc}"
                if attempt >= retry_count:
                    break
                time.sleep(min(2**attempt, 8))
        else:
            continue

        if "data" in locals():
            break
    else:
        return last_error or "[Gemini error] No Gemini models configured."

    candidates = data.get("candidates", [])
    if not candidates:
        return "[Gemini error] No candidates returned."

    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts)


def get_llm_fn():
    provider = os.getenv("LLM_PROVIDER", "mock").strip().lower()
    if provider == "openai":
        return openai_llm
    if provider == "ollama":
        return ollama_llm
    if provider == "gemini":
        return gemini_llm
    return mock_llm


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file/folder list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs[:12]:
        article = doc.metadata.get("article")
        article_suffix = f" ({article})" if article else ""
        print(f"  - {doc.id}: {doc.metadata['source']}{article_suffix}")
    if len(docs) > 12:
        print(f"  ... {len(docs) - 12} more")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    embedding_backend = getattr(embedder, "_backend_name", embedder.__class__.__name__)
    cache_name = re.sub(r"\W+", "_", embedding_backend, flags=re.UNICODE).strip("_").lower()
    cache_path = EMBEDDING_CACHE_DIR / f"manual_test_store_{cache_name}.json"
    store = EmbeddingStore(
        collection_name="manual_test_store",
        embedding_fn=embedder,
        persist_path=cache_path,
        embedding_backend=embedding_backend,
    )
    store.add_documents(docs)

    print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")
    print(f"Embedding cache: {cache_path}")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        article = result["metadata"].get("article")
        article_suffix = f" article={article}" if article else ""
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}{article_suffix}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=get_llm_fn())
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=3))
    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())
