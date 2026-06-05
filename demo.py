from __future__ import annotations

import time
from pathlib import Path

import gradio as gr

from src.chunking import (
    FixedSizeChunker,
    ParentChildChunker,
    RecursiveChunker,
    SentenceChunker,
)
from src.embeddings import LocalEmbedder
from src.models import Document
from src.store import EmbeddingStore

LAW_FILES = {
    "data/md/13_2023_ND-CP.md": {"doc_type": "nghị_định", "year": 2023, "topic": "bảo_vệ_dữ_liệu"},
    "data/md/2018_775 + 776_24-2018-QH14.md": {"doc_type": "luật", "year": 2018, "topic": "an_ninh_mạng"},
    "data/md/2023_867 + 868_20-2023-QH15.md": {"doc_type": "luật", "year": 2023, "topic": "giao_dịch_điện_tử"},
    "data/md/356_2025_ND-CP.md": {"doc_type": "nghị_định", "year": 2025, "topic": "bảo_vệ_dữ_liệu"},
    "data/md/91_2025_QH15.md": {"doc_type": "luật", "year": 2025, "topic": "bảo_vệ_dữ_liệu"},
}

EXAMPLE_QUERIES = [
    "Dữ liệu cá nhân nhạy cảm bao gồm những gì?",
    "Trách nhiệm của doanh nghiệp nước ngoài xử lý dữ liệu người Việt?",
    "Chữ ký điện tử có giá trị pháp lý không?",
    "Mức phạt vi phạm bảo vệ dữ liệu cá nhân?",
    "Quyền của chủ thể dữ liệu cá nhân gồm những gì?",
]

STRATEGIES = {
    "ParentChildChunker (custom)": "parent_child",
    "SentenceChunker(3)": "sentence_3",
    "SentenceChunker(5)": "sentence_5",
    "RecursiveChunker(500)": "recursive",
    "FixedSizeChunker(500)": "fixed_size",
}


def load_stores():
    print("Loading Vietnamese embedding model...")
    embedder = LocalEmbedder("AITeamVN/Vietnamese_Embedding")

    stores: dict[str, EmbeddingStore] = {}
    parent_map: dict[str, str] = {}

    chunkers = {
        "sentence_3": SentenceChunker(max_sentences_per_chunk=3),
        "sentence_5": SentenceChunker(max_sentences_per_chunk=5),
        "recursive": RecursiveChunker(chunk_size=500),
        "fixed_size": FixedSizeChunker(chunk_size=500, overlap=50),
    }
    pc_chunker = ParentChildChunker(child_max_sentences=3)

    for key, chunker in chunkers.items():
        stores[key] = EmbeddingStore(collection_name=key, embedding_fn=embedder)
    stores["parent_child"] = EmbeddingStore(collection_name="parent_child", embedding_fn=embedder)

    for filepath, meta in LAW_FILES.items():
        path = Path(filepath)
        text = path.read_text(encoding="utf-8")

        for key, chunker in chunkers.items():
            print(f"  Chunking {path.name} with {key}...")
            chunks = chunker.chunk(text)
            docs = [
                Document(id=f"{path.stem}_{key}_{i}", content=c, metadata={**meta, "source": path.name})
                for i, c in enumerate(chunks)
            ]
            stores[key].add_documents(docs)

        print(f"  Chunking {path.name} with parent_child...")
        pc_pairs = pc_chunker.chunk(text)
        for i, pair in enumerate(pc_pairs):
            doc_id = f"{path.stem}_pc_{i}"
            parent_map[doc_id] = pair["parent"]
            stores["parent_child"].add_documents([Document(
                id=doc_id,
                content=pair["child"],
                metadata={**meta, "source": path.name, "parent_id": doc_id},
            )])

    print("\nReady! Store sizes:")
    for key, store in stores.items():
        print(f"  {key}: {store.get_collection_size()} chunks")
    return stores, parent_map


stores, parent_map = load_stores()


def search(query: str, strategy: str, top_k: int) -> str:
    if not query.strip():
        return "Vui lòng nhập câu hỏi."

    store_key = STRATEGIES[strategy]
    store = stores[store_key]

    start = time.time()
    results = store.search(query, top_k=top_k)
    elapsed = (time.time() - start) * 1000

    chunk_count = store.get_collection_size()
    output_parts = [
        f"**⏱ Thời gian: {elapsed:.0f} ms** | Strategy: **{strategy}** | "
        f"Chunks: {chunk_count} | Top-{top_k}\n"
    ]

    for i, r in enumerate(results, 1):
        score = r["score"]
        source = r["metadata"].get("source", "?")
        doc_type = r["metadata"].get("doc_type", "?")
        year = r["metadata"].get("year", "?")
        topic = r["metadata"].get("topic", "?")
        content = r["content"].strip()

        output_parts.append(f"---\n### Kết quả #{i}  —  score = {score:.4f}")
        output_parts.append(f"📄 **{source}** | {doc_type} | {year} | {topic}\n")
        output_parts.append(f"```\n{content}\n```")

        if store_key == "parent_child":
            pid = r["metadata"].get("parent_id")
            if pid and pid in parent_map:
                parent_text = parent_map[pid]
                output_parts.append(
                    f"\n<details><summary>📂 Parent chunk (Điều gốc) — "
                    f"{len(parent_text)} ký tự</summary>\n\n```\n{parent_text}\n```\n</details>"
                )

    return "\n".join(output_parts)


def search_all(query: str, top_k: int) -> str:
    if not query.strip():
        return "Vui lòng nhập câu hỏi."

    output_parts = [f"## So sánh tất cả strategies — Top-{top_k}\n"]

    for strategy_name, store_key in STRATEGIES.items():
        store = stores[store_key]
        start = time.time()
        results = store.search(query, top_k=top_k)
        elapsed = (time.time() - start) * 1000

        top1_score = results[0]["score"] if results else 0
        top1_source = results[0]["metadata"].get("source", "?") if results else "?"
        scores_str = ", ".join(f"{r['score']:.4f}" for r in results)

        output_parts.append(f"### {strategy_name}  ⏱ {elapsed:.0f}ms")
        output_parts.append(f"Top-1: **{top1_score:.4f}** ({top1_source}) | Scores: [{scores_str}]\n")

    return "\n".join(output_parts)


with gr.Blocks(title="Vietnamese Law RAG Demo — E5", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        "# 🇻🇳 Vietnamese Law RAG Demo\n"
        "**Nhóm E5** — Embedding & Vector Store | AITeamVN/Vietnamese_Embedding (1024-dim)\n\n"
        "5 strategies: **ParentChildChunker** (custom), SentenceChunker(3), "
        "SentenceChunker(5), RecursiveChunker, FixedSizeChunker"
    )

    with gr.Row():
        with gr.Column(scale=3):
            query_box = gr.Textbox(
                label="Câu hỏi về luật Việt Nam",
                placeholder="Nhập câu hỏi...",
                lines=2,
            )
        with gr.Column(scale=1):
            strategy_dd = gr.Dropdown(
                choices=list(STRATEGIES.keys()),
                value="ParentChildChunker (custom)",
                label="Chunking Strategy",
            )
            top_k_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Top-K")
            with gr.Row():
                search_btn = gr.Button("🔍 Tìm kiếm", variant="primary")
                compare_btn = gr.Button("⚡ So sánh tất cả", variant="secondary")

    gr.Markdown("**Câu hỏi mẫu:**")
    with gr.Row():
        for q in EXAMPLE_QUERIES:
            gr.Button(q, size="sm").click(fn=lambda x=q: x, outputs=query_box)

    output_md = gr.Markdown(label="Kết quả")

    search_btn.click(fn=search, inputs=[query_box, strategy_dd, top_k_slider], outputs=output_md)
    query_box.submit(fn=search, inputs=[query_box, strategy_dd, top_k_slider], outputs=output_md)
    compare_btn.click(fn=search_all, inputs=[query_box, top_k_slider], outputs=output_md)

app.launch(inbrowser=True)
