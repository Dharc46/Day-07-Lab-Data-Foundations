from __future__ import annotations

import sys
import io
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.chunking import SentenceChunker, ParentChildChunker
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

QUERIES = [
    "Dữ liệu cá nhân nhạy cảm bao gồm những gì?",
    "Trách nhiệm của doanh nghiệp nước ngoài xử lý dữ liệu người Việt?",
    "Chữ ký điện tử có giá trị pháp lý không?",
    "Mức phạt vi phạm bảo vệ dữ liệu cá nhân?",
    "Quyền của chủ thể dữ liệu cá nhân gồm những gì?",
]

GOLD_ANSWERS = [
    "12 loại: nguồn gốc chủng tộc/dân tộc, quan điểm chính trị/tôn giáo, đời sống riêng tư, sức khỏe, sinh trắc học, di truyền, đời sống tình dục, dữ liệu tội phạm, vị trí, tên đăng nhập/mật khẩu, tài chính/tín dụng, hành vi trên mạng (NĐ 356/2025, Điều 4)",
    "Phải tuân thủ Luật BVDLCN Việt Nam nếu trực tiếp tham gia hoặc liên quan đến xử lý DLCN của công dân VN và người gốc Việt (Luật 91/2025, Điều 1 Khoản 2c)",
    "Có — chữ ký điện tử chuyên dùng bảo đảm an toàn hoặc chữ ký số có giá trị pháp lý tương đương chữ ký tay trên văn bản giấy (Luật GDĐT 20/2023, Điều 23 Khoản 2)",
    "Phạt tiền tối đa 03 tỷ đồng cho vi phạm hành chính trong lĩnh vực BVDLCN (Luật 91/2025, Điều 8 Khoản 5)",
    "6 nhóm quyền: được biết, đồng ý/rút lại đồng ý, xem/chỉnh sửa, yêu cầu cung cấp/xóa/hạn chế/phản đối, khiếu nại/khởi kiện, yêu cầu cơ quan bảo vệ DLCN (Luật 91/2025, Điều 4 Khoản 1)",
]

OUTPUT_FILE = Path("results/benchmark_output.txt")


def log(msg: str, file) -> None:
    print(msg)
    file.write(msg + "\n")


def build_sentence_store(embedder):
    chunker = SentenceChunker(max_sentences_per_chunk=5)
    store = EmbeddingStore(collection_name="sentence", embedding_fn=embedder)
    stats = {}
    for filepath, meta in LAW_FILES.items():
        path = Path(filepath)
        text = path.read_text(encoding="utf-8")
        chunks = chunker.chunk(text)
        docs = [
            Document(id=f"{path.stem}_s_{i}", content=chunk, metadata={**meta, "source": str(path)})
            for i, chunk in enumerate(chunks)
        ]
        store.add_documents(docs)
        stats[path.name] = {"count": len(chunks), "avg": sum(len(c) for c in chunks) // max(len(chunks), 1)}
    return store, stats


def build_parent_child_store(embedder):
    chunker = ParentChildChunker(child_max_sentences=3)
    store = EmbeddingStore(collection_name="parent_child", embedding_fn=embedder)
    parent_map: dict[str, str] = {}
    stats = {}
    child_count_per_file = {}
    for filepath, meta in LAW_FILES.items():
        path = Path(filepath)
        text = path.read_text(encoding="utf-8")
        pairs = chunker.chunk(text)
        count = 0
        for i, pair in enumerate(pairs):
            doc_id = f"{path.stem}_pc_{i}"
            parent_map[doc_id] = pair["parent"]
            docs = [Document(
                id=doc_id,
                content=pair["child"],
                metadata={**meta, "source": str(path), "parent_id": doc_id},
            )]
            store.add_documents(docs)
            count += 1
        child_lengths = [len(p["child"]) for p in pairs]
        parent_lengths = [len(p["parent"]) for p in pairs]
        stats[path.name] = {
            "children": count,
            "avg_child": sum(child_lengths) // max(count, 1),
            "parents": len(set(p["parent"] for p in pairs)),
            "avg_parent": sum(len(p) for p in set(p["parent"] for p in pairs)) // max(len(set(p["parent"] for p in pairs)), 1),
        }
    return store, parent_map, stats


def run_queries(store, queries, f, strategy_name, parent_map=None):
    log(f"\n{'=' * 70}", f)
    log(f"=== {strategy_name} ===", f)
    log(f"{'=' * 70}", f)

    scores_per_query = []
    for i, (query, gold) in enumerate(zip(queries, GOLD_ANSWERS), 1):
        results = store.search(query, top_k=3)
        top1 = results[0]

        log(f"\n{'─' * 70}", f)
        log(f"Query {i}: {query}", f)
        log(f"Gold Answer: {gold}", f)
        log(f"{'─' * 70}", f)

        for j, r in enumerate(results, 1):
            src = Path(r["metadata"].get("source", "")).name
            score = r["score"]
            content = r["content"].replace("\n", " ")
            tag = " << TOP-1" if j == 1 else ""
            log(f"\n  #{j} [score={score:.4f}] (source: {src}){tag}", f)
            log(f"     metadata: doc_type={r['metadata'].get('doc_type')}, year={r['metadata'].get('year')}, topic={r['metadata'].get('topic')}", f)
            log(f"     child chunk:", f)
            for line_idx in range(0, min(len(content), 500), 100):
                log(f"       {content[line_idx:line_idx+100]}", f)

            if parent_map and r["metadata"].get("parent_id"):
                parent_text = parent_map.get(r["metadata"]["parent_id"], "")
                parent_preview = parent_text[:300].replace("\n", " ")
                log(f"     parent chunk (preview):", f)
                log(f"       {parent_preview}...", f)

        top3_scores = ", ".join(f"{r['score']:.4f}" for r in results)
        log(f"\n  Summary: top-3 scores = [{top3_scores}]", f)
        scores_per_query.append(top1["score"])

    return scores_per_query


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    f = open(OUTPUT_FILE, "w", encoding="utf-8")

    log(f"Benchmark run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f)
    log("=" * 70, f)

    log("\nLoading Vietnamese embedding model...", f)
    embedder = LocalEmbedder("AITeamVN/Vietnamese_Embedding")
    log(f"Model: {embedder._backend_name}", f)

    # --- Strategy 1: SentenceChunker(5) ---
    log("\n--- Building SentenceChunker(5) store ---", f)
    sent_store, sent_stats = build_sentence_store(embedder)
    for name, s in sent_stats.items():
        log(f"  {name}: {s['count']} chunks (avg {s['avg']} chars)", f)
    log(f"  Total: {sent_store.get_collection_size()} chunks", f)

    # --- Strategy 2: ParentChildChunker ---
    log("\n--- Building ParentChildChunker store ---", f)
    pc_store, parent_map, pc_stats = build_parent_child_store(embedder)
    for name, s in pc_stats.items():
        log(f"  {name}: {s['parents']} parents (avg {s['avg_parent']} chars), {s['children']} children (avg {s['avg_child']} chars)", f)
    log(f"  Total: {pc_store.get_collection_size()} child chunks", f)

    # --- Run queries ---
    sent_scores = run_queries(sent_store, QUERIES, f, "SentenceChunker(max_sentences=5)")
    pc_scores = run_queries(pc_store, QUERIES, f, "ParentChildChunker(child_sentences=3)", parent_map=parent_map)

    # --- Comparison summary ---
    log(f"\n{'=' * 70}", f)
    log("=== COMPARISON SUMMARY ===", f)
    log(f"{'=' * 70}", f)
    log(f"Model: AITeamVN/Vietnamese_Embedding (dim=1024)", f)
    log(f"Documents: {len(LAW_FILES)}", f)
    log(f"SentenceChunker(5) total chunks: {sent_store.get_collection_size()}", f)
    log(f"ParentChildChunker  total chunks: {pc_store.get_collection_size()}", f)
    log("", f)
    log(f"{'Query':<6} {'SentenceChunker(5)':>20} {'ParentChild':>20} {'Winner':>15}", f)
    log(f"{'-'*6} {'-'*20} {'-'*20} {'-'*15}", f)
    sent_wins = 0
    pc_wins = 0
    for i, (ss, ps) in enumerate(zip(sent_scores, pc_scores), 1):
        winner = "SentenceChunker" if ss > ps else "ParentChild" if ps > ss else "Tie"
        if ss > ps:
            sent_wins += 1
        elif ps > ss:
            pc_wins += 1
        log(f"Q{i:<5} {ss:>20.4f} {ps:>20.4f} {winner:>15}", f)
    log("", f)
    avg_sent = sum(sent_scores) / len(sent_scores)
    avg_pc = sum(pc_scores) / len(pc_scores)
    log(f"{'Avg':<6} {avg_sent:>20.4f} {avg_pc:>20.4f}", f)
    log(f"\nSentenceChunker wins: {sent_wins}, ParentChild wins: {pc_wins}", f)

    f.close()
    print(f"\nResults saved to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
