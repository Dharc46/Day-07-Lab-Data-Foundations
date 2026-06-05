import os
import sys
from pathlib import Path
from src.chunking import FixedSizeChunker, SentenceChunker, RecursiveChunker, ChunkingStrategyComparator
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent
from src.models import Document
from src.embeddings import _mock_embed, LocalEmbedder

DATA_FILES = {
    "13_2023_ND-CP.md": {"doc_type": "nghi_dinh", "year": 2023, "authority": "chinh_phu"},
    "2018_775 + 776_24-2018-QH14.md": {"doc_type": "luat", "year": 2018, "authority": "quoc_hoi"},
    "2023_867 + 868_20-2023-QH15.md": {"doc_type": "luat", "year": 2023, "authority": "quoc_hoi"},
    "356_2025_ND-CP.md": {"doc_type": "nghi_dinh", "year": 2025, "authority": "chinh_phu"},
    "91_2025_QH15.md": {"doc_type": "luat", "year": 2025, "authority": "quoc_hoi"}
}

def load_data():
    docs = []
    for filename, meta in DATA_FILES.items():
        filepath = Path("data") / filename
        if not filepath.exists():
            continue
        content = filepath.read_text(encoding="utf-8")
        docs.append(Document(id=filename, content=content, metadata=meta))
    return docs

def main():
    docs = load_data()
    
    with open("benchmark_results.txt", "w", encoding="utf-8") as f:
        f.write("=== CHUNKING STRATEGY COMPARISON ===\n")
        comparator = ChunkingStrategyComparator()
        for doc in docs:
            f.write(f"\nDocument: {doc.id}\n")
            stats = comparator.compare(doc.content, chunk_size=500)
            for name, info in stats.items():
                f.write(f"  Strategy: {name}\n")
                f.write(f"    Chunk Count: {info['count']}\n")
                f.write(f"    Avg Length: {info['avg_length']:.2f}\n")

        f.write("\n=== RUNNING BENCHMARK QUERIES ===\n")
        chunker = RecursiveChunker(chunk_size=500)
        
        import pickle
        cache_path = Path("vn_law_store_cache.pkl")
        
        # We will use LocalEmbedder with Vietnamese_Embedding for real vector search
        print("Loading Vietnamese_Embedding model...")
        embedder = LocalEmbedder(model_name='AITeamVN/Vietnamese_Embedding')
        store = EmbeddingStore(collection_name="vn_law_store", embedding_fn=embedder)
        
        if cache_path.exists():
            print("Loading cached embeddings from vn_law_store_cache.pkl...")
            with open(cache_path, "rb") as cf:
                store._store = pickle.load(cf)
            f.write("Loaded cached embeddings from disk.\n")
        else:
            print("No cache found. Embedding chunks from scratch (this may take 1-2 minutes)...")
            chunked_docs = []
            for doc in docs:
                chunks = chunker.chunk(doc.content)
                for idx, chunk in enumerate(chunks):
                    meta = doc.metadata.copy()
                    meta["doc_id"] = doc.id
                    chunked_docs.append(Document(
                        id=f"{doc.id}_chunk_{idx}",
                        content=chunk,
                        metadata=meta
                    ))
            store.add_documents(chunked_docs)
            with open(cache_path, "wb") as cf:
                pickle.dump(store._store, cf)
            f.write("Computed and cached embeddings to disk.\n")
            
        f.write(f"Indexed {store.get_collection_size()} chunks in EmbeddingStore.\n")

        queries = [
            {
                "id": 1,
                "query": "Dữ liệu cá nhân nhạy cảm bao gồm những gì?",
                "filter": {"year": 2025, "doc_type": "nghi_dinh"}
            },
            {
                "id": 2,
                "query": "Trách nhiệm của doanh nghiệp nước ngoài xử lý dữ liệu người Việt?",
                "filter": {"year": 2025, "doc_type": "luat"}
            },
            {
                "id": 3,
                "query": "Chữ ký điện tử có giá trị pháp lý không?",
                "filter": {"year": 2023, "doc_type": "luat"}
            },
            {
                "id": 4,
                "query": "Mức phạt vi phạm bảo vệ dữ liệu cá nhân?",
                "filter": {"year": 2025, "doc_type": "luat"}
            },
            {
                "id": 5,
                "query": "Quyền của chủ thể dữ liệu cá nhân gồm những gì?",
                "filter": {"year": 2025, "doc_type": "luat"}
            }
        ]

        def mock_llm(prompt):
            return "Tác nhân RAG đã tìm kiếm và trả lời thành công dựa trên ngữ cảnh được cung cấp."

        agent = KnowledgeBaseAgent(store=store, llm_fn=mock_llm)

        for q in queries:
            f.write(f"\n--- Query {q['id']}: {q['query']} ---\n")
            if q["filter"]:
                f.write(f"Using filter: {q['filter']}\n")
                results = store.search_with_filter(q["query"], top_k=3, metadata_filter=q["filter"])
            else:
                results = store.search(q["query"], top_k=3)
                
            for idx, res in enumerate(results):
                f.write(f"Top-{idx+1} Chunk (Score: {res['score']:.4f}, Doc: {res['id']}):\n")
                f.write(f"   {res['content'][:250].strip()}...\n\n")
                
            ans = agent.answer(q["query"], top_k=3)
            f.write(f"Agent Answer: {ans}\n")
            
    print("Benchmark completed. Results written to benchmark_results.txt")

if __name__ == "__main__":
    main()
