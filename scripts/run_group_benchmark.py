from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import LocalEmbedder, RecursiveChunker


def find_matches(content: str, needles: list[str]) -> list[str]:
    matches = []
    lowered = content.lower()
    for needle in needles:
        if needle.startswith("Điều "):
            if re.search(rf"{re.escape(needle)}(?!\d)", content, flags=re.IGNORECASE):
                matches.append(needle)
        elif needle.lower() in lowered:
            matches.append(needle)
    return matches


DOCS = [
    (
        "13_2023_ND_CP_bao_ve_du_lieu_ca_nhan.md",
        {"doc_type": "decree", "year": "2023", "topic": "personal_data_protection", "law_no": "13/2023/NĐ-CP"},
    ),
    (
        "24_2018_QH14_luat_an_ninh_mang.md",
        {"doc_type": "law", "year": "2018", "topic": "cybersecurity", "law_no": "24/2018/QH14"},
    ),
    (
        "20_2023_QH15_luat_giao_dich_dien_tu.md",
        {"doc_type": "law", "year": "2023", "topic": "electronic_transactions", "law_no": "20/2023/QH15"},
    ),
    (
        "356_2025_ND_CP_huong_dan_luat_bvdlcn.md",
        {"doc_type": "decree", "year": "2025", "topic": "personal_data_protection", "law_no": "356/2025/NĐ-CP"},
    ),
    (
        "91_2025_QH15_luat_bao_ve_du_lieu_ca_nhan.md",
        {"doc_type": "law", "year": "2025", "topic": "personal_data_protection", "law_no": "91/2025/QH15"},
    ),
]


BENCHMARKS = [
    {
        "query": "Dữ liệu cá nhân nhạy cảm bao gồm những gì?",
        "gold": "12 loại dữ liệu cá nhân nhạy cảm trong NĐ 356/2025, Điều 4",
        "filter": {"law_no": "356/2025/NĐ-CP"},
        "needles": ["Điều 4", "Dữ liệu cá nhân nhạy cảm", "nguồn gốc chủng tộc", "tài khoản ngân hàng"],
    },
    {
        "query": "Trách nhiệm của doanh nghiệp nước ngoài xử lý dữ liệu người Việt?",
        "gold": "Luật 91/2025, Điều 1 Khoản 2c về đối tượng nước ngoài liên quan đến xử lý dữ liệu cá nhân của công dân Việt Nam và người gốc Việt",
        "filter": {"law_no": "91/2025/QH15"},
        "needles": ["Điều 1", "nước ngoài", "công dân Việt Nam", "người gốc Việt"],
    },
    {
        "query": "Chữ ký điện tử có giá trị pháp lý không?",
        "gold": "Luật GDĐT 20/2023, Điều 23 Khoản 2 về giá trị pháp lý tương đương chữ ký tay",
        "filter": {"law_no": "20/2023/QH15"},
        "needles": ["Điều 23", "giá trị pháp lý", "chữ ký tay", "chữ ký số"],
    },
    {
        "query": "Mức phạt vi phạm bảo vệ dữ liệu cá nhân?",
        "gold": "Luật 91/2025, Điều 8 Khoản 5: phạt tiền tối đa 03 tỷ đồng",
        "filter": {"law_no": "91/2025/QH15"},
        "needles": ["Điều 8", "03 tỷ đồng", "vi phạm hành chính", "phạt tiền"],
    },
    {
        "query": "Quyền của chủ thể dữ liệu cá nhân gồm những gì?",
        "gold": "Luật 91/2025, Điều 4 Khoản 1: 6 nhóm quyền của chủ thể dữ liệu cá nhân",
        "filter": {"law_no": "91/2025/QH15"},
        "needles": ["Điều 4", "Quyền", "Được biết", "Đồng ý", "khiếu nại"],
    },
]


def main() -> None:
    base = Path("data/group_legal_docs")
    chunker = RecursiveChunker(chunk_size=1800)
    records = []

    for filename, meta in DOCS:
        text = (base / filename).read_text(encoding="utf-8")
        for chunk_index, chunk in enumerate(chunker.chunk(text)):
            records.append(
                {
                    "content": chunk,
                    "metadata": {**meta, "source": filename, "chunk_index": chunk_index},
                }
            )

    print("Strategy: RecursiveChunker(chunk_size=1800) + metadata filter + AITeamVN/Vietnamese_Embedding")
    print(f"Total chunks: {len(records)}")

    embedder = LocalEmbedder("AITeamVN/Vietnamese_Embedding")
    embeddings = np.asarray(
        embedder.model.encode(
            [record["content"] for record in records],
            normalize_embeddings=True,
            batch_size=16,
            show_progress_bar=False,
        )
    )

    relevant_count = 0
    for index, benchmark in enumerate(BENCHMARKS, start=1):
        query_embedding = np.asarray(
            embedder.model.encode([benchmark["query"]], normalize_embeddings=True, show_progress_bar=False)
        )[0]
        scores = embeddings @ query_embedding
        candidates = [
            record_index
            for record_index, record in enumerate(records)
            if all(record["metadata"].get(key) == value for key, value in benchmark["filter"].items())
        ]
        top_ids = sorted(candidates, key=lambda record_index: float(scores[record_index]), reverse=True)[:3]

        print("\n" + "=" * 90)
        print(f"Q{index}: {benchmark['query']}")
        print(f"Gold: {benchmark['gold']}")
        print(f"Filter: {benchmark['filter']}")

        relevant_in_top3 = False
        for rank, record_index in enumerate(top_ids, start=1):
            record = records[record_index]
            content = " ".join(record["content"].split())
            matched = find_matches(content, benchmark["needles"])
            if matched:
                relevant_in_top3 = True
            print(
                f"Top {rank}: score={float(scores[record_index]):.3f} "
                f"source={record['metadata']['source']} chunk={record['metadata']['chunk_index']} "
                f"matched={matched or '-'}"
            )
            print(f"Preview: {content[:420]}")

        relevant_count += int(relevant_in_top3)
        print(f"Relevant in top-3: {'YES' if relevant_in_top3 else 'NO'}")

    print("\n" + "=" * 90)
    print(f"Top-3 relevant: {relevant_count}/{len(BENCHMARKS)}")


if __name__ == "__main__":
    main()
