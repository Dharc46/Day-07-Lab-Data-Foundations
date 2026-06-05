from __future__ import annotations

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.embeddings import LocalEmbedder
from src.chunking import compute_similarity

embedder = LocalEmbedder("AITeamVN/Vietnamese_Embedding")

pairs = [
    (
        "Dữ liệu cá nhân nhạy cảm cần được bảo vệ nghiêm ngặt",
        "Thông tin riêng tư nhạy cảm phải được bảo mật chặt chẽ",
        "high",
    ),
    (
        "Chữ ký điện tử có giá trị pháp lý tương đương chữ ký tay",
        "Hợp đồng điện tử được pháp luật công nhận hiệu lực",
        "high",
    ),
    (
        "Mức phạt vi phạm bảo vệ dữ liệu cá nhân tối đa 03 tỷ đồng",
        "Thời tiết hôm nay nắng đẹp, nhiệt độ khoảng 30 độ C",
        "low",
    ),
    (
        "Quyền của chủ thể dữ liệu bao gồm quyền được biết và quyền đồng ý",
        "Quyền riêng tư của người dùng bao gồm quyền truy cập và từ chối",
        "high",
    ),
    (
        "Doanh nghiệp nước ngoài xử lý dữ liệu phải tuân thủ luật Việt Nam",
        "Công ty nước ngoài thu thập thông tin người dùng Việt phải theo quy định pháp luật",
        "high",
    ),
]

print("=== Similarity Predictions ===\n")
for i, (a, b, prediction) in enumerate(pairs, 1):
    vec_a = embedder(a)
    vec_b = embedder(b)
    score = compute_similarity(vec_a, vec_b)
    actual = "high" if score >= 0.5 else "low"
    correct = "YES" if actual == prediction else "NO"
    print(f"Pair {i}: score={score:.4f}  predicted={prediction}  actual={actual}  correct={correct}")
    print(f"  A: {a}")
    print(f"  B: {b}")
    print()
