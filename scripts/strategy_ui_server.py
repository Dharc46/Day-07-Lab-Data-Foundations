from __future__ import annotations

import json
import pickle
import re
import sys
from hashlib import md5
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import FixedSizeChunker, LocalEmbedder, RecursiveChunker, SentenceChunker


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "group_legal_docs"
CACHE_DIR = ROOT / ".cache" / "strategy_ui"
MODEL_NAME = "AITeamVN/Vietnamese_Embedding"


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
        "id": 1,
        "query": "Dữ liệu cá nhân nhạy cảm bao gồm những gì?",
        "gold": "12 loại: nguồn gốc chủng tộc/dân tộc, quan điểm chính trị/tôn giáo, đời sống riêng tư, sức khỏe, sinh trắc học, di truyền, đời sống tình dục, dữ liệu tội phạm, vị trí, tên đăng nhập/mật khẩu, tài chính/tín dụng, hành vi trên mạng (NĐ 356/2025, Điều 4)",
        "filter": {"law_no": "356/2025/NĐ-CP"},
        "needles": ["Điều 4", "Dữ liệu cá nhân nhạy cảm", "nguồn gốc chủng tộc", "tài khoản ngân hàng"],
    },
    {
        "id": 2,
        "query": "Trách nhiệm của doanh nghiệp nước ngoài xử lý dữ liệu người Việt?",
        "gold": "Phải tuân thủ Luật BVDLCN Việt Nam nếu trực tiếp tham gia hoặc liên quan đến xử lý DLCN của công dân VN và người gốc Việt (Luật 91/2025, Điều 1 Khoản 2c)",
        "filter": {"law_no": "91/2025/QH15"},
        "needles": ["Điều 1", "nước ngoài", "công dân Việt Nam", "người gốc Việt"],
    },
    {
        "id": 3,
        "query": "Chữ ký điện tử có giá trị pháp lý không?",
        "gold": "Có; chữ ký điện tử chuyên dùng bảo đảm an toàn hoặc chữ ký số có giá trị pháp lý tương đương chữ ký tay trên văn bản giấy (Luật GDĐT 20/2023, Điều 23 Khoản 2)",
        "filter": {"law_no": "20/2023/QH15"},
        "needles": ["Điều 23", "giá trị pháp lý", "chữ ký tay", "chữ ký số"],
    },
    {
        "id": 4,
        "query": "Mức phạt vi phạm bảo vệ dữ liệu cá nhân?",
        "gold": "Phạt tiền tối đa 03 tỷ đồng cho vi phạm hành chính trong lĩnh vực BVDLCN (Luật 91/2025, Điều 8 Khoản 5)",
        "filter": {"law_no": "91/2025/QH15"},
        "needles": ["Điều 8", "03 tỷ đồng", "vi phạm hành chính", "phạt tiền"],
    },
    {
        "id": 5,
        "query": "Quyền của chủ thể dữ liệu cá nhân gồm những gì?",
        "gold": "6 nhóm quyền: được biết, đồng ý/rút lại đồng ý, xem/chỉnh sửa, yêu cầu cung cấp/xóa/hạn chế/phản đối, khiếu nại/khởi kiện, yêu cầu cơ quan bảo vệ DLCN (Luật 91/2025, Điều 4 Khoản 1)",
        "filter": {"law_no": "91/2025/QH15"},
        "needles": ["Điều 4", "Quyền", "Được biết", "Đồng ý", "khiếu nại"],
    },
]


@dataclass(frozen=True)
class StrategyConfig:
    key: str
    member: str
    label: str
    description: str


STRATEGIES = [
    StrategyConfig(
        key="fixed",
        member="Member 1",
        label="FixedSizeChunker",
        description="Fixed-size chunks, 1800 chars, 180 overlap.",
    ),
    StrategyConfig(
        key="sentence",
        member="Member 2",
        label="SentenceChunker",
        description="Groups up to 8 detected sentences per chunk.",
    ),
    StrategyConfig(
        key="recursive",
        member="Member 3",
        label="RecursiveChunker",
        description="Splits by paragraph, line, sentence, word, then character.",
    ),
    StrategyConfig(
        key="article",
        member="Member 4",
        label="ArticleAwareChunker",
        description="Splits legal text by Markdown article headings like '#### Điều X'.",
    ),
]


class ArticleAwareChunker:
    def __init__(self, chunk_size: int = 1800) -> None:
        self.chunk_size = chunk_size
        self.fallback = RecursiveChunker(chunk_size=chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []
        matches = list(re.finditer(r"(?m)^####\s+Điều\s+\d+[^\n]*", text))
        if not matches:
            return self.fallback.chunk(text)

        chunks: list[str] = []
        if matches[0].start() > 0:
            chunks.extend(self.fallback.chunk(text[: matches[0].start()].strip()))

        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            article = text[match.start() : end].strip()
            if len(article) <= self.chunk_size:
                chunks.append(article)
            else:
                chunks.extend(self.fallback.chunk(article))
        return [chunk for chunk in chunks if chunk.strip()]


def make_chunker(strategy_key: str):
    if strategy_key == "fixed":
        return FixedSizeChunker(chunk_size=1800, overlap=180)
    if strategy_key == "sentence":
        return SentenceChunker(max_sentences_per_chunk=8)
    if strategy_key == "recursive":
        return RecursiveChunker(chunk_size=1800)
    if strategy_key == "article":
        return ArticleAwareChunker(chunk_size=1800)
    raise ValueError(f"Unknown strategy: {strategy_key}")


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


def is_article_needle(needle: str) -> bool:
    return re.search(r"\bĐiều\s+\d+\b", needle, flags=re.IGNORECASE) is not None


def score_matches(
    matches: list[str],
    needles: list[str],
    exact_article: bool,
) -> tuple[float, float, bool]:
    if not needles:
        return 0.0, 0.0, False

    content_needles = [needle for needle in needles if not is_article_needle(needle)]
    content_matches = [match for match in matches if not is_article_needle(match)]
    coverage = len(matches) / len(needles)
    content_coverage = len(content_matches) / len(content_needles) if content_needles else coverage

    is_relevant = (
        coverage >= 0.5
        or content_coverage >= 0.5
        or (exact_article and bool(content_matches))
    )
    return coverage, content_coverage, is_relevant


def article_number_from_content(content: str) -> int | None:
    match = re.search(r"####\s+Điều\s+(\d+)", content)
    if not match:
        return None
    return int(match.group(1))


def expected_article_from_needles(needles: list[str]) -> int | None:
    for needle in needles:
        match = re.search(r"Điều\s+(\d+)", needle)
        if match:
            return int(match.group(1))
    return None


class BenchmarkEngine:
    def __init__(self) -> None:
        self.embedder: LocalEmbedder | None = None

    def get_embedder(self) -> LocalEmbedder:
        if self.embedder is None:
            self.embedder = LocalEmbedder(MODEL_NAME)
        return self.embedder

    def build_records(self, strategy_key: str) -> list[dict]:
        chunker = make_chunker(strategy_key)
        records = []
        for filename, meta in DOCS:
            text = (DATA_DIR / filename).read_text(encoding="utf-8")
            for chunk_index, chunk in enumerate(chunker.chunk(text)):
                records.append(
                    {
                        "content": chunk,
                        "metadata": {
                            **meta,
                            "source": filename,
                            "chunk_index": chunk_index,
                            "article_number": article_number_from_content(chunk),
                        },
                    }
                )
        return records

    def cache_path(self, strategy_key: str) -> Path:
        return CACHE_DIR / f"{strategy_key}_{MODEL_NAME.replace('/', '_')}.pkl"

    def cache_signature(self, strategy_key: str) -> str:
        parts = [MODEL_NAME, strategy_key, "relevance-v2"]
        for filename, _meta in DOCS:
            path = DATA_DIR / filename
            stat = path.stat()
            parts.append(f"{filename}:{stat.st_size}:{stat.st_mtime_ns}")
        return md5("|".join(parts).encode("utf-8")).hexdigest()

    def load_index(self, strategy_key: str) -> tuple[list[dict], np.ndarray]:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_path(strategy_key)
        signature = self.cache_signature(strategy_key)
        if cache_path.exists():
            with cache_path.open("rb") as cache_file:
                payload = pickle.load(cache_file)
            if payload.get("signature") == signature or (
                "signature" not in payload
                and "records" in payload
                and "embeddings" in payload
            ):
                return payload["records"], payload["embeddings"]

        records = self.build_records(strategy_key)
        embedder = self.get_embedder()
        embeddings = np.asarray(
            embedder.model.encode(
                [record["content"] for record in records],
                normalize_embeddings=True,
                batch_size=16,
                show_progress_bar=False,
            )
        )
        with cache_path.open("wb") as cache_file:
            pickle.dump({"signature": signature, "records": records, "embeddings": embeddings}, cache_file)
        return records, embeddings

    def normalize_benchmarks(self, benchmarks: list[dict] | None) -> list[dict]:
        source = benchmarks or BENCHMARKS
        normalized = []
        for index, benchmark in enumerate(source, start=1):
            query = str(benchmark.get("query", "")).strip()
            if not query:
                continue
            needles = benchmark.get("needles", [])
            if isinstance(needles, str):
                needles = [part.strip() for part in re.split(r"[\n,;]+", needles) if part.strip()]
            normalized.append(
                {
                    "id": benchmark.get("id", index),
                    "query": query,
                    "gold": str(benchmark.get("gold", "")).strip(),
                    "filter": dict(benchmark.get("filter") or {}),
                    "needles": [str(needle).strip() for needle in needles if str(needle).strip()],
                }
            )
        if not normalized:
            raise ValueError("At least one benchmark query is required.")
        return normalized

    def build_answer(self, query: dict, results: list[dict]) -> str:
        if not results:
            return "Không tìm thấy ngữ cảnh phù hợp trong knowledge base."

        best = next((result for result in results if result["relevant"]), results[0])
        source = best["source"]
        article = f" Điều {best['article_number']}" if best.get("article_number") else ""
        evidence = best["preview"]
        return (
            f"Dựa trên {source}{article}, câu trả lời liên quan nhất là: {evidence} "
            "Cần đối chiếu gold answer và văn bản gốc để xác nhận chi tiết pháp lý."
        )

    def run_strategy(
        self,
        strategy: StrategyConfig,
        benchmarks: list[dict],
        query_embeddings: np.ndarray,
    ) -> dict:
        records, embeddings = self.load_index(strategy.key)
        per_query = []
        top3_relevant = 0
        top1_relevant = 0
        exact_article_hits = 0
        total_coverage = 0.0
        total_best_length = 0
        total_mrr = 0.0

        for benchmark, query_embedding in zip(benchmarks, query_embeddings):
            scores = embeddings @ query_embedding
            candidates = [
                index
                for index, record in enumerate(records)
                if all(record["metadata"].get(key) == value for key, value in benchmark["filter"].items())
            ]
            ranked = sorted(candidates, key=lambda index: float(scores[index]), reverse=True)
            top_ids = ranked[:3]
            results = []
            first_relevant_rank = None
            expected_article = expected_article_from_needles(benchmark["needles"])
            top1_exact_article = False
            top1_coverage = 0.0
            top1_length = 0

            for rank, record_index in enumerate(top_ids, start=1):
                record = records[record_index]
                content = " ".join(record["content"].split())
                matches = find_matches(content, benchmark["needles"])
                exact_article = (
                    expected_article is not None
                    and record["metadata"].get("article_number") == expected_article
                )
                coverage, content_coverage, is_relevant = score_matches(
                    matches,
                    benchmark["needles"],
                    exact_article,
                )
                if is_relevant and first_relevant_rank is None:
                    first_relevant_rank = rank
                if rank == 1:
                    top1_exact_article = exact_article
                    top1_coverage = coverage
                    top1_length = len(content)
                results.append(
                    {
                        "rank": rank,
                        "score": round(float(scores[record_index]), 3),
                        "source": record["metadata"]["source"],
                        "chunk_index": record["metadata"]["chunk_index"],
                        "article_number": record["metadata"].get("article_number"),
                        "exact_article": exact_article,
                        "coverage": round(coverage, 3),
                        "content_coverage": round(content_coverage, 3),
                        "matches": matches,
                        "relevant": is_relevant,
                        "preview": content[:520],
                    }
                )

            if first_relevant_rank is not None:
                top3_relevant += 1
                total_mrr += 1.0 / first_relevant_rank
                if first_relevant_rank == 1:
                    top1_relevant += 1
            if top1_exact_article:
                exact_article_hits += 1
            total_coverage += top1_coverage
            total_best_length += top1_length

            per_query.append(
                {
                    "id": benchmark["id"],
                    "query": benchmark["query"],
                    "gold": benchmark["gold"],
                    "filter": benchmark["filter"],
                    "results": results,
                    "answer": self.build_answer(benchmark, results),
                    "relevant_in_top3": first_relevant_rank is not None,
                    "first_relevant_rank": first_relevant_rank,
                    "expected_article": expected_article,
                    "top1_exact_article": top1_exact_article,
                    "top1_coverage": round(top1_coverage, 3),
                    "top1_length": top1_length,
                }
            )

        query_count = len(benchmarks)
        top3_rate = top3_relevant / query_count
        top1_rate = top1_relevant / query_count
        exact_article_rate = exact_article_hits / query_count
        coverage_rate = total_coverage / query_count
        mrr = total_mrr / query_count
        avg_best_length = total_best_length / query_count
        length_quality = max(0.0, min(1.0, 1.0 - ((avg_best_length - 1800) / 3600)))
        score = (
            (top3_rate * 25)
            + (top1_rate * 20)
            + (exact_article_rate * 25)
            + (coverage_rate * 20)
            + (mrr * 5)
            + (length_quality * 5)
        )
        return {
            "key": strategy.key,
            "member": strategy.member,
            "label": strategy.label,
            "description": strategy.description,
            "chunk_count": len(records),
            "query_count": query_count,
            "top3_relevant": top3_relevant,
            "top1_relevant": top1_relevant,
            "exact_article_hits": exact_article_hits,
            "top3_rate": round(top3_rate, 3),
            "top1_rate": round(top1_rate, 3),
            "exact_article_rate": round(exact_article_rate, 3),
            "coverage_rate": round(coverage_rate, 3),
            "mrr": round(mrr, 3),
            "avg_best_length": round(avg_best_length),
            "length_quality": round(length_quality, 3),
            "score": round(score, 1),
            "queries": per_query,
        }

    def run(self, strategy_keys: list[str] | None = None, benchmarks: list[dict] | None = None) -> dict:
        selected = [strategy for strategy in STRATEGIES if strategy_keys is None or strategy.key in strategy_keys]
        benchmark_items = self.normalize_benchmarks(benchmarks)
        embedder = self.get_embedder()
        query_embeddings = np.asarray(
            embedder.model.encode(
                [benchmark["query"] for benchmark in benchmark_items],
                normalize_embeddings=True,
                batch_size=16,
                show_progress_bar=False,
            )
        )
        results = [
            self.run_strategy(strategy, benchmarks=benchmark_items, query_embeddings=query_embeddings)
            for strategy in selected
        ]
        ranked = sorted(results, key=lambda item: item["score"], reverse=True)
        return {
            "model": MODEL_NAME,
            "benchmarks": benchmark_items,
            "results": results,
            "ranking": ranked,
            "best": ranked[0] if ranked else None,
        }


ENGINE = BenchmarkEngine()


INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Strategy Benchmark Console</title>
  <style>
    :root {
      --bg: #f7f7f4;
      --ink: #202124;
      --muted: #676b73;
      --line: #d8d7d0;
      --panel: #ffffff;
      --accent: #0f766e;
      --accent-2: #8a4b0f;
      --bad: #9f1239;
      --good: #166534;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--ink);
      letter-spacing: 0;
    }
    header {
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      position: sticky;
      top: 0;
      z-index: 10;
    }
    h1 {
      font-size: 20px;
      margin: 0;
      line-height: 1.2;
    }
    .subtle { color: var(--muted); font-size: 13px; }
    main { padding: 20px 24px 40px; max-width: 1440px; margin: 0 auto; }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      margin-bottom: 16px;
    }
    button {
      border: 1px solid #0b625d;
      background: var(--accent);
      color: white;
      height: 36px;
      padding: 0 14px;
      border-radius: 6px;
      font-weight: 700;
      cursor: pointer;
    }
    button.secondary {
      background: white;
      color: var(--accent);
      border-color: var(--line);
    }
    button:disabled { opacity: 0.55; cursor: wait; }
    .grid {
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .panel h2 {
      font-size: 15px;
      margin: 0;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fbfbfa;
    }
    .panel-body { padding: 14px; }
    textarea {
      width: 100%;
      min-height: 84px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      font-family: Arial, Helvetica, sans-serif;
      font-size: 13px;
      line-height: 1.45;
      color: var(--ink);
      background: #fff;
    }
    .benchmark-list { display: grid; gap: 8px; }
    label.benchmark {
      display: grid;
      grid-template-columns: 20px 1fr;
      gap: 8px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      cursor: pointer;
    }
    label.benchmark strong { display: block; font-size: 13px; line-height: 1.35; }
    select {
      width: 100%;
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 0 8px;
      margin-top: 8px;
    }
    .strategy-list { display: grid; gap: 10px; }
    label.strategy {
      display: grid;
      grid-template-columns: 20px 1fr;
      gap: 8px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      cursor: pointer;
    }
    label.strategy strong { display: block; font-size: 14px; }
    label.strategy span { display: block; margin-top: 2px; }
    .status {
      min-height: 38px;
      display: flex;
      align-items: center;
      color: var(--muted);
      font-size: 14px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      text-align: left;
      padding: 10px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }
    th { background: #fbfbfa; font-size: 12px; color: #42464d; }
    .pill {
      display: inline-flex;
      align-items: center;
      height: 22px;
      padding: 0 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      background: #ece9df;
      color: #3b3326;
      white-space: nowrap;
    }
    .pill.good { background: #dcfce7; color: var(--good); }
    .pill.bad { background: #ffe4e6; color: var(--bad); }
    .cards { display: grid; gap: 12px; margin-top: 16px; }
    .result-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      overflow: hidden;
    }
    .result-head {
      padding: 12px 14px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid var(--line);
      background: #fbfbfa;
    }
    .result-head strong { font-size: 15px; }
    .query {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
    }
    .query:last-child { border-bottom: none; }
    .query-title {
      display: flex;
      gap: 10px;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 8px;
    }
    .gold { color: var(--muted); margin: 4px 0 8px; line-height: 1.45; }
    .answer {
      margin: 8px 0;
      padding: 10px;
      border-left: 3px solid var(--accent);
      background: #f1f8f6;
      line-height: 1.45;
    }
    .retrieved {
      display: grid;
      gap: 8px;
      margin-top: 8px;
    }
    .chunk {
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
    }
    .chunk-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 12px;
    }
    .preview {
      color: #2f3338;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }
    .best {
      border-left: 4px solid var(--accent);
    }
    @media (max-width: 900px) {
      header { position: static; align-items: flex-start; flex-direction: column; }
      .grid { grid-template-columns: 1fr; }
      main { padding: 14px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Vietnamese Legal RAG Strategy Benchmark</h1>
      <div class="subtle">Compare four member strategies on the shared 5-query benchmark.</div>
    </div>
    <div class="subtle">Model: AITeamVN/Vietnamese_Embedding</div>
  </header>
  <main>
    <div class="toolbar">
      <button id="runAll">Run Selected Strategies</button>
      <button id="selectAll" class="secondary">Select All</button>
      <button id="clear" class="secondary">Clear</button>
      <div id="status" class="status">Ready. First run can take several minutes while embeddings are cached.</div>
    </div>
    <div class="grid">
      <aside class="panel">
        <h2>Members And Strategies</h2>
        <div class="panel-body">
          <div class="strategy-list" id="strategyList"></div>
        </div>
        <h2>Benchmark Questions</h2>
        <div class="panel-body">
          <div class="subtle" style="margin-bottom:8px">
            Select one or more prepared benchmark questions.
          </div>
          <div class="benchmark-list" id="benchmarkList"></div>
          <div class="toolbar" style="margin-top:10px;margin-bottom:0">
            <button id="selectBenchmarks" class="secondary">Select All</button>
            <button id="clearBenchmarks" class="secondary">Clear</button>
          </div>
        </div>
        <h2>Chat Question</h2>
        <div class="panel-body">
          <div class="subtle" style="margin-bottom:8px">
            Optional: type one extra question. The selected source filter helps compare strategies.
          </div>
          <textarea id="chatQuestion" placeholder="Nhập câu hỏi của bạn..."></textarea>
          <select id="chatFilter">
            <option value="">Auto source filter</option>
            <option value="356/2025/NĐ-CP">NĐ 356/2025/NĐ-CP</option>
            <option value="91/2025/QH15">Luật 91/2025/QH15</option>
            <option value="20/2023/QH15">Luật GDĐT 20/2023/QH15</option>
            <option value="24/2018/QH14">Luật An ninh mạng 24/2018/QH14</option>
            <option value="13/2023/NĐ-CP">NĐ 13/2023/NĐ-CP</option>
          </select>
        </div>
      </aside>
      <section>
        <div class="panel">
          <h2>Ranking</h2>
          <div class="panel-body">
            <div id="rankingEmpty" class="subtle">Run the benchmark to see the best strategy.</div>
            <div class="subtle" id="rankingNote" hidden style="margin-bottom:10px">
              Score is normalized to 100: Top-3 25%, Top-1 20%, exact article 25%, keyword coverage 20%, MRR 5%, concise chunk 5%. Article-only matches are not counted as relevant.
            </div>
            <table id="rankingTable" hidden>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Member</th>
                  <th>Strategy</th>
                  <th>Chunks</th>
                  <th>Top-3</th>
                  <th>Top-1</th>
                  <th>Exact Article</th>
                  <th>Coverage</th>
                  <th>Avg Len</th>
                  <th>MRR</th>
                  <th>Score</th>
                </tr>
              </thead>
              <tbody id="rankingBody"></tbody>
            </table>
          </div>
        </div>
        <div id="results" class="cards"></div>
      </section>
    </div>
  </main>
  <script>
    const strategies = [
      { key: "fixed", member: "Member 1", label: "FixedSizeChunker", description: "Fixed-size chunks, 1800 chars, 180 overlap." },
      { key: "sentence", member: "Member 2", label: "SentenceChunker", description: "Groups up to 8 detected sentences per chunk." },
      { key: "recursive", member: "Member 3", label: "RecursiveChunker", description: "Paragraph/line/sentence/word fallback." },
      { key: "article", member: "Member 4", label: "ArticleAwareChunker", description: "Splits by legal article headings." }
    ];

    const strategyList = document.getElementById("strategyList");
    const statusEl = document.getElementById("status");
    const runAll = document.getElementById("runAll");
    const benchmarkList = document.getElementById("benchmarkList");
    const chatQuestion = document.getElementById("chatQuestion");
    const chatFilter = document.getElementById("chatFilter");
    let defaultBenchmarks = [];

    function renderStrategies() {
      strategyList.innerHTML = strategies.map((s) => `
        <label class="strategy">
          <input type="checkbox" value="${s.key}" checked />
          <span>
            <strong>${s.member}: ${s.label}</strong>
            <span class="subtle">${s.description}</span>
          </span>
        </label>
      `).join("");
    }

    async function loadDefaultBenchmarks() {
      const response = await fetch("/api/benchmarks");
      const data = await response.json();
      defaultBenchmarks = data.benchmarks;
      renderBenchmarkChoices();
    }

    function renderBenchmarkChoices() {
      benchmarkList.innerHTML = defaultBenchmarks.map((benchmark, index) => `
        <label class="benchmark">
          <input type="checkbox" value="${index}" checked />
          <span>
            <strong>Q${benchmark.id}. ${benchmark.query}</strong>
            <span class="subtle">${benchmark.gold}</span>
          </span>
        </label>
      `).join("");
    }

    function selectedKeys() {
      return Array.from(strategyList.querySelectorAll("input:checked")).map((input) => input.value);
    }

    function setBusy(isBusy) {
      runAll.disabled = isBusy;
      document.getElementById("selectAll").disabled = isBusy;
      document.getElementById("clear").disabled = isBusy;
      document.getElementById("selectBenchmarks").disabled = isBusy;
      document.getElementById("clearBenchmarks").disabled = isBusy;
    }

    function selectedBenchmarks() {
      const selected = Array.from(benchmarkList.querySelectorAll("input:checked"))
        .map((input) => defaultBenchmarks[Number(input.value)])
        .filter(Boolean);

      const customQuestion = chatQuestion.value.trim();
      if (customQuestion) {
        selected.push(makeCustomBenchmark(customQuestion));
      }
      return selected;
    }

    function inferLawNo(question) {
      const text = question.toLowerCase();
      if (chatFilter.value) return chatFilter.value;
      if (text.includes("chữ ký") || text.includes("giao dịch điện tử") || text.includes("hợp đồng điện tử")) {
        return "20/2023/QH15";
      }
      if (text.includes("an ninh mạng") || text.includes("không gian mạng")) {
        return "24/2018/QH14";
      }
      if (text.includes("nhạy cảm") || text.includes("dữ liệu cá nhân cơ bản")) {
        return "356/2025/NĐ-CP";
      }
      if (text.includes("dữ liệu cá nhân") || text.includes("chủ thể dữ liệu") || text.includes("mức phạt")) {
        return "91/2025/QH15";
      }
      return "";
    }

    function makeCustomBenchmark(question) {
      const lawNo = inferLawNo(question);
      const words = question
        .split(/\s+/)
        .map((word) => word.replace(/[^\p{L}\p{N}]/gu, ""))
        .filter((word) => word.length >= 4)
        .slice(0, 8);
      return {
        id: "Chat",
        query: question,
        gold: "Custom chat question",
        filter: lawNo ? { law_no: lawNo } : {},
        needles: words
      };
    }

    function pill(value, good) {
      return `<span class="pill ${good ? "good" : "bad"}">${value}</span>`;
    }

    function renderRanking(data) {
      const rankingTable = document.getElementById("rankingTable");
      const rankingEmpty = document.getElementById("rankingEmpty");
      const rankingBody = document.getElementById("rankingBody");
      document.getElementById("rankingNote").hidden = false;
      rankingEmpty.hidden = true;
      rankingTable.hidden = false;
      rankingBody.innerHTML = data.ranking.map((r, index) => `
        <tr>
          <td>${index + 1}</td>
          <td>${r.member}</td>
          <td><strong>${r.label}</strong><br><span class="subtle">${r.description}</span></td>
          <td>${r.chunk_count}</td>
          <td>${r.top3_relevant}/${r.query_count}</td>
          <td>${r.top1_relevant}/${r.query_count}</td>
          <td>${r.exact_article_hits}/${r.query_count}</td>
          <td>${Math.round(r.coverage_rate * 100)}%</td>
          <td>${r.avg_best_length}</td>
          <td>${r.mrr}</td>
          <td><strong>${r.score}/100</strong></td>
        </tr>
      `).join("");
    }

    function renderResults(data) {
      const bestKey = data.best ? data.best.key : "";
      document.getElementById("results").innerHTML = data.results.map((strategy) => `
        <article class="result-card ${strategy.key === bestKey ? "best" : ""}">
          <div class="result-head">
            <div>
              <strong>${strategy.member}: ${strategy.label}</strong>
              <div class="subtle">${strategy.description}</div>
            </div>
            <div>
              ${pill(`Top-3 ${strategy.top3_relevant}/${strategy.query_count}`, strategy.top3_relevant >= Math.ceil(strategy.query_count * 0.8))}
              ${pill(`Top-1 ${strategy.top1_relevant}/${strategy.query_count}`, strategy.top1_relevant >= Math.ceil(strategy.query_count * 0.8))}
            </div>
          </div>
          ${strategy.queries.map((q) => `
            <section class="query">
              <div class="query-title">
                <div>
                  <strong>Q${q.id}. ${q.query}</strong>
                  <div class="gold">Gold: ${q.gold}</div>
                  <div class="subtle">Filter: ${JSON.stringify(q.filter)}</div>
                </div>
                ${pill(q.relevant_in_top3 ? "Relevant" : "Miss", q.relevant_in_top3)}
              </div>
              <div class="answer"><strong>Answer:</strong> ${q.answer}</div>
              <div class="retrieved">
                ${q.results.map((r) => `
                  <div class="chunk">
                    <div class="chunk-meta">
                      <span>Top ${r.rank}</span>
                      <span>score ${r.score}</span>
                      <span>${r.source}</span>
                      <span>chunk ${r.chunk_index}</span>
                      ${r.article_number ? `<span>Điều ${r.article_number}</span>` : ""}
                      <span>coverage ${Math.round(r.coverage * 100)}%</span>
                      <span>${r.exact_article ? "exact article" : "article miss"}</span>
                      <span>${r.matches.length ? "matched: " + r.matches.join(", ") : "matched: -"}</span>
                    </div>
                    <div class="preview">${r.preview}</div>
                  </div>
                `).join("")}
              </div>
            </section>
          `).join("")}
        </article>
      `).join("");
    }

    async function runBenchmark() {
      const keys = selectedKeys();
      if (!keys.length) {
        statusEl.textContent = "Select at least one strategy.";
        return;
      }
      setBusy(true);
      statusEl.textContent = "Running benchmark. This can take a while on the first run.";
      try {
        const benchmarks = selectedBenchmarks();
        if (!benchmarks.length) {
          throw new Error("Select at least one benchmark or type a chat question.");
        }
        const response = await fetch("/api/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ strategies: keys, benchmarks })
        });
        if (!response.ok) {
          throw new Error(await response.text());
        }
        const data = await response.json();
        renderRanking(data);
        renderResults(data);
        statusEl.textContent = `Done. Best strategy: ${data.best.member} - ${data.best.label}.`;
      } catch (error) {
        statusEl.textContent = `Error: ${error.message}`;
      } finally {
        setBusy(false);
      }
    }

    document.getElementById("selectAll").addEventListener("click", () => {
      strategyList.querySelectorAll("input").forEach((input) => { input.checked = true; });
    });
    document.getElementById("clear").addEventListener("click", () => {
      strategyList.querySelectorAll("input").forEach((input) => { input.checked = false; });
    });
    document.getElementById("selectBenchmarks").addEventListener("click", () => {
      benchmarkList.querySelectorAll("input").forEach((input) => { input.checked = true; });
    });
    document.getElementById("clearBenchmarks").addEventListener("click", () => {
      benchmarkList.querySelectorAll("input").forEach((input) => { input.checked = false; });
    });
    runAll.addEventListener("click", runBenchmark);
    renderStrategies();
    loadDefaultBenchmarks();
  </script>
</body>
</html>
"""


class StrategyUiHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(INDEX_HTML)
            return
        if parsed.path == "/api/benchmarks":
            self._send_json({"benchmarks": BENCHMARKS})
            return
        if parsed.path == "/api/run":
            query = parse_qs(parsed.query)
            raw_keys = query.get("strategies", [""])[0]
            strategy_keys = [key for key in raw_keys.split(",") if key] or None
            try:
                self._send_json(ENGINE.run(strategy_keys=strategy_keys))
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self.send_error(404, "Not found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(body or "{}")
            strategy_keys = payload.get("strategies") or None
            benchmarks = payload.get("benchmarks") or None
            self._send_json(ENGINE.run(strategy_keys=strategy_keys, benchmarks=benchmarks))
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = ThreadingHTTPServer(("127.0.0.1", port), StrategyUiHandler)
    print(f"Strategy UI running at http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
