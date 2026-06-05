# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Tên sinh viên]
**Nhóm:** [Tên nhóm]
**Ngày:** [Ngày nộp]

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
>  Nghĩa là hai vector có nội dung gần nhau


**Ví dụ HIGH similarity:**
- Sentence A: Con mèo ngồi trên thảm
- Sentence B: Một chú mèo nằm trên tấm thảm
- Tại sao tương đồng: Cùng chủ thể (mèo), cùng vị trí (trên thảm), chỉ khác cách diễn đạt

**Ví dụ LOW similarity:**
- Sentence A: Con mèo ngồi trên thảm
- Sentence B: Thị trường chứng khoán tăng 5% hôm nay
- Tại sao khác: Hoàn toàn khác domain, không có từ hay ý nghĩa chung

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Viết 1-2 câu:* Vì các đoạn văn có độ dài và ngắn khác nhau và euclidean distance sẽ không nhận ra sự tương đồng chữa 1 bài văn và 1 bản tóm tắt dù chúng dùng cùng nội dung

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = 23`
> *Đáp án:* 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25` → 25 chunks. Overlap nhiều hơn giúp giữ được context giữa các chunk liền kề, tránh mất thông tin ở ranh giới, đổi lại cần lưu trữ nhiều hơn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Vietnamese law — an ninh mạng, bảo vệ dữ liệu cá nhân, giao dịch điện tử

**Tại sao nhóm chọn domain này?**
> Luật Việt Nam là dữ liệu public domain, dễ tiếp cận và có cấu trúc rõ ràng theo Chương/Điều/Khoản. Nội dung pháp lý có tính chính xác cao nên dễ kiểm tra retrieval quality — gold answer có thể trích dẫn trực tiếp từ điều luật cụ thể. Các văn bản liên quan đến công nghệ (an ninh mạng, dữ liệu cá nhân, giao dịch điện tử) phù hợp với kiến thức ngành CNTT.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | NĐ 13/2023/NĐ-CP — Bảo vệ dữ liệu cá nhân | Chính phủ | 88,231 | doc_type: nghị_định, year: 2023, topic: bảo_vệ_dữ_liệu |
| 2 | Luật 24/2018/QH14 — An ninh mạng | Quốc hội | 82,915 | doc_type: luật, year: 2018, topic: an_ninh_mạng |
| 3 | Luật 20/2023/QH15 — Giao dịch điện tử | Quốc hội | 77,391 | doc_type: luật, year: 2023, topic: giao_dịch_điện_tử |
| 4 | NĐ 356/2025/NĐ-CP — Chi tiết Luật BVDLCN | Chính phủ | 147,603 | doc_type: nghị_định, year: 2025, topic: bảo_vệ_dữ_liệu |
| 5 | Luật 91/2025/QH15 — Bảo vệ dữ liệu cá nhân | Quốc hội | 71,777 | doc_type: luật, year: 2025, topic: bảo_vệ_dữ_liệu |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| doc_type | str | "luật" / "nghị_định" | Lọc theo loại văn bản — luật quy định nguyên tắc, nghị định quy định chi tiết |
| year | int | 2025 | Lọc theo năm ban hành — ưu tiên văn bản mới nhất khi có xung đột |
| topic | str | "bảo_vệ_dữ_liệu" | Lọc theo chủ đề — thu hẹp phạm vi tìm kiếm khi query thuộc domain cụ thể |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| NĐ 13/2023 | FixedSizeChunker (`fixed_size`) | 137 | 497 | Không — cắt giữa câu/điều |
| NĐ 13/2023 | SentenceChunker (`by_sentences`) | 215 | 315 | Tốt — giữ trọn câu |
| NĐ 13/2023 | RecursiveChunker (`recursive`) | 530 | 127 | Trung bình — chunk quá nhỏ |
| Luật An ninh mạng | FixedSizeChunker (`fixed_size`) | 128 | 499 | Không — cắt giữa câu/điều |
| Luật An ninh mạng | SentenceChunker (`by_sentences`) | 130 | 490 | Tốt — giữ trọn câu |
| Luật An ninh mạng | RecursiveChunker (`recursive`) | 893 | 70 | Kém — chunk quá nhỏ |
| Luật GDĐT | FixedSizeChunker (`fixed_size`) | 115 | 498 | Không — cắt giữa câu/điều |
| Luật GDĐT | SentenceChunker (`by_sentences`) | 155 | 368 | Tốt — giữ trọn câu |
| Luật GDĐT | RecursiveChunker (`recursive`) | 611 | 93 | Kém — chunk quá nhỏ |

### Strategy Của Tôi

**Loại:** SentenceChunker (tuned: `max_sentences_per_chunk=5`)

**Mô tả cách hoạt động:**
> SentenceChunker tách văn bản theo ranh giới câu (dấu `. `, `! `, `? `, `.\n`) rồi gom nhóm mỗi 5 câu thành 1 chunk. So với default (3 câu), 5 câu giữ được trọn vẹn ý của từng Khoản trong điều luật. Strategy này không cắt giữa câu nên mỗi chunk luôn có nghĩa hoàn chỉnh, phù hợp cho retrieval.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Văn bản luật Việt Nam có cấu trúc Điều → Khoản → Điểm, mỗi khoản thường gồm 3-6 câu. SentenceChunker với 5 câu/chunk giữ được context trọn vẹn của từng khoản. Baseline cho thấy RecursiveChunker tạo chunk quá nhỏ (70-127 chars) và FixedSizeChunker cắt giữa câu — cả hai đều mất context.

**Code snippet (nếu custom):**
```python
chunker = SentenceChunker(max_sentences_per_chunk=5)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| NĐ 13/2023 | best baseline (SentenceChunker default) | 215 | 315 | Chờ kết quả embeddings |
| NĐ 13/2023 | **SentenceChunker(5) của tôi** | — | — | Chờ kết quả embeddings |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | | | | |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng regex `re.split(r'(?<=[.!?])[\s]', text)` để tách câu dựa trên dấu chấm, chấm than, chấm hỏi theo sau bởi whitespace. Sau đó gom các câu thành nhóm theo `max_sentences_per_chunk`, strip whitespace và lọc bỏ chuỗi rỗng. Xử lý edge case text rỗng trả về list rỗng.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Algorithm thử từng separator theo thứ tự ưu tiên (paragraph → line → sentence → word → character). Base case: nếu `len(text) <= chunk_size` hoặc hết separator thì trả về `[text]`. Với separator rỗng `""`, cắt cứng theo `chunk_size`. Mỗi phần sau khi split nếu vẫn quá lớn sẽ recurse với danh sách separator còn lại.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Mỗi document được chuyển thành record dict chứa `id`, `content`, `metadata`, và `embedding` (vector từ embedding function), lưu vào list `self._store`. Khi search, embed query rồi tính dot product giữa query embedding và tất cả stored embeddings, sort giảm dần theo score và trả về top_k kết quả.

**`search_with_filter` + `delete_document`** — approach:
> Filter trước, search sau: lọc `self._store` theo metadata_filter (tất cả key-value phải match), rồi chạy similarity search trên tập đã lọc. Delete dùng list comprehension giữ lại các record có `id != doc_id`, so sánh length trước/sau để trả về True/False.

### KnowledgeBaseAgent

**`answer`** — approach:
> Gọi `store.search(question, top_k)` để lấy top-k chunks liên quan. Nối các chunk content bằng `"\n\n"` thành context block, rồi build prompt theo format `"Context:\n{context}\n\nQuestion: {question}\nAnswer:"`. Truyền prompt này vào `llm_fn` và trả về kết quả.

### Test Results

```
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

42 passed in 0.06s
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | | | high / low | | |
| 2 | | | high / low | | |
| 3 | | | high / low | | |
| 4 | | | high / low | | |
| 5 | | | high / low | | |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:*

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Dữ liệu cá nhân nhạy cảm bao gồm những gì? | 12 loại: nguồn gốc chủng tộc/dân tộc, quan điểm chính trị/tôn giáo, đời sống riêng tư, sức khỏe, sinh trắc học, di truyền, đời sống tình dục, dữ liệu tội phạm, vị trí, tên đăng nhập/mật khẩu, tài chính/tín dụng, hành vi trên mạng (NĐ 356/2025, Điều 4) |
| 2 | Trách nhiệm của doanh nghiệp nước ngoài xử lý dữ liệu người Việt? | Phải tuân thủ Luật BVDLCN Việt Nam nếu trực tiếp tham gia hoặc liên quan đến xử lý DLCN của công dân VN và người gốc Việt (Luật 91/2025, Điều 1 Khoản 2c) |
| 3 | Chữ ký điện tử có giá trị pháp lý không? | Có — chữ ký điện tử chuyên dùng bảo đảm an toàn hoặc chữ ký số có giá trị pháp lý tương đương chữ ký tay trên văn bản giấy (Luật GDĐT 20/2023, Điều 23 Khoản 2) |
| 4 | Mức phạt vi phạm bảo vệ dữ liệu cá nhân? | Phạt tiền tối đa 03 tỷ đồng cho vi phạm hành chính trong lĩnh vực BVDLCN (Luật 91/2025, Điều 8 Khoản 5) |
| 5 | Quyền của chủ thể dữ liệu cá nhân gồm những gì? | 6 nhóm quyền: được biết, đồng ý/rút lại đồng ý, xem/chỉnh sửa, yêu cầu cung cấp/xóa/hạn chế/phản đối, khiếu nại/khởi kiện, yêu cầu cơ quan bảo vệ DLCN (Luật 91/2025, Điều 4 Khoản 1) |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bao nhiêu queries trả về chunk relevant trong top-3?** __ / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
