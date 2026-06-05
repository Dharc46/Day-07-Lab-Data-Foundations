# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Tài Khoa
**Nhóm:** E5
**Ngày:** 6/5/2026

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
> *Viết 1-2 câu:* Vì các đoạn văn có độ dài và ngắn khác nhau và euclidean distance sẽ không nhận ra sự tương đồng giữa 1 bài văn và 1 bản tóm tắt dù chúng dùng cùng nội dung

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

**Loại:** ParentChildChunker (custom)

**Mô tả cách hoạt động:**
> ParentChildChunker tạo 2 tầng chunk: parent là nguyên 1 Điều luật (tách bằng regex `#### Điều \d+`), child là từng nhóm 3 câu trong Điều đó (dùng SentenceChunker bên trong). Khi search, embedding match trên child chunk (nhỏ, chính xác), nhưng mỗi child giữ reference đến parent chunk (toàn bộ Điều) để trả về đầy đủ context. Kỹ thuật này phổ biến trong production RAG (LlamaIndex, LangChain đều có built-in).

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Văn bản luật Việt Nam có cấu trúc tự nhiên theo Điều — đây chính là đơn vị tham chiếu chuẩn trong pháp luật. Child chunk nhỏ giúp embedding chính xác hơn (avg 294-429 chars), trong khi parent chunk (avg 1058-2542 chars) giữ trọn ngữ cảnh Điều luật. Benchmark cho thấy ParentChildChunker thắng SentenceChunker(5) ở 4/5 queries với avg score cao hơn (0.6581 vs 0.6506).

**Code snippet (nếu custom):**
```python
class ParentChildChunker:
    def __init__(self, parent_separator=r"(?=####\s+Điều\s+\d+)", child_max_sentences=3):
        self.parent_separator = parent_separator
        self.child_chunker = SentenceChunker(max_sentences_per_chunk=child_max_sentences)

    def chunk(self, text):
        parents = re.split(self.parent_separator, text)
        results = []
        for parent in parents:
            children = self.child_chunker.chunk(parent)
            for child in children:
                results.append({"child": child, "parent": parent})
        return results
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| NĐ 13/2023 | best baseline (SentenceChunker default) | 215 | 315 | Chunk nhỏ, thiếu context khoản |
| NĐ 13/2023 | **ParentChild của tôi** | 230 children / 45 parents | child 294, parent 1509 | Tốt — child chính xác, parent giữ trọn Điều |
| Luật An ninh mạng | best baseline (SentenceChunker default) | 130 | 490 | Chunk vừa nhưng đôi khi tách khoản |
| Luật An ninh mạng | **ParentChild của tôi** | 148 children / 44 parents | child 429, parent 1448 | Tốt — top-3 đều từ đúng Điều luật |
| Luật GDĐT | best baseline (SentenceChunker default) | 155 | 368 | Chunk nhỏ, tách giữa điều khoản |
| Luật GDĐT | **ParentChild của tôi** | 178 children / 54 parents | child 320, parent 1058 | Tốt — Q3 score 0.6368 > baseline 0.6159 |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Avg Score | Top-3 Relevant | Điểm mạnh | Điểm yếu |
|-----------|----------|-----------|----------------|-----------|----------|
| Tôi (Nguyễn Tài Khoa) | ParentChildChunker (child=3 câu, parent=Điều) | 0.6581 | 5/5 | Child chunk chính xác, parent giữ trọn Điều để trả context đầy đủ | Q2 score thấp (0.5313), child có thể quá nhỏ cho query mơ hồ |
| Nguyễn Thanh Đạt | RecursiveChunker(1800) + metadata filter (topic, law_no, doc_type) | — | 5/5 | Metadata filter thu hẹp search space hiệu quả, chunk lớn giữ context | Chunk 1800 chars quá lớn, có thể chứa nhiều Điều trong 1 chunk |
| Nguyễn Khôi Lâm | Article-level chunking (tách theo `#### Điều`) + lexical/article boost | — | 5/5 | Top-1 đúng 5/5, tách đúng đơn vị Điều luật tự nhiên, có domain boost | Cần custom boost phức tạp, khó tái sử dụng cho domain khác |
| Mai Văn Thuyên | RecursiveChunker(500) | 0.6502 | 5/5 | Đơn giản, giữ cấu trúc đoạn tốt | Chunk không theo đơn vị Điều, thiếu context khi điều khoản dài |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Article-level chunking (Nguyễn Khôi Lâm) cho kết quả tốt nhất vì tách đúng theo đơn vị tham chiếu pháp lý tự nhiên (Điều). ParentChildChunker của tôi là bước tiến tương tự vì parent cũng là Điều, nhưng thêm tầng child nhỏ giúp embedding chính xác hơn. RecursiveChunker(1800) của Thanh Đạt kết hợp metadata filter cũng hiệu quả nhưng chunk quá lớn dễ lẫn nhiều Điều.

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
| 1 | Dữ liệu cá nhân nhạy cảm cần được bảo vệ nghiêm ngặt | Thông tin riêng tư nhạy cảm phải được bảo mật chặt chẽ | high | 0.8442 | Đúng |
| 2 | Chữ ký điện tử có giá trị pháp lý tương đương chữ ký tay | Hợp đồng điện tử được pháp luật công nhận hiệu lực | high | 0.5902 | Đúng |
| 3 | Mức phạt vi phạm bảo vệ dữ liệu cá nhân tối đa 03 tỷ đồng | Thời tiết hôm nay nắng đẹp, nhiệt độ khoảng 30 độ C | low | 0.0870 | Đúng |
| 4 | Quyền của chủ thể dữ liệu bao gồm quyền được biết và quyền đồng ý | Quyền riêng tư của người dùng bao gồm quyền truy cập và từ chối | high | 0.5832 | Đúng |
| 5 | Doanh nghiệp nước ngoài xử lý dữ liệu phải tuân thủ luật Việt Nam | Công ty nước ngoài thu thập thông tin người dùng Việt phải theo quy định pháp luật | high | 0.6239 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Pair 2 (0.5902) thấp hơn kỳ vọng — "chữ ký điện tử" và "hợp đồng điện tử" đều thuộc domain giao dịch điện tử nhưng model nhận ra đây là 2 khái niệm pháp lý khác nhau (chữ ký vs hợp đồng), không chỉ match keyword "điện tử". Điều này cho thấy Vietnamese_Embedding biểu diễn semantic meaning chứ không chỉ dựa trên từ vựng chung — nó phân biệt được ý nghĩa cụ thể ngay cả khi các câu cùng domain.

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
| 1 | Dữ liệu cá nhân nhạy cảm bao gồm những gì? | NĐ 356/2025 Điều 4: Danh mục DLCN nhạy cảm — liệt kê 12 loại (chủng tộc, chính trị, sức khỏe, sinh trắc học…). Parent chunk chứa trọn Điều 4. | 0.6716 | Có | Trả về đúng điều luật liệt kê 12 loại DLCN nhạy cảm |
| 2 | Trách nhiệm của DN nước ngoài xử lý dữ liệu người Việt? | Luật 91/2025: Trách nhiệm bên xử lý DLCN — phải có thỏa thuận, hợp đồng, thực hiện biện pháp bảo vệ. Parent chunk chứa trọn điều về trách nhiệm. | 0.5313 | Một phần | Trả về trách nhiệm bên xử lý DLCN chung, chưa đặc thù "nước ngoài" |
| 3 | Chữ ký điện tử có giá trị pháp lý không? | Luật GDĐT Điều 23: Chữ ký điện tử chuyên dùng/chữ ký số có giá trị pháp lý tương đương chữ ký tay. Parent chunk chứa trọn Điều 23. | 0.6368 | Có | Trả về đúng Điều 23 Khoản 2 về giá trị pháp lý |
| 4 | Mức phạt vi phạm bảo vệ dữ liệu cá nhân? | Luật 91/2025 Điều 8: Mức phạt tối đa, top-3 đều cùng Điều 8. Parent chunk chứa trọn điều xử phạt với mức 03 tỷ đồng. | 0.7038 | Có | Trả về đúng điều khoản xử phạt, child #3 chứa mức 03 tỷ |
| 5 | Quyền của chủ thể dữ liệu cá nhân gồm những gì? | Luật 91/2025 Điều 4: 6 nhóm quyền — được biết, đồng ý, xem/chỉnh sửa, yêu cầu xóa/hạn chế, khiếu nại, yêu cầu cơ quan BVDLCN. Parent chunk chứa trọn Điều 4. | 0.7468 | Có | Trả về đúng điều khoản liệt kê đầy đủ 6 nhóm quyền |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Nguyễn Khôi Lâm kết hợp lexical boost với semantic search — không chỉ dựa vào embedding mà còn tăng điểm cho chunk có heading chứa keyword query. Điều này giải quyết được weakness của pure semantic search khi query paraphrase khác từ vựng với văn bản gốc (ví dụ "doanh nghiệp nước ngoài" vs "cơ quan, tổ chức, cá nhân nước ngoài"). Nguyễn Thanh Đạt cho thấy RecursiveChunker(1800) kết hợp metadata filter (topic, law_no, doc_type) thu hẹp search space hiệu quả. Mai Văn Thuyên cho thấy RecursiveChunker đơn giản vẫn đạt kết quả tốt (5/5 relevant) nếu chunk_size phù hợp.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Nhóm chọn slide buổi học là chủ đề, dùng enriching để lấy add thêm metadata xem có đồ thị code.. chunking theo slide thì sẽ có vấn đề khi có 2 slide topic giống nhau, thì thêm filter. Có nhóm làm về luật zalo thì có gợi ý là dùng recursive có thể phù hợp với luật vì ta cần độ chính xác cao. 
**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ bổ sung metadata `article_number` và `chapter` ngay từ bước loading, thay vì chỉ có `doc_type`/`year`/`topic`. Metadata chi tiết hơn giúp filter trước khi search semantic, giảm false positive. Ngoài ra sẽ thêm lexical/BM25 score kết hợp với cosine similarity để xử lý tốt hơn các query có từ khóa pháp lý cụ thể.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 3 / 5 |
| **Tổng** | | **98 / 100** |
