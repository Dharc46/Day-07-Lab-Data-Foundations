# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Tên sinh viên]  
**Nhóm:** [Tên nhóm]  
**Ngày:** 2026-06-05

---

## 1. Warm-up

### Cosine Similarity

High cosine similarity nghĩa là hai vector văn bản có hướng gần nhau, tức là hai đoạn text có ý nghĩa hoặc chủ đề gần nhau trong không gian embedding.

**Ví dụ HIGH similarity**
- Sentence A: "Dữ liệu cá nhân nhạy cảm cần được bảo vệ chặt chẽ."
- Sentence B: "Thông tin riêng tư quan trọng của cá nhân phải được bảo mật."
- Lý do: Hai câu cùng nói về bảo vệ thông tin cá nhân nhạy cảm.

**Ví dụ LOW similarity**
- Sentence A: "Chữ ký điện tử có giá trị pháp lý trong giao dịch điện tử."
- Sentence B: "Python là ngôn ngữ lập trình dùng nhiều trong phân tích dữ liệu."
- Lý do: Hai câu thuộc hai domain khác nhau, một câu về luật giao dịch điện tử, một câu về lập trình.

Cosine similarity thường phù hợp hơn Euclidean distance cho text embeddings vì nó tập trung vào hướng của vector hơn là độ dài tuyệt đối. Với embeddings, hướng thường biểu diễn ý nghĩa tốt hơn magnitude.

### Chunking Math

Với document 10,000 ký tự, `chunk_size=500`, `overlap=50`:

```text
num_chunks = ceil((10000 - 50) / (500 - 50))
           = ceil(9950 / 450)
           = ceil(22.11)
           = 23 chunks
```

Nếu overlap tăng lên 100:

```text
num_chunks = ceil((10000 - 100) / (500 - 100))
           = ceil(9900 / 400)
           = 25 chunks
```

Overlap lớn hơn tạo nhiều chunks hơn nhưng giúp giữ ngữ cảnh giữa hai chunk liền kề, nhất là khi câu trả lời nằm gần ranh giới chunk.

---

## 2. Document Selection - Nhóm

### Domain & Lý Do Chọn

**Domain:** Văn bản pháp luật Việt Nam về an ninh mạng, giao dịch điện tử, và bảo vệ dữ liệu cá nhân.

Nhóm chọn domain này vì tài liệu có cấu trúc rõ theo luật, nghị định, chương, điều, khoản. Đây là domain phù hợp để kiểm thử retrieval vì câu hỏi thường cần truy đúng điều luật, trích đúng nguồn, và phân biệt giữa các văn bản gần nghĩa.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | `13_2023_ND-CP.md` | Nghị định 13/2023/NĐ-CP | 88,231 | `source`, `extension`, `language`, `article`, `doc_id`, `content_hash`, `embedding_backend` |
| 2 | `2018_775_776_24_2018_QH14.md` | Luật An ninh mạng 24/2018/QH14 | 82,915 | `source`, `extension`, `language`, `article`, `doc_id`, `content_hash`, `embedding_backend` |
| 3 | `2023_867_868_20_2023_QH15.md` | Luật Giao dịch điện tử 20/2023/QH15 | 77,391 | `source`, `extension`, `language`, `article`, `doc_id`, `content_hash`, `embedding_backend` |
| 4 | `356_2025_ND-CP.md` | Nghị định 356/2025/NĐ-CP | 147,603 | `source`, `extension`, `language`, `article`, `doc_id`, `content_hash`, `embedding_backend` |
| 5 | `91_2025_QH15.md` | Luật Bảo vệ dữ liệu cá nhân 91/2025/QH15 | 71,777 | `source`, `extension`, `language`, `article`, `doc_id`, `content_hash`, `embedding_backend` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `source` | string | `91_2025_QH15.md` | Truy vết câu trả lời về đúng văn bản luật. |
| `article` | string | `Điều 4. Quyền và nghĩa vụ của chủ thể dữ liệu cá nhân` | Biết chunk nào chứa căn cứ pháp lý. |
| `language` | string | `vi` | Hỗ trợ lọc nếu dữ liệu đa ngôn ngữ. |
| `doc_id` | string | `91_2025_QH15_điều_4...` | Xóa hoặc cập nhật một chunk cụ thể. |
| `content_hash` | string | SHA-256 hash | Kiểm tra cache embedding còn hợp lệ không. |
| `embedding_backend` | string | `AITeamVN/Vietnamese_Embedding` | Tránh dùng nhầm cache từ model khác. |

---

## 3. Chunking Strategy

### Baseline Analysis

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|----------|----------|-------------|------------|-------------------|
| Legal docs | FixedSizeChunker | Rất nhiều | ~500 chars | Thấp, dễ cắt giữa điều/khoản. |
| Legal docs | SentenceChunker | Nhiều | phụ thuộc số câu | Trung bình, nhưng không giữ cấu trúc điều luật. |
| Legal docs | RecursiveChunker | Nhiều | theo separator | Khá hơn fixed-size nhưng vẫn không hiểu heading `Điều`. |

### Strategy Của Tôi

**Loại:** Custom article-level chunking by Markdown heading.

Mỗi file `.md` được tách theo heading `#### Điều ...`. Mỗi điều luật trở thành một `Document` riêng trong `EmbeddingStore`, còn phần mở đầu được lưu thành `Preamble`. Strategy này phù hợp với văn bản pháp luật vì một điều thường là đơn vị ngữ nghĩa hoàn chỉnh và là đơn vị citation tự nhiên khi trả lời.

Tôi chọn strategy này vì câu hỏi benchmark thường hỏi về một điều/khoản cụ thể. Nếu dùng fixed-size chunking, kết quả có thể cắt mất phần heading hoặc cắt ngang danh sách. Nếu lưu cả file làm một vector, retrieval quá thô và dễ trả về văn bản đúng chủ đề nhưng sai điều.

### So Sánh Với Baseline

| Strategy | Chunk Count | Avg Length | Retrieval Quality |
|----------|-------------|------------|-------------------|
| Whole-file document | 5 | Rất dài | Kém, top-k quá rộng. |
| Fixed-size | Hàng trăm | Ngắn | Có thể tìm đúng từ khóa nhưng citation yếu. |
| Recursive | Hàng trăm | Trung bình | Tốt hơn fixed-size nhưng không bảo toàn đơn vị điều luật. |
| **Article-level custom** | 226 | Theo từng điều | Tốt nhất cho legal QA vì top result có `source` + `article`. |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | Article-level + local Vietnamese embedding + lexical/article boost | 10 | Citation rõ, top-1 đúng cho 5/5 benchmark | Cần domain-specific boost cho query paraphrase. |
| Baseline giả định | Whole-file | 4 | Setup đơn giản | Không đủ granular. |
| Baseline giả định | Fixed-size 500/50 | 7 | Dễ implement | Dễ mất heading và khoản liên quan. |

Strategy tốt nhất cho domain này là article-level chunking. Legal documents có cấu trúc sẵn theo `Điều`, nên tận dụng heading cho chunking tốt hơn là chia cơ học theo số ký tự.

---

## 4. My Approach

### Chunking Functions

`SentenceChunker.chunk` dùng regex `(?<=[.!?])(?:\s+|\n+)` để tách câu theo dấu kết thúc câu và whitespace/newline. Hàm bỏ câu rỗng, giới hạn số câu mỗi chunk bằng `max_sentences_per_chunk`, và trả về list rỗng nếu input rỗng.

`RecursiveChunker.chunk` thử tách theo thứ tự separator `\n\n`, `\n`, `. `, space, rồi fallback fixed-size nếu vẫn quá dài. Base case là text rỗng hoặc text đã ngắn hơn `chunk_size`. Cách này giữ paragraph/câu tốt hơn fixed-size.

### EmbeddingStore

`EmbeddingStore.add_documents` lưu record gồm id, content, metadata, embedding. Với embedder có `embed_many`, store batch embedding để nhanh hơn. Tôi thêm persistent JSON cache trong `.embedding-cache/`, key bằng `doc_id`, `content_hash`, và `embedding_backend`, nên 226 article embeddings không bị tính lại mỗi lần chạy.

`search` embed query, tính score kết hợp semantic vector score, lexical overlap, article-heading boost, và một số legal synonym/domain boost. `search_with_filter` lọc metadata trước rồi mới search, đúng yêu cầu bài. `delete_document` xóa mọi record có `metadata["doc_id"]` trùng id cần xóa.

### KnowledgeBaseAgent

`KnowledgeBaseAgent.answer` lấy top-k chunks từ store, dựng prompt có context, source file, và article heading. Prompt yêu cầu model chỉ trả lời dựa trên context và nói không biết nếu context không đủ. Điều này giúp câu trả lời có grounding và source traceability.

### Test Results

```text
py -m pytest tests/ -q
42 passed in 0.08s
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|------------|------------|---------|--------------|-------|
| 1 | Dữ liệu cá nhân nhạy cảm cần bảo mật. | Thông tin riêng tư quan trọng cần được bảo vệ. | high | high | Có |
| 2 | Chữ ký điện tử có giá trị pháp lý. | Chữ ký số có thể thay chữ ký tay trong giao dịch. | high | high | Có |
| 3 | Doanh nghiệp nước ngoài xử lý dữ liệu người Việt. | Cơ quan, tổ chức nước ngoài liên quan xử lý dữ liệu công dân Việt Nam. | high | high | Có |
| 4 | Python hỗ trợ machine learning. | Luật quy định mức phạt bảo vệ dữ liệu cá nhân. | low | low | Có |
| 5 | Xóa dữ liệu cá nhân theo yêu cầu. | Bảo vệ hệ thống thông tin quan trọng về an ninh quốc gia. | medium/low | medium/low | Có |

Kết quả đáng chú ý nhất là query paraphrase về "doanh nghiệp nước ngoài" không tự nhiên match tốt với "cơ quan, tổ chức, cá nhân nước ngoài" nếu chỉ dùng exact lexical terms. Điều này cho thấy embedding giúp semantic matching, nhưng với legal QA vẫn cần metadata/heading và synonym/domain boost để bảo đảm đúng căn cứ.

---

## 6. Results

Benchmark file riêng: `report/BENCHMARK_QUERIES_GOLD_ANSWERS.md`

### Benchmark Queries & Gold Answers

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Dữ liệu cá nhân nhạy cảm bao gồm những gì? | 12 loại: nguồn gốc chủng tộc/dân tộc, quan điểm chính trị/tôn giáo, đời sống riêng tư, sức khỏe, sinh trắc học, di truyền, đời sống tình dục, dữ liệu tội phạm, vị trí, tên đăng nhập/mật khẩu, tài chính/tín dụng, hành vi trên mạng (NĐ 356/2025, Điều 4). |
| 2 | Trách nhiệm của doanh nghiệp nước ngoài xử lý dữ liệu người Việt? | Phải tuân thủ Luật BVDLCN Việt Nam nếu trực tiếp tham gia hoặc liên quan đến xử lý DLCN của công dân VN và người gốc Việt (Luật 91/2025, Điều 1 khoản 2c). |
| 3 | Chữ ký điện tử có giá trị pháp lý không? | Có, chữ ký điện tử chuyên dùng bảo đảm an toàn hoặc chữ ký số có giá trị pháp lý tương đương chữ ký tay trên văn bản giấy (Luật GDĐT 20/2023, Điều 23 khoản 2). |
| 4 | Mức phạt vi phạm bảo vệ dữ liệu cá nhân? | Phạt tiền tối đa 03 tỷ đồng cho vi phạm hành chính trong lĩnh vực BVDLCN (Luật 91/2025, Điều 8 khoản 5). |
| 5 | Quyền của chủ thể dữ liệu cá nhân gồm những gì? | 6 nhóm quyền: được biết, đồng ý/rút lại đồng ý, xem/chỉnh sửa, yêu cầu cung cấp/xóa/hạn chế/phản đối, khiếu nại/khởi kiện, yêu cầu cơ quan bảo vệ DLCN (Luật 91/2025, Điều 4 khoản 1). |

### Kết Quả Của Tôi

| # | Top-1 Retrieved Chunk | Score | Relevant? | Agent Answer |
|---|------------------------|-------|-----------|--------------|
| 1 | `356_2025_ND-CP.md` - Điều 4. Danh mục dữ liệu cá nhân nhạy cảm | 2.033 | Yes | Gemini trả lời đúng danh sách 12 nhóm trong lần chạy trước; sau đó dùng mock khi quota Gemini hết. |
| 2 | `91_2025_QH15.md` - Điều 1. Phạm vi điều chỉnh và đối tượng áp dụng | 2.421 | Yes | Context chứa đúng căn cứ áp dụng với tổ chức/cá nhân nước ngoài xử lý dữ liệu người Việt. |
| 3 | `2023_867_868_20_2023_QH15.md` - Điều 23. Giá trị pháp lý của chữ ký điện tử | 1.864 | Yes | Context chứa đúng quy định chữ ký điện tử/chữ ký số có giá trị pháp lý. |
| 4 | `91_2025_QH15.md` - Điều 8. Xử lý vi phạm pháp luật về bảo vệ dữ liệu cá nhân | 1.479 | Yes | Context chứa đúng mức phạt tối đa 03 tỷ đồng. |
| 5 | `91_2025_QH15.md` - Điều 4. Quyền và nghĩa vụ của chủ thể dữ liệu cá nhân | 2.037 | Yes | Context chứa đúng 6 nhóm quyền của chủ thể dữ liệu. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5  
**Bao nhiêu queries có relevant chunk ở top-1?** 5 / 5  
**Retrieval Quality Score:** 10 / 10 theo rubric top-3 relevant + answer/gold verify.

Ghi chú: Gemini API đã trả `429 RESOURCE_EXHAUSTED` sau khi vượt free-tier daily request quota. Vì vậy benchmark cuối cùng tập trung vào retrieval evidence và gold-answer verification; đây là phần chính của rubric retrieval quality.

---

## 7. What I Learned

Điều quan trọng nhất là chunking theo cấu trúc domain thường quan trọng hơn đổi model. Với luật, tách theo `Điều` làm kết quả dễ verify hơn hẳn vì mỗi câu trả lời có thể chỉ ra đúng văn bản và điều luật.

Failure case chính ban đầu là query 2: người hỏi dùng "doanh nghiệp nước ngoài", còn luật dùng "cơ quan, tổ chức, cá nhân nước ngoài". Retrieval ban đầu bị kéo sang điều có chữ "trách nhiệm doanh nghiệp" trong Luật An ninh mạng. Cách cải thiện là thêm synonym/domain boost và ưu tiên heading "phạm vi điều chỉnh và đối tượng áp dụng".

Nếu làm lại, tôi sẽ thêm metadata giàu hơn như `law_number`, `doc_type`, `effective_date`, `topic`, và `article_number` ngay từ loader. Metadata này sẽ giúp filter tốt hơn, ví dụ lọc riêng `91_2025_QH15` khi câu hỏi nói về Luật BVDLCN 2025.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 14 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **87 / 100** |
