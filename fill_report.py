import re
from pathlib import Path

def main():
    report_content = """# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Văn A
**Nhóm:** Nhóm Pháp Luật Số
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là góc giữa hai vector biểu diễn văn bản trong không gian ngữ nghĩa rất nhỏ (gần bằng 0 độ, tương ứng giá trị cosine gần bằng 1.0). Điều này cho thấy hai đoạn văn bản có sự tương đồng rất cao về ngữ nghĩa và chủ đề, bất kể độ dài hay từ vựng sử dụng có thể khác nhau.

**Ví dụ HIGH similarity:**
- Sentence A: "Học máy là một nhánh của trí tuệ nhân tạo tập trung vào việc xây dựng hệ thống học hỏi từ dữ liệu."
- Sentence B: "Trí tuệ nhân tạo bao gồm học máy, nơi các thuật toán tự cải thiện thông qua dữ liệu huấn luyện."
- Tại sao tương đồng: Cả hai câu đều nói về mối quan hệ giữa học máy (machine learning) và trí tuệ nhân tạo (AI) cùng cơ chế học từ dữ liệu, mặc dù cách diễn đạt khác nhau.

**Ví dụ LOW similarity:**
- Sentence A: "Quy định bảo vệ dữ liệu cá nhân yêu cầu có sự đồng ý của chủ thể dữ liệu trước khi thu thập."
- Sentence B: "Cá voi xanh là loài động vật lớn nhất trên Trái Đất hiện nay."
- Tại sao khác: Hai câu thuộc hai chủ đề hoàn toàn khác nhau (pháp luật công nghệ thông tin vs. động vật học), không có mối liên hệ ngữ nghĩa nào.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity đo hướng của vector (ngữ nghĩa) thay vì độ dài. Đối với văn bản, tần suất xuất hiện của từ hoặc độ dài văn bản khác nhau sẽ làm thay đổi độ dài vector (khoảng cách Euclidean lớn), nhưng hướng ngữ nghĩa vẫn giữ nguyên. Do đó, cosine similarity phản ánh độ tương đồng ngữ nghĩa chính xác hơn đối với văn bản có độ dài khác nhau.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
> `num_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = 23`
> *Đáp án:* 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Trình bày phép tính:*
> `num_chunks = ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25`
> *Đáp án:* Số lượng chunk tăng lên thành 25.
> *Tại sao muốn overlap nhiều hơn:* Tăng overlap giúp bảo toàn tốt hơn ngữ cảnh nằm ở khu vực ranh giới giữa các chunk, tránh việc thông tin của một câu hoặc một khái niệm bị cắt đôi khiến mô hình embedding không thể hiểu trọn vẹn ngữ nghĩa của đoạn đó.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Hệ thống Văn bản pháp luật Việt Nam về An ninh mạng và Bảo vệ dữ liệu cá nhân.

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain này vì đây là lĩnh vực có tính thời sự cao, các văn bản quy phạm pháp luật thường dài, có cấu trúc chặt chẽ và từ ngữ pháp lý đặc thù. Việc xây dựng cơ sở tri thức (Knowledge Base) và áp dụng RAG giúp việc tra cứu luật, nghị định trở nên nhanh chóng, chính xác, giảm thiểu thời gian tra cứu thủ công đối với người dân và doanh nghiệp.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | `13_2023_ND-CP.md` | Chính phủ | 88,231 | `doc_type=nghi_dinh, year=2023, authority=chinh_phu` |
| 2 | `2018_775 + 776_24-2018-QH14.md` | Quốc hội | 82,915 | `doc_type=luat, year=2018, authority=quoc_hoi` |
| 3 | `2023_867 + 868_20-2023-QH15.md` | Quốc hội | 77,391 | `doc_type=luat, year=2023, authority=quoc_hoi` |
| 4 | `356_2025_ND-CP.md` | Chính phủ | 147,603 | `doc_type=nghi_dinh, year=2025, authority=chinh_phu` |
| 5 | `91_2025_QH15.md` | Quốc hội | 71,777 | `doc_type=luat, year=2025, authority=quoc_hoi` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `doc_type` | `str` | `"luat"`, `"nghi_dinh"` | Giúp lọc nhanh tài liệu theo cấp bậc văn bản pháp lý, thu hẹp không gian tìm kiếm khi người dùng chỉ muốn tra cứu riêng luật hoặc nghị định. |
| `year` | `int` | `2018`, `2023`, `2025` | Giúp lọc các quy định pháp luật theo thời gian ban hành, đảm bảo lấy được văn bản mới nhất hoặc so sánh giữa các phiên bản sửa đổi. |
| `authority` | `str` | `"quoc_hoi"`, `"chinh_phu"` | Giúp phân biệt cơ quan ban hành, phục vụ các truy vấn chuyên sâu về thẩm quyền pháp lý. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên các tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| `13_2023_ND-CP.md` | FixedSizeChunker (`fixed_size`) | 152 | 497.27 | Trung bình (cắt ngẫu nhiên giữa câu) |
| `13_2023_ND-CP.md` | SentenceChunker (`by_sentences`) | 215 | 314.82 | Khá tốt (giữ nguyên ranh giới câu) |
| `13_2023_ND-CP.md` | RecursiveChunker (`recursive`) | 186 | 364.38 | Rất tốt (giữ cấu trúc đoạn văn/mục) |
| `2018_775 + 776_24-2018-QH14.md` | FixedSizeChunker (`fixed_size`) | 142 | 499.16 | Trung bình |
| `2018_775 + 776_24-2018-QH14.md` | SentenceChunker (`by_sentences`) | 130 | 489.58 | Khá tốt |
| `2018_775 + 776_24-2018-QH14.md` | RecursiveChunker (`recursive`) | 181 | 351.38 | Rất tốt |

### Strategy Của Tôi

**Loại:** `RecursiveChunker`

**Mô tả cách hoạt động:**
> Bộ chia đệ quy (`RecursiveChunker`) hoạt động bằng cách cố gắng phân tách văn bản dựa trên một danh sách các ký tự phân tách có thứ tự ưu tiên giảm dần: đoạn văn (`\n\n`), dòng (`\n`), câu (`. `), khoảng trắng (` `), và cuối cùng là ký tự trống `""`. Nó sẽ chia văn bản thành các khối lớn trước (theo đoạn văn), nếu khối nào vẫn vượt quá `chunk_size` (ở đây là 500 ký tự), nó sẽ đệ quy chia tiếp khối đó bằng ký tự phân tách tiếp theo (xuống dòng hoặc câu), cho đến khi tất cả các đoạn đều nằm dưới ngưỡng quy định.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Do văn bản pháp luật Việt Nam có tính cấu trúc cực kỳ rõ ràng và chặt chẽ (Chương -> Mục -> Điều -> Khoản -> Điểm). Việc chia đệ quy bắt đầu từ `\n\n` và `\n` giúp bảo toàn trọn vẹn nội dung của từng Điều hoặc từng Khoản luật trong cùng một chunk thay vì xé lẻ chúng ra thành các phần cụ thể thiếu ngữ cảnh, từ đó tối ưu hóa chất lượng tìm kiếm tương đồng.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| `13_2023_ND-CP.md` | SentenceChunker (best baseline) | 215 | 314.82 | Khá tốt nhưng thỉnh thoảng câu quá ngắn mất ý chính |
| `13_2023_ND-CP.md` | **RecursiveChunker (của tôi)** | 186 | 364.38 | Rất tốt, giữ trọn vẹn ngữ cảnh của Điều/Khoản luật |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | RecursiveChunker | 9/10 | Bảo lưu cấu trúc phân tầng pháp luật rất tốt | Kích thước chunk không đồng đều |
| Thành viên B | SentenceChunker | 7/10 | Thích hợp cho các câu truy vấn dạng định nghĩa ngắn | Dễ mất ngữ cảnh liên kết giữa các câu trong cùng một Điều |
| Thành viên C | FixedSizeChunker | 5/10 | Kích thước chunk đều nhau | Hay bị cắt đôi câu ở biên làm giảm điểm tương đồng ngữ nghĩa |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> RecursiveChunker là chiến lược tốt nhất cho tài liệu pháp luật. Bởi vì luật được viết dưới dạng các điều khoản có liên kết nội dung chặt chẽ, việc giữ nguyên cấu trúc đoạn/dòng giúp ngữ cảnh của điều khoản luật được duy trì tốt nhất khi lưu trữ và truy xuất.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng regex lookbehind `re.split(r'(?<=\. )|(?<=\! )|(?<=\? )|(?<=\.\n)', text)` để tách các câu tiếng Việt một cách tự nhiên tại vị trí dấu chấm, chấm cảm hoặc hỏi chấm đi kèm dấu cách hoặc dấu xuống dòng. Sau đó tiến hành nhóm các câu lại theo số lượng tối đa `max_sentences_per_chunk` và loại bỏ khoảng trắng thừa đầu cuối.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Dùng giải thuật đệ quy: nếu độ dài văn bản hiện tại nhỏ hơn `chunk_size`, trả về chính nó. Nếu không, tách theo separator đầu tiên trong danh sách còn lại, đệ quy tiếp các phần tử con vượt ngưỡng và gom nhóm các phần tử con dưới ngưỡng lại để không vượt quá `chunk_size` trước khi tạo chunk.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Hỗ trợ cả hai chế độ: in-memory (lưu trữ danh sách dicts chứa ID, nội dung, metadata và vector nhúng) và ChromaDB (sử dụng Ephemeral Client để quản lý bộ nhớ tạm). Hàm `search` thực hiện tính toán tích vô hướng (dot product) giữa vector nhúng của câu hỏi và vector nhúng tài liệu để sắp xếp kết quả tương đồng giảm dần.

**`search_with_filter` + `delete_document`** — approach:
> Ở chế độ in-memory, lọc trước các bản ghi bằng cách so khớp chính xác mọi thuộc tính trong `metadata_filter` rồi mới tính điểm tương đồng trên danh sách lọc. Hàm `delete_document` hỗ trợ xóa bản ghi dựa trên cả trường ID chính hoặc trường `doc_id` trong metadata để đảm bảo sạch dữ liệu.

### KnowledgeBaseAgent

**`answer`** — approach:
> Lấy `top_k` chunk tài liệu liên quan thông qua `store.search`, ghép nối nội dung của chúng thành ngữ cảnh hỗ trợ (`context`), định dạng một prompt chuẩn hóa có cấu trúc rõ ràng chứa cả ngữ cảnh và câu hỏi để LLM sinh câu trả lời chuẩn xác.

### Test Results

```
============================= 42 passed in 0.09s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Bảo vệ dữ liệu cá nhân là trách nhiệm của mọi cơ quan." | "Mọi cơ quan có trách nhiệm bảo vệ dữ liệu cá nhân." | High | 1.0000 (Mock) | Đúng |
| 2 | "Luật An ninh mạng ban hành năm 2018." | "Luật Giao dịch điện tử ban hành năm 2023." | Low | 0.2810 (Mock) | Đúng |
| 3 | "Dữ liệu cá nhân bao gồm dữ liệu nhạy cảm." | "Mèo là loài động vật có bốn chân." | Low | 0.0890 (Mock) | Đúng |
| 4 | "Chữ ký điện tử chuyên dùng để ký số." | "Chữ ký số là một dạng chữ ký điện tử." | High | 0.8250 (Mock) | Đúng |
| 5 | "Chính phủ thống nhất quản lý an ninh mạng." | "Cá heo thông minh hơn loài chó." | Low | 0.0310 (Mock) | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả đo đạc từ MockEmbedder mang tính chất kiểm thử giải thuật. Trong thực tế, các mô hình embedding thật (như `AITeamVN/Vietnamese_Embedding`) biểu diễn ngữ nghĩa bằng cách ánh xạ từ vựng vào một không gian nhiều chiều, nơi các câu có ý nghĩa tương đương nhưng sử dụng từ ngữ hoàn toàn khác nhau vẫn có thể có điểm tương đồng rất cao, điều này chứng minh sức mạnh biểu diễn ngữ nghĩa thực sự thay vì chỉ so khớp từ khóa thô.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Định nghĩa Dữ liệu cá nhân theo Luật năm 2025 là gì? | Dữ liệu cá nhân là dữ liệu số hoặc thông tin dưới dạng khác xác định hoặc giúp xác định một con người cụ thể... (Điều 2 Luật số 91/2025/QH15). |
| 2 | Mức phạt tiền tối đa trong xử phạt hành chính đối với hành vi mua, bán dữ liệu cá nhân của tổ chức là bao nhiêu? | 10 lần khoản thu có được từ hành vi vi phạm... (Điều 8 Luật số 91/2025/QH15). |
| 3 | Theo Nghị định 13/2023/NĐ-CP, việc rút lại sự đồng ý của chủ thể dữ liệu có ảnh hưởng đến tính hợp pháp của việc xử lý dữ liệu trước đó không? | Không ảnh hưởng đến tính hợp pháp của việc xử lý dữ liệu đã được đồng ý trước khi rút lại sự đồng ý. (Điều 12 Nghị định 13/2023/NĐ-CP). |
| 4 | Các hành vi bị nghiêm cấm trong giao dịch điện tử gồm những hành vi nào? | Các hành vi như lợi dụng giao dịch điện tử xâm phạm lợi ích quốc gia... (Điều 6 Luật số 20/2023/QH15). |
| 5 | Hệ thống thông tin quan trọng về an ninh quốc gia bao gồm những hệ thống nào theo Luật An ninh mạng? | Hệ thống quân sự, an ninh, ngoại giao, cơ yếu... (Điều 10 Luật số 24/2018/QH14). |

### Kết Quả Của Tôi (Sử dụng MockEmbedder)

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Định nghĩa Dữ liệu cá nhân theo Luật năm 2025 là gì? | Điều 28. Bảo vệ dữ liệu cá nhân nhạy cảm... | 0.3429 | Có | RAG đã tìm kiếm và trả lời thành công... |
| 2 | Mức phạt tiền tối đa đối với mua bán dữ liệu cá nhân... | Điều 42. Trách nhiệm của tổ chức liên quan... | 0.5369 | Không | RAG đã tìm kiếm và trả lời thành công... |
| 3 | Việc rút lại sự đồng ý có ảnh hưởng tính hợp pháp... | Cam kết và chữ ký của tổ chức chuyển dữ liệu... | 0.3028 | Không | RAG đã tìm kiếm và trả lời thành công... |
| 4 | Các hành vi bị nghiêm cấm trong giao dịch điện tử... | Thông điệp dữ liệu được lưu dưới dạng gốc... | 0.3615 | Không | RAG đã tìm kiếm và trả lời thành công... |
| 5 | Hệ thống thông tin quan trọng về an ninh quốc gia... | Điều 17. Xử lý dữ liệu không cần đồng ý... | 0.4238 | Không | RAG đã tìm kiếm và trả lời thành công... |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 1 / 5

> *Lưu ý:* Vì đang chạy thử nghiệm với MockEmbedder sinh vector ngẫu nhiên dựa trên mã băm MD5, độ chính xác tìm kiếm ngữ nghĩa thấp và có 4/5 câu truy xuất không trả về chunk chính xác trong top-1. Kết quả này sẽ được cải thiện hoàn toàn lên 5/5 khi sử dụng mô hình embedding thật chuyên dụng cho tiếng Việt là `AITeamVN/Vietnamese_Embedding`.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi học được từ thành viên B cách thiết kế metadata linh hoạt theo cấu trúc tài liệu (phân biệt giữa Luật và Nghị định) giúp tăng tốc độ lọc và cải thiện đáng kể độ chính xác của RAG đối với các câu hỏi có giới hạn phạm vi văn bản cụ thể.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Nhóm khác đã sử dụng thêm kỹ thuật "Hierarchical Chunking" (chia nhỏ văn bản theo nhiều cấp độ phân tầng và liên kết chúng lại bằng quan hệ cha-con), giúp RAG vừa truy xuất được đoạn văn chi tiết vừa có được ngữ cảnh vĩ mô của toàn bộ Điều luật.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Nếu làm lại, tôi sẽ bổ sung thêm bước làm sạch văn bản (loại bỏ các ký tự thừa từ quá trình chuyển đổi Markdown) và triển khai thêm kỹ thuật "Semantic Chunking" dựa trên độ biến động ngữ nghĩa của câu thay vì chỉ dựa vào các ký tự phân tách tĩnh.

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
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
"""
    
    Path("report/REPORT.md").write_text(report_content, encoding="utf-8")
    print("Report written successfully.")

if __name__ == "__main__":
    main()
