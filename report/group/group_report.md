Dựa trên 4 report (Đạt, Thuyên, Khoa, Lâm), có thể so sánh như sau:

| Thành viên | Strategy                                               | Top-1 đúng         | Top-3 relevant | Nhận xét                                                                                                                  |
| ---------- | ------------------------------------------------------ | ------------------ | -------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Đạt        | Recursive + metadata filter + Vietnamese Embedding     | 4/5                | 4/5            | Q2 retrieve đúng văn bản nhưng sai điều khoản.                                                                            |
| Thuyên     | RecursiveChunker                                       | Chưa chứng minh rõ | 5/5            | Tốt hơn fixed-size nhưng vẫn chunk theo cấu trúc chung, không theo Điều luật.                                             |
| Khoa       | Parent-Child (Parent = Điều, Child = 3 câu)            | 4/5                | 5/5            | Giữ được context Điều luật, retrieval chính xác hơn recursive. Q2 vẫn bị lệch sang điều trách nhiệm xử lý dữ liệu chung.  |
| Lâm        | Article-level chunking + article boost + lexical boost | 5/5                | 5/5            | Truy đúng Điều luật cho toàn bộ benchmark, citation rõ ràng.                                                              |

## Kết quả so sánh

### Hạng 1: Article-level Chunking (Lâm)

Lý do:

* Chunk được tạo đúng theo đơn vị pháp lý tự nhiên là **Điều**.
* Benchmark của nhóm đều hỏi theo dạng:

  * "Điều nào quy định..."
  * "Quyền gì..."
  * "Mức phạt bao nhiêu..."
* Trong văn bản luật, một Điều thường là một đơn vị ngữ nghĩa hoàn chỉnh.
* Có thêm:

  * article metadata
  * lexical boost
  * article heading boost
* Đạt:

  * Top-1 đúng: 5/5
  * Top-3 relevant: 5/5. 

### Hạng 2: Parent-Child Chunking (Khoa)

Ưu điểm:

* Child chunk nhỏ → embedding chính xác hơn.
* Parent chunk là toàn bộ Điều → giữ ngữ cảnh khi trả lời.

Đây là kiến trúc rất phổ biến trong production RAG.

Tuy nhiên:

* Query Q2 vẫn retrieve nhầm Điều về trách nhiệm xử lý dữ liệu thay vì Điều 1 Khoản 2c.
* Chưa tận dụng mạnh article heading như strategy của Lâm. 

### Hạng 3: Recursive + Metadata Filter (Đạt)

Ưu điểm:

* Metadata filter (`law_no`) giúp thu hẹp search space.
* Chunk lớn nên ít mất context.

Nhược điểm:

* Chunk 1800 ký tự chứa nhiều Điều/Khoản.
* Semantic search tìm đúng văn bản nhưng có thể nhảy sang Điều khác trong cùng văn bản.
* Q2 là ví dụ điển hình. 

### Hạng 4: RecursiveChunker thuần (Thuyên)

Ưu điểm:

* Đơn giản.
* Dễ triển khai.

Nhược điểm:

* Không tận dụng cấu trúc pháp luật.
* Điều luật dài có thể bị tách thành nhiều chunk.
* Citation và traceability yếu hơn 3 cách còn lại. 

## Kết luận nhóm

**Strategy retrieval tốt nhất là Article-level Chunking (tách theo Điều luật) kết hợp metadata/article boost.** 

Nguyên nhân:

1. Văn bản pháp luật vốn đã có cấu trúc ngữ nghĩa rõ ràng theo Điều/Khoản.
2. Điều luật chính là đơn vị được dùng để trích dẫn và trả lời câu hỏi.
3. Retrieval không chỉ cần đúng văn bản mà còn phải đúng Điều.
4. Các benchmark của nhóm đều đánh giá theo Điều/Khoản cụ thể.
5. Kết quả thực nghiệm cho thấy chỉ strategy này đạt:

   * Top-1 đúng: **5/5**
   * Top-3 relevant: **5/5**. 

Nếu viết ngắn gọn cho phần report nhóm:

> Qua so sánh 4 cách tiếp cận, Article-level Chunking cho kết quả retrieval tốt nhất đối với domain văn bản pháp luật Việt Nam. Nguyên nhân là mỗi Điều luật đã là một đơn vị ngữ nghĩa hoàn chỉnh và cũng là đơn vị trích dẫn pháp lý tự nhiên. Việc chunk theo Điều kết hợp metadata và article boost giúp hệ thống truy xuất đúng căn cứ pháp lý hơn so với Recursive Chunking hoặc Parent-Child Chunking. Kết quả benchmark đạt 5/5 câu hỏi có relevant chunk ở top-1 và 5/5 câu hỏi có relevant chunk trong top-3, cao nhất trong nhóm.
