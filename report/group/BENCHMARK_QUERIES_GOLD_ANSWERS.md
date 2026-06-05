# Benchmark Queries & Gold Answers

Domain: Vietnamese cybersecurity, electronic transaction, and personal data protection law.

Dataset folder: `data/luat_an_ninh_mang_ve_bao_ve_du_lieu_ca_nhan`

| # | Query | Gold Answer | Supporting Chunk |
|---|-------|-------------|------------------|
| 1 | Dữ liệu cá nhân nhạy cảm bao gồm những gì? | 12 loại: nguồn gốc chủng tộc/dân tộc, quan điểm chính trị/tôn giáo, đời sống riêng tư, sức khỏe, sinh trắc học, di truyền, đời sống tình dục, dữ liệu tội phạm, vị trí, tên đăng nhập/mật khẩu, tài chính/tín dụng, hành vi trên mạng. | `356_2025_ND-CP.md`, Điều 4 |
| 2 | Trách nhiệm của doanh nghiệp nước ngoài xử lý dữ liệu người Việt? | Phải tuân thủ Luật BVDLCN Việt Nam nếu trực tiếp tham gia hoặc liên quan đến xử lý dữ liệu cá nhân của công dân Việt Nam và người gốc Việt. | `91_2025_QH15.md`, Điều 1 khoản 2(c) |
| 3 | Chữ ký điện tử có giá trị pháp lý không? | Có. Chữ ký điện tử chuyên dùng bảo đảm an toàn hoặc chữ ký số có giá trị pháp lý tương đương chữ ký của cá nhân trên văn bản giấy. | `2023_867_868_20_2023_QH15.md`, Điều 23 khoản 2 |
| 4 | Mức phạt vi phạm bảo vệ dữ liệu cá nhân? | Phạt tiền tối đa 03 tỷ đồng cho vi phạm hành chính trong lĩnh vực bảo vệ dữ liệu cá nhân. | `91_2025_QH15.md`, Điều 8 khoản 5 |
| 5 | Quyền của chủ thể dữ liệu cá nhân gồm những gì? | 6 nhóm quyền: được biết, đồng ý/rút lại đồng ý, xem/chỉnh sửa, yêu cầu cung cấp/xóa/hạn chế/phản đối, khiếu nại/khởi kiện, yêu cầu cơ quan bảo vệ dữ liệu cá nhân. | `91_2025_QH15.md`, Điều 4 khoản 1 |

Notes:
- Query 1 is factual and list-based.
- Query 2 tests paraphrase handling because the query says "doanh nghiệp nước ngoài" while the statute says "cơ quan, tổ chức, cá nhân nước ngoài".
- Query 3 checks retrieval across the electronic transaction law.
- Query 4 asks for a numeric legal threshold.
- Query 5 checks a multi-item rights list.
