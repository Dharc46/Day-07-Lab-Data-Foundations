# Bao Cao Lab 7: Embedding & Vector Store

**Ho ten:** Nguyen Thanh Dat  
**Nhom:** Vietnamese Digital Law RAG  
**Ngay:** 2026-06-05

---

## 1. Warm-up

### Cosine Similarity

Cosine similarity do muc do cung huong giua hai embedding vectors. Trong retrieval, query va chunk co cosine similarity cao khi chung noi ve cung mot noi dung phap ly, vi du cung hoi ve "quyen cua chu the du lieu" hoac "hanh vi bi nghiem cam".

**High similarity:**
- A: "Chu the du lieu ca nhan co quyen yeu cau xoa du lieu khong?"
- B: "Quyen cua chu the du lieu ca nhan bao gom yeu cau cung cap, xoa, han che xu ly du lieu."
- Ly do: ca hai deu noi ve quyen cua chu the du lieu ca nhan.

**Low similarity:**
- A: "Bien phap bao ve an ninh mang gom nhung gi?"
- B: "Hop dong dien tu duoc giao ket theo nguyen tac nao?"
- Ly do: mot cau thuoc an ninh mang, cau kia thuoc giao dich dien tu.

Cosine similarity phu hop voi text embeddings vi no tap trung vao huong cua vector, tuc y nghia tuong doi, thay vi do dai tuyet doi cua vector. Euclidean distance co the bi anh huong boi scale va khong on dinh bang khi embedding da duoc normalize.

### Chunking Math

Document dai 10,000 ky tu, `chunk_size=500`, `overlap=50`:

```text
num_chunks = ceil((10000 - 50) / (500 - 50))
           = ceil(9950 / 450)
           = 23
```

Neu overlap tang len 100:

```text
num_chunks = ceil((10000 - 100) / (500 - 100))
           = ceil(9900 / 400)
           = 25
```

Overlap lon hon lam tang so chunk, nhung giup giu ngu canh quanh ranh gioi cat, rat huu ich voi van ban phap luat vi mot khoan/dieu co the bi tach ngang.

---

## 2. Document Selection - Nhom

**Domain:** Vietnamese digital law and personal data protection RAG.

Nhom chon bo van ban phap luat dang Markdown trong file `md.zip`. Domain nay phu hop voi RAG vi nguoi dung thuong hoi theo ngon ngu tu nhien, trong khi cau tra loi can duoc grounding vao dieu/khoan cua van ban nguon. Bo du lieu co 5 van ban ve bao ve du lieu ca nhan, an ninh mang, va giao dich dien tu.

### Data Inventory

| # | Ten tai lieu | File trong repo | Loai | So ky tu | So headings | Metadata chinh |
|---|--------------|-----------------|------|----------|-------------|----------------|
| 1 | Nghi dinh 13/2023/ND-CP ve bao ve du lieu ca nhan | `data/group_legal_docs/13_2023_ND_CP_bao_ve_du_lieu_ca_nhan.md` | decree | 68,035 | 67 | source, doc_type, year, topic, law_no |
| 2 | Luat An ninh mang 24/2018/QH14 | `data/group_legal_docs/24_2018_QH14_luat_an_ninh_mang.md` | law | 63,831 | 51 | source, doc_type, year, topic, law_no |
| 3 | Luat Giao dich dien tu 20/2023/QH15 | `data/group_legal_docs/20_2023_QH15_luat_giao_dich_dien_tu.md` | law | 57,257 | 67 | source, doc_type, year, topic, law_no |
| 4 | Nghi dinh 356/2025/ND-CP huong dan Luat BVDLCN | `data/group_legal_docs/356_2025_ND_CP_huong_dan_luat_bvdlcn.md` | decree | 109,398 | 74 | source, doc_type, year, topic, law_no |
| 5 | Luat Bao ve du lieu ca nhan 91/2025/QH15 | `data/group_legal_docs/91_2025_QH15_luat_bao_ve_du_lieu_ca_nhan.md` | law | 53,636 | 47 | source, doc_type, year, topic, law_no |

### Metadata Schema

| Truong metadata | Kieu | Vi du | Vai tro trong retrieval |
|----------------|------|-------|--------------------------|
| source | string | `91_2025_QH15_luat_bao_ve_du_lieu_ca_nhan.md` | Trich nguon trong answer |
| doc_type | string | `law`, `decree` | Loc theo luat/nghi dinh |
| year | string | `2025` | Uu tien van ban moi |
| topic | string | `personal_data_protection` | Loc nhanh theo mien noi dung |
| law_no | string | `91/2025/QH15` | Loc dung van ban khi query neu so hieu |
| chunk_index | int | `12` | Debug va trace chunk |

---

## 3. Chunking Strategy - Nhom

### Baseline Analysis

Chay tren 5,000 ky tu dau cua 3 van ban mau, `chunk_size=1200`:

| Tai lieu | Strategy | Chunk Count | Avg Length | Nhan xet |
|----------|----------|-------------|------------|----------|
| ND 13/2023 | Fixed-size | 5 | 1096.0 | De kiem soat size nhung co the cat ngang dieu/khoan |
| ND 13/2023 | Sentence | 6 | 818.5 | Giu cau tot, nhung markdown legal co nhieu bullet dai |
| ND 13/2023 | Recursive | 6 | 821.7 | Can bang tot, uu tien heading/newline |
| Luat An ninh mang | Fixed-size | 5 | 1096.0 | Co nguy co cat giua danh sach bien phap |
| Luat An ninh mang | Sentence | 8 | 621.6 | Chunk ngan hon nhung de tach khoan lien quan |
| Luat An ninh mang | Recursive | 6 | 830.2 | Giu cau truc dieu/khoan kha tot |
| Luat Giao dich dien tu | Fixed-size | 5 | 1096.0 | Kich thuoc deu |
| Luat Giao dich dien tu | Sentence | 12 | 414.9 | Qua nhieu chunk nho |
| Luat Giao dich dien tu | Recursive | 5 | 998.8 | Tot nhat cho markdown co heading |

### Strategy Cua Toi

**Loai:** `RecursiveChunker(chunk_size=1800)` + metadata filter + model `AITeamVN/Vietnamese_Embedding`.

Van ban phap luat co cau truc ro: chuong, muc, dieu, khoan, bullet. Recursive chunking uu tien paragraph/newline truoc khi fallback sang word/character, nen giu duoc ngu canh hon fixed-size. Metadata filter theo `topic`, `law_no`, `doc_type` giup thu hep tap ung vien truoc semantic search, rat quan trong vi cac van ban co nhieu cum tu lap lai nhu "du lieu ca nhan", "co quan, to chuc, ca nhan".

Model nhom chon:

```bash
pip install sentence-transformers
python3 -c "from src import LocalEmbedder; e = LocalEmbedder('AITeamVN/Vietnamese_Embedding'); print(e._backend_name, len(e('xin chào')))"
```

Ket qua tren may:

```text
AITeamVN/Vietnamese_Embedding 1024
```

### So Sanh Voi Thanh Vien Khac

| Thanh vien | Strategy | Retrieval Score (/10) | Diem manh | Diem yeu |
|------------|----------|----------------------|-----------|----------|
| Toi | Recursive + metadata filter + Vietnamese embedding | 8 | Hop van ban markdown phap ly, co trace source | Can them article-aware metadata de top-1 chuan hon |
| Ban A | Fixed-size + Vietnamese embedding | 6 | Don gian, chunk deu | Co the cat ngang Dieu/Khoan |
| Ban B | Sentence chunking + Vietnamese embedding | 7 | Giu cau day du | Tao nhieu chunk nho, mat ngu canh cua dieu |

Ket luan nhom: recursive chunking la baseline tot nhat, nhung de dat diem cao hon voi van ban phap luat nen bo sung parser tach theo `#### Điều X` va gan metadata `article_number`, `chapter`.

---

## 4. My Approach - Ca Nhan

### Chunking Functions

`SentenceChunker.chunk` normalize whitespace, tach cau theo dau `.`, `!`, `?`, sau do gom toi da `max_sentences_per_chunk` cau vao mot chunk. `RecursiveChunker.chunk` dung base case text rong/text ngan hon `chunk_size`, neu qua dai thi thu lan luot separator tu paragraph den word/character fallback.

### EmbeddingStore

`EmbeddingStore` dung in-memory list de on dinh cho test. Moi record luu `id`, `doc_id`, `content`, `metadata`, `embedding`; search tinh cosine similarity, sort giam dan theo `score`, va tra ve top-k. `search_with_filter` loc metadata truoc khi search semantic; `delete_document` xoa theo `doc_id`.

### KnowledgeBaseAgent

Agent goi `store.search`, build prompt co `Retrieved context` va `Question`, yeu cau chi tra loi dua tren context. Neu khong co context, agent tra ve cau an toan thay vi suy dien.

### Local Embedder

`LocalEmbedder` dung `sentence-transformers`. Tren may Windows bi loi SSL certificate khi tai Hugging Face, nen code co retry voi HTTP backend `verify=False` trong moi truong lab. Sau khi model da cache, cac lan chay sau nhanh hon.

### Test Results

```text
pytest tests/ -v
42 passed in 0.13s
```

---

## 5. Similarity Predictions

Model dung de benchmark: `AITeamVN/Vietnamese_Embedding`, 1024 dimensions.

| Pair | Sentence A | Sentence B | Du doan | Ghi chu |
|------|------------|------------|---------|---------|
| 1 | "du lieu ca nhan nhay cam la gi" | "du lieu gan lien voi quyen rieng tu cua ca nhan" | high | Cung noi ve dinh nghia BVDLCN |
| 2 | "chu the du lieu co quyen xoa du lieu" | "yeu cau cung cap, xoa, han che xu ly du lieu ca nhan" | high | Cung noi ve quyen cua chu the du lieu |
| 3 | "hanh vi bi nghiem cam trong giao dich dien tu" | "gia mao, lam sai lech thong diep du lieu" | high | Cung thuoc Dieu 6 Luat GDDT |
| 4 | "bien phap bao ve an ninh mang" | "chu ky dien tu nuoc ngoai" | low | Khac topic |
| 5 | "thoi han phan hoi yeu cau xoa du lieu" | "giao ket hop dong dien tu" | low | Khac van ban va y dinh |

Ket qua tong quan: model tieng Viet tot hon mock embedding trong viec nhan dien cum tu phap ly, nhung top-1 van co the sai khi chunk qua dai hoac nhieu dieu khoan co tu lap lai.

---

## 6. Results - Benchmark Queries

Benchmark dung 281 chunks tao boi `RecursiveChunker(chunk_size=1800)`, batch encoded bang `AITeamVN/Vietnamese_Embedding`.

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Du lieu ca nhan nhay cam bao gom nhung gi? | 12 loai trong ND 356/2025, Dieu 4. |
| 2 | Trach nhiem cua doanh nghiep nuoc ngoai xu ly du lieu nguoi Viet? | Phai tuan thu Luat BVDLCN Viet Nam neu truc tiep tham gia hoac lien quan den xu ly DLCN cua cong dan Viet Nam/nguoi goc Viet; Luat 91/2025, Dieu 1 Khoan 2c. |
| 3 | Chu ky dien tu co gia tri phap ly khong? | Co; chu ky dien tu chuyen dung bao dam an toan hoac chu ky so co gia tri phap ly tuong duong chu ky tay; Luat 20/2023, Dieu 23 Khoan 2. |
| 4 | Muc phat vi pham bao ve du lieu ca nhan? | Phat tien toi da 03 ty dong; Luat 91/2025, Dieu 8 Khoan 5. |
| 5 | Quyen cua chu the du lieu ca nhan gom nhung gi? | 6 nhom quyen trong Luat 91/2025, Dieu 4 Khoan 1. |

### Ket Qua Ca Nhan

| # | Metadata filter | Top-1 retrieved | Score | Relevant top-3? | Nhan xet |
|---|-----------------|-----------------|-------|-----------------|----------|
| 1 | `law_no=356/2025/ND-CP` | ND 356/2025, chunk 2, Dieu 4 | 0.654 | Yes | Top-1 dung gold answer ve danh muc du lieu ca nhan nhay cam. |
| 2 | `law_no=91/2025/QH15` | Luat 91/2025, chunk 41, Dieu 37 | 0.519 | Partial/No | Dung van ban va noi ve trach nhiem xu ly DLCN, nhung khong dung Dieu 1 Khoan 2c ve doi tuong nuoc ngoai. |
| 3 | `law_no=20/2023/QH15` | Luat 20/2023, chunk 20, Dieu 23 | 0.623 | Yes | Top-1 dung dieu ve gia tri phap ly cua chu ky dien tu. |
| 4 | `law_no=91/2025/QH15` | Luat 91/2025, chunk 8, Dieu 8 | 0.697 | Yes | Top-1 dung dieu ve xu ly vi pham va muc phat 03 ty dong. |
| 5 | `law_no=91/2025/QH15` | Luat 91/2025, chunk 4, Dieu 4 | 0.740 | Yes | Top-1 dung dieu ve quyen cua chu the du lieu ca nhan. |

**Precision theo van ban trong top-3:** 5 / 5  
**Gold answer relevant trong top-3:** 4 / 5, Q2 partial vi retrieve dung van ban/trach nhiem nhung lech dieu khoan gold.  
**Top-1 dung dieu khoan chinh xac:** 4 / 5  
**Ket luan:** strategy hoat dong tot khi co `law_no` metadata filter. Loi con lai cho thay legal RAG nen gan them `article_number` de query nhu Q2 co the truy dung Dieu 1 Khoan 2c.

---

## 7. Evaluation

**Retrieval Precision:** Với metadata filter, top-3 gan nhu luon dung van ban. Tuy nhien, top-1 chua on dinh khi mot van ban co nhieu dieu lap lai cum tu "du lieu ca nhan".

**Chunk Coherence:** Recursive chunking giu doan markdown tot hon fixed-size. Tuy vay, neu chunk chua ca nhieu khoan/phu luc, model co the xep hang sai dieu.

**Metadata Utility:** `topic`, `law_no`, `doc_type` cai thien retrieval ro ret. Truong con thieu quan trong la `article_number`, `chapter`, `effective_date`.

**Grounding Quality:** Agent prompt co context va source/score, giup cau tra loi duoc grounding. Voi phap ly, nen bat agent trich source theo `law_no + article_number`.

**Data Strategy Impact:** Chat luong schema va chunking quan trong hon viec chi doi model. Model Vietnamese embedding giup semantic tot hon mock, nhung khong thay the duoc document parsing tot.

---

## 8. Failure Cases Va Cai Thien

1. **Query khong noi ro van ban:** "quyen cua toi la gi?" qua mo ho. Cai thien: hoi lai nguoi dung hoac them classifier topic.

2. **Chunk qua dai/chua nhieu dieu:** Top-1 co the la phu luc hoac dieu lien quan nhung khong phai gold article. Cai thien: tach theo `#### Điều X`.

3. **Thieu metadata dieu/khoan:** Chi filter theo `topic` van con nhieu ung vien. Cai thien: gan `article_number`, `chapter`, `section`.

4. **Cau hoi can cap nhat phap ly:** Neu co van ban moi hon, retrieval phai uu tien `year`/`effective_date`. Cai thien: metadata ve hieu luc va quan he thay the/sua doi.

5. **SSL/Hugging Face tai model:** May Windows co the loi certificate. Cai thien: cache model truoc buoi demo va giu fallback retry trong `LocalEmbedder`.

---

## 9. What I Learned

Trong domain phap luat, retrieval khong chi la embedding. Document parsing va metadata gan voi cau truc phap ly moi la phan quyet dinh: dung van ban, dung dieu, dung khoan. Vietnamese embedding giup hon mock embedding, nhung neu chunking khong ton trong `Điều/Khoản` thi van co the lay sai context.

Neu lam tiep, toi se viet them `LegalArticleChunker` de tach moi `#### Điều X` thanh mot chunk, keo theo metadata `article_number`, `article_title`, `chapter`, `law_no`. Khi do benchmark se cong bang hon va agent co the trich dan nguon chinh xac.

---

## 10. Ket Luan

Bo du lieu nhom la 5 van ban phap luat ve bao ve du lieu ca nhan, an ninh mang, va giao dich dien tu. Strategy tot nhat hien tai la recursive chunking + metadata filter + `AITeamVN/Vietnamese_Embedding`. De nang chat luong len muc san sang demo, buoc tiep theo nen la article-aware chunking va metadata theo Dieu/Khoan.

---

## Tu Danh Gia

| Tieu chi | Loai | Diem tu danh gia |
|----------|------|-------------------|
| Warm-up | Ca nhan | 5 / 5 |
| Document selection | Nhom | 10 / 10 |
| Chunking strategy | Nhom | 13 / 15 |
| My approach | Ca nhan | 10 / 10 |
| Similarity predictions | Ca nhan | 4 / 5 |
| Results | Ca nhan | 8 / 10 |
| Core implementation (tests) | Ca nhan | 30 / 30 |
| Demo | Nhom | 4 / 5 |
| **Tong** | | **84 / 100** |
