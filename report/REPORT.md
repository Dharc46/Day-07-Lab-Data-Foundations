# Bao Cao Lab 7: Embedding & Vector Store

**Ho ten:** Nguyen Thanh Dat  
**Nhom:** AI Restaurant Chooser for Groups  
**Ngay:** 2026-06-05

---

## 1. Warm-up

### Cosine Similarity

Cosine similarity do muc do cung huong giua hai vector embedding. Trong RAG, query va chunk co cosine similarity cao khi chung noi ve cung y dinh hoac cung chu de, du cau chu co the khac nhau.

**Vi du HIGH similarity:**
- Sentence A: "Find a cheap Korean restaurant for a group dinner."
- Sentence B: "Recommend a low-budget Korean place suitable for five people."
- Ly do: ca hai deu noi ve mon Han, ngan sach thap, va an theo nhom.

**Vi du LOW similarity:**
- Sentence A: "Find a vegetarian restaurant near school."
- Sentence B: "Explain how vector databases store embeddings."
- Ly do: mot cau hoi ve chon quan an, cau kia ve ky thuat vector store.

Cosine similarity duoc uu tien hon Euclidean distance cho text embeddings vi no tap trung vao huong cua vector, phu hop voi y nghia/nguyen tac semantic similarity. Do lon vector co the bi anh huong boi model hoac normalization, nen khoang cach Euclidean khong phai luc nao cung phan anh muc do lien quan.

### Chunking Math

Document dai 10,000 ky tu, `chunk_size=500`, `overlap=50`.

Cong thuc:

```text
num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))
           = ceil((10000 - 50) / (500 - 50))
           = ceil(9950 / 450)
           = 23
```

Neu overlap tang len 100:

```text
num_chunks = ceil((10000 - 100) / (500 - 100))
           = ceil(9900 / 400)
           = 25
```

Overlap lon hon lam tang so chunk vi buoc nhay nho hon. Doi lai, overlap giup giu ngu canh o ranh gioi giua hai chunk, dac biet khi thong tin quan trong bi nam gan diem cat.

---

## 2. Document Selection - Nhom

**Domain:** AI Restaurant Chooser for Groups.

Nhom chon domain nay vi bai toan chon quan an theo nhom can dung ca semantic search va metadata filtering. Nguoi dung thuong hoi bang ngon ngu tu nhien nhu "quan nao ngoi lau duoc" hoac "nhom co nguoi an chay", trong khi du lieu can co truong co cau truc nhu khu vuc, gia, cuisine, group_friendly.

Dataset duoi day la **mock/synthetic dataset** dung cho lab, mo phong du lieu that nhung khong phai thong tin kinh doanh thuc te.

### Data Inventory

| # | Ten tai lieu | Nguon | So ky tu | Metadata da gan |
|---|--------------|-------|----------|-----------------|
| 1 | Pho Harmony | synthetic | 209 | district, price_level, group_friendly, cuisine, dietary_options, source, updated_at |
| 2 | Seoul Corner | synthetic | 206 | district, price_level, group_friendly, cuisine, dietary_options, source, updated_at |
| 3 | Green Table | synthetic | 219 | district, price_level, group_friendly, cuisine, dietary_options, source, updated_at |
| 4 | Bistro Late | synthetic | 194 | district, price_level, group_friendly, cuisine, dietary_options, source, updated_at |
| 5 | Solo Sushi | synthetic | 172 | district, price_level, group_friendly, cuisine, dietary_options, source, updated_at |

### Metadata Schema

| Truong metadata | Kieu | Vi du gia tri | Tai sao huu ich cho retrieval? |
|----------------|------|---------------|-------------------------------|
| district | string | District 1 | Loc theo khu vuc gan nguoi dung |
| price_level | string | low, medium, high | Loc theo ngan sach |
| group_friendly | bool | True | Loai quan khong phu hop di nhom |
| cuisine | string | Korean, Vegetarian | Loc theo loai mon |
| dietary_options | string | vegan allergy-friendly | Tim quan cho nguoi an chay/di ung |
| updated_at | string | 2026-06-05 | Biet du lieu co con moi khong |

---

## 3. Chunking Strategy

### Baseline Analysis

Chay `ChunkingStrategyComparator().compare()` voi `chunk_size=120` tren 3 tai lieu mau:

| Tai lieu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Pho Harmony | FixedSizeChunker | 2 | 110.5 | Trung binh, co the cat ngang y |
| Pho Harmony | SentenceChunker | 1 | 209.0 | Tot, giu tron cau |
| Pho Harmony | RecursiveChunker | 2 | 104.0 | Tot, uu tien cau/word khi can |
| Seoul Corner | FixedSizeChunker | 2 | 109.0 | Trung binh |
| Seoul Corner | SentenceChunker | 1 | 206.0 | Tot |
| Seoul Corner | RecursiveChunker | 2 | 102.5 | Tot |
| Green Table | FixedSizeChunker | 2 | 115.5 | Trung binh |
| Green Table | SentenceChunker | 1 | 219.0 | Tot nhung chunk hoi dai |
| Green Table | RecursiveChunker | 3 | 72.3 | Tot, chunk gon hon |

### Strategy Cua Toi

**Loai:** RecursiveChunker + metadata filter + `top_k=3`.

RecursiveChunker cat theo thu tu tu nhien: paragraph, newline, sentence separator, word, roi character fallback. Cach nay giu duoc y nghia cua doan van khi tai lieu co cau truc, nhung van dam bao chunk khong qua dai neu gap mot doan lien tuc.

Toi chon strategy nay vi tai lieu nha hang thuong co nhieu thong tin gan nhau: ten quan, khu vuc, gia, loai mon, do phu hop voi nhom, gio mo cua. Recursive chunking giu cac thong tin nay trong cung mot vung ngu canh tot hon fixed-size. Metadata filter giup giam nhieu truoc khi search, vi cac dieu kien nhu gia, khu vuc, cuisine thuong rat ro.

### So Sanh Voi Baseline

| Tai lieu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Restaurant docs | Fixed-size baseline | 2 moi doc | 109-116 | On nhung co nguy co cat ngang thong tin |
| Restaurant docs | Sentence baseline | 1 moi doc | 206-219 | Giu nghia tot nhung it hat retrieval hon |
| Restaurant docs | **Recursive + metadata filter** | 2-3 moi doc | 72-104 | Can bang giua ngu canh va do gon |

### So Sanh Voi Thanh Vien Khac

| Thanh vien | Strategy | Retrieval Score (/10) | Diem manh | Diem yeu |
|-----------|----------|----------------------|-----------|----------|
| Toi | Recursive + metadata filter | 8 | Giu cau truc, loc metadata tot | Mock embedding lam score khong on dinh |
| Ban A | Fixed-size | 6 | De kiem soat size | Co the cat ngang y |
| Ban B | Sentence chunking | 7 | Giu tron cau | Chunk dai neu tai lieu nhieu cau lien quan |

Strategy tot nhat cho domain nay la recursive chunking ket hop metadata filtering. Ly do la domain co nhieu rang buoc co cau truc, nen loc metadata truoc giup giam nhieu, sau do semantic search chi can xep hang trong tap ung vien da hop ly.

---

## 4. My Approach - Ca Nhan

### Chunking Functions

**`SentenceChunker.chunk`:** Toi normalize whitespace, dung regex de tach cau theo `.`, `!`, `?`, sau do gom toi da `max_sentences_per_chunk` cau vao mot chunk. Cach nay giu lai dau cau va loai bo chunk rong.

**`RecursiveChunker.chunk` / `_split`:** Thuat toan co base case la text rong hoac text ngan hon `chunk_size`. Neu doan qua dai, no thu tung separator theo thu tu uu tien; neu khong con separator thi fallback cat theo ky tu de khong crash.

### EmbeddingStore

**`add_documents` + `search`:** Store dung in-memory list de on dinh cho test. Moi document duoc normalize thanh record gom `id`, `doc_id`, `content`, `metadata`, `embedding`; search tao embedding cho query, tinh cosine similarity, sort giam dan theo `score`, va tra ve toi da `top_k`.

**`search_with_filter` + `delete_document`:** Metadata filter duoc ap dung truoc semantic search bang exact match. Delete xoa tat ca record co `doc_id` trung voi input va tra ve `True` neu collection size giam.

### KnowledgeBaseAgent

**`answer`:** Agent goi `store.search(question, top_k)`, build prompt co phan `Retrieved context` va `Question`, yeu cau chi tra loi dua tren context. Neu khong co context, agent tra ve cau an toan thay vi tu suy dien.

### Test Results

```text
pytest tests/ -v
42 passed in 0.13s
```

**So tests pass:** 42 / 42

---

## 5. Similarity Predictions

Ket qua dung mock embedding cua repo, nen score chi mang tinh deterministic smoke test, khong phai semantic model that.

| Pair | Sentence A | Sentence B | Du doan | Actual Score | Dung? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Need a cheap Korean place for a group dinner. | Seoul Corner is a low-price Korean restaurant for groups. | high | 0.158 | Tuong doi |
| 2 | Vegetarian options are required. | The restaurant has vegan bowls and allergy notes. | high | 0.037 | Mot phan |
| 3 | We need a quiet place to talk. | Comfortable seating is good for sitting a long time. | high | 0.119 | Tuong doi |
| 4 | Find sushi for two people. | A large team needs Vietnamese noodles under 100k. | low | 0.096 | Khong ro |
| 5 | Open until late evening. | The price is high and seating is limited. | low | -0.138 | Dung |

Ket qua bat ngo nhat la pair 4 co score duong du khac chu de. Dieu nay cho thay mock embedding chi phu hop cho test deterministic, khong thay the duoc embedding semantic that khi can chat luong retrieval cao.

---

## 6. Results - Ca Nhan

### Benchmark Queries & Gold Answers

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Tim quan phu hop cho nhom 5 nguoi, gia duoi 100k/nguoi. | Pho Harmony hoac Seoul Corner; ca hai low price va group_friendly. |
| 2 | Co quan nao gan truong, phu hop an toi nhom khong? | Seoul Corner gan university area, phu hop nhom 5-6 nguoi, mo den 22:00. |
| 3 | Nhom co nguoi an chay thi nen chon quan nao? | Green Table la lua chon tot nhat; Pho Harmony/Bistro Late co vegetarian options. |
| 4 | Quan nao phu hop de ngoi lau va noi chuyen? | Bistro Late hoac Green Table vi co seating thoai mai/quiet seating. |
| 5 | Neu nhom thich mon Han va ngan sach thap thi nen chon quan nao? | Seoul Corner. |

### Ket Qua Cua Toi

| # | Query | Top-1 Retrieved Chunk | Score | Relevant? | Agent Answer |
|---|-------|----------------------|-------|-----------|--------------|
| 1 | Group of 5 under 100k | Pho Harmony: Vietnamese, 85k/person, group tables | 0.048 | Yes | Chon Pho Harmony; Seoul Corner cung la ung vien |
| 2 | Near university for dinner group | Seoul Corner: university area, Korean, open until 22:00 | -0.133 | Yes | Chon Seoul Corner |
| 3 | Someone vegetarian | Pho Harmony top-1, Green Table top-2 | 0.258 | Yes | Green Table tot nhat neu uu tien vegetarian |
| 4 | Sit long and talk | Seoul Corner top-1, Bistro Late top-3 | 0.140 | Partly | Nen dung metadata/score threshold vi mock embedding xep hang chua tot |
| 5 | Low budget Korean group | Seoul Corner | -0.065 | Yes | Chon Seoul Corner |

**Bao nhieu queries tra ve chunk relevant trong top-3?** 5 / 5  
**Top-1 relevant:** 4 / 5

---

## 7. Evaluation

**Retrieval Precision:** Top-3 co chunk relevant cho 5/5 benchmark queries khi dung metadata filter. Top-1 thinh thoang chua tot vi mock embedding khong nam semantic similarity thuc su.

**Chunk Coherence:** Recursive chunking giu y tot hon fixed-size vi uu tien paragraph/sentence/word. Cac chunk khong bi rong va phan lon nam duoi muc `chunk_size`.

**Metadata Utility:** Metadata nhu `district`, `price_level`, `group_friendly`, `cuisine`, `dietary_options` cai thien ket qua ro ret. Vi du query Korean low budget duoc loc truoc bang `cuisine=Korean` va `price_level=low`.

**Grounding Quality:** Agent build prompt tu retrieved context va yeu cau chi tra loi dua tren context. Neu khong retrieve duoc chunk nao, agent tra ve thong bao khong du thong tin trong knowledge base.

**Data Strategy Impact:** Chat luong du lieu quan trong hon viec chon model trong lab nay. Neu thieu metadata ve gia, khu vuc, gio mo cua, hoac dietary options, agent se kho tra loi dung cac cau hoi co rang buoc.

---

## 8. Failure Cases

1. **Query qua mo ho:** "Quan nao ngon?" khong co tieu chi ve gia, khu vuc, mon an, hay nhom may nguoi. Cai thien bang cach hoi lai user hoac them prompt clarification.

2. **Thieu metadata gia/khu vuc:** Neu document chi co review text, search co the tra ve quan hay nhung khong dung ngan sach. Cai thien bang schema bat buoc `price_level`, `district`, `group_friendly`.

3. **Chunk qua ngan:** Neu cat moi field thanh mot chunk rieng, thong tin "gia thap" co the tach khoi "phu hop nhom". Cai thien bang recursive chunking va chunk size du de giu thong tin lien quan.

4. **Mock embedding khong semantic:** Mock embedding co the cho score am voi chunk dung hoac score duong voi chunk nhieu. Cai thien bang local `all-MiniLM-L6-v2` hoac OpenAI embedding khi lam demo that.

5. **Khong co score threshold:** Store hien tra ve top-k ngay ca khi score thap. Cai thien bang them threshold va cau tra loi "khong du thong tin" khi diem qua thap.

---

## 9. What I Learned

Dieu hay nhat hoc duoc tu so sanh strategy la metadata filtering thuong quan trong ngang, thậm chi hon chunking, trong cac domain co rang buoc ro nhu chon nha hang. Chunking tot giup giu ngu canh, nhung metadata moi giup loai bo ung vien sai ngay tu dau.

Neu lam lai, toi se tang chat luong dataset: moi quan nen co gia cap nhat, gio mo cua, do on, suc chua nhom, tag an chay/di ung, va review ngan gon. Toi cung se dung embedding model that de danh gia retrieval semantic cong bang hon.

---

## 10. Ket Luan

Data quality quan trong hon model selection trong lab nay. Strategy tot nhat cho domain AI Restaurant Chooser for Groups la recursive chunking ket hop metadata filtering va `top_k=3`. RAG giup agent giam hallucination vi cau tra loi duoc grounding bang retrieved context, dong thoi co duong lui an toan khi knowledge base khong du thong tin.

---

## Tu Danh Gia

| Tieu chi | Loai | Diem tu danh gia |
|----------|------|-------------------|
| Warm-up | Ca nhan | 5 / 5 |
| Document selection | Nhom | 9 / 10 |
| Chunking strategy | Nhom | 14 / 15 |
| My approach | Ca nhan | 10 / 10 |
| Similarity predictions | Ca nhan | 4 / 5 |
| Results | Ca nhan | 9 / 10 |
| Core implementation (tests) | Ca nhan | 30 / 30 |
| Demo | Nhom | 4 / 5 |
| **Tong** | | **85 / 100** |
