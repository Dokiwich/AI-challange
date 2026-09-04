# 🎬 AIC 2026 Video Retrieval System V4.0 (Graph-Enhanced Architecture)

Hệ thống truy vấn video (Text-to-Video / Keyframe Retrieval) phục vụ cuộc thi AI Challenge (AIC), xây dựng trên kiến trúc V4.0 kết hợp sức mạnh của VectorDB (Qdrant), GraphDB (Neo4j), và Grounding DINO. Hệ thống không chỉ "nhìn" bằng độ tương đồng ngữ nghĩa CLIP mà còn "hiểu" được ngữ cảnh, đếm được số lượng và theo vết đối tượng qua thời gian.

---

## 🌟 Tính Năng & Công Nghệ Lõi (V4.0)

1. **Neo4j Graph Database ("Hòa mạng" Chuỗi Sự Kiện & Đối Tượng)**
   - Lưu trữ toàn bộ thông tin hàng triệu đối tượng (JSON) được nhận diện trước. Trình biên dịch Cypher mạnh mẽ giúp ép buộc tính thứ tự (TRAKE) và lọc kết quả siêu tốc thay vì dùng LLM duyệt từng ảnh.
2. **Qdrant VectorDB (Truy Xuất Tốc Độ Cao)**
   - Lưu trữ vector nhúng của CLIP, xử lý tìm kiếm trên tập 50GB video siêu mượt, loại bỏ thắt cổ chai CPU/RAM của Numpy mảng truyền thống.
3. **Grounding DINO (Open-Vocabulary Sniper)**
   - Mô hình Zero-Shot đóng vai trò "lính bắn tỉa", xác nhận chính xác các tính từ phức tạp và đếm số lượng vật thể khi bảng từ vựng tĩnh (Neo4j) bó tay. Tốc độ cực nhanh vì chỉ chạy trên top 300 khung hình.
4. **Skip-Aware Monotonic DP (Temporal Alignment)**
   - Thuật toán quy hoạch động bắt dính sự kiện theo thời gian mà không cần tải LLM nặng nề để xem video, tự động bắt mượt các pha hành động.
5. **AI Query Parser (OpenRouter / Gemma)**
   - Phân tích câu hỏi QA hoặc KIS để trích xuất ngữ cảnh bằng OpenRouter (`google/gemma-4-26b-a4b-it:free`), siêu tốc và an toàn, giải phóng hoàn toàn gánh nặng VRAM của máy trạm.

---

## 📁 Cấu Trúc Thư Mục (Directory Tree)

```text
├── app.py                     # Giao diện Web Streamlit UI chính (Visual QA, KIS, TRAKE, Batch)
├── main_pipeline.py           # CLI Pipeline chạy hàng loạt file đề thi (.txt) -> Tự động nén submission.zip
├── evaluate_pipeline.py       # Script chấm điểm và phân tích hệ thống
├── requirements.txt           # Danh sách thư viện (neo4j, qdrant-client, ultralytics...)
├── submission/                # Nơi xuất kết quả nộp bài tự động
├── de-thi/                    # Nơi để các file truy vấn .txt của BTC
│
├── core/                      # Trái tim của hệ thống (Core Engine V4)
│   ├── retrieval_engine.py    # Điều phối 4 Track, nay tích hợp VLM QA Generator
│   ├── base_retriever.py      # Load mô hình CLIP
│   ├── query_compiler.py      # Biên dịch truy vấn bằng LLM
│   ├── meta_router.py         # Định tuyến thông minh tới các Track
│   ├── graph_matcher.py       # Trình biên dịch Neo4j Cypher ép buộc chuỗi thời gian cho TRAKE
│   ├── evidence_engine.py     # Thẩm phán chứng cứ 3-Level Veto
│   ├── temporal_alignment.py  # Thuật toán Skip-Aware Monotonic DP (Fallback)
│   └── submission_exporter.py # Đóng gói kết quả thành CSV chuẩn BTC
│
└── utils/                     # Tiện ích
    └── download_from_drive.py # Tool tải dữ liệu từ Google Drive
```

*Lưu ý: Các đoạn script trích xuất dữ liệu gốc (`scripts/`) và thư mục `csv/` tạm đã được gộp và xóa đi nhằm làm sạch dự án cho giai đoạn truy vấn chính thức.*

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy

### 1. Chuẩn Bị Dữ Liệu Cốt Lõi (Bắt Buộc)
Do giới hạn lưu trữ của Git, toàn bộ các file dữ liệu khổng lồ **KHÔNG** có sẵn trong repo này. Khi cài đặt trên máy chủ mới, bạn cần thao tác copy thủ công:
- **Thư mục Vector:** Copy thư mục `data/features/` (Chứa các file `clip_features.npy` ~1.25GB) vào chung thư mục `data/` của dự án để tìm kiếm bằng CLIP.
- **Thư mục Đồ thị:** Copy thư mục `data/objects/` (Chứa hàng chục nghìn file JSON xuất từ CLIP/DINO) vào `data/objects/` của dự án để chuẩn bị nạp vào Neo4j.
*(Các file cấu hình nhẹ như `data/mapping/map_keyframes.json` đã có sẵn)*

### 2. Cấu Hình Biến Môi Trường
1. Copy file mẫu `.env.example` thành `.env`
2. Mở `.env` và điền khóa API của OpenRouter để kích hoạt Trí tuệ nhân tạo:
```env
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxx
OPENROUTER_MODEL=google/gemma-4-26b-a4b-it:free
```

### 3. Khởi Chạy Cơ Sở Dữ Liệu (Docker)
Hệ thống cần 2 Database chạy song song (Yêu cầu phải cài đặt Docker):
```bash
# Khởi chạy Qdrant (VectorDB) trên cổng 6333
docker run -d -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant

# Khởi chạy Neo4j (GraphDB) trên cổng 7687
docker run -d -p 7474:7474 -p 7687:7687 -v neo4j_data:/data -v neo4j_logs:/logs -e NEO4J_AUTH=neo4j/password neo4j:latest
```

### 4. Nạp Dữ Liệu Khởi Tạo (Seed Database)
Sau khi Neo4j đã chạy, bạn bắt buộc phải chạy kịch bản nạp dữ liệu Object JSON vào đồ thị (Chỉ cần chạy 1 lần duy nhất trên máy mới):
```bash
python scripts/import_json_objects_to_neo4j.py
```
*(Quá trình này sẽ mất một lúc do số lượng JSON rất lớn).*

### 5. Khởi Chạy Ứng Dụng
**Mở Giao diện Web Trực Quan (Streamlit):**
```bash
streamlit run app.py
```
**Hoặc Chạy tự động (Batch Pipeline) không cần UI:**
```bash
python main_pipeline.py --dir de-thi --engine_mode meta
```
Hệ thống sẽ duyệt qua mọi file truy vấn `.txt`, nội suy đề KIS/QA/TRAKE và nén toàn bộ kết quả vào `submission.zip`.

> **[LƯU Ý QUAN TRỌNG]**: Đọc thêm [CONFIG_GUIDE.md](CONFIG_GUIDE.md) để biết cách sử dụng tính năng "Giỏ hàng Chốt hạng" (Ranked Submission Builder) và tối ưu hóa VRAM.
