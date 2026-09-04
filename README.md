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

### 1. Cài đặt môi trường & Database
Khuyến nghị sử dụng Python 3.9 - 3.11:
```bash
# Tạo và kích hoạt môi trường ảo
python -m venv .venv
.\.venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt
```

> **[LƯU Ý QUAN TRỌNG]**: Vui lòng tham khảo file [CONFIG_GUIDE.md](file:///d:/AI%20challange/codeing/CONFIG_GUIDE.md) để biết cách cấu hình `.env`, thiết lập Docker cho Neo4j/Qdrant, và hướng dẫn sử dụng tính năng "Giỏ hàng Chốt hạng" (Ranked Submission Builder).

**Khởi chạy Docker:**
Yêu cầu bật Docker cục bộ để chạy hai database (cổng 7687 cho Neo4j và 6333 cho Qdrant).

### 2. Cấu hình file `.env`
Thiết lập API Key OpenRouter để AI phân tích cấu trúc truy vấn:
```env
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxx
OPENROUTER_MODEL=google/gemma-4-26b-a4b-it:free
```

### 3. Chuẩn bị dữ liệu (Quan trọng)
Vì giới hạn kỹ thuật của Git, các file dữ liệu khổng lồ **không được đưa lên GitHub**. Khi cài đặt máy chủ mới, bạn phải copy thủ công các dữ liệu sau vào dự án:
1. **Thư mục Vector:** Copy `data/features/` (Chứa các file `clip_features.npy` ~1.25GB) để tìm kiếm bằng CLIP.
2. **Thư mục Đồ thị (Nếu dùng Neo4j):** Copy `data/objects/` (Hàng chục nghìn file JSON). Sau khi copy, chạy script sau để nạp chúng vào cơ sở dữ liệu Neo4j:
   ```bash
   python scripts/import_json_objects_to_neo4j.py
   ```
Các file cấu hình nhẹ như `data/mapping/map_keyframes.json` đã có sẵn trong Git.

### 4. Khởi chạy Pipeline và Ứng dụng
**Mở Giao diện Web (Streamlit):**
```bash
streamlit run app.py
```

**Chạy tự động (Batch Pipeline):**
```bash
python main_pipeline.py --dir de-thi --engine_mode meta
```
Hệ thống sẽ duyệt qua mọi file `.txt`, nội suy đề KIS/QA/TRAKE và nén kết quả vào `submission.zip`.
