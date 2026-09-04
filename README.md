# 🎬 AIC 2026 Video Retrieval System V4.0 (Graph-Enhanced Architecture)

Hệ thống truy vấn video (Text-to-Video / Keyframe Retrieval) phục vụ cuộc thi AI Challenge (AIC), xây dựng trên kiến trúc V4.0 kết hợp sức mạnh của VectorDB (Qdrant), GraphDB (Neo4j), và Grounding DINO. Hệ thống không chỉ "nhìn" bằng độ tương đồng ngữ nghĩa CLIP mà còn "hiểu" được ngữ cảnh, đếm được số lượng và theo vết đối tượng qua thời gian.

---

## 🌟 Tính Năng & Công Nghệ Lõi (V4.0)

1. **Neo4j Graph Database ("Hòa mạng" Chuỗi Sự Kiện)**
   - Giải quyết bài toán suy luận đa thực thể và định vị chuỗi thời gian (TRAKE). Trình biên dịch Cypher ép buộc tính thứ tự (`e1.timestamp < e2.timestamp`) trả về danh sách `matched_timestamps` chính xác.
2. **Qdrant VectorDB (Truy Xuất Tốc Độ Cao)**
   - Lưu trữ vector nhúng của CLIP, xử lý tìm kiếm trên tập 50GB video siêu mượt, loại bỏ thắt cổ chai CPU/RAM của Numpy mảng truyền thống.
3. **YOLOv8 & ByteTrack (Counting & Tracking)**
   - Dữ liệu bounding box và tracking được nạp vào GraphDB, đè bẹp điểm yếu "mù đếm số" và định vị tuyệt đối của mô hình CLIP.
4. **VideoMAE V2 (Video Context)**
   - Trích xuất hành động liên tục, kết hợp sức mạnh với CLIP để phân tích ngữ cảnh tĩnh và động.
5. **AI Query Parser (OpenRouter / Gemma)**
   - Phân tích câu hỏi QA hoặc KIS để trích xuất ngữ cảnh. Hỗ trợ "User-in-the-loop" trả lời câu hỏi QA một cách an toàn. chuẩn API của OpenRouter (`google/gemma-4-26b-a4b-it:free`).

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

### 3. Chuẩn bị dữ liệu
Các file đồ thị sự kiện đã được gộp lại tại `data/event_graph_merged_clean_395.csv`. Đảm bảo bạn đã tải bộ dữ liệu `clip_features.npy` và `map_keyframes.json` vào thư mục `data/`.

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
