# Hướng dẫn Cấu hình Hệ thống AIC 2026

Tài liệu này hướng dẫn chi tiết cách thiết lập môi trường, cấu hình cơ sở dữ liệu và tinh chỉnh các tính năng AI của hệ thống truy vấn video.

## 1. Yêu cầu hệ thống
- **Hệ điều hành:** Windows/Linux/macOS
- **Python:** 3.9 - 3.11
- **Docker:** Bắt buộc (để chạy Qdrant và Neo4j)
- **VRAM tối thiểu:** 4GB (để chạy mô hình CLIP và Grounding DINO cục bộ ở chế độ tiết kiệm).

---

## 2. Thiết lập Cơ sở dữ liệu (Database)

Hệ thống yêu cầu 2 cơ sở dữ liệu chạy song song qua Docker.

### A. Qdrant (Vector Database)
Lưu trữ toàn bộ vector CLIP của khung hình video.
```bash
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

### B. Neo4j (Graph Database)
Hỗ trợ truy xuất chuỗi sự kiện không gian thời gian (TRAKE) và liên kết ngữ nghĩa.
```bash
docker run -p 7474:7474 -p 7687:7687 \
  -v neo4j_data:/data -v neo4j_logs:/logs \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

---

## 3. Cấu hình biến môi trường (`.env`)
Tạo một file `.env` ở thư mục gốc của dự án (ngang hàng với `app.py`). Tuyệt đối **không đưa file này lên Github hay chia sẻ công khai** vì chứa khóa bảo mật.

Nội dung file `.env` chuẩn:
```env
# OpenRouter API (Cho tính năng AI Semantic Compiler)
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxx
OPENROUTER_MODEL=google/gemma-4-26b-a4b-it:free

# Qdrant Config
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Neo4j Config
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

---

## 4. Cấu hình Giao diện Người dùng (UI)

Khi chạy lệnh `streamlit run app.py`, hệ thống sẽ tải giao diện điều khiển. Một số tinh chỉnh quan trọng ở thanh Sidebar:

- **AI Mode vs Offline Mode:** 
  - *Bật AI Mode:* Hệ thống dùng OpenRouter để phân tích cấu trúc 6D của truy vấn, tìm ra "Hành động cốt lõi", "Chuỗi thời gian" (TRAKE).
  - *Tắt AI Mode:* Tìm kiếm truyền thống cục bộ 100%, tốc độ siêu nhanh nhưng độ chính xác chuỗi hành động giảm.
- **ASR (Lời thoại):** Bật nếu truy vấn chứa câu lệnh trích xuất từ âm thanh/lời nói (phải có file Text ASR đi kèm).

---

## 5. Hướng dẫn sử dụng "Ranked Submission Builder" (Giỏ hàng Chốt hạng)

Để đạt điểm tối đa (R@k) trong luật thi AIC, hệ thống cung cấp công cụ chốt đáp án thông minh.
- **Auto-Increment:** Khi duyệt qua kết quả AI, các khung hình chưa được chọn sẽ hiện số `0`. Nếu bấm `OK`, hệ thống tự động đưa khung hình đó vào Top hạng kế tiếp đang trống (Top 1, Top 2...).
- **Chốt TRAKE (Chuỗi sự kiện):** Với các câu hỏi TRAKE, mỗi Hạng (Rank) sẽ bao gồm một chuỗi các khung hình (Event 1, Event 2...). Việc gắp tay vào Giỏ Hàng sẽ tự động gom các khung hình này lại theo đúng 1 Video duy nhất để xuất ra file CSV hợp lệ.
- **Xuất CSV:** Hệ thống tự động lấp đầy 100 dòng. Các vị trí bạn chốt tay sẽ được giữ nguyên, những vị trí trống sẽ được hệ thống lấp đầy bằng kết quả của AI để đảm bảo chuẩn định dạng nộp bài của BTC.
