# AIC Video Retrieval System 🎬🔍

Hệ thống truy vấn video (Text-to-Video / Keyframe Retrieval) phục vụ cuộc thi AI Challenge (AIC), xây dựng trên mô hình CLIP (Contrastive Language-Image Pre-training) cùng giao diện trực quan Streamlit.

---

## 🌟 Tính Năng Chính
- **Text Query Retrieval**: Tìm kiếm cảnh/khung hình (Keyframes) bằng câu lệnh tiếng Việt hoặc tiếng Anh (tự động dịch đa ngữ qua Deep Translator).
- **Trực quan hóa kết quả (Streamlit Web UI)**: Hiển thị top-K keyframes, thông tin video, mốc thời gian (FPS/PTS), điểm tương đồng (Cosine Similarity).
- **Multi-query & Batch Processing**: Tự động duyệt và xử lý các bộ đề thi (`.txt`, KIS, QA).
- **Xuất file nộp bài chuẩn định dạng**: Hỗ trợ xuất file nộp bài định dạng CSV/TXT theo quy chuẩn của BTC.
- **Tiện ích tải dữ liệu**: Hỗ trợ tải trực tiếp dữ liệu/keyframes từ Google Drive bằng `gdown`.

---

## 📁 Cấu Trúc Dự Án
```text
├── core/
│   ├── retrieval_engine.py      # Module nạp đặc trưng, xử lý truy vấn CLIP
│   └── submission_exporter.py   # Module xuất kết quả nộp bài chuẩn AIC
├── utils/
│   └── download_from_drive.py   # Công cụ tải & giải nén dữ liệu từ Google Drive
├── app.py                       # Giao diện Web tương tác (Streamlit)
├── build_dataset_map.py         # Script xây dựng bản đồ map keyframes
├── extract_image_features.py    # Script trích xuất vector đặc trưng hình ảnh
├── main_baseline.py             # Script baseline tìm kiếm cơ bản
├── main_pipeline.py             # Pipeline tự động chạy các bộ đề thi
├── requirements.txt             # Danh sách thư viện phụ thuộc
├── .env.example                 # File cấu hình mẫu
└── .gitignore                   # Cấu hình bỏ qua file nặng & thông tin bảo mật
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy

### 1. Cài đặt môi trường
Khuyến nghị sử dụng Python 3.9 - 3.11:
```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo
# Trên Windows:
.venv\Scripts\activate
# Trên Linux/macOS:
source .venv/bin/activate

# Cài đặt các thư viện
pip install -r requirements.txt
```

### 2. Cấu hình file `.env`
Tạo file `.env` từ file mẫu:
```bash
cp .env.example .env
```

### 3. Chuẩn bị dữ liệu
Đặt các file dữ liệu (không đẩy lên GitHub vì dung lượng lớn) vào thư mục gốc:
- `clip_features.npy`: File ma trận vector đặc trưng đã trích xuất.
- `map_keyframes.json`: File danh sách đường dẫn tương ứng với vector.
- `Keyframes_*`: Thư mục chứa ảnh keyframe.

Nếu cần tải từ Google Drive:
```bash
python utils/download_from_drive.py "<GOOGLE_DRIVE_URL_HOẶC_ID>"
```

### 4. Khởi chạy ứng dụng Web (Streamlit)
```bash
streamlit run app.py
```

### 5. Chạy pipeline batch query
```bash
python main_pipeline.py --query-dir "THUNGHIEM-bo-de-thi" --top-k 100
```

---

## 🔒 Lưu Ý Bảo Mật & Đẩy Lên GitHub
- Tuyệt đối **không** push file `.env` chứa token/key cá nhân.
- Không push các file dữ liệu lớn (`.npy`, `.pt`, `Keyframes_*`, `clip-features-*`) vì vượt quá giới hạn 100MB của GitHub.
- Sử dụng file `.gitignore` đã được cấu hình sẵn trong repository.
