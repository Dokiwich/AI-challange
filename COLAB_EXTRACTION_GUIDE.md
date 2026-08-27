# Hướng Dẫn Trích Xuất Event Graph (Chia Việc Cho 5 Người/5 Acc)

Vì số lượng video quá lớn (955 video, mỗi video 30 phút), việc chạy trên 1 tài khoản Colab duy nhất sẽ mất nhiều ngày. Tài liệu này hướng dẫn cách **chia nhỏ danh sách video ra làm 5 phần bằng nhau**, giao cho 5 người (hoặc 5 tài khoản Gmail/Colab khác nhau) chạy song song để tiết kiệm thời gian.

Mọi người sẽ cùng trỏ vào một thư mục gốc trên Google Drive đã được chia sẻ chung.

## Bước 1: Chuẩn bị dữ liệu chung trên Google Drive
1. Tạo 1 thư mục Mẹ (Ví dụ: `AIC_2026_Videos`) chứa toàn bộ các video.
2. Tải file code `build_event_graph_colab.py` (bản ĐÃ TỐI ƯU) lên cùng thư mục đó hoặc thư mục gốc.
3. Click chuột phải vào thư mục Mẹ -> Chọn **Share (Chia sẻ)** cho 4 tài khoản Gmail còn lại với quyền Editor (Người chỉnh sửa).
4. **Trên 4 tài khoản kia:** Vào mục *Shared with me (Được chia sẻ với tôi)*, click chuột phải vào thư mục đó -> Chọn **Add shortcut to Drive (Thêm phím tắt vào Drive)**.

## Bước 2: Tạo Notebook trên Google Colab
Mỗi người (hoặc mỗi tài khoản) truy cập [Google Colab](https://colab.research.google.com/) và tạo **1 Notebook mới**. Đảm bảo đã bật GPU (`Runtime` > `Change runtime type` > `T4 GPU`).

Chạy 3 lệnh thiết lập môi trường chung cho cả 5 người:

**Cell 1: Kết nối Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```
**Cell 2: Cài đặt thư viện**
```bash
!pip install ultralytics transformers pandas opencv-python-headless tqdm lapx
```
**Cell 3: Chuyển hướng tới thư mục code**
```bash
%cd /content/drive/MyDrive/Thu_Muc_Chua_Script
```

## Bước 3: Phân công chạy 5 phần (Mỗi người chạy 1 lệnh riêng biệt)

Code đã được tích hợp sẵn tính năng tự động chia đều danh sách 955 video ra làm 5 phần. Bạn chỉ cần copy đúng dòng lệnh được giao:

### Người thứ 1 (Phần 0)
```bash
!python build_event_graph_colab.py --video_dir "/content/drive/MyDrive/AIC_2026_Videos" --out_csv "/content/drive/MyDrive/event_graph_part0.csv" --total_parts 5 --part_idx 0
```

### Người thứ 2 (Phần 1)
```bash
!python build_event_graph_colab.py --video_dir "/content/drive/MyDrive/AIC_2026_Videos" --out_csv "/content/drive/MyDrive/event_graph_part1.csv" --total_parts 5 --part_idx 1
```

### Người thứ 3 (Phần 2)
```bash
!python build_event_graph_colab.py --video_dir "/content/drive/MyDrive/AIC_2026_Videos" --out_csv "/content/drive/MyDrive/event_graph_part2.csv" --total_parts 5 --part_idx 2
```

### Người thứ 4 (Phần 3)
```bash
!python build_event_graph_colab.py --video_dir "/content/drive/MyDrive/AIC_2026_Videos" --out_csv "/content/drive/MyDrive/event_graph_part3.csv" --total_parts 5 --part_idx 3
```

### Người thứ 5 (Phần 4)
```bash
!python build_event_graph_colab.py --video_dir "/content/drive/MyDrive/AIC_2026_Videos" --out_csv "/content/drive/MyDrive/event_graph_part4.csv" --total_parts 5 --part_idx 4
```

> **Giải thích lệnh:**
> - `--total_parts 5`: Báo cho code biết tổng cộng có 5 người đang chia nhau làm.
> - `--part_idx`: Đánh dấu phần việc của từng người (từ 0 đến 4). Code sẽ tự động lọc danh sách video và chỉ chạy phần của riêng người đó.
> - `--out_csv`: **Quan trọng!** Tên file CSV xuất ra phải khác nhau (part0, part1...) để không bị ghi đè lẫn nhau trên Google Drive chung.

## Bước 4: Tính năng Resume (Chạy tiếp khi bị ngắt)
Nếu Colab của ai đó bị ngắt kết nối giữa chừng (vượt quá 4 tiếng), người đó chỉ cần:
1. Đổi sang một Gmail dự phòng khác (hoặc nhờ người khác chạy dùm phần của mình).
2. Chạy lại y hệt dòng lệnh được giao ban đầu.
3. Code sẽ tự động đọc file `event_graph_partX.csv` tương ứng, và in ra: `🔄 [RESUME] Đã tìm thấy ... video đã xử lý`. 
4. Nó sẽ **tự động bỏ qua các video đã làm xong** và chạy tiếp phần còn lại!

## Bước 5: Thu thập kết quả
Khi cả 5 người đều nhận được thông báo `🎉 Tuyệt vời! Toàn bộ video của phần này đã được xử lý xong.`, trên Google Drive sẽ có 5 file CSV (từ `part0` đến `part4`). Bạn chỉ cần tải 5 file này về máy tính gộp lại là có toàn bộ Event Graph!
