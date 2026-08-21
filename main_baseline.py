import os
import zipfile
import json
import numpy as np
import pandas as pd
import torch
import clip

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "ViT-B/32"  # Trùng khớp với mô hình trích xuất của BTC
NPY_FEATURES_PATH = "data/features/clip_features.npy"
KEYFRAME_MAP_PATH = "data/mapping/map_keyframes.json"

print(f"[INFO] Đang chạy trên thiết bị: {DEVICE}")

# ==========================================
# 1. TẢI MÔ HÌNH CLIP & DỮ LIỆU BASELINE
# ==========================================
print("[INFO] Đang tải mô hình CLIP-ViT-B-32...")
model, preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
model.eval()

print(f"[INFO] Đang nạp đặc trưng CLIP từ '{NPY_FEATURES_PATH}'...")
if not os.path.exists(NPY_FEATURES_PATH):
    raise FileNotFoundError(f"Không tìm thấy file: {NPY_FEATURES_PATH}. Hãy chạy 'build_dataset_map.py' trước!")

image_features = np.load(NPY_FEATURES_PATH)
image_features = torch.from_numpy(image_features).to(DEVICE).float()
# Chuẩn hóa (Normalize) các vector ảnh để tính Cosine Similarity chính xác
image_features /= image_features.norm(dim=-1, keepdim=True)
print(f"[INFO] Đã nạp xong {image_features.shape[0]} vector đặc trưng ảnh.")

print(f"[INFO] Đang nạp file map Keyframe từ '{KEYFRAME_MAP_PATH}'...")
if not os.path.exists(KEYFRAME_MAP_PATH):
    raise FileNotFoundError(f"Không tìm thấy file: {KEYFRAME_MAP_PATH}. Hãy chạy 'build_dataset_map.py' trước!")

with open(KEYFRAME_MAP_PATH, "r", encoding="utf-8") as f:
    keyframe_map = json.load(f)
print(f"[INFO] Đã nạp xong mapping cho {len(keyframe_map)} keyframes.")

# Hàm hỗ trợ phân tích định dạng file map của BTC
def parse_mapping(map_item):
    if isinstance(map_item, dict):
        video = map_item.get("video") or map_item.get("video_name")
        frame = map_item.get("frame") or map_item.get("frame_idx") or map_item.get("frame_id")
        return video, int(frame) if frame is not None else 1
    elif isinstance(map_item, str):
        # Ví dụ nếu map_item là "L01_V001/0000.jpg"
        parts = map_item.replace("\\", "/").split('/')
        video = parts[0]
        # Loại bỏ phần đuôi mở rộng để lấy số Frame ID dạng số nguyên
        frame_str = parts[-1].split('.')[0]
        try:
            frame = int(frame_str)
        except ValueError:
            frame = 1
        return video, frame
    elif isinstance(map_item, (list, tuple)) and len(map_item) >= 2:
        return map_item[0], int(map_item[1])
    return None, None


# ==========================================
# 2. HÀM TÌM KIẾM TƯƠNG ĐỒNG VECTOR
# ==========================================
def search_query(query_text, top_k=100):
    """
    Nhận vào 1 câu truy vấn, chuyển thành vector bằng CLIP, 
    so khớp với toàn bộ database ảnh và trả về danh sách top_k kết quả.
    """
    # Mã hóa câu chữ sang vector đặc trưng
    text_inputs = clip.tokenize([query_text], truncate=True).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Tính Cosine Similarity bằng tích vô hướng ma trận
    similarity = (text_features @ image_features.T).squeeze(0)
    
    # Lấy ra top_k kết quả cao điểm nhất
    actual_k = min(top_k, image_features.shape[0])
    top_values, top_indices = similarity.topk(actual_k)
    top_indices = top_indices.tolist()
    
    results = []
    for idx in top_indices:
        video_name, frame_id = parse_mapping(keyframe_map[idx])
        if video_name is not None and frame_id is not None:
            results.append((video_name, frame_id))
            
    return results


# ==========================================
# 3. ĐỌC ĐỀ BÀI & XUẤT FILE CSV KẾT QUẢ
# ==========================================
def process_challenge_round(query_txt_file, output_csv_name, query_type="kis"):
    """
    Đọc danh sách câu hỏi từ file .txt, thực hiện tìm kiếm,
    và ghi trực tiếp kết quả ra file .csv theo đúng quy chuẩn BTC.
    """
    if not os.path.exists(query_txt_file):
        print(f"[WARNING] Không tìm thấy file đề bài: {query_txt_file}. Bỏ qua.")
        return

    print(f"\n[PROCESS] Đang xử lý file câu hỏi: {query_txt_file}...")
    
    # Đọc các câu truy vấn từ file .txt (mỗi dòng là 1 câu hỏi)
    with open(query_txt_file, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]
        
    csv_rows = []
    
    # Tạo thư mục 'submission' nếu chưa có
    os.makedirs("submission", exist_ok=True)
    
    for i, q in enumerate(queries):
        print(f" -> Đang tìm kiếm câu {i+1}/{len(queries)}: '{q[:50]}...'")
        top_matches = search_query(q, top_k=100) # Đội thi được nộp tối đa 100 dòng cho mỗi câu
        
        for video, frame in top_matches:
            if query_type == "kis":
                # Định dạng KIS: <video_name>, <frame_id>
                csv_rows.append([video, frame])
            elif query_type == "qa":
                # Định dạng Q&A: <video_name>, <frame_id>, "<answer>"
                csv_rows.append([video, frame, "yes"])
                
    # Ghi dữ liệu ra file CSV chuẩn UTF-8, không có dòng tiêu đề (no header row)
    df = pd.DataFrame(csv_rows)
    output_path = os.path.join("submission", output_csv_name)
    df.to_csv(output_path, index=False, header=False, encoding="utf-8")
    print(f"[SUCCESS] Đã xuất file kết quả: {output_path} ({len(df)} dòng)")


# ==========================================
# 4. TỰ ĐỘNG ĐÓNG GÓI THÀNH FILE ZIP NỘP BÀI
# ==========================================
def zip_submission(zip_name="my_team_submission.zip"):
    """
    Nén toàn bộ thư mục 'submission' thành file .zip theo đúng yêu cầu BTC.
    """
    if not os.path.exists("submission"):
        print("[ERROR] Thư mục submission không tồn tại!")
        return
        
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk("submission"):
            for file in files:
                file_path = os.path.join(root, file)
                # Đảm bảo nén giữ đúng cấu trúc submission/<file.csv>
                zipf.write(file_path, file_path)
                
    print(f"\n[CONGRATS] Đã nén thành công file nộp bài: '{zip_name}'")
    print("[HINT] Bạn có thể lấy trực tiếp file này tải lên hệ thống chấm điểm của BTC!")


# ==========================================
# CHẠY TOÀN BỘ PIPELINE
# ==========================================
if __name__ == "__main__":
    # 1. Xử lý các câu hỏi dạng Textual KIS (ví dụ file query-1-kis.txt)
    process_challenge_round("query-1-kis.txt", "query-1-kis.csv", query_type="kis")
    
    # 2. Xử lý các câu hỏi dạng Hỏi-Đáp Q&A (ví dụ file query-3-qa.txt)
    process_challenge_round("query-3-qa.txt", "query-3-qa.csv", query_type="qa")
    
    # 3. Tiến hành đóng gói ZIP nộp bài
    zip_submission("my_team_submission.zip")
