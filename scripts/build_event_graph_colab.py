"""
build_event_graph_colab.py - Event Extraction Pipeline cho Google Colab (Chạy song song)
Tích hợp YOLOv8 (Tracking) + VideoMAE V2 (Action Recognition) -> Xuất ra file CSV.
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
import argparse
import shutil
from tqdm import tqdm

# Models
from ultralytics import YOLO
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

YOLO_MODEL = "yolov8n.pt"  
VIDEOMAE_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_LENGTH = 16  # VideoMAE yêu cầu 16 frames

# [TỐI ƯU HÓA] Các classes cần trích xuất hành động (Bỏ qua ghế, bàn, ly, chén...)
TARGET_CLASSES = {'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'}
# [TỐI ƯU HÓA] Bước nhảy frame (Chỉ xử lý 1/N frame để chạy lẹ hơn)
FRAME_SKIP = 3  
# [TỐI ƯU HÓA] Kích thước tối thiểu của vật thể (pixel)
MIN_BOX_SIZE = 50 

class ColabEventExtractor:
    def __init__(self):
        print(f"[INIT] Thiết bị xử lý: {DEVICE}")
        
        print("[INIT] Load YOLOv8...")
        self.yolo = YOLO(YOLO_MODEL)

        print(f"[INIT] Load VideoMAE V2...")
        self.action_processor = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_MODEL)
        self.action_model = VideoMAEForVideoClassification.from_pretrained(VIDEOMAE_MODEL).to(DEVICE)
        self.action_model.eval()

        self.id_to_label = self.action_model.config.id2label

    def recognize_action(self, frames_list: list) -> str:
        """Dự đoán hành động từ 16 frames của vật thể."""
        if len(frames_list) < CLIP_LENGTH:
            return "unknown"
            
        # Chọn 16 frames cách đều nhau
        indices = np.linspace(0, len(frames_list) - 1, num=CLIP_LENGTH, dtype=int)
        sampled_frames = [frames_list[i] for i in indices]
        
        processed_frames = []
        for f in sampled_frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            processed_frames.append(rgb)

        inputs = self.action_processor(list(processed_frames), return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = self.action_model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            
        action_name = self.id_to_label[predicted_class_idx]
        return action_name.lower().replace("_", " ")

    def process_video(self, video_path: str) -> list:
        """Xử lý 1 video và trả về danh sách các sự kiện (Events)"""
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        
        # [TỐI ƯU HÓA IO] Copy video từ Google Drive vào ổ cứng cục bộ của Colab để đọc mượt hơn
        local_video_path = f"/content/{os.path.basename(video_path)}"
        try:
            shutil.copy2(video_path, local_video_path)
        except Exception:
            local_video_path = video_path # Nếu lỗi (chạy ở máy cá nhân), dùng luôn đường dẫn gốc
            
        cap = cv2.VideoCapture(local_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Dictionary lưu buffer frames cho từng track_id để đưa vào VideoMAE
        # track_id -> {"label": str, "frames": list, "start_time": float}
        track_buffers = {}
        
        events = []
        frame_idx = 0
        
        print(f"\n▶️ Bắt đầu xử lý video {video_id} ({total_frames} frames)...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Thay vì dùng tqdm lồng nhau (hay bị lỗi hiển thị ở Colab), ta in ra mỗi 500 frame
            if frame_idx % 500 == 0:
                percent = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                print(f"   ⏳ {video_id}: {frame_idx}/{total_frames} frames ({percent:.1f}%)")
            
            # [TỐI ƯU HÓA] Bỏ qua frame để chạy nhanh hơn gấp N lần
            if frame_idx % FRAME_SKIP != 0:
                continue

            timestamp_sec = frame_idx / fps
            
            # 1. Chạy YOLO Tracking
            # Sử dụng persist=True để gán ID theo dõi liên tục cho vật thể
            results = self.yolo.track(frame, persist=True, verbose=False, tracker="botsort.yaml")
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                
                for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    label_str = self.yolo.names[cls_id]
                    
                    # Cắt hình ảnh vật thể (Crop)
                    obj_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                    if obj_crop.size == 0:
                        continue
                        
                    if track_id not in track_buffers:
                        track_buffers[track_id] = {
                            "label": label_str,
                            "frames": [],
                            "start_time": timestamp_sec
                        }
                    
                    track_buffers[track_id]["frames"].append(obj_crop)
                    
                    # Nếu đã thu thập đủ lượng frame cho 1 Action Clip (vd: 16 frames ~ 0.5s)
                    if len(track_buffers[track_id]["frames"]) >= CLIP_LENGTH:
                        # 2. Đưa vào VideoMAE để nhận diện hành động
                        action = self.recognize_action(track_buffers[track_id]["frames"])
                        
                        if action != "unknown":
                            events.append({
                                "video_id": video_id,
                                "timestamp": track_buffers[track_id]["start_time"],
                                "entity_id": f"{video_id}_obj_{track_id}",
                                "entity_type": label_str,
                                "action": action
                            })
                        
                        # Xóa buffer để bắt đầu nhận diện hành động tiếp theo của vật thể này
                        track_buffers[track_id]["frames"] = []
                        track_buffers[track_id]["start_time"] = timestamp_sec

        cap.release()
        
        # Xóa file video cục bộ để giải phóng dung lượng Colab
        if local_video_path.startswith("/content/") and os.path.exists(local_video_path):
            os.remove(local_video_path)
            
        print(f"✅ Hoàn thành video {video_id}! Tìm thấy {len(events)} sự kiện.")
        return events

def main():
    parser = argparse.ArgumentParser(description="Chạy trích xuất Event Graph song song trên Colab")
    parser.add_argument("--video_dir", type=str, required=True, help="Thư mục chứa video")
    parser.add_argument("--total_parts", type=int, default=5, help="Tổng số phần chia (chạy song song)")
    parser.add_argument("--part_idx", type=int, required=True, help="Chỉ số phần hiện tại (0 đến total_parts-1)")
    parser.add_argument("--out_csv", type=str, default="event_graph.csv", help="Tên file xuất ra")
    args = parser.parse_args()

    # Lấy danh sách video (Sử dụng Cache để tránh 5 Tab cùng cày nát Google Drive API)
    list_cache_file = os.path.join(args.video_dir, "cached_video_list.txt")
    
    if os.path.exists(list_cache_file):
        print(f"📦 Đang đọc danh sách video từ file cache: {list_cache_file}")
        with open(list_cache_file, "r", encoding="utf-8") as f:
            all_videos = [line.strip() for line in f if line.strip()]
        print(f"✅ Đã tải xong danh sách {len(all_videos)} video từ Cache.")
    else:
        print(f"🔍 Đang quét tìm video trong {args.video_dir}...")
        print("⏳ (Lưu ý: Quét file trên Google Drive có thể mất từ 2-5 phút, vui lòng kiên nhẫn...)")
        
        all_videos = []
        for root, dirs, files in os.walk(args.video_dir):
            # Tối ưu hóa cực mạnh: Bỏ qua các thư mục chứa hàng triệu file rác (Keyframes, Objects, Metadata)
            # Dòng này giúp os.walk KHÔNG đi sâu vào các thư mục đó, tránh treo Google Drive.
            dirs[:] = [d for d in dirs if not any(skip in d for skip in ["Keyframes", "build_event_graph_colab.py", "Metadata", "clip-features-32", "map-keyframes", "objects", "media-info"])]
            
            for f in files:
                if f.endswith((".mp4", ".avi", ".mkv")):
                    all_videos.append(os.path.join(root, f))
                    if len(all_videos) % 50 == 0:
                        print(f"   ... Đã tìm thấy {len(all_videos)} video ...")
        
        all_videos = sorted(all_videos)
        print(f"✅ Quét xong! Tổng cộng tìm thấy {len(all_videos)} video.")
        
        # Lưu cache cho các tab khác
        try:
            with open(list_cache_file, "w", encoding="utf-8") as f:
                for v in all_videos:
                    f.write(v + "\n")
            print(f"💾 Đã lưu cache vào {list_cache_file} để các Tab khác chạy cực nhanh.")
        except Exception as e:
            print(f"⚠️ Không thể lưu file cache (nhưng tiến trình vẫn tiếp tục): {e}")
    
    if not all_videos:
        print(f"❌ Không tìm thấy video nào trong {args.video_dir} và các thư mục con")
        return
        
    # Lọc các video thuộc Part hiện tại
    part_videos = all_videos[args.part_idx :: args.total_parts]
    
    # Tính năng Resume: Bỏ qua các video đã xử lý
    processed_video_ids = set()
    if os.path.exists(args.out_csv):
        try:
            df_existing = pd.read_csv(args.out_csv)
            if "video_id" in df_existing.columns:
                processed_video_ids = set(df_existing["video_id"].astype(str).unique())
                print(f"🔄 [RESUME] Đã tìm thấy {len(processed_video_ids)} video đã xử lý trong {args.out_csv}. Sẽ bỏ qua các video này.")
        except Exception as e:
            print(f"⚠️ Lỗi đọc file CSV cũ: {e}")

    # Chỉ giữ lại các video chưa xử lý
    videos_to_process = []
    for v_path in part_videos:
        v_name = os.path.basename(v_path)
        v_id = os.path.splitext(v_name)[0]
        if v_id not in processed_video_ids:
            videos_to_process.append(v_path)

    print(f"📦 Part {args.part_idx + 1}/{args.total_parts}: Cần xử lý {len(videos_to_process)}/{len(part_videos)} videos")
    
    if not videos_to_process:
        print("🎉 Tuyệt vời! Toàn bộ video của phần này đã được xử lý xong.")
        return

    extractor = ColabEventExtractor()
    
    # Tạo Header nếu file CSV chưa tồn tại
    if not os.path.exists(args.out_csv):
        pd.DataFrame(columns=["video_id", "timestamp", "entity_id", "entity_type", "action"]).to_csv(args.out_csv, index=False)
    
    # Xử lý và lưu ngay sau MỖI VIDEO
    for v_path in tqdm(videos_to_process, desc=f"Part {args.part_idx}"):
        v_name = os.path.basename(v_path)
        try:
            events = extractor.process_video(v_path)
            if events:
                df = pd.DataFrame(events)
                df.to_csv(args.out_csv, mode='a', header=False, index=False)
        except Exception as e:
            print(f"⚠️ Lỗi video {v_name}: {e}")
            
    print(f"✅ Hoàn tất Part {args.part_idx}!")

if __name__ == "__main__":
    main()
