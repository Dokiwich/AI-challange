"""
build_event_graph.py - Event Extraction Pipeline
Tích hợp YOLOv8 (Tracking) + VideoMAE V2 (Action Recognition) -> Neo4j Graph DB.
"""

import os
import cv2
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv

# Models
from ultralytics import YOLO
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from neo4j import GraphDatabase

load_dotenv()

# =========================================================================
# CẤU HÌNH NEO4J & MODELS
# =========================================================================

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

YOLO_MODEL = "yolov8n.pt"  # Có thể đổi sang yolov8s.pt / yolov8m.pt
VIDEOMAE_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"  # Kinetics-400 Action Recognition
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_LENGTH = 16  # VideoMAE thường yêu cầu 16 frames

class EventGraphBuilder:
    def __init__(self):
        print("[INIT] Khởi tạo Neo4j Driver...")
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self._init_neo4j_schema()

        print("[INIT] Load YOLOv8...")
        self.yolo = YOLO(YOLO_MODEL)

        print(f"[INIT] Load VideoMAE V2 trên {DEVICE}...")
        self.action_processor = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_MODEL)
        self.action_model = VideoMAEForVideoClassification.from_pretrained(VIDEOMAE_MODEL).to(DEVICE)
        self.action_model.eval()

        self.id_to_label = self.action_model.config.id2label

    def _init_neo4j_schema(self):
        """Tạo index cho Neo4j để tăng tốc Cypher queries."""
        try:
            with self.driver.session() as session:
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (v:Video) REQUIRE v.id IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
            print("[Neo4j] Schema & Index đã sẵn sàng.")
        except Exception as e:
            print(f"[Neo4j] Cảnh báo lỗi Schema (có thể do DB chưa bật): {e}")

    def recognize_action(self, frames_list: list) -> str:
        """Dự đoán hành động từ 16 frames của vật thể."""
        if len(frames_list) < CLIP_LENGTH:
            return "unknown"
            
        # Chọn 16 frames cách đều nhau (uniform sampling) nếu có nhiều hơn 16
        indices = np.linspace(0, len(frames_list) - 1, num=CLIP_LENGTH, dtype=int)
        sampled_frames = [frames_list[i] for i in indices]
        
        # Resize về kích thước chuẩn (224x224 RGB)
        processed_frames = []
        for f in sampled_frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            # VideoMAEImageProcessor sẽ handle resize/normalize
            processed_frames.append(rgb)

        inputs = self.action_processor(list(processed_frames), return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = self.action_model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            
        action_name = self.id_to_label[predicted_class_idx]
        return action_name.lower().replace("_", " ")

    def write_to_neo4j(self, video_id: str, track_id: int, label_str: str, action: str, timestamp_sec: float):
        """Ghi dữ liệu (Entity)-[:PERFORMS]->(Action) vào Đồ thị"""
        if action == "unknown":
            return
            
        entity_id = f"{video_id}_obj_{track_id}"
        frame_id = f"{video_id}_{int(timestamp_sec)}"

        cypher = """
        MERGE (v:Video {id: $video_id})
        MERGE (f:Frame {id: $frame_id, timestamp: $timestamp_sec})
        MERGE (v)-[:HAS_FRAME]->(f)
        MERGE (e:Entity {id: $entity_id, type: $label_str})
        MERGE (f)-[:CONTAINS]->(e)
        WITH e
        // Action relationship (Dạng simplified: Entity tự Performs Action)
        // Nếu cần target object, sẽ thêm logic tìm bbox giao nhau sau
        MERGE (e)-[act:PERFORMS {action: $action}]->(e) 
        """
        try:
            with self.driver.session() as session:
                session.run(
                    cypher, 
                    video_id=video_id, 
                    frame_id=frame_id, 
                    timestamp_sec=timestamp_sec,
                    entity_id=entity_id,
                    label_str=label_str,
                    action=action
                )
        except Exception as e:
            print(f"[Neo4j Error] {e}")

    def process_video(self, video_path: str, video_id: str):
        print(f"\n[Processing] Video: {video_id}")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        # Buffer lưu frames (crops) cho từng track ID
        track_buffers = defaultdict(list)
        track_labels = {}
        
        # Chạy tracking YOLOv8
        results = self.yolo.track(video_path, stream=True, persist=True, verbose=False)
        
        frame_count = 0
        for r in tqdm(results, desc="Tracking & Action Recognition"):
            frame_count += 1
            timestamp_sec = frame_count / fps
            
            if r.boxes is None or r.boxes.id is None:
                continue

            orig_img = r.orig_img
            boxes = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.int().cpu().tolist()
            clss = r.boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, clss):
                # Chỉ xử lý người (class 0) - có thể mở rộng tùy ý
                if cls != 0: 
                    continue
                    
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_img.shape[1], x2), min(orig_img.shape[0], y2)
                
                # Crop vật thể
                crop = orig_img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                    
                track_buffers[track_id].append(crop)
                track_labels[track_id] = self.yolo.names[cls]

                # Nếu đủ số lượng frame, chạy Action Recognition và reset buffer
                if len(track_buffers[track_id]) >= CLIP_LENGTH:
                    action = self.recognize_action(track_buffers[track_id])
                    label_str = track_labels[track_id]
                    
                    # Ghi vào Neo4j
                    self.write_to_neo4j(video_id, track_id, label_str, action, timestamp_sec)
                    
                    # Clear buffer để tính action ở cửa sổ thời gian tiếp theo
                    track_buffers[track_id] = []

        cap.release()
        print(f"[Done] Xử lý xong {video_id}")

    def close(self):
        self.driver.close()


if __name__ == "__main__":
    builder = EventGraphBuilder()
    
    # Test trên 1 video
    test_video_path = "data/test_video.mp4"
    if os.path.exists(test_video_path):
        builder.process_video(test_video_path, "L01_V001")
    else:
        print(f"⚠️ Vui lòng cung cấp video test tại {test_video_path} để chạy thử.")
    
    builder.close()
