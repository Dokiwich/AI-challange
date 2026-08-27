"""
init_qdrant_db.py
Upload clip_features.npy and videoma_merged.npy to Qdrant Vector Database.
"""

import os
import json
import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

def upload_clip_to_qdrant(client: QdrantClient, clip_npy_path: str, map_path: str):
    collection_name = "aic_clip_frames"
    
    print(f"\n--- Tải đặc trưng CLIP từ {clip_npy_path} ---")
    if not os.path.exists(clip_npy_path):
        print(f"Không tìm thấy {clip_npy_path}")
        return

    raw_features = np.load(clip_npy_path)
    # L2 Normalization giống trong RetrievalEngine để dùng COSINE similarity chính xác
    features = torch.from_numpy(raw_features).float()
    features /= features.norm(dim=-1, keepdim=True)
    features = features.numpy()
    
    vector_size = features.shape[1]
    
    # Tạo/Re-create collection
    print(f"Tạo collection '{collection_name}' với vector size = {vector_size}...")
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    
    # Nạp mapping
    with open(map_path, 'r', encoding='utf-8') as f:
        keyframe_map = json.load(f)
        
    points = []
    print("Đang chuẩn bị payload cho Qdrant...")
    for idx, (vec, meta) in enumerate(zip(features, keyframe_map)):
        # Qdrant Point Payload
        vname = meta.get("video") or meta.get("video_name", "")
        kf = meta.get("keyframe_idx") or 1
        frame = meta.get("frame") or meta.get("frame_idx") or kf
        pts = float(meta.get("pts_time", 0.0))
        
        payload = {
            "video": vname,
            "keyframe_idx": int(kf),
            "frame": int(frame),
            "pts_time": pts,
            "original_index": idx  # Lưu lại index gốc để tra cứu ngược nếu cần
        }
        
        points.append(
            PointStruct(
                id=idx,
                vector=vec.tolist(),
                payload=payload
            )
        )
        
    print(f"Bắt đầu upload {len(points)} vectors lên Qdrant...")
    batch_size = 500
    for i in tqdm(range(0, len(points), batch_size)):
        client.upsert(
            collection_name=collection_name,
            points=points[i:i+batch_size]
        )
        
    print("✅ Đã đẩy toàn bộ dữ liệu CLIP lên Qdrant thành công!")

if __name__ == "__main__":
    print("Khởi tạo kết nối Qdrant...")
    try:
        client = QdrantClient("localhost", port=6333)
        # Test connection
        client.get_collections()
        
        upload_clip_to_qdrant(
            client=client, 
            clip_npy_path="data/features/clip_features.npy",
            map_path="data/mapping/map_keyframes.json"
        )
    except Exception as e:
        print(f"❌ Lỗi kết nối Qdrant (Đảm bảo Docker container qdrant đang chạy): {e}")
