import os
import glob
import json
import numpy as np
from tqdm import tqdm

def build_dataset(
    features_dir="data/features/clip-features-32-aic25-b1/clip-features-32",
    media_info_dir="data/media-info-aic25-b1/media-info",
    output_npy="data/features/clip_features.npy",
    output_map="data/mapping/map_keyframes.json",
    fps=25.0
):
    """
    Gộp toàn bộ các file vector .npy riêng lẻ của từng video thành 1 file duy nhất 'clip_features.npy'
    và tạo file 'map_keyframes.json' ánh xạ vị trí từng dòng sang (video, keyframe_idx, frame, pts_time).
    """
    if not os.path.exists(features_dir):
        print(f"[ERROR] Không tìm thấy thư mục: {features_dir}")
        return

    npy_files = sorted(glob.glob(os.path.join(features_dir, "*.npy")))
    print(f"[INFO] Tìm thấy {len(npy_files)} file .npy trong '{features_dir}'")

    # Đọc trước toàn bộ media-info
    media_info_map = {}
    if os.path.exists(media_info_dir):
        for json_path in glob.glob(os.path.join(media_info_dir, "*.json")):
            vname = os.path.splitext(os.path.basename(json_path))[0]
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    media_info_map[vname] = json.load(f)
            except Exception:
                pass
        print(f"[INFO] Đã nạp metadata từ {len(media_info_map)} file media-info.")

    all_features = []
    keyframe_map = []

    for npy_path in tqdm(npy_files, desc="Đang tổng hợp vector & tạo mapping"):
        video_name = os.path.splitext(os.path.basename(npy_path))[0]
        feats = np.load(npy_path)  # shape: (num_frames, 512)
        all_features.append(feats)

        num_keyframes = feats.shape[0]
        video_length_sec = media_info_map.get(video_name, {}).get("length", 0)
        
        # Nếu có thông tin độ dài video, tính tổng số frames thực tế theo FPS = 25
        if video_length_sec > 0:
            total_video_frames = video_length_sec * fps
        else:
            total_video_frames = num_keyframes * 100.0

        for kf_idx in range(1, num_keyframes + 1):
            # Tính frame index thực tế trong video gốc
            real_frame = round(kf_idx * (total_video_frames / num_keyframes))
            pts_time = round(real_frame / fps, 3)
            
            keyframe_map.append({
                "video": video_name,
                "keyframe_idx": kf_idx,
                "frame": real_frame,
                "frame_idx": real_frame,
                "pts_time": pts_time
            })

    print("[INFO] Đang ghép các ma trận vector...")
    combined_features = np.vstack(all_features).astype(np.float32)

    os.makedirs(os.path.dirname(output_npy), exist_ok=True)
    os.makedirs(os.path.dirname(output_map), exist_ok=True)

    print(f"[INFO] Đang lưu '{output_npy}' với shape: {combined_features.shape}...")
    np.save(output_npy, combined_features)

    print(f"[INFO] Đang lưu '{output_map}' ({len(keyframe_map)} keyframes)...")
    with open(output_map, "w", encoding="utf-8") as f:
        json.dump(keyframe_map, f, ensure_ascii=False)

    print("\n✅ HOÀN THÀNH TỔNG HỢP DỮ LIỆU!")
    print(f" - File ma trận tổng hợp: {output_npy} (Shape: {combined_features.shape})")
    print(f" - File ánh xạ index: {output_map} (Số lượng: {len(keyframe_map)})")

if __name__ == "__main__":
    build_dataset()

