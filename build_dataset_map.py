import os
import glob
import json
import numpy as np
from tqdm import tqdm

def build_dataset(
    features_dir="clip-features-32-aic25-b1/clip-features-32",
    output_npy="clip_features.npy",
    output_map="map_keyframes.json"
):
    """
    Gộp toàn bộ các file vector .npy riêng lẻ của từng video thành 1 file duy nhất 'clip_features.npy'
    và tạo file 'map_keyframes.json' ánh xạ vị trí từng dòng sang (video, frame_id).
    """
    if not os.path.exists(features_dir):
        print(f"[ERROR] Không tìm thấy thư mục: {features_dir}")
        return

    npy_files = sorted(glob.glob(os.path.join(features_dir, "*.npy")))
    print(f"[INFO] Tìm thấy {len(npy_files)} file .npy trong '{features_dir}'")

    all_features = []
    keyframe_map = []

    for npy_path in tqdm(npy_files, desc="Đang tổng hợp vector & tạo mapping"):
        video_name = os.path.splitext(os.path.basename(npy_path))[0]
        feats = np.load(npy_path)  # shape: (num_frames, 512)
        all_features.append(feats)

        # Mỗi dòng i trong file .npy tương ứng với keyframe (i + 1)
        num_frames = feats.shape[0]
        for frame_idx in range(1, num_frames + 1):
            keyframe_map.append({
                "video": video_name,
                "frame": frame_idx
            })

    print("[INFO] Đang ghép các ma trận vector...")
    combined_features = np.vstack(all_features).astype(np.float32)

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
