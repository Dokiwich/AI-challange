import os
import glob
import numpy as np

def merge_features(input_dir="data/features/videoma", output_file="data/features/videoma_merged.npy"):
    """
    Gộp các file videomae_features_part_*.npy thành một file duy nhất
    """
    print(f"Đang quét các file .npy tại {input_dir}...")
    npy_files = glob.glob(os.path.join(input_dir, "*.npy"))
    
    if not npy_files:
        print("Không tìm thấy file nào! Hãy kiểm tra lại đường dẫn.")
        return
        
    merged_dict = {}
    for f in npy_files:
        # Load dictionary từ file numpy (allow_pickle=True là bắt buộc khi lưu dict)
        chunk_data = np.load(f, allow_pickle=True).item()
        merged_dict.update(chunk_data)
        print(f"Đã gộp {len(chunk_data)} videos từ {os.path.basename(f)}")
        
    print(f"\nTổng số video đã trích xuất: {len(merged_dict)}")
    
    # Lưu ra file đích
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, merged_dict)
    print(f"✅ Đã lưu file gộp thành công tại: {output_file}")

if __name__ == "__main__":
    # Bạn hãy tải các file npy trên Drive về thư mục data/features/videoma/
    # Sau đó chạy script này: python scripts/merge_videoma_features.py
    merge_features()
