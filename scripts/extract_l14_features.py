import os
import glob
import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class KeyframeDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.preprocess(img)
            return tensor, path
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros((3, 224, 224)), path

def extract_features_vit_l14(
    keyframes_dir="data/keyframes",
    output_dir="data/features/clip-features-L14-aic25",
    batch_size=128
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Đang nạp mô hình ViT-L/14 trên {device.upper()}...")
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    # Tìm đệ quy TẤT CẢ các thư mục chứa file .jpg
    print("[INFO] Đang quét toàn bộ thư mục tìm ảnh .jpg...")
    all_jpgs = glob.glob(os.path.join(keyframes_dir, "**", "*.jpg"), recursive=True)
    
    if not all_jpgs:
        print(f"[ERROR] Không tìm thấy ảnh .jpg nào trong {keyframes_dir}")
        return

    video_map = {}
    for path in all_jpgs:
        # Đường dẫn: .../L21_V001/0001.jpg -> parent là L21_V001
        parent_dir = os.path.basename(os.path.dirname(path))
        if parent_dir not in video_map:
            video_map[parent_dir] = []
        video_map[parent_dir].append(path)

    # Lọc ra các video chưa xử lý
    videos_to_process = {}
    for vname, paths in video_map.items():
        out_npy_path = os.path.join(output_dir, f"{vname}.npy")
        if not os.path.exists(out_npy_path):
            videos_to_process[vname] = sorted(paths)

    if not videos_to_process:
        print("[INFO] Đã trích xuất xong tất cả video!")
        return

    print(f"[INFO] Cần xử lý {len(videos_to_process)} videos (tổng cộng đã tìm thấy {len(video_map)} videos). Bắt đầu trích xuất...")

    # Gom tất cả đường dẫn ảnh, sắp xếp theo video để gom nhóm dễ dàng
    flat_image_paths = []
    for vname in sorted(videos_to_process.keys()):
        flat_image_paths.extend(videos_to_process[vname])

    dataset = KeyframeDataset(flat_image_paths, preprocess)
    # Tăng num_workers lên 8 để tận dụng CPU decode ảnh nhanh hơn, chỉ 1 DataLoader nên không sợ spawn overhead
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8 if device=="cuda" else 0)

    current_video = None
    current_features = []

    with torch.no_grad():
        for batch_tensors, batch_paths in tqdm(dataloader, desc="Extracting features"):
            batch_tensors = batch_tensors.to(device)
            features = model.encode_image(batch_tensors)
            features /= features.norm(dim=-1, keepdim=True)
            features = features.cpu().numpy()

            for i, path in enumerate(batch_paths):
                vname = os.path.basename(os.path.dirname(path))
                
                if current_video is None:
                    current_video = vname

                if vname != current_video:
                    # Lưu feature cho video trước đó
                    out_npy_path = os.path.join(output_dir, f"{current_video}.npy")
                    np.save(out_npy_path, np.array(current_features, dtype=np.float32))
                    
                    # Bắt đầu video mới
                    current_video = vname
                    current_features = []

                current_features.append(features[i])

    # Lưu video cuối cùng
    if current_video is not None and len(current_features) > 0:
        out_npy_path = os.path.join(output_dir, f"{current_video}.npy")
        np.save(out_npy_path, np.array(current_features, dtype=np.float32))

    print(f"✅ Hoàn tất trích xuất ViT-L/14. Dữ liệu lưu tại: {output_dir}")

if __name__ == "__main__":
    extract_features_vit_l14()
