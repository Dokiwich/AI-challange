import os
import glob
import numpy as np
import torch
import clip
from PIL import Image

def extract_features_from_folder(image_folder: str, output_npy: str = "image_features.npy"):
    """
    Trích xuất vector đặc trưng từ thư mục chứa ảnh và lưu thành file .npy
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Nạp mô hình CLIP trên {device.upper()}...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
    image_paths.sort()

    if not image_paths:
        print(f"[CẢNH BÁO] Không tìm thấy ảnh trong thư mục '{image_folder}'!")
        return

    print(f"[INFO] Tìm thấy {len(image_paths)} ảnh. Đang trích xuất đặc trưng...")
    
    all_features = []
    with torch.no_grad():
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            feature = model.encode_image(img_tensor)
            feature /= feature.norm(dim=-1, keepdim=True)
            all_features.append(feature.cpu().numpy().squeeze())

    features_array = np.array(all_features, dtype=np.float32)
    np.save(output_npy, features_array)
    print(f"✅ Đã lưu vector đặc trưng vào '{output_npy}'. Shape: {features_array.shape}")

if __name__ == "__main__":
    folder = input("👉 Nhập đường dẫn thư mục ảnh: ").strip()
    if folder and os.path.exists(folder):
        extract_features_from_folder(folder)
    else:
        print("Thư mục không tồn tại!")
