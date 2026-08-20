import os
import numpy as np
import torch
import clip

def load_image_embeddings(npy_path: str) -> np.ndarray:
    """
    Đọc file .npy chứa vector đặc trưng (embeddings) của các khung hình.
    Kỳ vọng shape: (N, D) - trong đó N là số frame, D là chiều vector (vd: 512).
    """
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Không tìm thấy file: {npy_path}")
    
    image_features = np.load(npy_path)
    print(f"[INFO] Đã load file '{npy_path}' với shape: {image_features.shape}")
    return image_features

def get_text_embedding(query_text: str, model, device: str) -> np.ndarray:
    """
    Chuyển câu mô tả văn bản thành vector đặc trưng bằng mô hình CLIP.
    """
    # Tokenize câu mô tả
    text_tokens = clip.tokenize([query_text]).to(device)
    
    with torch.no_grad():
        # Trích xuất đặc trưng văn bản
        text_features = model.encode_text(text_tokens)
        # Chuẩn hóa L2 vector
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Chuyển về dạng NumPy array (1, D)
    return text_features.cpu().numpy()

def compute_cosine_similarity(image_embeddings: np.ndarray, text_embedding: np.ndarray) -> np.ndarray:
    """
    Tính Cosine Similarity giữa vector chữ và toàn bộ vector ảnh.
    Công thức: Cosine Similarity = (A . B) / (||A|| * ||B||)
    """
    # Chuẩn hóa L2 cho các vector ảnh (tránh lỗi nếu file .npy chưa chuẩn hóa)
    img_norms = np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    img_norms[img_norms == 0] = 1e-10  # Tránh chia cho 0
    normalized_images = image_embeddings / img_norms

    # Chuẩn hóa L2 cho vector text
    text_norm = np.linalg.norm(text_embedding)
    if text_norm == 0:
        text_norm = 1e-10
    normalized_text = text_embedding / text_norm

    # Tính tích vô hướng (Dot product) giữa matrix ảnh (N, D) và vector text (D,)
    similarities = np.dot(normalized_images, normalized_text.squeeze())
    return similarities

def search_top_frames(similarities: np.ndarray, top_k: int = 5):
    """
    Sắp xếp và in ra Top K khung hình có điểm số cao nhất.
    """
    # Lấy top K chỉ số có điểm tương đồng cao nhất
    top_indices = np.argsort(similarities)[::-1][:top_k]

    print("\n" + "=" * 50)
    print(f"🎯 KẾT QUẢ TOP {top_k} KHUNG HÌNH PHÙ HỢP NHẤT:")
    print("=" * 50)
    print(f"{'Hạng':<6} | {'Chỉ số Frame (Index)':<22} | {'Điểm tương đồng':<15}")
    print("-" * 50)
    for rank, idx in enumerate(top_indices, start=1):
        score = similarities[idx]
        print(f"{rank:<6} | Khung hình #{idx:<14} | {score:.4f} ({score * 100:.2f}%)")
    print("=" * 50 + "\n")
    return top_indices

def main():
    # 1. Cấu hình thiết bị & nạp mô hình CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Sử dụng thiết bị: {device.upper()}")
    print("[INFO] Đang nạp mô hình CLIP (ViT-B/32)...")
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()

    # 2. Đọc file vector ảnh (.npy)
    npy_path = input("👉 Nhập đường dẫn file vector ảnh (.npy) [mặc định: image_features.npy]: ").strip()
    if not npy_path:
        npy_path = "image_features.npy"

    try:
        image_embeddings = load_image_embeddings(npy_path)
    except Exception as e:
        print(f"[LỖI] {e}")
        return

    # 3. Vòng lặp nhận câu mô tả từ bàn phím và tìm kiếm
    while True:
        query_text = input("\n📝 Nhập câu mô tả (hoặc gõ 'exit' để thoát): ").strip()
        if query_text.lower() in ["exit", "quit", "q"]:
            print("Đã thoát chương trình.")
            break
        if not query_text:
            print("Vui lòng nhập câu mô tả hợp lệ!")
            continue

        # Số lượng top frame muốn hiển thị
        top_k_input = input("👉 Số lượng kết quả hiển thị [mặc định: 5]: ").strip()
        top_k = int(top_k_input) if top_k_input.isdigit() and int(top_k_input) > 0 else 5

        # 4. Trích xuất Text Embedding bằng CLIP
        print(f"[INFO] Đang mã hóa câu mô tả: '{query_text}'...")
        text_embedding = get_text_embedding(query_text, model, device)

        # 5. Tính Cosine Similarity
        similarities = compute_cosine_similarity(image_embeddings, text_embedding)

        # 6. Sắp xếp & in Top K kết quả
        search_top_frames(similarities, top_k=top_k)

if __name__ == "__main__":
    main()
