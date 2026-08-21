import numpy as np

def create_sample_npy(output_path="image_features.npy", num_frames=100, dim=512):
    """
    Tạo file .npy giả lập chứa vector đặc trưng của các khung hình để test code.
    Mặc định: 100 khung hình, mỗi vector có 512 chiều (chuẩn ViT-B/32).
    """
    # Sinh dữ liệu ngẫu nhiên
    features = np.random.randn(num_frames, dim).astype(np.float32)
    
    # Chuẩn hóa L2 vector
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / norms
    
    np.save(output_path, features)
    print(f"✅ Đã tạo file mẫu '{output_path}' thành công! Shape: {features.shape}")

if __name__ == "__main__":
    create_sample_npy()
