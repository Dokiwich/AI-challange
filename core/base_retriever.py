"""
Base Visual Retriever Interface & CLIP Implementation
Cung cấp lớp trừu tượng thống nhất cho các mô hình trích xuất đặc trưng thị giác (CLIP ViT-B/32, SigLIP, OpenCLIP).
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional
import torch
import numpy as np
import clip
import logging

logger = logging.getLogger(__name__)

class BaseVisualRetriever(ABC):
    """Lớp trừu tượng cho mọi Visual Embedding Retriever"""
    
    @abstractmethod
    def encode_text(self, text_or_texts: Union[str, List[str]]) -> torch.Tensor:
        """Mã hóa văn bản thành vector chuẩn hóa L2 kích thước (B, D)"""
        pass

    @abstractmethod
    def compute_similarity(self, text_features: torch.Tensor, image_features: torch.Tensor) -> np.ndarray:
        """Tính ma trận Cosine Similarity giữa text features và image features"""
        pass


class CLIPVisualRetriever(BaseVisualRetriever):
    """Hiện thực hóa bằng mô hình OpenAI CLIP ViT-B/32 chuẩn công nghiệp"""
    
    def __init__(self, model_name: str = "ViT-L/14", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[CLIPVisualRetriever] Loading {model_name} on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self._text_cache = {}

    def encode_text(self, text_or_texts: Union[str, List[str]]) -> torch.Tensor:
        """Mã hóa văn bản với RAM cache để tối ưu tốc độ"""
        if isinstance(text_or_texts, str):
            text_list = [text_or_texts]
            is_single = True
        else:
            text_list = text_or_texts
            is_single = False

        # Kiểm tra cache
        uncached_indices = []
        uncached_texts = []
        results = [None] * len(text_list)

        for idx, t in enumerate(text_list):
            t_clean = t.strip()
            if t_clean in self._text_cache:
                results[idx] = self._text_cache[t_clean]
            else:
                uncached_indices.append(idx)
                uncached_texts.append(t_clean)

        if uncached_texts:
            tokens = clip.tokenize(uncached_texts, truncate=True).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            
            for local_idx, global_idx in enumerate(uncached_indices):
                feat_single = feats[local_idx:local_idx+1]
                t_key = uncached_texts[local_idx]
                self._text_cache[t_key] = feat_single
                results[global_idx] = feat_single

        final_tensor = torch.cat(results, dim=0)
        return final_tensor

    def compute_similarity(self, text_features: torch.Tensor, image_features: torch.Tensor) -> np.ndarray:
        """Tính toán tích vô hướng (Cosine Similarity) trên GPU/CPU"""
        with torch.no_grad():
            # Đồng bộ kiểu dữ liệu (tránh lỗi float16 vs float32 khi dùng model to như ViT-L/14)
            text_features = text_features.to(image_features.dtype)
            sims = (text_features @ image_features.T).cpu().numpy()
        return sims
