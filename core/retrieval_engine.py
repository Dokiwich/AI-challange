import os
import glob
import json
import numpy as np
import torch
import clip
from PIL import Image
from deep_translator import GoogleTranslator

class RetrievalEngine:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RetrievalEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        features_path: str = "clip_features.npy",
        map_path: str = "map_keyframes.json",
        model_name: str = "ViT-B/32",
        device: str = None
    ):
        if getattr(self, "_initialized", False):
            return

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[RetrievalEngine] Khởi tạo trên thiết bị: {self.device}")

        # 1. Nạp Model CLIP
        print(f"[RetrievalEngine] Đang nạp mô hình CLIP {model_name}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        # 2. Nạp Ma trận đặc trưng
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Không tìm thấy file: {features_path}")
        
        print(f"[RetrievalEngine] Đang nạp ma trận đặc trưng từ '{features_path}'...")
        raw_features = np.load(features_path)
        self.image_features = torch.from_numpy(raw_features).to(self.device).float()
        # Chuẩn hóa L2 trên GPU/CPU
        self.image_features /= self.image_features.norm(dim=-1, keepdim=True)
        print(f"[RetrievalEngine] Đã nạp và chuẩn hóa {self.image_features.shape[0]} vector (dim={self.image_features.shape[1]}).")

        # 3. Nạp File Mapping
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Không tìm thấy file: {map_path}")
        
        print(f"[RetrievalEngine] Đang nạp map keyframes từ '{map_path}'...")
        with open(map_path, "r", encoding="utf-8") as f:
            self.keyframe_map = json.load(f)
        print(f"[RetrievalEngine] Đã nạp {len(self.keyframe_map)} bản ghi ánh xạ.")

        # Xây dựng cấu trúc tra cứu nhanh theo video_name -> list indices
        self.video_to_indices = {}
        for idx, item in enumerate(self.keyframe_map):
            vname = item.get("video") or item.get("video_name")
            if vname:
                if vname not in self.video_to_indices:
                    self.video_to_indices[vname] = []
                self.video_to_indices[vname].append(idx)

        # 4. Quét trước các thư mục Keyframes để tìm ảnh nhanh
        self.keyframe_roots = self._scan_keyframe_roots()
        print(f"[RetrievalEngine] Tìm thấy các thư mục keyframes: {self.keyframe_roots}")

        # Bộ dịch tự động
        self.translator = GoogleTranslator(source='auto', target='en')

        self._initialized = True

    def _scan_keyframe_roots(self):
        roots = []
        current_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        for entry in os.listdir(current_dir):
            full_path = os.path.join(current_dir, entry)
            if os.path.isdir(full_path):
                if "keyframe" in entry.lower():
                    sub_kf = os.path.join(full_path, "keyframes")
                    if os.path.isdir(sub_kf):
                        roots.append(sub_kf)
                    else:
                        roots.append(full_path)
        if not roots:
            roots.append(current_dir)
        return roots

    def get_extracted_videos(self) -> set[str]:
        """Lấy danh sách các video đã có thư mục ảnh thực tế trên ổ cứng"""
        extracted = set()
        for root in self.keyframe_roots:
            if os.path.isdir(root):
                for entry in os.listdir(root):
                    if os.path.isdir(os.path.join(root, entry)):
                        extracted.add(entry)
        return extracted

    def get_image_path(self, video_name: str, frame_id: int) -> str | None:
        """
        Tìm kiếm đường dẫn thực tế của ảnh keyframe theo video_name và frame_id
        """
        candidates = [
            f"{frame_id:03d}.jpg",
            f"{frame_id:04d}.jpg",
            f"{frame_id:05d}.jpg",
            f"{frame_id:06d}.jpg",
            f"{frame_id}.jpg",
            f"{frame_id:03d}.png",
            f"{frame_id:04d}.png",
            f"{frame_id}.png"
        ]

        for root in self.keyframe_roots:
            video_dir = os.path.join(root, video_name)
            if os.path.isdir(video_dir):
                for cand in candidates:
                    img_path = os.path.join(video_dir, cand)
                    if os.path.isfile(img_path):
                        return img_path
                    
            img_cand = os.path.join(root, f"{video_name}_{frame_id:03d}.jpg")
            if os.path.isfile(img_cand):
                return img_cand

        return None

    def translate_to_en(self, text: str) -> str:
        """Tự động dịch câu truy vấn tiếng Việt sang tiếng Anh cho CLIP"""
        try:
            # Kiểm tra xem có chứa ký tự tiếng Việt có dấu không
            if re.search(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', text, re.IGNORECASE):
                return self.translator.translate(text)
            return text
        except Exception:
            return text

    def encode_text(self, query_text: str) -> torch.Tensor:
        """Trích xuất vector văn bản 512 chiều từ câu truy vấn"""
        tokens = clip.tokenize([query_text], truncate=True).to(self.device)
        with torch.no_grad():
            text_feature = self.model.encode_text(tokens)
            text_feature /= text_feature.norm(dim=-1, keepdim=True)
        return text_feature

    def search(
        self,
        query_text: str,
        top_k: int = 100,
        video_filter: str | list[str] | None = None,
        auto_translate: bool = True
    ) -> tuple[list[dict], str]:
        """
        Tìm kiếm khung hình phù hợp nhất với câu truy vấn.
        Trả về (results, translated_query)
        """
        processed_query = query_text.strip()
        if auto_translate:
            processed_query = self.translate_to_en(processed_query)

        text_feature = self.encode_text(processed_query)

        if video_filter:
            if isinstance(video_filter, str):
                video_filter = [video_filter]
            
            target_indices = []
            for vf in video_filter:
                target_indices.extend(self.video_to_indices.get(vf, []))
            
            if not target_indices:
                return [], processed_query
            
            target_indices = torch.tensor(target_indices, device=self.device, dtype=torch.long)
            sub_features = self.image_features[target_indices]
            similarity = (text_feature @ sub_features.T).squeeze(0)
            
            actual_k = min(top_k, similarity.shape[0])
            top_scores, top_pos = similarity.topk(actual_k)
            
            top_indices = target_indices[top_pos].tolist()
            top_scores = top_scores.tolist()
        else:
            similarity = (text_feature @ self.image_features.T).squeeze(0)
            actual_k = min(top_k, self.image_features.shape[0])
            top_scores, top_indices = similarity.topk(actual_k)
            top_scores = top_scores.tolist()
            top_indices = top_indices.tolist()

        results = []
        for idx, score in zip(top_indices, top_scores):
            item = self.keyframe_map[idx]
            v_name = item.get("video") or item.get("video_name")
            f_id = int(item.get("frame") or item.get("frame_idx") or item.get("frame_id") or 1)
            img_path = self.get_image_path(v_name, f_id)

            results.append({
                "index": idx,
                "video": v_name,
                "frame": f_id,
                "score": float(score),
                "image_path": img_path
            })

        return results, processed_query

    def search_trake(
        self,
        event_queries: list[str],
        top_k_videos: int = 10,
        video_filter: list[str] | None = None,
        auto_translate: bool = True
    ) -> tuple[list[dict], list[str]]:
        if not event_queries:
            return [], []

        processed_events = []
        for eq in event_queries:
            eq_proc = self.translate_to_en(eq.strip()) if auto_translate else eq.strip()
            processed_events.append(eq_proc)

        event_features = [self.encode_text(eq) for eq in processed_events]
        
        candidate_videos = set()
        video_scores = {}

        for feat in event_features:
            sim = (feat @ self.image_features.T).squeeze(0)
            top_val, top_idx = sim.topk(min(150, self.image_features.shape[0]))
            for v, idx in zip(top_val.tolist(), top_idx.tolist()):
                vname = self.keyframe_map[idx]["video"]
                if video_filter and vname not in video_filter:
                    continue
                candidate_videos.add(vname)
                video_scores[vname] = video_scores.get(vname, 0.0) + v

        sorted_videos = sorted(candidate_videos, key=lambda x: video_scores[x], reverse=True)[:top_k_videos]

        trake_results = []
        for vname in sorted_videos:
            v_indices = self.video_to_indices.get(vname, [])
            if not v_indices:
                continue

            v_indices_tensor = torch.tensor(v_indices, device=self.device, dtype=torch.long)
            v_feats = self.image_features[v_indices_tensor]

            chosen_frames = []
            min_frame = 0
            valid_sequence = True

            for feat in event_features:
                sim = (feat @ v_feats.T).squeeze(0)
                sorted_score_pos = sim.argsort(descending=True)

                found = False
                for pos in sorted_score_pos.tolist():
                    global_idx = v_indices[pos]
                    f_id = int(self.keyframe_map[global_idx]["frame"])
                    if f_id > min_frame:
                        chosen_frames.append({
                            "frame": f_id,
                            "score": float(sim[pos]),
                            "image_path": self.get_image_path(vname, f_id)
                        })
                        min_frame = f_id
                        found = True
                        break
                
                if not found:
                    valid_sequence = False
                    break

            if valid_sequence and len(chosen_frames) == len(processed_events):
                trake_results.append({
                    "video": vname,
                    "frames": chosen_frames,
                    "total_score": sum(cf["score"] for cf in chosen_frames)
                })

        trake_results.sort(key=lambda x: x["total_score"], reverse=True)
        return trake_results, processed_events
