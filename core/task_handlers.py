"""
AIC 2026 Task Handlers (KIS, Visual QA, TRAKE)
Tích hợp các xử lý chuyên biệt cho từng tác vụ của đề thi AIC:
1. KIS Handler (Top-K Multimodal Retrieval)
2. QA Handler (Multimodal Evidence + Strict Answer Normalization)
3. TRAKE Handler (3-Stage Temporal Refinement: Coarse -> Dense -> Local +/-10s)
"""

import os
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class QAnswerNormalizer:
    """Bộ chuẩn hóa câu trả lời Visual QA theo quy chế AIC 2026"""
    
    @staticmethod
    def normalize_answer(raw_answer: str, expected_type: Optional[str] = None) -> str:
        """Chuẩn hóa đáp án loại bỏ từ thừa, mạo từ và khoảng trắng"""
        if not raw_answer:
            return "yes"

        ans = raw_answer.strip().lower()
        ans = re.sub(r'^(the answer is|it is|i think it is|có|đáp án là)\s*', '', ans, flags=re.IGNORECASE)
        ans = re.sub(r'^(a|an|the)\s+', '', ans, flags=re.IGNORECASE)
        ans = ans.strip(' .!?,;:"\'')

        # Chuẩn hóa Boolean yes/no
        if ans in ["yes", "có", "true", "đúng", "chính xác"]:
            return "yes"
        if ans in ["no", "không", "false", "sai"]:
            return "no"

        # Chuẩn hóa số
        num_map = {
            "một": "1", "hai": "2", "ba": "3", "bốn": "4", "năm": "5",
            "sáu": "6", "bảy": "7", "tám": "8", "chín": "9", "mười": "10",
            "one": "1", "two": "2", "three": "3", "four": "4", "five": "5"
        }
        if ans in num_map:
            return num_map[ans]

        # Chuẩn hóa màu sắc
        color_map = {
            "đỏ": "red", "xanh": "blue", "vàng": "yellow", "trắng": "white",
            "đen": "black", "cam": "orange", "hồng": "pink", "xám": "gray"
        }
        if ans in color_map:
            return color_map[ans]

        return ans if ans else "yes"


class TRAKE3StageLocalizer:
    """Định vị chuỗi sự kiện thời gian 3 giai đoạn cho bài toán TRAKE"""
    
    def __init__(self, temporal_engine, image_resolver=None):
        self.temporal_engine = temporal_engine
        self.image_resolver = image_resolver

    def refine_trake_sequence(
        self,
        event_embeddings: Any,
        image_features: Any,
        keyframe_map: List[Dict[str, Any]],
        video_to_indices: Dict[str, List[int]],
        top_k_videos: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Quy trình 3 giai đoạn định vị thời gian:
        - Stage 1: Coarse Video Scoring
        - Stage 2: Dense Monotonic DP trên Top candidate videos
        - Stage 3: Local Temporal Refinement quanh khung hình cực đại
        """
        S = event_embeddings.shape[0]
        # Stage 1: Coarse Scoring
        sim_matrix = (event_embeddings @ image_features.T)
        candidate_videos = set()
        video_scores = {}

        coarse_k = min(250, image_features.shape[0])
        for s_idx in range(S):
            top_val, top_idx = sim_matrix[s_idx].topk(coarse_k)
            for v, idx in zip(top_val.tolist(), top_idx.tolist()):
                vname = keyframe_map[idx].get("video") or keyframe_map[idx].get("video_name")
                candidate_videos.add(vname)
                video_scores[vname] = video_scores.get(vname, 0.0) + v

        top_candidates = sorted(candidate_videos, key=lambda x: video_scores[x], reverse=True)[:top_k_videos * 2]

        # Stage 2 & 3: Monotonic DP + Local Refinement per candidate video
        trake_results = []
        for vname in top_candidates:
            v_indices = video_to_indices.get(vname, [])
            T_v = len(v_indices)
            if T_v < S:
                continue

            v_feats = image_features[v_indices]
            score_matrix = (event_embeddings @ v_feats.T).cpu().numpy().astype(np.float32)

            v_timestamps = np.array([
                float(keyframe_map[idx]["pts_time"]) if keyframe_map[idx].get("pts_time") is not None
                else float(keyframe_map[idx].get("frame", 0))
                for idx in v_indices
            ])

            best_score, best_path, _, c_temp, _ = self.temporal_engine.align_sequence_monotonic_dp(
                score_matrix, v_timestamps, gap_type="SHORT"
            )

            if not best_path:
                continue

            chosen_frames = []
            for s_idx, local_f in enumerate(best_path):
                global_idx = v_indices[local_f]
                item = keyframe_map[global_idx]
                _kf = item.get("keyframe_idx")
                kf_id = int(_kf) if _kf is not None else 1
                _frame = item.get("frame")
                if _frame is None:
                    _frame = item.get("frame_idx")
                if _frame is None:
                    _frame = kf_id
                real_frame = int(_frame)
                
                img_path = self.image_resolver(vname, kf_id, real_frame) if self.image_resolver else item.get("image_path")
                
                chosen_frames.append({
                    "keyframe_idx": kf_id,
                    "frame": real_frame,
                    "pts_time": float(item.get("pts_time", 0.0)),
                    "score": float(score_matrix[s_idx, local_f]),
                    "image_path": img_path
                })

            trake_results.append({
                "video": vname,
                "frames": chosen_frames,
                "total_score": float(best_score),
                "temporal_confidence": float(c_temp)
            })

        trake_results.sort(key=lambda x: x["total_score"], reverse=True)
        return trake_results[:top_k_videos]

class QAHandler:
    """Xử lý truy vấn QA theo chuẩn AIC 2026: Joint Ranking Top-100 Answers"""
    
    def __init__(self, vlm_endpoint: Optional[str] = None):
        self.vlm_endpoint = vlm_endpoint
        self.normalizer = QAnswerNormalizer()
        
    def _fast_deterministic_extract(self, item: Dict[str, Any], query_en: str, constraints: List[Any]) -> Tuple[Optional[str], float]:
        """Giả lập Fast Extractor (OCR/CV/Rules). Trả về (Answer, Confidence)"""
        # Giả lập OCR có confidence cao nếu frame khớp mạnh với câu có số
        has_num = any(ch.isdigit() for ch in query_en)
        if has_num and item.get("score", 0.0) > 0.8:
            # Mô phỏng đọc ra 1 số
            return "5", 0.95
            
        # Giả lập OCR mờ / không chắc chắn
        if has_num and 0.5 < item.get("score", 0.0) <= 0.8:
            return "O5", 0.45
            
        return None, 0.0

    def _vlm_verifier(self, item: Dict[str, Any], query_en: str) -> Dict[str, Any]:
        """Giả lập gọi VLM (LLM Vision) trả về cấu trúc JSON"""
        # Trong thực tế, sẽ truyền image_path và query_en vào LLM Vision API
        return {
            "answer": "yes" if "is there" in query_en.lower() else "unknown",
            "confidence": 0.85,
            "evidence_supported": True
        }

    def process_qa_candidates(
        self,
        candidates: List[Dict[str, Any]],
        query_en: str,
        intent_flags: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Quy trình QA: Fast Extractor -> Confidence Gate -> Selective VLM -> Joint Score Rank 100
        """
        results = []
        constraints = intent_flags.get("constraints", [])
        
        for item in candidates:
            retrieval_score = float(item.get("final_score", item.get("score", 0.0)))
            evidence_score = float(item.get("cec", 0.5))
            
            # 1. Fast Extractor
            ans_ext, conf_ext = self._fast_deterministic_extract(item, query_en, constraints)
            
            final_answer = ans_ext
            final_conf = conf_ext
            used_vlm = False
            
            # 2. Confidence & Evidence Gate
            # Kích hoạt VLM nếu Fast Extractor tự tin thấp hoặc không trích xuất được
            if conf_ext < 0.80 or not final_answer:
                vlm_res = self._vlm_verifier(item, query_en)
                final_answer = vlm_res.get("answer", "yes")
                final_conf = vlm_res.get("confidence", 0.5)
                used_vlm = True
                
            final_answer = self.normalizer.normalize_answer(final_answer)
            
            # 3. Joint Score (alpha=0.4, beta=0.2, gamma=0.4)
            joint_score = (0.4 * (retrieval_score / 10000.0)) + (0.2 * evidence_score) + (0.4 * final_conf)
            
            # Lưu kết quả
            res_item = dict(item)
            res_item["qa_answer"] = final_answer
            res_item["qa_confidence"] = final_conf
            res_item["joint_score"] = float(joint_score)
            res_item["used_vlm"] = used_vlm
            results.append(res_item)
            
        # 4. Joint Score Ranking (Re-rank)
        results.sort(key=lambda x: x["joint_score"], reverse=True)
        return results[:100]
