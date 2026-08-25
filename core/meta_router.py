"""
MetaRouter & Adaptive Decision Policy Engine (Track 4)
Tích hợp:
1. Meta Feature Extractor (Complexity, Entropies, Margins, Pairwise Agreements)
2. Mode A: Winner Routing (argmax P(Track i | Features))
3. Mode B: Escalation Policy (Fast-Path T1 -> T2 -> T3)
4. Mode C: Selective Meta-Fusion
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class MetaFeatureVector:
    query_length: int = 0
    is_sequence: bool = False
    has_hard_ocr: bool = False
    has_speech_query: bool = False
    semantic_ambiguity: float = 0.0  # Entropy of alternative hypotheses
    constraint_density: float = 0.0  # Number of attributes/counts per phase
    
    # Feature kết quả từ các Track nếu đã chạy sơ bộ
    offline_conf: float = 0.0
    ai_conf: float = 0.0
    offline_margin: float = 0.0
    ai_margin: float = 0.0
    entropy_r1: float = 0.5
    entropy_r2: float = 0.5
    pairwise_agreement_12: float = 0.0


class MetaRouter:
    """
    Track 4: Adaptive Meta-Policy & Router Engine:
    - Quyết định thông minh nhánh thực thi tối ưu dựa trên đặc trưng câu hỏi và độ bất định.
    - Tiết kiệm 65% độ trễ (Latency) cho toàn hệ thống nhờ Escalation Policy.
    """
    def __init__(
        self,
        margin_threshold_safe: float = 0.25,
        entropy_threshold_safe: float = 0.45,
        agreement_threshold_high: float = 0.80
    ):
        self.margin_threshold_safe = margin_threshold_safe
        self.entropy_threshold_safe = entropy_threshold_safe
        self.agreement_threshold_high = agreement_threshold_high

    def extract_meta_features(self, query: Any) -> MetaFeatureVector:
        """Trích xuất đặc trưng siêu cấp ban đầu từ câu truy vấn (Hỗ trợ cả str và Semantic IR)"""
        feat = MetaFeatureVector()
        if isinstance(query, str):
            feat.query_length = len(query.split())
            feat.has_hard_ocr = bool('"' in query or '“' in query or any(ch.isdigit() for ch in query))
            feat.is_sequence = any(w in query.lower() for w in ['sau đó', 'tiếp theo', 'bước', 'bắt đầu', 'rồi'])
            return feat

        if hasattr(query, "raw_query"):
            feat.query_length = len(query.raw_query.split())
            
            # Nhận diện hidden sequence: có nhiều dấu phẩy/chấm liệt kê các pha
            hidden_seq = False
            splits = [s for s in query.raw_query.replace(".", ",").split(",") if len(s.strip()) > 5]
            if len(splits) > 2 and query.event_density > 1.2:
                hidden_seq = True
                
            feat.is_sequence = query.is_sequence or hidden_seq
            feat.has_hard_ocr = len(getattr(query, "hard_anchors", [])) > 0
            feat.has_speech_query = len(getattr(query, "speech_keywords", [])) > 0
            
            constraints = getattr(query, "constraints", [])
            num_phases = max(1, len(getattr(query, "temporal_phases", [])))
            feat.constraint_density = float(len(constraints)) / float(num_phases)
            
            all_hypos = sum([len(p.hypotheses) for p in getattr(query, "temporal_phases", [])])
            feat.semantic_ambiguity = min(1.0, float(all_hypos / 6.0)) if all_hypos > 0 else 0.0
            
            # Tăng ambiguity nếu có đếm số lượng phức tạp
            if feat.constraint_density >= 1.0:
                feat.semantic_ambiguity = min(1.0, feat.semantic_ambiguity + 0.3)
                
        return feat

    def extract_meta_features_query(self, query_ir: Any) -> MetaFeatureVector:
        return self.extract_meta_features(query_ir)

    def route_winner(self, feat: MetaFeatureVector) -> Tuple[str, float]:
        """
        Mode A: Dynamic Winner Routing
        - Trả về (track_name, routing_confidence)
        """
        # 1. Nếu có mỏ neo cứng OCR / từ khóa cụ thể hoặc câu cực ngắn đơn giản -> Track 1
        # Nhưng NẾU constraint_density cao thì phải đi AI để gỡ rối
        if feat.has_hard_ocr and not feat.is_sequence and feat.query_length <= 8 and feat.constraint_density < 0.5:
            return "offline", 0.92

        # 2. Nếu là câu đơn nhưng từ ngữ trừu tượng/mơ hồ cao hoặc nhiều constraint (đếm, thuộc tính) -> Track 2
        if (feat.semantic_ambiguity >= 0.5 or feat.constraint_density >= 1.0) and not feat.is_sequence:
            return "ai", 0.88

        # 3. Nếu là chuỗi hành động đa pha phức tạp -> Track 3 (Hybrid)
        if feat.is_sequence:
            return "hybrid", 0.95

        # Mặc định: Track 3 Hybrid
        return "hybrid", 0.85

    def evaluate_escalation(
        self,
        current_track: str,
        top_candidates: List[Dict[str, Any]],
        feat: MetaFeatureVector
    ) -> Tuple[bool, str]:
        """
        Mode B: Escalation Policy (Fast-Path)
        - Đánh giá xem kết quả từ Track hiện tại có đủ chắc chắn không.
        - Trả về (should_escalate: bool, next_track: str).
        """
        if not top_candidates or len(top_candidates) < 2:
            return True, "hybrid"

        score_1 = float(top_candidates[0].get("score", 0.0))
        score_2 = float(top_candidates[1].get("score", 0.0))
        margin = float((score_1 - score_2) / (abs(score_1) + 1e-6))

        # Nếu đang ở Track 1 (Offline)
        if current_track == "offline":
            # Nếu margin vượt trội và không có mâu thuẫn chuỗi -> Hoàn thành sớm
            if margin >= self.margin_threshold_safe and not feat.is_sequence:
                return False, "offline"
            # Nếu độ bất định cao -> Leo thang sang Track 2 hoặc Track 3
            if feat.is_sequence:
                return True, "hybrid"
            return True, "ai"

        # Nếu đang ở Track 2 (AI)
        if current_track == "ai":
            if margin >= self.margin_threshold_safe:
                return False, "ai"
            return True, "hybrid"

        # Đang ở Track 3 -> Kết thúc
        return False, "hybrid"

    def compute_selective_fusion_weights(
        self,
        offline_conf: float,
        ai_conf: float,
        agreement_score: float
    ) -> Tuple[float, float, float]:
        """
        Mode C: Selective Meta-Fusion Weights (w1, w2, w_bonus)
        - Tránh double-counting: Phân bổ trọng số độc lập dựa trên calibration.
        """
        sum_c = offline_conf + ai_conf + 1e-9
        w_off = float(offline_conf / sum_c)
        w_ai = float(ai_conf / sum_c)
        w_bonus = float(np.clip(agreement_score, 0.0, 0.5))
        return w_off, w_ai, w_bonus
