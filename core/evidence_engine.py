"""
EvidenceEngine V2 (Constraint Graph Evidence Judge)
Tích hợp:
1. Hard Required vs Soft Preferred Constraint Satisfaction
2. Confusion-Aware Fuzzy OCR & ASR Matching (0 <-> O, 1 <-> I <-> l)
3. Constraint Satisfaction Rate (CSR) Calculation
4. Robust Percentile Calibration & Tier Ladder Scoring
5. Consensus != Truth (Strict Evidence Requirements for TIER_0)
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class EvidenceEngine:
    """
    EvidenceEngine V2 (Judge of Truth):
    Đóng vai trò thẩm phán khách quan, đánh giá thỏa mãn các ràng buộc trong đồ thị ngữ nghĩa (Constraint Graph).
    """
    def __init__(self):
        self.tier_offsets = {
            "TIER_0": 10000.0,  # Top-1 Target: Khớp Hard Required Constraints + Chuỗi hoàn hảo
            "TIER_1": 8000.0,   # Rất liên quan: Chuỗi hành động chuẩn xác + Visual mạnh
            "TIER_2": 6000.0,   # Liên quan cao: Đúng thực thể chính + Lời thoại ASR
            "TIER_3": 4000.0,   # Liên quan vừa: Visual tương đồng cao
            "TIER_4": 2000.0,   # Bối cảnh chung / Semantic Fallback
            "TIER_5": 0.0       # Bị phủ quyết / Vi phạm Hard Constraint
        }
        
        # Bảng quy đổi ký tự nhầm lẫn quang học (Optical Confusion Matrix)
        self.char_confusion = {
            '0': 'o', 'o': '0',
            '1': 'i', 'i': '1', 'l': '1',
            '8': 'b', 'b': '8',
            '5': 's', 's': '5',
            '2': 'z', 'z': '2'
        }

    def calibrate_percentile(self, scores: np.ndarray, p_low: float = 5.0, p_high: float = 100.0) -> np.ndarray:
        """Chuẩn hóa điểm số theo phân vị nền thấp đến điểm đỉnh"""
        if len(scores) == 0:
            return scores
            
        min_v = float(np.percentile(scores, p_low))
        max_v = float(scores.max()) if p_high >= 100.0 else float(np.percentile(scores, p_high))
        
        if max_v > min_v:
            norm_scores = (scores - min_v) / (max_v - min_v)
            return np.clip(norm_scores, 0.0, 1.0)
        else:
            std = float(scores.std())
            if std > 1e-6:
                return np.clip((scores - scores.mean()) / (2.0 * std) + 0.5, 0.0, 1.0)
            return np.zeros_like(scores)

    def fuzzy_match_ocr_anchor(self, target_anchor: str, candidate_text: str) -> Tuple[bool, float]:
        """
        Khớp mờ có nhận thức nhầm lẫn quang học (Confusion-Aware Fuzzy Match)
        - Cho phép 0 <-> O, 1 <-> I, nhưng phạt nghiêm khắc sai lệch chữ số khác.
        """
        if not target_anchor or not candidate_text:
            return False, 0.0

        t_clean = target_anchor.lower().strip()
        c_clean = candidate_text.lower().strip()

        # 1. Khớp chính xác hoàn hảo
        if t_clean in c_clean:
            return True, 1.0

        # 2. Khớp nhầm lẫn ký tự quang học
        norm_t = "".join([self.char_confusion.get(ch, ch) for ch in t_clean])
        norm_c = "".join([self.char_confusion.get(ch, ch) for ch in c_clean])
        if norm_t in norm_c:
            return True, 0.9

        # 3. Levenshtein substring similarity
        len_t = len(t_clean)
        if len_t >= 4:
            # Quét cửa sổ độ dài len_t trên candidate_text
            for i in range(max(1, len(c_clean) - len_t + 1)):
                sub = c_clean[i:i + len_t]
                diffs = sum(1 for a, b in zip(t_clean, sub) if a != b)
                sim = 1.0 - (diffs / float(len_t))
                if sim >= 0.80:
                    return True, float(sim)

        return False, 0.0

    def evaluate_constraint_satisfaction(
        self,
        item: Dict[str, Any],
        intent_flags: Dict[str, Any]
    ) -> Tuple[float, bool]:
        """
        Tính Constraint Satisfaction Rate (CSR) và kiểm tra Hard Constraints
        Trả về: (csr_score in [0, 1], hard_constraints_passed: bool)
        """
        constraints = []
        
        # 1. Ràng buộc Hard Anchors (OCR)
        anchors = intent_flags.get("anchors", [])
        if anchors:
            constraints.append(("HARD", "OCR", bool(item.get("has_anchor_match", False))))

        # 2. Ràng buộc Lời thoại ASR
        if intent_flags.get("has_speech", False):
            asr_norm = float(item.get("asr_norm", 0.0))
            constraints.append(("SOFT", "ASR", asr_norm >= 0.4))

        # 3. Ràng buộc Chuỗi thời gian (Temporal DP)
        if intent_flags.get("is_sequence", False):
            seq_norm = float(item.get("seq_norm", 0.0))
            constraints.append(("HARD", "TEMPORAL", seq_norm >= 0.35))

        # 4. Ràng buộc Visual tối thiểu
        visual_norm = float(item.get("visual_norm", 0.0))
        constraints.append(("HARD", "VISUAL", visual_norm >= 0.35))

        if not constraints:
            return 1.0, True

        satisfied_weights = 0.0
        total_weights = 0.0
        hard_passed = True

        for c_priority, c_type, c_passed in constraints:
            weight = 2.0 if c_priority == "HARD" else 1.0
            total_weights += weight
            if c_passed:
                satisfied_weights += weight
            else:
                if c_priority == "HARD":
                    hard_passed = False

        csr = satisfied_weights / total_weights if total_weights > 0 else 1.0
        return float(csr), hard_passed

    def assign_tier(
        self,
        visual_norm: float,
        seq_norm: float,
        asr_norm: float,
        has_anchor_match: bool,
        consensus_bonus: float,
        intent_flags: Dict[str, Any],
        hard_passed: bool,
        csr: float
    ) -> Tuple[str, float]:
        """
        Đánh giá bằng chứng đa chiều và gán Tier (Evidence Judge):
        - TIER_0: Bắt buộc thỏa mãn 100% Hard Constraints + Evidence mạnh
        - Consensus Bonus chỉ là gia số hỗ trợ, không thay thế Hard Evidence.
        """
        is_seq_query = intent_flags.get("is_sequence", False)
        has_speech_query = intent_flags.get("has_speech", False)
        has_ocr_query = intent_flags.get("has_ocr_quote", False)

        # 1. TIER 0 (Top-1 Gold Target)
        if hard_passed and csr >= 0.85:
            # Trường hợp có OCR anchor khớp
            if has_ocr_query and has_anchor_match and visual_norm >= 0.45:
                return "TIER_0", 1.0 + visual_norm + (seq_norm * 0.5 if is_seq_query else 0.0) + consensus_bonus

            # Trường hợp chuỗi hành động hoàn hảo
            if is_seq_query and seq_norm >= 0.70 and visual_norm >= 0.55:
                return "TIER_0", visual_norm + seq_norm * 1.5 + consensus_bonus

            # Trường hợp ASR khớp mạnh
            if has_speech_query and asr_norm >= 0.80 and visual_norm >= 0.40:
                return "TIER_0", visual_norm * 0.8 + asr_norm * 1.8 + consensus_bonus

            # Visual đơn lẻ cực mạnh
            if visual_norm >= 0.90:
                return "TIER_0", visual_norm * 1.6 + consensus_bonus

        # 2. TIER 1 (High Multimodal / Strong Action Sequence)
        if (is_seq_query and seq_norm >= 0.50) or (visual_norm >= 0.75):
            return "TIER_1", visual_norm + seq_norm * 1.0 + consensus_bonus

        if has_speech_query and asr_norm >= 0.60:
            return "TIER_1", visual_norm + asr_norm * 1.2 + consensus_bonus

        # 3. TIER 2 (Entity Match)
        if visual_norm >= 0.65 or asr_norm >= 0.45:
            return "TIER_2", visual_norm + seq_norm * 0.3 + asr_norm * 0.5 + consensus_bonus

        # 4. TIER 3 (General Visual Match)
        if visual_norm >= 0.45:
            return "TIER_3", visual_norm + seq_norm * 0.1

        # 5. TIER 4 (Semantic Fallback)
        return "TIER_4", visual_norm

    def rank_candidates(
        self,
        candidate_items: List[Dict[str, Any]],
        intent_flags: Dict[str, Any],
        diversity_top_2: bool = False
    ) -> List[Dict[str, Any]]:
        """Xếp hạng toàn bộ ứng viên theo Disjoint Tier Ladder Scoring & CSR Judge"""
        if not candidate_items:
            return []

        for item in candidate_items:
            v_norm = item.get("visual_norm", 0.0)
            seq_norm = item.get("seq_norm", 0.0)
            asr_norm = item.get("asr_norm", 0.0)
            has_anchor = item.get("has_anchor_match", False)
            consensus_bonus = item.get("consensus_bonus", 0.0)

            # Đánh giá Constraint Satisfaction
            csr, hard_passed = self.evaluate_constraint_satisfaction(item, intent_flags)
            item["csr"] = csr
            item["hard_passed"] = hard_passed

            tier_name, continuous_score = self.assign_tier(
                visual_norm=v_norm,
                seq_norm=seq_norm,
                asr_norm=asr_norm,
                has_anchor_match=has_anchor,
                consensus_bonus=consensus_bonus,
                intent_flags=intent_flags,
                hard_passed=hard_passed,
                csr=csr
            )
            
            base_offset = self.tier_offsets.get(tier_name, 0.0)
            raw_score = float(item.get("score", 0.0))
            final_score = base_offset + (continuous_score * 10.0) + raw_score

            item["tier"] = tier_name
            item["continuous_score"] = float(continuous_score)
            item["final_score"] = float(final_score)

        # Sắp xếp giảm dần theo final_score
        candidate_items.sort(key=lambda x: x["final_score"], reverse=True)

        # Lọc đa dạng hóa cho Top-2 nếu được yêu cầu
        if diversity_top_2 and len(candidate_items) >= 2:
            top1_video = candidate_items[0].get("video")
            top2_idx = 1
            while top2_idx < len(candidate_items) and candidate_items[top2_idx].get("video") == top1_video:
                top2_idx += 1
            if top2_idx < len(candidate_items) and top2_idx != 1:
                promoted = candidate_items.pop(top2_idx)
                candidate_items.insert(1, promoted)

        return candidate_items
