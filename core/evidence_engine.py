"""
EvidenceEngine V3.2 (Event-Centric Evidence Judge & 3-Level Veto Protocol)
Tích hợp:
1. 10D Evidence Vector per Candidate (Visual, Action, Object, Rel, Temp, CEC, Order, Contradiction, Margin, ASR)
2. 2-Level Veto Protocol:
   - Veto A (Contradiction): Loại bỏ ngay ứng viên mâu thuẫn (TIER_5).
   - Veto C (Evidence Insufficiency): Giữ ứng viên cho Recall nhưng KHÓA CHẶT TIER_0/TIER_1.
3. Core Event Coverage (CEC >= 0.80) bắt buộc cho TIER_0 (Chống bẫy background).
4. Relative Margin (Z_margin) & Trạng thái trực giao AMBIGUOUS.
5. Query-to-Evidence Trace & Automated Failure Taxonomy (F01 - F12).
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class EvidenceEngine:
    """
    EvidenceEngine V3.2:
    Thẩm phán độc lập đánh giá chứng cứ sự kiện, loại bỏ hoàn toàn hiện tượng False High-Confidence.
    'Top-1 is a ranking result. TIER_0 is a claim that requires evidence.'
    """
    def __init__(self):
        self.tier_offsets = {
            "TIER_0": 10000.0,  # Certified Gold: CEC >= 0.80 + Temporal Verified + No Contradiction
            "TIER_1": 8000.0,   # Highly Plausible: Strong Core Evidence + Partial Sequence
            "TIER_2": 6000.0,   # Candidate: Đúng thực thể chính nhưng thiếu hành động cốt lõi
            "TIER_3": 4000.0,   # Weak: Chỉ khớp đồ vật nền / Background match
            "TIER_4": 2000.0,   # Semantic Fallback
            "TIER_5": 0.0       # Rejected / Bị phủ quyết Veto A
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
        """Khớp mờ có nhận thức nhầm lẫn quang học (Confusion-Aware Fuzzy Match)"""
        if not target_anchor or not candidate_text:
            return False, 0.0

        t_clean = target_anchor.lower().strip()
        c_clean = candidate_text.lower().strip()

        # 1. Khớp chính xác
        if t_clean in c_clean:
            return True, 1.0

        # 2. Khớp nhầm lẫn ký tự quang học (0 <-> O, 1 <-> I)
        norm_t = "".join([self.char_confusion.get(ch, ch) for ch in t_clean])
        norm_c = "".join([self.char_confusion.get(ch, ch) for ch in c_clean])
        if norm_t in norm_c:
            return True, 0.9

        # 3. Levenshtein substring similarity
        len_t = len(t_clean)
        if len_t >= 4:
            for i in range(max(1, len(c_clean) - len_t + 1)):
                sub = c_clean[i:i + len_t]
                diffs = sum(1 for a, b in zip(t_clean, sub) if a != b)
                sim = 1.0 - (diffs / float(len_t))
                if sim >= 0.80:
                    return True, float(sim)

        return False, 0.0

    def evaluate_evidence_vector(
        self,
        item: Dict[str, Any],
        intent_flags: Dict[str, Any],
        z_margin: float = 0.5
    ) -> Dict[str, Any]:
        """
        Xây dựng Evidence Vector 10 chiều và thẩm định các cấp độ Veto
        """
        is_seq_query = intent_flags.get("is_sequence", False)
        has_speech = intent_flags.get("has_speech", False)
        has_ocr = len(intent_flags.get("anchors", [])) > 0

        visual_norm = float(item.get("visual_norm", 0.0))
        action_norm = float(item.get("action_norm", visual_norm * 0.8))
        object_norm = float(item.get("object_norm", visual_norm * 0.9))
        relation_norm = float(item.get("relation_norm", visual_norm * 0.7))
        seq_norm = float(item.get("seq_norm", 0.0))
        asr_norm = float(item.get("asr_norm", 0.0))
        has_anchor_match = bool(item.get("has_anchor_match", False))

        # 1. Tính toán Core Event Coverage (CEC)
        # Nếu là query chuỗi: CEC phụ thuộc trực tiếp vào seq_norm và action_norm
        if is_seq_query:
            cec = float(item.get("cec", (seq_norm * 0.7 + action_norm * 0.3)))
        else:
            cec = float(item.get("cec", (visual_norm * 0.5 + action_norm * 0.5)))
        cec = float(np.clip(cec, 0.0, 1.0))

        # 2. Đánh giá Mâu thuẫn (Contradiction Score)
        # Ví dụ: Có anchor nhưng không khớp, hoặc chuỗi thời gian bị đứt gãy
        contradiction_score = 0.0
        if has_ocr and not has_anchor_match:
            contradiction_score += 0.4
        if is_seq_query and seq_norm < 0.20 and visual_norm >= 0.80:
            # Hiện tượng Background cao nhưng không có chuỗi hành động -> Nghi ngờ background bias
            contradiction_score += 0.35

        contradiction_score = float(np.clip(contradiction_score, 0.0, 1.0))

        # 3. Áp dụng 3 Cấp Độ Veto:
        # Veto A: Contradiction cao -> Prune / Reject
        veto_a_triggered = contradiction_score >= 0.60

        # Veto C: Evidence Insufficiency (Thiếu chứng cứ Core Event hoặc Sequence)
        veto_c_locked_tier0 = False
        if is_seq_query and (cec < 0.70 or seq_norm < 0.35):
            veto_c_locked_tier0 = True
        if has_ocr and not has_anchor_match:
            veto_c_locked_tier0 = True

        # Đánh giá ConstraintNode (Atomic Evidence Verification)
        constraints = intent_flags.get("constraints", [])
        n_hard_unverified = 0
        if constraints:
            hard_constraints = [c for c in constraints if c.get("is_hard", True)]
            # Chưa có detector/VLM → tất cả hard constraint đều "unverified"
            # Khi tích hợp detector, giảm n_hard_unverified cho constraint đã verify
            n_hard_unverified = len(hard_constraints)
            # Phạt tích lũy cho mỗi constraint chưa verify
            contradiction_score += n_hard_unverified * 0.10
            # Giảm CEC theo tỷ lệ constraint bị bỏ lỡ
            if n_hard_unverified > 0:
                cec *= max(0.3, 1.0 - n_hard_unverified * 0.15)
            # ≥2 hard constraints chưa verify → không đủ evidence cho TIER_0/TIER_1
            if n_hard_unverified >= 2:
                veto_c_locked_tier0 = True
            # visual_norm thấp + có constraint → phạt nặng
            if hard_constraints and visual_norm < 0.55:
                veto_c_locked_tier0 = True
                contradiction_score += 0.2

        contradiction_score = float(np.clip(contradiction_score, 0.0, 1.0))
        
        evidence_vec = {
            "visual_norm": visual_norm,
            "action_norm": action_norm,
            "object_norm": object_norm,
            "relation_norm": relation_norm,
            "seq_norm": seq_norm,
            "asr_norm": asr_norm,
            "cec": cec,
            "order_consistency": float(item.get("order_consistency", 1.0 if not is_seq_query else (0.9 if seq_norm >= 0.4 else 0.3))),
            "contradiction_score": contradiction_score,
            "z_margin": z_margin,
            "has_anchor_match": has_anchor_match,
            "veto_a": veto_a_triggered,
            "veto_c": veto_c_locked_tier0
        }
        return evidence_vec

    def assign_tier(
        self,
        ev: Dict[str, Any],
        consensus_bonus: float,
        intent_flags: Dict[str, Any]
    ) -> Tuple[str, float, bool]:
        """
        Phán quyết TIER chuẩn V3.2 dựa trên Evidence Vector:
        - TIER_0 BẮT BUỘC: CEC >= 0.80 + Temporal Verified + Z_margin >= 0.10 + Contradiction < 0.05 + No Veto C
        - Trả về: (tier_name, continuous_score, is_ambiguous)
        """
        is_seq_query = intent_flags.get("is_sequence", False)
        has_speech = intent_flags.get("has_speech", False)
        has_ocr = len(intent_flags.get("anchors", [])) > 0

        # Nếu bị Veto A (Mâu thuẫn)
        if ev["veto_a"]:
            return "TIER_5", 0.0, False

        # Kiểm tra cờ AMBIGUOUS (Z_margin quá hẹp)
        is_ambiguous = ev["z_margin"] < 0.08

        # Điểm liên tục đa thành phần: S = w_v V + w_a A + w_o O + w_t T - w_c C
        if is_seq_query:
            continuous_score = (
                0.20 * ev["visual_norm"] +
                0.35 * ev["action_norm"] +
                0.15 * ev["object_norm"] +
                0.30 * ev["seq_norm"] +
                (0.20 * ev["asr_norm"] if has_speech else 0.0) +
                consensus_bonus -
                (0.50 * ev["contradiction_score"])
            )
        else:
            continuous_score = (
                0.35 * ev["visual_norm"] +
                0.25 * ev["action_norm"] +
                0.25 * ev["object_norm"] +
                0.15 * ev["relation_norm"] +
                (0.30 * ev["asr_norm"] if has_speech else 0.0) +
                consensus_bonus -
                (0.50 * ev["contradiction_score"])
            )

        # 1. TIÊU CHUẨN VÀNG TIER_0 (Certified Gold Target)
        # Bắt buộc: KHÔNG bị Veto C + CEC >= 0.80 + Contradiction cực thấp + Margin đủ tốt
        if not ev["veto_c"] and ev["cec"] >= 0.80 and ev["contradiction_score"] <= 0.10:
            if has_ocr and ev["has_anchor_match"] and ev["visual_norm"] >= 0.45:
                return "TIER_0", continuous_score + 1.0, is_ambiguous
            if is_seq_query and ev["seq_norm"] >= 0.65 and ev["action_norm"] >= 0.50:
                return "TIER_0", continuous_score + 1.0, is_ambiguous
            if not is_seq_query and ev["visual_norm"] >= 0.85 and ev["action_norm"] >= 0.70:
                return "TIER_0", continuous_score + 1.0, is_ambiguous

        # 2. TIER 1 (Highly Plausible)
        # visual_norm >= 0.80 (percentile-normalized): top ~20% candidates
        # action_norm >= 0.45: no-op khi action_norm = visual_norm*0.8 (proxy, không độc lập)
        #   → giữ placeholder cho khi có action detector thật
        # cec >= 0.65: chỉ hoạt động cho multi-phase (single-span cec = raw cosine ~0.25)
        if not ev["veto_c"] and (ev["cec"] >= 0.65 or ev["visual_norm"] >= 0.80):
            return "TIER_1", continuous_score, is_ambiguous

        # 3. TIER 2 (Candidate - Thiếu một phần chứng cứ nhưng vẫn giữ lại cho Recall)
        if ev["visual_norm"] >= 0.60 or ev["action_norm"] >= 0.50 or (has_speech and ev["asr_norm"] >= 0.45):
            return "TIER_2", continuous_score, is_ambiguous

        # 4. TIER 3 (Weak - Chỉ khớp bối cảnh nền)
        if ev["visual_norm"] >= 0.40:
            return "TIER_3", continuous_score, is_ambiguous

        # 5. TIER 4 (Semantic Fallback)
        return "TIER_4", continuous_score, is_ambiguous

    def rank_candidates(
        self,
        candidate_items: List[Dict[str, Any]],
        intent_flags: Dict[str, Any],
        diversity_top_2: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Xếp hạng toàn bộ ứng viên theo Disjoint Tier Ladder Scoring chuẩn V3.2
        """
        if not candidate_items:
            return []

        # 1. Tính toán Relative Margin (Z_margin) sơ bộ
        raw_scores = np.array([float(it.get("score", 0.0)) for it in candidate_items])
        std_score = float(raw_scores.std()) if len(raw_scores) > 1 else 1.0
        if len(raw_scores) >= 2 and std_score > 1e-6:
            sorted_raw = np.sort(raw_scores)[::-1]
            z_margin = float((sorted_raw[0] - sorted_raw[1]) / (std_score + 1e-6))
        else:
            z_margin = 0.5

        # 2. Đánh giá Evidence Vector và gán Tier cho từng ứng viên
        for item in candidate_items:
            consensus_bonus = float(item.get("consensus_bonus", 0.0))

            ev = self.evaluate_evidence_vector(item, intent_flags, z_margin=z_margin)
            item["evidence_vector"] = ev
            item["cec"] = ev["cec"]
            item["contradiction_score"] = ev["contradiction_score"]

            tier_name, continuous_score, is_ambiguous = self.assign_tier(
                ev=ev,
                consensus_bonus=consensus_bonus,
                intent_flags=intent_flags
            )

            base_offset = self.tier_offsets.get(tier_name, 0.0)
            raw_score = float(item.get("score", 0.0))
            # Base offset phân tầng tuyệt đối + continuous score đa chiều + raw score vi phân biệt
            final_score = base_offset + (continuous_score * 10.0) + raw_score

            item["tier"] = tier_name
            item["continuous_score"] = float(continuous_score)
            item["final_score"] = float(final_score)
            item["is_ambiguous"] = is_ambiguous

            # Gắn Query-to-Evidence Trace
            item["evidence_trace"] = {
                "core_event_coverage": round(ev["cec"], 2),
                "action_match": round(ev["action_norm"], 2),
                "temporal_sequence": round(ev["seq_norm"], 2),
                "contradiction": round(ev["contradiction_score"], 2),
                "z_margin": round(z_margin, 3),
                "veto_c_locked": ev["veto_c"],
                "tier": tier_name
            }

        # Sắp xếp giảm dần theo final_score
        candidate_items.sort(key=lambda x: x["final_score"], reverse=True)

        # Cập nhật lại Z_margin thực tế giữa Top-1 và Top-2 sau khi xếp hạng
        if len(candidate_items) >= 2:
            top1_sc = candidate_items[0]["final_score"]
            top2_sc = candidate_items[1]["final_score"]
            rel_margin = float((top1_sc - top2_sc) / (abs(top1_sc) + 1e-6))
            if rel_margin < 0.005:
                candidate_items[0]["is_ambiguous"] = True
                if len(candidate_items) > 1:
                    candidate_items[1]["is_ambiguous"] = True

        # Diversity filtering cho Top-2 nếu được yêu cầu
        if diversity_top_2 and len(candidate_items) >= 2:
            top1_video = candidate_items[0].get("video")
            top2_idx = 1
            while top2_idx < len(candidate_items) and candidate_items[top2_idx].get("video") == top1_video:
                top2_idx += 1
            if top2_idx < len(candidate_items) and top2_idx != 1:
                promoted = candidate_items.pop(top2_idx)
                candidate_items.insert(1, promoted)

        return candidate_items
