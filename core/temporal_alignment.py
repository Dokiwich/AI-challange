"""
TemporalAlignmentEngine V3.2 (Skip-Aware Monotonic DP & Event Reasoning Engine)
Tích hợp:
1. Skip-Aware Monotonic DP: Cho phép nhảy cóc các pha phụ khi keyframe bị trích xuất thưa (Sampling Gap).
2. Adaptive Temporal Windows W(q): Khớp chính xác vi hành động (0.5-5s) đến chuỗi đa pha (5-60s).
3. Transition Consistency T(P_i, P_i+1, F_j, F_k) trên trục thời gian.
4. Core Event Coverage (CEC) & Weighted Semantic Coverage (WSC).
5. Duration-Normalized Gap Penalty & Temporal Alignment Confidence (C_temporal).
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional

class TemporalAlignmentEngine:
    """
    TemporalAlignmentEngine V3.2:
    Định vị chuỗi sự kiện thời gian trên video, hỗ trợ Skip State chống rớt keyframe và Adaptive Temporal Windows.
    """
    def __init__(self, tau: float = 45.0, lambda_gap: float = 0.12, lambda_order: float = 0.5):
        self.tau = tau
        self.lambda_gap = lambda_gap
        self.lambda_order = lambda_order

    def gaussian_smoothing(self, scores: np.ndarray, window_size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """Làm mượt điểm số trên trục thời gian của 1 video bằng Gaussian Filter 1D"""
        if len(scores) <= 2:
            return scores
            
        radius = window_size // 2
        x = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / (kernel.sum() + 1e-9)
        
        smoothed = np.convolve(scores, kernel, mode='same')
        return smoothed

    def compute_composite_phase_weights(
        self,
        score_matrix: np.ndarray,
        semantic_importances: Optional[List[float]] = None,
        alpha: float = 0.6,
        beta: float = 0.4
    ) -> np.ndarray:
        """
        Tính toán trọng số pha chuẩn hóa: w_s = alpha * I_semantic(s) + beta * R_retrieval(s)
        """
        S, T = score_matrix.shape
        if S == 0:
            return np.array([], dtype=np.float32)
        if S == 1:
            return np.array([1.0], dtype=np.float32)

        weights = np.zeros(S, dtype=np.float32)
        sem_imp = semantic_importances if (semantic_importances and len(semantic_importances) == S) else [0.6] * S

        for s in range(S):
            row = score_matrix[s]
            i_sem = float(sem_imp[s])

            # 1. Normalized Entropy: H_hat in [0, 1]
            if T > 1 and row.max() > row.min():
                exp_row = np.exp(np.clip(row - row.max(), -15.0, 0.0))
                prob = exp_row / (exp_row.sum() + 1e-9)
                entropy = - np.sum(prob * np.log(prob + 1e-12))
                h_norm = float(entropy / (np.log(T) + 1e-9))
                h_norm = float(np.clip(h_norm, 0.0, 1.0))
            else:
                h_norm = 0.5

            # 2. Robust Sigmoid Z-Score: Z_hat in [0, 1]
            mean_v = float(row.mean())
            std_v = float(row.std())
            if std_v > 1e-6:
                z = (float(row.max()) - mean_v) / std_v
                z_hat = float(1.0 / (1.0 + np.exp(- 0.5 * (z - 2.0))))
            else:
                z_hat = 0.5

            r_retrieval = 0.5 * (1.0 - h_norm) + 0.5 * z_hat
            weights[s] = alpha * i_sem + beta * r_retrieval

        # Partition of unity: Tổng w'_s = 1.0
        total_w = float(weights.sum())
        if total_w > 1e-6:
            weights = weights / total_w
        else:
            weights = np.ones(S, dtype=np.float32) / float(S)

        return weights

    def align_sequence_monotonic_dp(
        self,
        score_matrix: np.ndarray,
        timestamps: np.ndarray,
        semantic_importances: Optional[List[float]] = None,
        is_core_flags: Optional[List[bool]] = None,
        start_at_beginning: bool = False,
        gap_type: str = "SHORT",
        allow_skip: bool = True
    ) -> Tuple[float, List[int], np.ndarray, float, float]:
        """
        Giải thuật Skip-Aware Monotonic DP V3.2:
        Trả về: (best_score, best_path, frame_bonuses, c_temporal, core_event_coverage)
        """
        S, T = score_matrix.shape
        if S == 0 or T == 0:
            return 0.0, [], np.zeros(T, dtype=np.float32), 0.0, 0.0

        # Nếu chỉ có 1 pha (Single-Phase)
        if S == 1:
            best_idx = int(np.argmax(score_matrix[0]))
            bonuses = np.zeros(T, dtype=np.float32)
            bonuses[best_idx] = float(score_matrix[0, best_idx])
            return float(score_matrix[0, best_idx]), [best_idx], bonuses, 1.0, 1.0

        # Trọng số pha chuẩn hóa w_s
        weights = self.compute_composite_phase_weights(score_matrix, semantic_importances)
        core_flags = is_core_flags if (is_core_flags and len(is_core_flags) == S) else [True] * S

        # Thời lượng video tổng thể (Duration)
        v_duration = max(1.0, float(timestamps[-1] - timestamps[0])) if T > 1 else 100.0

        # Adaptive Temporal Window W(q) theo gap_type
        if gap_type == "IMMEDIATE":
            w_min, w_max = 0.5, 10.0
            tau_scaled = 8.0
        elif gap_type == "LONG":
            w_min, w_max = 5.0, min(180.0, v_duration * 0.85)
            tau_scaled = 60.0
        else:  # SHORT
            w_min, w_max = 1.0, min(60.0, v_duration * 0.60)
            tau_scaled = 30.0

        # Bảng DP và Backpointer: dp[s, t]
        dp = np.full((S, T), -1e9, dtype=np.float32)
        parent = np.full((S, T), -1, dtype=np.int32)
        skipped = np.zeros((S, T), dtype=bool)

        # Khởi tạo pha đầu tiên s = 0
        for t in range(T):
            init_bonus = 0.0
            if start_at_beginning:
                init_bonus = - 0.25 * float(timestamps[t] - timestamps[0]) / (v_duration + 1e-6)
            dp[0, t] = weights[0] * float(score_matrix[0, t]) + init_bonus

        # Quy hoạch động qua các pha s = 1..S-1
        for s in range(1, S):
            w_s = weights[s]
            is_core = core_flags[s]
            # Mức phạt khi nhảy cóc pha (Core phase phạt nặng 0.5, Support phase phạt nhẹ 0.15)
            skip_pen = 0.45 if is_core else 0.15

            for t in range(s, T):
                sim_current = float(score_matrix[s, t])
                
                # 1. Nhánh Match: Tìm k < t tối ưu
                best_val = -1e9
                best_k = -1
                for k in range(t):
                    if dp[s - 1, k] <= -1e8:
                        continue
                    delta_t = float(timestamps[t] - timestamps[k])
                    if delta_t <= 0.0:
                        continue

                    # Duration-normalized gap penalty
                    delta_t_norm = delta_t / (tau_scaled + 1e-6)
                    gap_pen = self.lambda_gap * (delta_t_norm ** 1.2)

                    # Phạt nếu vượt ngoài Adaptive Window W(q)
                    if delta_t > w_max:
                        gap_pen += 0.35 * ((delta_t - w_max) / tau_scaled)

                    val = dp[s - 1, k] + (w_s * sim_current) - gap_pen
                    if val > best_val:
                        best_val = val
                        best_k = k

                # 2. Nhánh Skip State (nếu allow_skip và cho phép bỏ qua pha s)
                if allow_skip and dp[s - 1, t] > -1e8:
                    skip_val = dp[s - 1, t] - skip_pen
                    if skip_val > best_val:
                        best_val = skip_val
                        best_k = t
                        skipped[s, t] = True

                dp[s, t] = best_val
                parent[s, t] = best_k

        # Tìm điểm kết thúc tối ưu ở pha S-1
        best_t_final = int(np.argmax(dp[S - 1]))
        best_score = float(dp[S - 1, best_t_final])

        if best_score <= -1e8:
            # Fallback nếu không tìm thấy đường đi đơn điệu
            return 0.0, [], np.zeros(T, dtype=np.float32), 0.0, 0.0

        # Truy vết ngược (Backtracking)
        best_path = [-1] * S
        curr_t = best_t_final
        matched_phases = 0
        core_matched_weight = 0.0
        total_core_weight = sum(weights[s] for s in range(S) if core_flags[s]) + 1e-6

        for s in range(S - 1, -1, -1):
            best_path[s] = curr_t
            if not skipped[s, curr_t]:
                matched_phases += 1
                if core_flags[s]:
                    core_matched_weight += weights[s]
            curr_t = parent[s, curr_t] if curr_t >= 0 else 0

        # Tính toán Core Event Coverage (CEC)
        cec = float(np.clip(core_matched_weight / total_core_weight, 0.0, 1.0))

        # Tính toán Temporal Alignment Confidence (C_temporal)
        coverage_ratio = float(matched_phases) / float(S)
        path_scores = [float(score_matrix[s, best_path[s]]) for s in range(S)]
        mean_path_score = float(np.mean(path_scores))
        c_temporal = float(np.clip(0.6 * coverage_ratio + 0.4 * (mean_path_score / (np.max(score_matrix) + 1e-9)), 0.0, 1.0))

        # Phân bổ Frame Bonuses cho video
        frame_bonuses = np.zeros(T, dtype=np.float32)
        for s_idx, t_idx in enumerate(best_path):
            if t_idx >= 0 and not skipped[s_idx, t_idx]:
                frame_bonuses[t_idx] += weights[s_idx] * float(score_matrix[s_idx, t_idx]) * 1.5

        # Lan tỏa Gaussian sang các khung hình lân cận
        if frame_bonuses.max() > 0:
            frame_bonuses = self.gaussian_smoothing(frame_bonuses, window_size=3, sigma=0.8)

        return best_score, best_path, frame_bonuses, c_temporal, cec
