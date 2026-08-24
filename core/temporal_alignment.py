"""
TemporalAlignmentEngine V2 (Precision Temporal & Relation Verifier)
Tích hợp:
1. Decoupled Composite Weighted DP: w_s = alpha * I_semantic + beta * R_retrieval
2. Normalized Entropy (H_hat) + Robust Sigmoid Z-Score (Z_hat)
3. Duration-Normalized Adaptive Gap Penalty
4. Temporal Alignment Confidence (C_temporal)
5. Gaussian Smoothing
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional

class TemporalAlignmentEngine:
    """
    TemporalAlignmentEngine V2:
    Cung cấp giải thuật Monotonic Dynamic Programming để khớp chuỗi hành động thời gian (Sequence Alignment),
    áp dụng trọng số thông tin chuẩn hóa và hàm phạt khoảng cách thời gian chuẩn hóa theo độ dài video.
    """
    def __init__(self, tau: float = 60.0, lambda_gap: float = 0.15, lambda_order: float = 0.5):
        """
        - tau: Ngưỡng khoảng cách thời gian (giây) tiêu chuẩn giữa 2 hành động
        - lambda_gap: Hệ số phạt khoảng cách thời gian chuẩn hóa
        - lambda_order: Hệ số phạt nếu thứ tự thời gian bị đảo lộn
        """
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
        kernel = kernel / kernel.sum()
        
        smoothed = np.convolve(scores, kernel, mode='same')
        return smoothed

    def compute_composite_phase_weights(
        self,
        score_matrix: np.ndarray,
        semantic_importances: Optional[List[float]] = None,
        alpha: float = 0.5,
        beta: float = 0.5
    ) -> np.ndarray:
        """
        Tính toán trọng số pha chuẩn hóa (Decoupled Composite Phase Weights):
        w_s = alpha * I_semantic(s) + beta * R_retrieval(s)
        - I_semantic(s): Tầm quan trọng ngữ nghĩa nội tại câu hỏi [0.1, 1.0]
        - R_retrieval(s): Độ tin cậy thống kê phân phối điểm CLIP [0.0, 1.0]
        - Partition of unity: Tổng w'_s = 1.0
        """
        S, T = score_matrix.shape
        if S == 0:
            return np.array([], dtype=np.float32)
        if S == 1:
            return np.array([1.0], dtype=np.float32)

        weights = np.zeros(S, dtype=np.float32)
        sem_imp = semantic_importances if (semantic_importances and len(semantic_importances) == S) else [0.5] * S

        for s in range(S):
            row = score_matrix[s]
            i_sem = float(sem_imp[s])

            # 1. Normalized Entropy: H_hat in [0, 1]
            if T > 1 and row.max() > row.min():
                # Softmax chuẩn hóa xác suất trên video
                exp_row = np.exp(np.clip(row - row.max(), -15.0, 0.0))
                prob = exp_row / (exp_row.sum() + 1e-9)
                entropy = - np.sum(prob * np.log(prob + 1e-12))
                h_norm = float(entropy / (np.log(T) + 1e-9))
                h_norm = float(np.clip(h_norm, 0.0, 1.0))
            else:
                h_norm = 0.5

            # 2. Robust Sigmoid Z-Score: Z_hat in [0, 1]
            std_val = float(row.std())
            if std_val > 1e-5:
                z_raw = float((row.max() - row.mean()) / (std_val + 1e-6))
                # Sigmoid chuẩn hóa Z_score về [0, 1]
                z_norm = float(1.0 / (1.0 + np.exp(- 0.5 * (z_raw - 2.0))))
            else:
                z_norm = 0.5

            # Retrieval Reliability R_retrieval
            r_retrieval = 0.5 * (1.0 - h_norm) + 0.5 * z_norm
            
            # Kết hợp decoupled
            w_raw = alpha * i_sem + beta * r_retrieval
            weights[s] = max(0.05, w_raw)

        # Partition of Unity (Chuẩn hóa tổng bằng 1.0)
        sum_w = float(weights.sum())
        if sum_w > 1e-6:
            weights = weights / sum_w
        else:
            weights = np.full(S, 1.0 / S, dtype=np.float32)

        return weights

    def align_sequence_monotonic_dp(
        self,
        score_matrix: np.ndarray,
        frame_timestamps: np.ndarray,
        semantic_importances: Optional[List[float]] = None,
        start_at_beginning: bool = False,
        gap_type: str = "SHORT"
    ) -> Tuple[float, List[int], np.ndarray, float]:
        """
        Monotonic DP Sequence Alignment V2:
        - score_matrix: (S, T) Ma trận similarity giữa S pha và T keyframes.
        - frame_timestamps: (T,) Mảng thời gian (giây) hoặc frame_id của từng keyframe.
        - semantic_importances: (S,) Tầm quan trọng ngữ nghĩa nội tại từng pha.
        - start_at_beginning: Ưu tiên pha 1 ở đầu video.
        - gap_type: 'SHORT' (phạt trung bình), 'LONG' (phạt nhẹ), 'IMMEDIATE' (phạt mạnh).
        
        Trả về:
        (effective_score, best_path, frame_bonuses, temporal_confidence)
        """
        S, T = score_matrix.shape
        if S == 0 or T == 0:
            return 0.0, [], np.zeros(T, dtype=np.float32), 0.0
            
        if S == 1:
            best_idx = int(np.argmax(score_matrix[0]))
            bonuses = np.zeros(T, dtype=np.float32)
            bonuses[best_idx] = float(score_matrix[0, best_idx])
            return float(score_matrix[0, best_idx]), [best_idx], bonuses, 1.0

        # 1. Tính toán Composite Phase Weights
        phase_weights = self.compute_composite_phase_weights(score_matrix, semantic_importances)

        # 2. Xác định độ dài video để chuẩn hóa khoảng cách thời gian (Duration-Normalized)
        t_min = float(frame_timestamps[0])
        t_max = float(frame_timestamps[-1])
        video_duration = max(1.0, t_max - t_min)

        # Điều chỉnh hệ số phạt theo gap_type
        gap_coeff = {"IMMEDIATE": 2.0, "SHORT": 1.0, "LONG": 0.3, "UNKNOWN": 0.8}.get(gap_type, 1.0)

        # DP Table: dp[s, t] là điểm tối ưu của chuỗi từ pha 0..s kết thúc tại frame t
        dp = np.full((S, T), -1e9, dtype=np.float32)
        backtrack = np.full((S, T), -1, dtype=np.int32)
        
        # Khởi tạo pha 0 (kết hợp trọng số w_0)
        w_0 = float(phase_weights[0])
        if start_at_beginning and T > 0:
            time_prior = np.exp(- np.maximum(0.0, frame_timestamps - t_min) / 120.0)
            dp[0] = score_matrix[0] * w_0 * (1.0 + 0.3 * time_prior)
        else:
            dp[0] = score_matrix[0] * w_0
        
        # Điền bảng DP
        for s in range(1, S):
            w_s = float(phase_weights[s])
            for t in range(T):
                current_time = float(frame_timestamps[t])
                current_val = float(score_matrix[s, t]) * w_s
                
                best_prev_score = -1e9
                best_prev_t = -1
                
                for t_prev in range(T):
                    prev_score = float(dp[s - 1, t_prev])
                    if prev_score <= -1e8:
                        continue
                        
                    prev_time = float(frame_timestamps[t_prev])
                    dt = current_time - prev_time
                    
                    # Strict Monotonicity: Pha sau bắt buộc xảy ra sau pha trước
                    if dt <= 0:
                        continue
                        
                    # Duration-Normalized Gap Penalty
                    dt_norm = dt / video_duration
                    penalty = self.lambda_gap * gap_coeff * np.log1p(10.0 * dt_norm)
                            
                    total_candidate_score = prev_score + current_val - penalty
                    if total_candidate_score > best_prev_score:
                        best_prev_score = total_candidate_score
                        best_prev_t = t_prev
                        
                dp[s, t] = best_prev_score
                backtrack[s, t] = best_prev_t

        # Tìm điểm kết thúc tối ưu tại pha cuối S-1
        last_row = dp[S - 1]
        best_end_t = int(np.argmax(last_row))
        best_score = float(last_row[best_end_t])
        
        if best_score <= -1e8:
            return -1e9, [], np.zeros(T, dtype=np.float32), 0.0

        # Tính Temporal Alignment Confidence (C_temporal)
        sorted_ends = np.sort(last_row[last_row > -1e8])[::-1]
        if len(sorted_ends) >= 2:
            second_score = float(sorted_ends[1])
            c_temporal = float((best_score - second_score) / (abs(best_score) + 1e-6))
            c_temporal = float(np.clip(c_temporal, 0.0, 1.0))
        else:
            c_temporal = 0.9

        # Truy vết ngược tìm đường đi tối ưu
        best_path = [best_end_t]
        curr_t = best_end_t
        for s in range(S - 1, 0, -1):
            prev_t = int(backtrack[s, curr_t])
            if prev_t == -1:
                return -1e9, [], np.zeros(T, dtype=np.float32), 0.0
            best_path.append(prev_t)
            curr_t = prev_t
            
        best_path.reverse()
        
        # Đánh giá độ hoàn chỉnh của chuỗi (Phase Completeness Factor)
        matched_phases = sum(1 for s_idx, f_idx in enumerate(best_path) if score_matrix[s_idx, f_idx] > 1e-4)
        completeness_factor = float((matched_phases / float(S)) ** 2)
        effective_score = best_score * completeness_factor

        # Gán điểm thưởng cho các khung hình thuộc chuỗi
        frame_bonuses = np.zeros(T, dtype=np.float32)
        for s_idx, f_idx in enumerate(best_path):
            frame_bonuses[f_idx] = effective_score * (1.0 + 0.1 * s_idx)
            
        return effective_score, best_path, frame_bonuses, c_temporal
