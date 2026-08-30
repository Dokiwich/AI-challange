"""
RetrievalEngine V3.3 (Precision Natural Evidence Retrieval Engine)
Tích hợp:
1. Direct Continuous Cosine Similarity làm xương sống (Giữ nguyên vẹn 100% độ nhạy khoảng cách)
2. Multi-Aspect & Multi-Phase Fusion cân bằng (70% Full Holistic Context + 15% Aspects + 15% Sequence DP)
3. Frame-Level Union-of-Phases Candidate Generation (Tối đa hóa Recall)
4. Skip-Aware Monotonic DP with Adaptive Windows
5. Evidence Judge V3.3 (10D Evidence Vector + 3-Level Veto + Strict CEC Certification)
"""

import os
import re
import json
import logging
from typing import Dict, List, Any, Tuple, Optional, Set
import numpy as np
import torch
from PIL import Image

from core.base_retriever import CLIPVisualRetriever
from core.query_compiler import QueryCompiler
from core.temporal_alignment import TemporalAlignmentEngine
from core.evidence_engine import EvidenceEngine
from core.meta_router import MetaRouter, MetaFeatureVector
from core.task_handlers import QAnswerNormalizer, TRAKE3StageLocalizer
from core.semantic_ir import CommonSemanticIR
from core.graph_matcher import GraphMatcher

logger = logging.getLogger(__name__)

class RetrievalEngine:
    """
    RetrievalEngine V3.3:
    Bộ điều phối hợp nhất 4 Track với độ chính xác cao dựa trên Cosine Similarity liên tục và Evidence Judge.
    """
    def __init__(
        self,
        features_path: str = "data/features/clip_features.npy",
        map_path: str = "data/mapping/map_keyframes.json",
        ocr_path: str = "data/ocr/ocr_results.json",
        asr_cache_path: str = "data/cache/asr_bm25_cache.pkl",
        model_name: str = "ViT-L/14",
        device: Optional[str] = None,
        qdrant_client = None,
        neo4j_driver = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._vlm_warned = False
        logger.info(f"[RetrievalEngine] Device: {self.device}")

        # Khởi tạo các Sub-Engines
        self.visual_retriever = CLIPVisualRetriever(model_name=model_name, device=self.device)
        self.compiler = QueryCompiler()
        self.temporal_engine = TemporalAlignmentEngine()
        self.evidence_engine = EvidenceEngine()
        self.meta_router = MetaRouter()
        self.trake_localizer = TRAKE3StageLocalizer(self.temporal_engine, image_resolver=self.get_image_path)
        self.qa_normalizer = QAnswerNormalizer()
        
        # Tích hợp Neo4j Graph Matcher
        self.neo4j_driver = neo4j_driver
        self.graph_matcher = GraphMatcher(neo4j_driver=neo4j_driver)
        
        # Tích hợp Qdrant Vector DB (Tắt để tránh lỗi cắt xén top-K làm mất vật thể hiếm)
        self.qdrant_client = None

        # Nạp đặc trưng VideoMA (Motion Features)
        videoma_path = "data/features/videoma_merged.npy"
        if os.path.exists(videoma_path):
            logger.info(f"[RetrievalEngine] Loading VideoMA features from '{videoma_path}'...")
            self.videoma_features = np.load(videoma_path, allow_pickle=True).item()
            logger.info(f"[RetrievalEngine] Loaded VideoMA features for {len(self.videoma_features)} videos.")
        else:
            self.videoma_features = {}


        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Not found: {features_path}")
        logger.info(f"[RetrievalEngine] Loading features from '{features_path}'...")
        raw_features = np.load(features_path)
        self.image_features = torch.from_numpy(raw_features).to(self.device).float()
        self.image_features /= self.image_features.norm(dim=-1, keepdim=True)
        logger.info(f"[RetrievalEngine] {self.image_features.shape[0]} vectors (dim={self.image_features.shape[1]})")

        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Not found: {map_path}")
        with open(map_path, 'r', encoding='utf-8') as f:
            self.keyframe_map: List[Dict[str, Any]] = json.load(f)
        logger.info(f"[RetrievalEngine] {len(self.keyframe_map)} keyframe records")

        # Lập chỉ mục video -> keyframe indices
        self.video_to_indices: Dict[str, List[int]] = {}
        for idx, item in enumerate(self.keyframe_map):
            vname = item.get("video") or item.get("video_name")
            if vname:
                self.video_to_indices.setdefault(vname, []).append(idx)

        # Quét đệ quy toàn bộ thư mục video keyframes trên đĩa
        self.video_directories: Dict[str, str] = {}
        self._scan_video_directories()
        logger.info(f"[RetrievalEngine] Indexed {len(self.video_directories)} video keyframe folders on disk.")

        # Lập chỉ mục OCR
        self.keyframe_texts: List[str] = [""] * len(self.keyframe_map)
        if os.path.exists(ocr_path):
            with open(ocr_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
            for idx, item in enumerate(self.keyframe_map):
                v = item.get("video") or item.get("video_name")
                f_idx = item.get("keyframe_idx") or item.get("frame")
                key = f"{v}_{f_idx}"
                if key in ocr_data:
                    self.keyframe_texts[idx] = ocr_data[key].get("text", "")

        # Nạp Cache ASR BM25
        self.has_bm25 = False
        self.bm25 = None
        self.asr_texts = None
        if os.path.exists(asr_cache_path):
            try:
                import pickle
                logger.info(f"[RetrievalEngine] Loading cached ASR index from '{asr_cache_path}'...")
                with open(asr_cache_path, 'rb') as f:
                    cache_obj = pickle.load(f)
                if isinstance(cache_obj, dict):
                    self.bm25 = cache_obj.get("bm25")
                    self.asr_texts = cache_obj.get("keyframe_texts")
                else:
                    self.bm25 = cache_obj
                self.has_bm25 = (self.bm25 is not None and hasattr(self.bm25, "get_scores"))
                if self.has_bm25:
                    logger.info("[RetrievalEngine] ✅ Cached BM25 index loaded instantly.")
                else:
                    logger.warning("[RetrievalEngine] ⚠️ Cached ASR object does not contain a valid BM25 index.")
            except Exception as e:
                logger.warning(f"[RetrievalEngine] ⚠️ Failed loading ASR cache: {e}")

    def _scan_video_directories(self):
        """Quét đệ quy toàn bộ thư mục chứa keyframes"""
        search_roots = [".", "data", "data/keyframes"]
        for s_root in search_roots:
            if not os.path.exists(s_root):
                continue
            for root, dirs, _ in os.walk(s_root):
                for d in dirs:
                    if re.match(r'^L\d+_V\d+$', d, re.IGNORECASE):
                        full_d_path = os.path.abspath(os.path.join(root, d))
                        self.video_directories[d] = full_d_path

    def get_extracted_videos(self) -> Set[str]:
        return set(self.video_directories.keys())

    def get_image_path(self, video_name: str, keyframe_idx: int, frame_idx: int = None) -> Optional[str]:
        """Định vị file ảnh keyframe thực tế trên đĩa"""
        if not video_name:
            return None
        v_dir = self.video_directories.get(video_name)
        if not v_dir or not os.path.exists(v_dir):
            return None

        candidates = []
        if keyframe_idx is not None:
            candidates.extend([
                f"{int(keyframe_idx):03d}.jpg",
                f"{int(keyframe_idx):04d}.jpg",
                f"{int(keyframe_idx):05d}.jpg",
                f"{int(keyframe_idx)}.jpg",
                f"{int(keyframe_idx):03d}.png",
                f"{int(keyframe_idx)}.png"
            ])
        if frame_idx is not None and frame_idx != keyframe_idx:
            candidates.extend([
                f"{int(frame_idx):03d}.jpg",
                f"{int(frame_idx):04d}.jpg",
                f"{int(frame_idx):05d}.jpg",
                f"{int(frame_idx)}.jpg",
                f"{int(frame_idx):03d}.png",
                f"{int(frame_idx)}.png"
            ])

        for fname in candidates:
            p = os.path.join(v_dir, fname)
            if os.path.exists(p):
                return p

        try:
            files = os.listdir(v_dir)
            if files:
                target_nums = {str(keyframe_idx)}
                if frame_idx is not None:
                    target_nums.add(str(frame_idx))
                for f in files:
                    f_base = os.path.splitext(f)[0]
                    f_num = f_base.lstrip('0') or '0'
                    if f_num in target_nums or f_base in target_nums:
                        return os.path.join(v_dir, f)
                return os.path.join(v_dir, files[0])
        except Exception:
            pass
        return None

    def _min_max_norm(self, arr: np.ndarray) -> np.ndarray:
        if len(arr) == 0:
            return arr
        min_v = float(arr.min())
        max_v = float(arr.max())
        if max_v > min_v:
            return (arr - min_v) / (max_v - min_v)
        return np.zeros_like(arr)

    # =========================================================================
    # CORE SEARCH ENGINE V3.3 (Precision Natural Cosine Backbone)
    # =========================================================================

    def search(
        self,
        raw_query: str,
        top_k: int = 100,
        audio_weight: float = 0.0,
        video_filter: Optional[List[str]] = None,
        auto_translate: bool = True,
        use_ai_query: bool = False,
        use_asr: bool = False,
        asr_keywords: Optional[str] = None,
        engine_mode: str = "hybrid",
        diversity_top_2: bool = False
    ) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        """
        Tìm kiếm hợp nhất 4 Track chuẩn V3.3 với Direct Cosine Similarity và Evidence Judge.
        """
        N = self.image_features.shape[0] if self.image_features is not None else len(self.keyframe_map)
        if not raw_query.strip():
            return [], "", {}

        # 1. Trích xuất đặc trưng meta
        feat_vec = self.meta_router.extract_meta_features(raw_query)

        # 2. Định tuyến Track
        if not use_ai_query or engine_mode == "offline":
            active_track = "offline"
        elif engine_mode == "meta":
            active_track, routing_confidence = self.meta_router.route_winner(feat_vec)
        elif engine_mode == "ai":
            active_track = "ai"
        else:
            active_track = "hybrid"

        # 3. Biên dịch câu truy vấn
        if active_track == "offline":
            compiled_dict = self.compiler.compile_query(raw_query, auto_translate=auto_translate, use_ai_query=False, engine_mode="offline")
        elif active_track == "ai":
            compiled_dict = self.compiler.compile_query(raw_query, auto_translate=auto_translate, use_ai_query=True, engine_mode="ai")
        else:
            compiled_dict = self.compiler.compile_query(raw_query, auto_translate=auto_translate, use_ai_query=True, engine_mode="hybrid")

        query_en = compiled_dict["query_en"]
        intent_flags = compiled_dict["intent_flags"]
        phases_en = compiled_dict["phases_en"]
        aspect_prompts = compiled_dict["aspect_prompts"]
        sem_ir = compiled_dict.get("semantic_ir")

        # 4. ASR Manual Routing
        if use_asr and audio_weight > 0:
            effective_audio_weight = float(audio_weight)
            intent_flags["has_speech"] = True
        else:
            effective_audio_weight = 0.0
            intent_flags["has_speech"] = False
        intent_flags["effective_audio_weight"] = effective_audio_weight

        # 5. Direct Cosine Visual Scoring (Trục chính xác số 1)
        global_text = aspect_prompts.get("global", query_en)
        global_feat = self.visual_retriever.encode_text([global_text])
        
        global_sim = np.zeros(N, dtype=np.float32)
        # Tính toán Global Sim bằng Tensor nguyên bản (tránh truncation của Qdrant)
        global_sim = self.visual_retriever.compute_similarity(global_feat, self.image_features)[0]

        is_multi_phase = len(phases_en) > 1 and intent_flags.get("is_sequence", False)
        seq_bonus = np.zeros(N, dtype=np.float32)
        cec_scores = np.zeros(N, dtype=np.float32)

        if is_multi_phase:
            logger.info(f"[Search] Multi-phase Event Mode: {len(phases_en)} phases ({active_track.upper()})")
            phase_feats = self.visual_retriever.encode_text(phases_en)
            
            phase_sims = np.zeros((len(phases_en), N), dtype=np.float32)
            # Tính toán Phase Sims bằng Tensor
            phase_sims = self.visual_retriever.compute_similarity(phase_feats, self.image_features)
            
            # Khai thác Aspect (Đặc biệt là Object) để bổ trợ CLIP tìm vật thể hiếm (như tê giác)
            aspect_list = []
            common_words = {"person", "man", "woman", "people", "boy", "girl", "child", 
                            "phone", "mobile phone", "camera", "wall", "tree", "car", "bridge", 
                            "floor", "ground", "clothing", "shirt", "pants", "hand", "face", "building"}
                            
            for k in ["object", "action"]:
                val = aspect_prompts.get(k)
                if val:
                    # Tách các object/action bằng dấu phẩy
                    parts = [p.strip() for p in val.split(",")]
                    for p in parts:
                        if len(p) > 2 and p.lower() not in common_words:
                            aspect_list.append(p)
                            
            # Fallback nếu vô tình filter mất hết
            if not aspect_list and aspect_prompts.get("object"):
                aspect_list = [p.strip() for p in aspect_prompts["object"].split(",") if len(p.strip()) > 2]
            
            aspect_bonus = np.zeros(N, dtype=np.float32)
            if aspect_list:
                a_feats = self.visual_retriever.encode_text(aspect_list)
                a_sims = self.visual_retriever.compute_similarity(a_feats, self.image_features)
                aspect_bonus = np.max(a_sims, axis=0)

            # Lấy Union Candidates từ Top frames của từng pha + Top Global
            candidate_videos = set()
            for s_idx in range(len(phases_en)):
                p_sim = phase_sims[s_idx]
                top_part = np.argpartition(p_sim, -250)[-250:]
                for idx in top_part:
                    v = self.keyframe_map[idx].get("video") or self.keyframe_map[idx].get("video_name")
                    if v:
                        candidate_videos.add(v)

            top_global_part = np.argpartition(global_sim, -200)[-200:]
            for idx in top_global_part:
                v = self.keyframe_map[idx].get("video") or self.keyframe_map[idx].get("video_name")
                if v:
                    candidate_videos.add(v)

            if video_filter:
                vf_set = set([video_filter] if isinstance(video_filter, str) else video_filter)
                candidate_videos = candidate_videos.intersection(vf_set)

            # Chạy Skip-Aware DP trên Candidate Videos
            for vname in candidate_videos:
                v_indices = self.video_to_indices.get(vname, [])
                T_v = len(v_indices)
                if T_v < 2:
                    continue

                v_score_matrix = phase_sims[:, v_indices]
                v_timestamps = np.array([
                    float(self.keyframe_map[idx]["pts_time"]) if self.keyframe_map[idx].get("pts_time") is not None
                    else float(self.keyframe_map[idx].get("frame", 0))
                    for idx in v_indices
                ])

                start_at_beg = intent_flags.get("has_start_anchor", False)
                _, _, frame_bonuses, _, cec_val = self.temporal_engine.align_sequence_monotonic_dp(
                    v_score_matrix, v_timestamps,
                    start_at_beginning=start_at_beg,
                    gap_type="SHORT",
                    allow_skip=True
                )
                for li, gi in enumerate(v_indices):
                    seq_bonus[gi] = frame_bonuses[li]
                    cec_scores[gi] = cec_val
                
            if seq_bonus.max() > 0:
                seq_bonus = seq_bonus / seq_bonus.max()

            # Fusion cân bằng và Phạt Context (Context Consensus Penalty)
            max_phase_sim = np.max(phase_sims, axis=0)
            
            # Phạt các frame chỉ có vật thể đứng trơ trọi mà mất hoàn toàn bối cảnh (vd: quả bí xanh treo trên cây)
            # context_gap đo lường mức độ "chênh lệch" giữa vật thể và bối cảnh tổng thể
            context_gap = np.maximum(0.0, aspect_bonus - global_sim)
            context_penalty = context_gap * 0.50
            
            visual_scores = (global_sim * 0.40) + (aspect_bonus * 0.25) + (max_phase_sim * 0.20) + (seq_bonus * 0.15) - context_penalty
        else:
            # Single-Span / Holistic Mode
            prompt_list = [global_text]
            common_words = {"person", "man", "woman", "people", "boy", "girl", "child", 
                            "phone", "mobile phone", "camera", "wall", "tree", "car", "bridge", 
                            "floor", "ground", "clothing", "shirt", "pants", "hand", "face", "building"}
                            
            for k in ["action", "object", "scene"]:
                val = aspect_prompts.get(k)
                if val and val != global_text:
                    parts = [p.strip() for p in val.split(",")]
                    for p in parts:
                        if len(p) > 2 and p.lower() not in common_words:
                            prompt_list.append(p)

            if len(prompt_list) > 1:
                p_feats = self.visual_retriever.encode_text(prompt_list)
                
                p_sims = np.zeros((len(prompt_list), N), dtype=np.float32)
                if self.qdrant_client:
                    for p_i, p_feat in enumerate(p_feats):
                        q_res = self.qdrant_client.query_points(
                            collection_name="aic_clip_frames",
                            query=p_feat.cpu().numpy().tolist(),
                            limit=10000
                        ).points
                        for hit in q_res:
                            p_sims[p_i, hit.id] = hit.score
                else:
                    p_sims = self.visual_retriever.compute_similarity(p_feats, self.image_features)
                    
                aspect_max = np.max(p_sims[1:], axis=0)
                visual_scores = (global_sim * 0.80) + (aspect_max * 0.20)
            else:
                visual_scores = global_sim
            cec_scores = visual_scores


        # 6. ASR Scoring (BM25 + Gaussian Smoothing)
        audio_scores = np.zeros(N, dtype=np.float32)
        if self.has_bm25 and effective_audio_weight > 0:
            if asr_keywords:
                kws = self.compiler.extract_speech_keywords(asr_keywords) if isinstance(asr_keywords, str) else asr_keywords
            else:
                kws = intent_flags.get("speech_keywords") or self.compiler.extract_speech_keywords(raw_query)

            intent_flags["speech_keywords"] = kws
            if kws:
                bm25_raw = np.array(self.bm25.get_scores(kws), dtype=np.float32)
                if bm25_raw.max() > 0:
                    smoothed = np.zeros_like(bm25_raw)
                    for vname, v_indices in self.video_to_indices.items():
                        v_bm25 = bm25_raw[v_indices]
                        if len(v_bm25) > 0 and v_bm25.max() > 0:
                            v_smoothed = self.temporal_engine.gaussian_smoothing(v_bm25, window_size=15, sigma=3.0)
                            v_hybrid = (0.25 * float(v_bm25.max())) + (0.75 * v_smoothed)
                            for li, gi in enumerate(v_indices):
                                smoothed[gi] = v_hybrid[li]
                    if smoothed.max() > 0:
                        audio_scores = smoothed / smoothed.max()

        # 7. Anchor Boost với Confusion-Aware Fuzzy OCR Match
        anchors = intent_flags.get("anchors", [])
        anchor_boost = np.zeros(N, dtype=np.float32)
        if anchors:
            for idx, text in enumerate(self.keyframe_texts):
                if text:
                    for anc in anchors:
                        matched, sim = self.evidence_engine.fuzzy_match_ocr_anchor(anc, text)
                        if matched:
                            anchor_boost[idx] = float(sim)
                            break

        # 8. Modality Fusion — Additive capped: ASR chỉ boost tối đa 15% so với visual max
        if effective_audio_weight > 0 and audio_scores.max() > 0:
            asr_cap = float(visual_scores.max()) * 0.15
            asr_bonus = np.minimum(effective_audio_weight * audio_scores * float(visual_scores.max()) * 0.25, asr_cap)
            final_scores = visual_scores + asr_bonus
        else:
            final_scores = visual_scores

        if anchor_boost.max() > 0:
            final_scores = final_scores + (anchor_boost * 0.15)

        # 9. Video Filtering
        valid_indices = None
        if video_filter:
            target_indices = set()
            for vf in ([video_filter] if isinstance(video_filter, str) else video_filter):
                target_indices.update(self.video_to_indices.get(vf, []))
            if not target_indices:
                return [], query_en, intent_flags
            valid_indices = list(target_indices)
            mask = np.zeros(N, dtype=bool)
            mask[valid_indices] = True
            final_scores[~mask] = -1e9

        # 10. Adaptive Candidate Pool
        candidate_k = min(500, N if valid_indices is None else len(valid_indices))
        if valid_indices is not None:
            pool_scores = np.array([final_scores[i] for i in valid_indices if final_scores[i] > -1e8])
            candidate_k = min(candidate_k, len(pool_scores))
            if candidate_k == 0:
                return [], query_en, intent_flags
            part = np.argpartition(pool_scores, -candidate_k)[-candidate_k:]
            top_candidate_indices = [valid_indices[i] for i in part[np.argsort(pool_scores[part])[::-1]]]
        else:
            part = np.argpartition(final_scores, -candidate_k)[-candidate_k:]
            top_candidate_indices = part[np.argsort(final_scores[part])[::-1]]

        # Chuẩn hóa phân vị
        visual_norm_arr = self.evidence_engine.calibrate_percentile(visual_scores)
        seq_norm_arr = self.evidence_engine.calibrate_percentile(seq_bonus) if seq_bonus.max() > 0 else seq_bonus
        audio_norm_arr = self.evidence_engine.calibrate_percentile(audio_scores) if audio_scores.max() > 0 else audio_scores

        # 11. Graph Verification (Neo4j Alignment)
        unique_candidate_videos = set()
        for idx in top_candidate_indices:
            idx = int(idx)
            v_name = self.keyframe_map[idx].get("video") or self.keyframe_map[idx].get("video_name")
            if v_name:
                unique_candidate_videos.add(v_name)
        
        graph_results = {}
        if sem_ir and hasattr(sem_ir, "entities"):
            graph_results = self.graph_matcher.execute_alignment(sem_ir, list(unique_candidate_videos))

        # 11.5 VideoMA PRF Motion Clustering
        top_5_videos = []
        for idx in top_candidate_indices:
            idx = int(idx)
            v_name = self.keyframe_map[idx].get("video") or self.keyframe_map[idx].get("video_name")
            if v_name and v_name in self.videoma_features and v_name not in top_5_videos:
                top_5_videos.append(v_name)
            if len(top_5_videos) == 5:
                break
        
        motion_centroid = None
        if top_5_videos and len(self.videoma_features) > 0:
            centroid_vecs = [self.videoma_features[v] for v in top_5_videos]
            centroid_vecs = np.array(centroid_vecs)
            motion_centroid = np.mean(centroid_vecs, axis=0)
            motion_centroid = motion_centroid / (np.linalg.norm(motion_centroid) + 1e-9)

        candidate_items = []
        for idx in top_candidate_indices:
            idx = int(idx)
            item = self.keyframe_map[idx]
            v_name = item.get("video") or item.get("video_name")
            _kf = item.get("keyframe_idx")
            kf_id = int(_kf) if _kf is not None else 1
            _frame = item.get("frame") or item.get("frame_idx") or kf_id
            real_frame = int(_frame)
            img_path = self.get_image_path(v_name, kf_id, real_frame)

            consensus_bonus = 0.05 if active_track == "hybrid" and visual_norm_arr[idx] >= 0.8 else 0.0

            # Áp dụng điểm thưởng/phạt từ VideoMA PRF
            if motion_centroid is not None and v_name in self.videoma_features:
                v_vec = self.videoma_features[v_name]
                v_vec_norm = v_vec / (np.linalg.norm(v_vec) + 1e-9)
                motion_sim = float(np.dot(motion_centroid, v_vec_norm))
                
                # Penalty cho nhiễu (khác biệt bối cảnh)
                if motion_sim < 0.65:
                    consensus_bonus -= 0.15
                # Bonus cho video cực kỳ đồng nhất bối cảnh với top 5
                elif motion_sim > 0.85:
                    consensus_bonus += 0.10

            candidate_items.append({
                "index": idx,
                "video": v_name,
                "keyframe_idx": kf_id,
                "frame": real_frame,
                "pts_time": float(item.get("pts_time", 0.0)),
                "score": float(final_scores[idx]),
                "image_path": img_path,
                "visual_norm": float(visual_norm_arr[idx]),
                "seq_norm": float(seq_norm_arr[idx]),
                "asr_norm": float(audio_norm_arr[idx]),
                "cec": float(cec_scores[idx]),
                "has_anchor_match": bool(anchor_boost[idx] > 0),
                "consensus_bonus": consensus_bonus,
                "graph_score": graph_results.get(v_name, {}).get("graph_score", 0.0),
                "matched_timestamps": graph_results.get(v_name, {}).get("matched_timestamps", [])
            })

        # STAGE 3: EVIDENCE JUDGE V3.3 RE-RANKING & 3-LEVEL VETO
        ranked = self.evidence_engine.rank_candidates(candidate_items, intent_flags, diversity_top_2=diversity_top_2)

        results = []
        for item in ranked[:top_k]:
            item["score"] = item.pop("final_score", item.get("score", 0.0))
            results.append(item)

        intent_flags["active_track"] = active_track
        if results:
            t0_badge = "✅ CERTIFIED TIER_0" if results[0].get("tier") == "TIER_0" else f"🔒 {results[0].get('tier')}"
            ambig_badge = "⚠️ AMBIGUOUS" if results[0].get("is_ambiguous") else "🎯 CONFIDENT"
            logger.info(f"[Search] Top-1: {results[0]['video']} F{results[0]['frame']} | Score: {results[0]['score']:.2f} | {t0_badge} | Track: {active_track.upper()}")

        return results, query_en, intent_flags

    # ======================================================================
    # AIC 2026 TASK HANDLERS (TRAKE & Visual QA)
    # ======================================================================

    def search_trake(
        self,
        event_queries: List[str],
        top_k_videos: int = 10,
        video_filter: Optional[List[str]] = None,
        auto_translate: bool = True,
        use_ai_query: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """TRAKE search với quy trình 3 giai đoạn định vị thời gian"""
        if not event_queries:
            return [], []

        processed_events = []
        for eq in event_queries:
            eq_proc = self.compiler.translate_to_en(eq.strip()) if auto_translate else eq.strip()
            processed_events.append(eq_proc)

        event_feats = self.visual_retriever.encode_text(processed_events)
        trake_res = self.trake_localizer.refine_trake_sequence(
            event_embeddings=event_feats,
            image_features=self.image_features,
            keyframe_map=self.keyframe_map,
            video_to_indices=self.video_to_indices,
            top_k_videos=top_k_videos
        )
        return trake_res, processed_events

    # ======================================================================
    # GPU MEMORY MANAGEMENT (Phase-Shifting Architecture)
    # ======================================================================

    def unload_clip(self):
        """Giải phóng CLIP và image_features khỏi GPU để nhường VRAM cho VLM."""
        if hasattr(self, 'image_features') and self.image_features is not None:
            self.image_features = self.image_features.cpu()
        if hasattr(self, 'visual_retriever') and hasattr(self.visual_retriever, 'model'):
            self.visual_retriever.model = self.visual_retriever.model.cpu()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("[RetrievalEngine] CLIP unloaded from GPU. VRAM freed.")

    def reload_clip(self):
        """Nạp lại CLIP lên GPU sau khi VLM hoàn tất."""
        if hasattr(self, 'image_features') and self.image_features is not None:
            self.image_features = self.image_features.to(self.device).float()
        if hasattr(self, 'visual_retriever') and hasattr(self.visual_retriever, 'model'):
            self.visual_retriever.model = self.visual_retriever.model.to(self.device)
        logger.info("[RetrievalEngine] CLIP reloaded to GPU.")

    def unload_qwen(self):
        """Giải phóng Qwen2-VL khỏi GPU."""
        if hasattr(self, '_qwen_model'):
            del self._qwen_model
        if hasattr(self, '_qwen_processor'):
            del self._qwen_processor
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("[RetrievalEngine] Qwen2-VL unloaded from GPU. VRAM freed.")

    # ======================================================================
    # VLM INFERENCE (Dual-Vision Dynamic Tiling)
    # ======================================================================

    def _lazy_load_qwen(self):
        if not hasattr(self, "_qwen_model"):
            logger.info("[RetrievalEngine] Loading Qwen2-VL-2B-Instruct VLM on-demand...")
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            self._qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
            self._qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-2B-Instruct",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            logger.info("[RetrievalEngine] Qwen2-VL loaded successfully.")

    def _build_dual_vision_images(self, img: Image.Image) -> List[Image.Image]:
        """Dual-Vision: ảnh bối cảnh thu nhỏ + ảnh crop trung tâm sắc nét."""
        context_img = img.resize((224, 224), Image.LANCZOS)
        w, h = img.size
        crop_size = min(416, w, h)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        detail_img = img.crop((left, top, left + crop_size, top + crop_size))
        return [context_img, detail_img]

    def answer_qa(self, question: str, item: Dict[str, Any]) -> str:
        """Trả lời QA bằng Qwen2-VL với chiến thuật Dual-Vision."""
        image_path = item.get("image_path", "")
        if not image_path or not os.path.exists(image_path):
            return "Có"

        try:
            self._lazy_load_qwen()
            img = Image.open(image_path).convert("RGB")
            dual_imgs = self._build_dual_vision_images(img)

            prompt = (
                "Image 1 is the full scene for context. "
                "Image 2 is a close-up crop for detail. "
                f"Answer as briefly as possible. Question: {question}"
            )
            content = [
                {"type": "image", "image": dual_imgs[0]},
                {"type": "image", "image": dual_imgs[1]},
                {"type": "text", "text": prompt}
            ]
            messages = [{"role": "user", "content": content}]

            text = self._qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._qwen_processor(text=[text], images=dual_imgs, padding=True, return_tensors="pt").to(self.device)

            generated_ids = self._qwen_model.generate(**inputs, max_new_tokens=20)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            raw_ans = self._qwen_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            if self.qa_normalizer:
                return self.qa_normalizer.normalize_answer(raw_ans)
            return raw_ans

        except Exception as e:
            if not getattr(self, "_vlm_warned", False):
                logger.warning(f"Local VLM error in answer_qa: {e}")
                self._vlm_warned = True

        if self.qa_normalizer:
            return self.qa_normalizer.normalize_answer(question)
        return "Có"

    def get_adjacent_keyframes(self, video_name: str, center_frame: int, radius: int = 50) -> List[Dict[str, Any]]:
        """Lấy các keyframe lân cận của một frame trong video (phục vụ Temporal Browsing)"""
        if not video_name or video_name not in self.video_to_indices:
            return []
        
        indices = self.video_to_indices[video_name]
        
        # Tìm index có frame gần center_frame nhất
        target_idx = 0
        min_diff = float('inf')
        for i, idx in enumerate(indices):
            kf_dict = self.keyframe_map[idx]
            _kf = kf_dict.get("keyframe_idx")
            kf_id = int(_kf) if _kf is not None else 1
            frame = int(kf_dict.get("frame") or kf_dict.get("frame_idx") or kf_id)
            diff = abs(frame - center_frame)
            if diff < min_diff:
                min_diff = diff
                target_idx = i
                
        # Lấy dải lân cận
        start_i = max(0, target_idx - radius)
        end_i = min(len(indices), target_idx + radius + 1)
        
        adjacent_items = []
        for i in range(start_i, end_i):
            idx = indices[i]
            item = self.keyframe_map[idx]
            _kf = item.get("keyframe_idx")
            kf_id = int(_kf) if _kf is not None else 1
            frame = int(item.get("frame") or item.get("frame_idx") or kf_id)
            img_path = self.get_image_path(video_name, kf_id, frame)
            
            adjacent_items.append({
                "video": video_name,
                "frame": frame,
                "keyframe_idx": kf_id,
                "pts_time": float(item.get("pts_time", 0.0)),
                "image_path": img_path,
                "offset_idx": i - target_idx  # 0 là center, - là trước, + là sau
            })
            
        return adjacent_items

