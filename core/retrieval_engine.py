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
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[RetrievalEngine] Device: {self.device}")

        # Khởi tạo các Sub-Engines
        self.visual_retriever = CLIPVisualRetriever(model_name=model_name, device=self.device)
        self.compiler = QueryCompiler()
        self.temporal_engine = TemporalAlignmentEngine()
        self.evidence_engine = EvidenceEngine()
        self.meta_router = MetaRouter()
        self.trake_localizer = TRAKE3StageLocalizer(self.temporal_engine, image_resolver=self.get_image_path)
        self.qa_normalizer = QAnswerNormalizer()

        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Not found: {features_path}")
        print(f"[RetrievalEngine] Loading features from '{features_path}'...")
        raw_features = np.load(features_path)
        self.image_features = torch.from_numpy(raw_features).to(self.device).float()
        self.image_features /= self.image_features.norm(dim=-1, keepdim=True)
        print(f"[RetrievalEngine] {self.image_features.shape[0]} vectors (dim={self.image_features.shape[1]})")

        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Not found: {map_path}")
        with open(map_path, 'r', encoding='utf-8') as f:
            self.keyframe_map: List[Dict[str, Any]] = json.load(f)
        print(f"[RetrievalEngine] {len(self.keyframe_map)} keyframe records")

        # Lập chỉ mục video -> keyframe indices
        self.video_to_indices: Dict[str, List[int]] = {}
        for idx, item in enumerate(self.keyframe_map):
            vname = item.get("video") or item.get("video_name")
            if vname:
                self.video_to_indices.setdefault(vname, []).append(idx)

        # Quét đệ quy toàn bộ thư mục video keyframes trên đĩa
        self.video_directories: Dict[str, str] = {}
        self._scan_video_directories()
        print(f"[RetrievalEngine] Indexed {len(self.video_directories)} video keyframe folders on disk.")

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
                print(f"[RetrievalEngine] Loading cached ASR index from '{asr_cache_path}'...")
                with open(asr_cache_path, 'rb') as f:
                    cache_obj = pickle.load(f)
                if isinstance(cache_obj, dict):
                    self.bm25 = cache_obj.get("bm25")
                    self.asr_texts = cache_obj.get("keyframe_texts")
                else:
                    self.bm25 = cache_obj
                self.has_bm25 = (self.bm25 is not None and hasattr(self.bm25, "get_scores"))
                if self.has_bm25:
                    print("[RetrievalEngine] ✅ Cached BM25 index loaded instantly.")
                else:
                    print("[RetrievalEngine] ⚠️ Cached ASR object does not contain a valid BM25 index.")
            except Exception as e:
                print(f"[RetrievalEngine] ⚠️ Failed loading ASR cache: {e}")

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
        N = self.image_features.shape[0]
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
        global_sim = self.visual_retriever.compute_similarity(global_feat, self.image_features)[0]

        is_multi_phase = len(phases_en) > 1 and intent_flags.get("is_sequence", False)
        seq_bonus = np.zeros(N, dtype=np.float32)
        cec_scores = np.zeros(N, dtype=np.float32)

        if is_multi_phase:
            print(f"[Search] Multi-phase Event Mode: {len(phases_en)} phases ({active_track.upper()})")
            phase_feats = self.visual_retriever.encode_text(phases_en)
            phase_sims = self.visual_retriever.compute_similarity(phase_feats, self.image_features)

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

            # Fusion cân bằng: 70% Global Holistic + 15% Max Phase + 15% DP Sequence
            max_phase_sim = np.max(phase_sims, axis=0)
            visual_scores = (global_sim * 0.70) + (max_phase_sim * 0.15) + (seq_bonus * 0.15)
        else:
            # Single-Span / Holistic Mode
            prompt_list = [global_text]
            for k in ["action", "object", "scene"]:
                if aspect_prompts.get(k) and aspect_prompts[k] != global_text:
                    prompt_list.append(aspect_prompts[k])

            if len(prompt_list) > 1:
                p_feats = self.visual_retriever.encode_text(prompt_list)
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
                "consensus_bonus": consensus_bonus
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
            print(f"[Search] Top-1: {results[0]['video']} F{results[0]['frame']} | Score: {results[0]['score']:.2f} | {t0_badge} | Track: {active_track.upper()}")

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

    def answer_qa(self, question: str, item: Dict[str, Any]) -> str:
        """Visual QA Answer normalizer"""
        raw_ans = "yes"
        return self.qa_normalizer.normalize_answer(raw_ans)
