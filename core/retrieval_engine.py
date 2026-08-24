"""
RetrievalEngine V3.0 (AIC 2026 Tournament-Grade 4-Track Engine)
Tích hợp:
1. BaseVisualRetriever & CLIP Visual Interface
2. Track 1 (Offline V2), Track 2 (AI Semantic V2), Track 3 (Hybrid Fusion), Track 4 (Meta-Policy / Winner Selection / Escalation)
3. 4-Stage Hierarchical Precision Verification
4. Decoupled Composite Weighted DP
5. Constraint Graph Evidence Judge
6. AIC Task Handlers: KIS, Visual QA Normalizer, TRAKE 3-Stage Localizer
"""

import os
import glob
import json
import re
import time
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional, Set, Union

from core.base_retriever import CLIPVisualRetriever
from core.query_compiler import QueryCompiler
from core.temporal_alignment import TemporalAlignmentEngine
from core.evidence_engine import EvidenceEngine
from core.meta_router import MetaRouter, MetaFeatureVector
from core.task_handlers import QAnswerNormalizer, TRAKE3StageLocalizer
from core.semantic_ir import CommonSemanticIR


class RetrievalEngine:
    def __init__(
        self,
        features_path: str = "data/features/clip_features.npy",
        map_path: str = "data/mapping/map_keyframes.json",
        model_name: str = "ViT-B/32",
        device: Optional[str] = None
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
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
        with open(map_path, "r", encoding="utf-8") as f:
            self.keyframe_map = json.load(f)
        print(f"[RetrievalEngine] {len(self.keyframe_map)} keyframe records")

        self.video_to_indices: Dict[str, List[int]] = {}
        for idx, item in enumerate(self.keyframe_map):
            vname = item.get("video") or item.get("video_name")
            if vname:
                if vname not in self.video_to_indices:
                    self.video_to_indices[vname] = []
                self.video_to_indices[vname].append(idx)

        self.video_to_dir = self._scan_video_directories()

        self.has_bm25 = False
        self._load_transcripts()
        self._initialized = True

    # ======================================================================
    # Data loading & Image resolution
    # ======================================================================

    def _load_transcripts(self, transcript_dir="data/transcripts", fps=25):
        import pickle
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")
        cache_file = os.path.join(cache_dir, "asr_bm25_cache.pkl")
        
        if os.path.exists(cache_file):
            try:
                print(f"[RetrievalEngine] Loading cached ASR index from '{cache_file}'...")
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                self.keyframe_texts = cached_data["keyframe_texts"]
                self.bm25 = cached_data["bm25"]
                self.has_bm25 = True
                print("[RetrievalEngine] ✅ Cached BM25 index loaded instantly.")
                return
            except Exception as e:
                print(f"[RetrievalEngine] Cache read error: {e}, rebuilding...")

        self.keyframe_texts = ["" for _ in range(len(self.keyframe_map))]
        if not os.path.exists(transcript_dir):
            return
        print(f"[RetrievalEngine] Loading ASR transcripts from {transcript_dir}...")
        for json_path in glob.glob(os.path.join(transcript_dir, "*.json")):
            vname = os.path.splitext(os.path.basename(json_path))[0]
            v_indices = self.video_to_indices.get(vname)
            if not v_indices:
                continue
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                segments = data if isinstance(data, list) else data.get("segments", [])
                for seg in segments:
                    text = seg.get("text", "").strip()
                    if not text:
                        continue
                    start_s = float(seg.get("start", 0))
                    end_s = float(seg.get("end", start_s))
                    mid_s = (start_s + end_s) / 2.0
                    
                    best_idx = None
                    best_diff = 999999
                    for idx in v_indices:
                        pts = self.keyframe_map[idx].get("pts_time")
                        if pts is not None:
                            diff = abs(float(pts) - mid_s)
                        else:
                            frame = int(self.keyframe_map[idx].get("frame", 0))
                            diff = abs((frame / float(fps)) - mid_s)
                        if diff < best_diff:
                            best_diff = diff
                            best_idx = idx
                            
                    if best_idx is not None and best_diff <= 15.0:
                        if self.keyframe_texts[best_idx]:
                            self.keyframe_texts[best_idx] += " " + text
                        else:
                            self.keyframe_texts[best_idx] = text
            except Exception as e:
                pass

        try:
            from rank_bm25 import BM25Okapi
            corpus_tokens = [t.lower().split() if t else [] for t in self.keyframe_texts]
            self.bm25 = BM25Okapi(corpus_tokens)
            self.has_bm25 = True
            
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump({"keyframe_texts": self.keyframe_texts, "bm25": self.bm25}, f)
            print(f"[RetrievalEngine] ✅ ASR BM25 index built and cached.")
        except Exception as e:
            print(f"[RetrievalEngine] BM25 indexing error: {e}")

    def _scan_video_directories(self) -> Dict[str, str]:
        """Quét và lập chỉ mục toàn bộ thư mục video keyframes trên đĩa"""
        video_to_dir = {}
        search_roots = [
            ".",
            "data",
            "data/keyframes",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "keyframes")
        ]
        for s_root in search_roots:
            if os.path.exists(s_root):
                for root, dirs, _ in os.walk(s_root):
                    for d in dirs:
                        if (d.startswith("L") and "_" in d) or d.startswith("V") or d.startswith("video_"):
                            if d not in video_to_dir:
                                video_to_dir[d] = os.path.abspath(os.path.join(root, d))
        print(f"[RetrievalEngine] Indexed {len(video_to_dir)} video keyframe folders on disk.")
        return video_to_dir

    def get_image_path(self, video_name: str, keyframe_idx: int, frame_idx: Optional[int] = None) -> Optional[str]:
        """Tìm đường dẫn ảnh thực tế trên đĩa cho khung hình"""
        video_dir = self.video_to_dir.get(video_name)
        if not video_dir or not os.path.exists(video_dir):
            return None

        candidates = []
        for fid in ([frame_idx, keyframe_idx] if frame_idx is not None else [keyframe_idx]):
            if fid is None:
                continue
            candidates.extend([
                f"{fid:03d}.jpg", f"{fid:04d}.jpg", f"{fid:05d}.jpg", f"{fid:06d}.jpg", f"{fid}.jpg",
                f"{fid:03d}.png", f"{fid:04d}.png", f"{fid:05d}.png", f"{fid:06d}.png", f"{fid}.png",
                f"{fid:03d}.jpeg", f"{fid:04d}.jpeg", f"{fid:05d}.jpeg", f"{fid:06d}.jpeg", f"{fid}.jpeg"
            ])

        for fname in candidates:
            p = os.path.join(video_dir, fname)
            if os.path.exists(p):
                return p
        return None

    def get_extracted_videos(self) -> Set[str]:
        """Lấy danh sách các video đã có thư mục ảnh keyframes trên đĩa"""
        return set(self.video_to_dir.keys())

    # ======================================================================
    # Visual RRF & Similarity Utilities
    # ======================================================================

    def _compute_rrf_visual(self, prompt_list: List[str], top_per: int = 2500, k_rrf: int = 60) -> np.ndarray:
        N = self.image_features.shape[0]
        if not prompt_list:
            return np.zeros(N, dtype=np.float32)

        prompt_weights = [1.0] * len(prompt_list)
        feats = self.visual_retriever.encode_text(prompt_list)
        sim_matrix = self.visual_retriever.compute_similarity(feats, self.image_features)  # (M, N)
        
        rrf_scores = np.zeros(N, dtype=np.float32)
        M = sim_matrix.shape[0]

        for p_idx in range(M):
            p_weight = prompt_weights[p_idx]
            scores = sim_matrix[p_idx]
            top_k_idx = np.argpartition(scores, -top_per)[-top_per:]
            sorted_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]

            ranks = np.arange(1, top_per + 1)
            rrf_scores[sorted_idx] += p_weight * (1.0 / (k_rrf + ranks))

        return rrf_scores

    def _min_max_norm(self, scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return scores
        min_v, max_v = float(scores.min()), float(scores.max())
        if max_v > min_v:
            return (scores - min_v) / (max_v - min_v)
        return np.zeros_like(scores)

    # ======================================================================
    # CORE SEARCH PIPELINE (Tracks 1, 2, 3, 4)
    # ======================================================================

    def search(
        self,
        query_text: str,
        top_k: int = 100,
        video_filter: Optional[List[str] | str] = None,
        auto_translate: bool = True,
        engine_mode: str = "meta",   # "offline" (T1) | "ai" (T2) | "hybrid" (T3) | "meta" (T4)
        use_ai_query: bool = True,
        use_asr: bool = False,
        audio_weight: float = 0.3,
        asr_keywords: Optional[List[str] | str] = None,
        diversity_top_2: bool = False
    ) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        """
        Universal Search Engine:
        Thực thi linh hoạt theo 4 Track độc lập hoặc điều phối thông minh qua Meta-Router.
        """
        raw_query = query_text.strip()
        N = self.image_features.shape[0]

        # 1. Khởi tạo và Phân tích câu hỏi
        offline_compiled = self.compiler.compile_offline_v2(raw_query, auto_translate=auto_translate)
        meta_feat = self.meta_router.extract_meta_features_query(offline_compiled)

        # 2. Xử lý Chế độ Track 4 (Meta-Policy / Winner Selection / Escalation)
        active_track = engine_mode
        if engine_mode == "meta":
            # Mode A: Winner Routing
            active_track = self.meta_router.route_winner(meta_feat)
            print(f"[MetaRouter] Routing query to: Track {active_track.upper()}")

        # 3. Biên dịch câu truy vấn theo Track được chọn
        if active_track == "offline":
            compiled_dict = self.compiler.compile_query(raw_query, auto_translate=auto_translate, engine_mode="offline")
        elif active_track == "ai":
            compiled_dict = self.compiler.compile_query(raw_query, auto_translate=auto_translate, use_ai_query=True, engine_mode="ai")
        else: # "hybrid"
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

        # 5. Visual Scoring (Single-Span RRF vs Multi-Phase Temporal DP)
        is_multi_phase = len(phases_en) > 1 and intent_flags.get("is_sequence")
        seq_bonus = np.zeros(N, dtype=np.float32)

        if is_multi_phase:
            print(f"[Search] Multi-phase mode: {len(phases_en)} phases ({active_track.upper()})")
            phase_sims_raw = []
            sem_importances = []
            
            for i, p in enumerate(phases_en):
                # Tạo aspect prompts hoặc hypotheses cho pha
                p_prompts = [p]
                if isinstance(sem_ir, CommonSemanticIR) and i < len(sem_ir.temporal_phases):
                    p_node = sem_ir.temporal_phases[i]
                    p_prompts = [h["prompt"] for h in p_node.hypotheses] or [p]
                    sem_importances.append(p_node.semantic_importance)
                else:
                    sem_importances.append(0.5)

                rrf = self._compute_rrf_visual(p_prompts, top_per=2500, k_rrf=60)
                phase_sims_raw.append(rrf)

            base_visual = np.max(phase_sims_raw, axis=0)
            S = len(phases_en)

            # Chạy Decoupled Composite Monotonic DP trên từng video
            for vname, v_indices in self.video_to_indices.items():
                T_v = len(v_indices)
                if T_v < S:
                    continue
                v_score_matrix = np.array([scores[v_indices] for scores in phase_sims_raw])
                v_timestamps = np.array([
                    float(self.keyframe_map[idx]["pts_time"]) if self.keyframe_map[idx].get("pts_time") is not None
                    else float(self.keyframe_map[idx].get("frame", 0))
                    for idx in v_indices
                ])
                
                start_at_beg = intent_flags.get("has_start_anchor", False)
                _, _, frame_bonuses, _ = self.temporal_engine.align_sequence_monotonic_dp(
                    v_score_matrix, v_timestamps, semantic_importances=sem_importances,
                    start_at_beginning=start_at_beg, gap_type="SHORT"
                )
                for li, gi in enumerate(v_indices):
                    seq_bonus[gi] = frame_bonuses[li]

            if seq_bonus.max() > 0:
                seq_bonus = seq_bonus / seq_bonus.max()

            visual_scores = (base_visual * 0.45) + (seq_bonus * 1.55)
        else:
            # Single-Span 6D Aspect Prompts
            prompt_list = list(aspect_prompts.values()) if isinstance(aspect_prompts, dict) else aspect_prompts
            if not prompt_list:
                prompt_list = [query_en]
            print(f"[Search] 6D Aspect mode: {len(prompt_list)} prompts ({active_track.upper()})")
            rrf_scores = self._compute_rrf_visual(prompt_list, top_per=2500, k_rrf=60)
            visual_scores = self._min_max_norm(rrf_scores)

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

        # 8. Modality Fusion
        if effective_audio_weight > 0 and audio_scores.max() > 0:
            final_scores = visual_scores * (1.0 + effective_audio_weight * audio_scores)
        else:
            final_scores = visual_scores

        if anchor_boost.max() > 0:
            final_scores = final_scores + (anchor_boost * 0.6)

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

        # 10. Adaptive Candidate Pool (100 / 300 / 500)
        candidate_k = min(500 if is_multi_phase else 250, N if valid_indices is None else len(valid_indices))
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

            # Consensus Bonus (nếu ở mode Hybrid)
            consensus_bonus = 0.15 if active_track == "hybrid" and visual_norm_arr[idx] >= 0.6 else 0.0

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
                "has_anchor_match": bool(anchor_boost[idx] > 0),
                "consensus_bonus": consensus_bonus
            })

        # 11. Constraint Graph Evidence Judge Re-ranking
        ranked = self.evidence_engine.rank_candidates(candidate_items, intent_flags, diversity_top_2=diversity_top_2)

        results = []
        for item in ranked[:top_k]:
            item["score"] = item.pop("final_score", item.get("score", 0.0))
            results.append(item)

        intent_flags["active_track"] = active_track
        if results:
            print(f"[Search] Top-1: {results[0]['score']:.2f} | Tier: {results[0].get('tier')} | CSR: {results[0].get('csr', 1.0):.2f} | Track: {active_track.upper()}")

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
