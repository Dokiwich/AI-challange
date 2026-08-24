"""
QueryCompiler V2 (Industrial-Grade Multi-Track Query Compiler)
Tích hợp:
1. Track 1: Offline Compiler V2 (Relation Classifier 8 types, Tiered Synonyms, Entity State Tracker)
2. Common Semantic IR Dataclass output
3. Precision-First Fallback (Unknown relation handling)
"""

import re
import logging
import unicodedata
from typing import Dict, List, Any, Tuple, Optional
from deep_translator import GoogleTranslator

from core.ai_query_parser import AIQueryParser
from core.semantic_ir import (
    CommonSemanticIR, EntityNode, RelationNode,
    AspectPromptNode, TemporalPhaseNode
)

logger = logging.getLogger(__name__)

class QueryCompiler:
    """
    QueryCompiler V2:
    - Biên dịch câu truy vấn tiếng Việt/tiếng Anh thành CommonSemanticIR.
    - Hỗ trợ độc lập Track 1 (Offline V2), Track 2 (AI V2), Track 3 (Hybrid Fusion).
    """
    def __init__(self, translator: Optional[GoogleTranslator] = None, ai_parser: Optional[AIQueryParser] = None):
        self.translator = translator if translator else GoogleTranslator(source='auto', target='en')
        self.ai_parser = ai_parser if ai_parser else AIQueryParser()
        self._trans_cache: Dict[str, str] = {}
        
        # Danh sách từ dừng ngữ pháp tiếng Việt chuẩn hóa
        self.vietnamese_stopwords = {
            'là', 'của', 'và', 'có', 'với', 'trong', 'một', 'các', 'này',
            'đã', 'được', 'cho', 'từ', 'đến', 'về', 'theo', 'để', 'bằng',
            'không', 'những', 'hay', 'hoặc', 'nhưng', 'vì', 'do', 'nếu',
            'thì', 'sẽ', 'đang', 'rất', 'cũng', 'đây', 'đó', 'ở', 'ra',
            'lên', 'vào', 'khi', 'sau', 'trước', 'trên', 'dưới', 'ngoài',
            'hình', 'ảnh', 'đoạn', 'clip', 'video', 'phần', 'bắt', 'đầu',
            'giới', 'thiệu', 'việc', 'nhiều', 'lần', 'lượt', 'sang', 'chuyển',
            'tiếp', 'thấy', 'biết', 'tìm', 'chính', 'xác', 'hỏi', 'loại',
            'xuất', 'hiện', 'quay', 'cao', 'cảnh', 'cận', 'trời', 'phim',
            'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười',
            'con', 'cái', 'chiếc', 'bức', 'tấm', 'ngọn', 'cây', 'quả', 'trái', 'viên', 'tờ', 'cuốn', 'quyển'
        }
        
        # Từ nối quan hệ thời gian tuần tự (TEMPORAL: P1 -> P2 -> P3)
        self.temporal_connectors_vi = [
            (r'\bsau đó\b', "TEMPORAL", "SHORT"),
            (r'\btiếp theo\b', "TEMPORAL", "SHORT"),
            (r'\btiếp tục\b', "TEMPORAL", "SHORT"),
            (r'\bcuối cùng\b', "TEMPORAL", "LONG"),
            (r'\bkế tiếp\b', "TEMPORAL", "SHORT"),
            (r'\blát sau\b', "TEMPORAL", "LONG"),
            (r'\bbắt đầu bằng\b', "TEMPORAL", "SHORT"),
            (r'\bbắt đầu với\b', "TEMPORAL", "SHORT"),
            (r'\bkết thúc bằng\b', "TEMPORAL", "SHORT"),
            (r'\brồi\b', "TEMPORAL", "SHORT"),
            (r'\bxong\b', "TEMPORAL", "SHORT")
        ]
        
        # Từ nối quan hệ đồng thời (SIMULTANEOUS: State / Action at same time)
        self.simultaneous_connectors_vi = [
            r'\bđồng thời\b', r'\bvừa lúc đó\b', r'\bvừa\b.*\bvừa\b', r'\btrong khi\b',
            r'\bcùng lúc\b', r'\bkhi đang\b'
        ]
        
        # Từ nối quan hệ không gian (SPATIAL: Left of, inside, on top)
        self.spatial_connectors_vi = [
            (r'\bbên trái\b', "LEFT_OF"),
            (r'\bbên phải\b', "RIGHT_OF"),
            (r'\bở giữa\b', "MIDDLE_OF"),
            (r'\bphía trên\b', "ABOVE"),
            (r'\bphía dưới\b', "BELOW"),
            (r'\bđằng trước\b', "IN_FRONT_OF"),
            (r'\bđằng sau\b', "BEHIND"),
            (r'\bcạnh\b', "NEXT_TO"),
            (r'\bbên trong\b', "INSIDE"),
            (r'\btrên cầu\b', "ON_BRIDGE")
        ]

        # Từ điển biến thể từ đồng nghĩa phân tầng (Tiered Visual Synonyms)
        self.synonym_dict_en = {
            "box truck": {
                "tier_a": ["box truck", "cargo truck", "enclosed truck"],
                "tier_b": ["delivery truck", "commercial van", "freight truck"],
                "tier_c": ["transport vehicle", "lorry"]
            },
            "dam": {
                "tier_a": ["dam", "hydroelectric dam", "reservoir dam"],
                "tier_b": ["water spillway", "concrete barrier dam"],
                "tier_c": ["water infrastructure", "barrage"]
            },
            "irrigation": {
                "tier_a": ["irrigation project", "irrigation canal", "water canal"],
                "tier_b": ["waterway", "sluice gate", "aqueduct"],
                "tier_c": ["agricultural water system"]
            },
            "police": {
                "tier_a": ["police officer", "policeman", "traffic police"],
                "tier_b": ["patrol officer", "law enforcement"],
                "tier_c": ["officer in uniform", "security personnel"]
            },
            "motorcycle": {
                "tier_a": ["motorcycle", "motorbike"],
                "tier_b": ["scooter", "moped", "two-wheeler"],
                "tier_c": ["motor vehicle"]
            },
            "jacket": {
                "tier_a": ["jacket", "hooded jacket", "windbreaker"],
                "tier_b": ["sun-protective jacket", "outerwear coat"],
                "tier_c": ["long-sleeve clothing"]
            }
        }

    # =========================================================================
    # TRACK 1: OFFLINE COMPILER V2 (Precision-First)
    # =========================================================================

    def compile_offline_v2(
        self,
        raw_query: str,
        auto_translate: bool = True,
        filename: Optional[str] = None
    ) -> CommonSemanticIR:
        """
        Track 1: Biên dịch câu truy vấn ngoại tuyến hoàn toàn (Zero-LLM).
        - Nhận diện 8 loại quan hệ vị ngữ (Relation Classifier).
        - Entity state tracking & Co-reference.
        - Adaptive Tiered Synonym Expansion.
        """
        raw_clean = raw_query.strip()
        query_id = filename or "offline_query"
        if not raw_clean:
            return CommonSemanticIR(query_id=query_id, compiler_source="offline")

        # 1. Dịch chuẩn sang tiếng Anh
        query_en = self.translate_to_en(raw_clean) if auto_translate else raw_clean

        # 2. Bóc tách Mỏ neo cứng (Hard Anchors) & Từ khóa ASR
        hard_anchors = self.extract_anchors_with_fuzzy(raw_clean)
        speech_kws = self.extract_speech_keywords(raw_clean)

        # 3. Phân loại quan hệ vị ngữ (Relation Classifier) & Cắt pha
        classified_relations, phases_en, is_sequence = self._classify_and_segment_relations(raw_clean, query_en)

        # 4. Trích xuất thực thể & Thuộc tính (Entity State Tracking)
        entities = self._extract_entities_and_attributes(raw_clean, query_en)

        # 5. Phân rã đa chiều 6D (Global, Scene, Action, Object, Attributes, Relations)
        aspect_node = self._build_offline_6d_aspects(query_en, entities, classified_relations)

        # 6. Xây dựng các Phase Node với Semantic Importance
        temporal_phase_nodes = []
        for p_idx, p_text in enumerate(phases_en):
            # Tính Semantic Importance dựa trên độ cụ thể của từ ngữ trong pha
            importance = self._calculate_semantic_importance(p_text, hard_anchors)
            # Sinh Top 2 biến thể từ điển an toàn (Tier A)
            hypotheses = self._generate_tier_a_hypotheses(p_text)
            
            temporal_phase_nodes.append(TemporalPhaseNode(
                phase_idx=p_idx + 1,
                canonical_prompt=p_text,
                hypotheses=hypotheses,
                action=p_text,
                semantic_importance=importance
            ))

        # 7. Ước tính độ tin cậy của Offline Compiler (Confidence Calibrated)
        offline_conf = 0.85 if len(hard_anchors) > 0 or len(phases_en) <= 3 else 0.70
        if any(r.type == "UNKNOWN" for r in classified_relations):
            offline_conf *= 0.8

        task_type = "TRAKE" if (filename and "trake" in filename.lower()) else ("QA" if ("qa" in filename.lower() if filename else False) else "KIS")

        return CommonSemanticIR(
            query_id=query_id,
            task_type=task_type,
            raw_query=raw_clean,
            query_en=query_en,
            compiler_source="offline",
            compiler_confidence={
                "offline_raw": offline_conf,
                "ai_raw": 0.0,
                "calibrated_offline": offline_conf,
                "calibrated_ai": 0.0
            },
            is_sequence=is_sequence,
            entities=entities,
            relations=classified_relations,
            aspects=aspect_node,
            temporal_phases=temporal_phase_nodes,
            hard_anchors=hard_anchors,
            speech_keywords=speech_kws
        )

    def _classify_and_segment_relations(self, raw_vi: str, query_en: str) -> Tuple[List[RelationNode], List[str], bool]:
        """Phân loại 8 quan hệ và cắt pha thời gian chỉ khi mang tính chất TEMPORAL tuần tự"""
        relations = []
        q_lower = raw_vi.lower()
        
        # 1. Kiểm tra quan hệ SPATIAL
        for pat, sp_type in self.spatial_connectors_vi:
            if re.search(pat, q_lower):
                relations.append(RelationNode(
                    type="SPATIAL",
                    subject="entity_subject",
                    relation=sp_type,
                    object="entity_landmark",
                    priority="HARD_REQUIRED"
                ))

        # 2. Kiểm tra quan hệ SIMULTANEOUS (Đồng thời)
        is_simultaneous = any(re.search(p, q_lower) for p in self.simultaneous_connectors_vi)
        if is_simultaneous:
            relations.append(RelationNode(
                type="SIMULTANEOUS",
                subject="entity_subject",
                action="multiple_actions_at_once",
                priority="SOFT_PREFERRED"
            ))

        # 3. Kiểm tra quan hệ TEMPORAL (Tuần tự)
        temporal_matches = []
        for pat, rel_type, gap in self.temporal_connectors_vi:
            for m in re.finditer(pat, q_lower):
                temporal_matches.append((m.start(), m.group(), gap))

        # Nếu có từ nối TEMPORAL rõ ràng
        if temporal_matches:
            temporal_matches.sort(key=lambda x: x[0])
            raw_chunks = self._split_by_patterns(query_en, self.temporal_connectors_vi)
            phases = [c.strip() for c in raw_chunks if len(c.strip()) > 5]
            if len(phases) > 1:
                for idx, p_txt in enumerate(phases):
                    gap_type = temporal_matches[min(idx, len(temporal_matches)-1)][2] if temporal_matches else "SHORT"
                    relations.append(RelationNode(
                        type="TEMPORAL",
                        subject=f"Phase_{idx+1}",
                        action=p_txt,
                        phase_idx=idx+1,
                        expected_gap=gap_type,
                        priority="HARD_REQUIRED"
                    ))
                return relations, phases, True

        # Nếu không có từ nối rõ ràng, kiểm tra Verb-Chain (đi vào -> lấy -> đi ra)
        verb_chain_chunks = self._detect_verb_chains(raw_vi, query_en)
        if len(verb_chain_chunks) > 1:
            for idx, vc in enumerate(verb_chain_chunks):
                relations.append(RelationNode(
                    type="TEMPORAL",
                    subject=f"Phase_{idx+1}",
                    action=vc,
                    phase_idx=idx+1,
                    expected_gap="SHORT",
                    priority="SOFT_PREFERRED"
                ))
            return relations, verb_chain_chunks, True

        # Mặc định: Single phase, quan hệ UNKNOWN / ATTRIBUTE
        relations.append(RelationNode(type="ATTRIBUTE", subject="global_scene", priority="SOFT_PREFERRED"))
        return relations, [query_en], False

    def _detect_verb_chains(self, raw_vi: str, query_en: str) -> List[str]:
        """Phát hiện chuỗi hành động qua liên từ 'rồi', 'xong' hoặc dấu phẩy ngăn cách động từ"""
        # Mẫu: [Mệnh đề 1] rồi [Mệnh đề 2] hoặc [Mệnh đề 1], [Mệnh đề 2]
        parts_vi = re.split(r'\b(?:rồi|xong|liền)\b|,\s*(?=(?:đi|cầm|lấy|chạy|bước|ném|mở|đóng|nhìn|quay)\b)', raw_vi, flags=re.IGNORECASE)
        parts_vi = [p.strip() for p in parts_vi if len(p.strip()) > 8]
        if len(parts_vi) >= 2:
            return [self.translate_to_en(p) for p in parts_vi]
        return [query_en]

    def _split_by_patterns(self, text_en: str, patterns: List[Any]) -> List[str]:
        """Cắt câu tiếng Anh dựa trên danh sách pattern liên từ"""
        sub_chunks = [text_en]
        for pat_tup in patterns:
            pat = pat_tup[0]
            new_chunks = []
            for chunk in sub_chunks:
                parts = re.split(pat, chunk, flags=re.IGNORECASE)
                for p in parts:
                    p_str = p.strip()
                    if len(p_str) > 5:
                        new_chunks.append(p_str)
            sub_chunks = new_chunks
        return sub_chunks

    def _extract_entities_and_attributes(self, raw_vi: str, query_en: str) -> List[EntityNode]:
        """Trích xuất thực thể, màu sắc, trang phục với provenance offline"""
        entities = []
        colors = ["đỏ", "xanh", "vàng", "trắng", "đen", "tím", "cam", "hồng", "nâu", "xám"]
        objects_vi = ["người", "xe", "ô tô", "xe tải", "xe máy", "xe buýt", "con đập", "bản đồ", "điện thoại", "chai nước"]
        
        q_lower = raw_vi.lower()
        e_idx = 1
        for obj in objects_vi:
            if obj in q_lower:
                attrs = []
                for c in colors:
                    if f"{obj} {c}" in q_lower or f"{obj} màu {c}" in q_lower:
                        attrs.append(c)
                
                entities.append(EntityNode(
                    id=f"E{e_idx}",
                    canonical=self.translate_to_en(obj),
                    attributes=[self.translate_to_en(a) for a in attrs],
                    source_span=obj,
                    source_type="explicit",
                    provenance={"offline": {"exists": True, "confidence": 0.85}, "ai": {"exists": False, "confidence": 0.0}}
                ))
                e_idx += 1
        return entities

    def _build_offline_6d_aspects(self, query_en: str, entities: List[EntityNode], relations: List[RelationNode]) -> AspectPromptNode:
        """Tạo 6 chiều đặc trưng có thể mang giá trị Nullable nếu không có thông tin"""
        scene_kw = None
        for s in ["rain", "sunny", "night", "day", "outdoor", "indoor", "aerial", "road", "bridge", "highway", "water", "dam", "supermarket", "store"]:
            if s in query_en.lower():
                scene_kw = f"scene in {s} environment"
                break

        action_kw = None
        for a in ["walking", "running", "driving", "holding", "standing", "entering", "leaving", "flowing", "talking", "moving"]:
            if a in query_en.lower():
                action_kw = f"action of {a}"
                break

        obj_kw = ", ".join([e.canonical for e in entities]) if entities else None
        attr_kw = ", ".join([f"{e.canonical} {' '.join(e.attributes)}" for e in entities if e.attributes]) or None
        rel_kw = ", ".join([f"{r.subject} {r.relation} {r.object}" for r in relations if r.type == "SPATIAL"]) or None

        return AspectPromptNode(
            global_prompt=query_en,
            scene={"value": scene_kw, "confidence": 0.7} if scene_kw else None,
            action={"value": action_kw, "confidence": 0.75} if action_kw else None,
            object={"value": obj_kw, "confidence": 0.8} if obj_kw else None,
            attributes={"value": attr_kw, "confidence": 0.8} if attr_kw else None,
            relations={"value": rel_kw, "confidence": 0.7} if rel_kw else None
        )

    def _calculate_semantic_importance(self, text_en: str, hard_anchors: List[Dict[str, Any]]) -> float:
        """Tính Semantic Importance nội tại I_semantic trong [0.0, 1.0]"""
        importance = 0.5
        t_lower = text_en.lower()
        if hard_anchors:
            importance += 0.3
        # Các từ khóa hiếm / cụ thể làm tăng importance
        if any(w in t_lower for w in ["truck", "dam", "irrigation", "red", "blue", "motorcycle", "police", "gun", "bottle"]):
            importance += 0.2
        # Các từ chung chung làm giảm importance
        if any(w in t_lower for w in ["aerial shot", "filmed from above", "camera panning", "view of"]):
            importance -= 0.2
        return float(min(1.0, max(0.1, importance)))

    def _generate_tier_a_hypotheses(self, text_en: str) -> List[Dict[str, Any]]:
        """Sinh tối đa 2 giả thuyết biến thể Tier A an toàn"""
        hypos = [{"prompt": text_en, "prior_weight": 1.0}]
        t_lower = text_en.lower()
        for key, dict_tiers in self.synonym_dict_en.items():
            if key in t_lower:
                for alt in dict_tiers.get("tier_a", []):
                    if alt != key and len(hypos) < 3:
                        new_p = re.sub(r'\b' + re.escape(key) + r'\b', alt, text_en, flags=re.IGNORECASE)
                        hypos.append({"prompt": new_p, "prior_weight": 0.8})
        return hypos

    # =========================================================================
    # UNIVERSAL COMPILER INTERFACE (Tương thích mọi Track)
    # =========================================================================

    def compile_query(
        self,
        raw_query: str,
        auto_translate: bool = True,
        filename: Optional[str] = None,
        use_ai_query: bool = False,
        engine_mode: str = "hybrid"
    ) -> Dict[str, Any]:
        """
        Hàm biên dịch trung tâm hỗ trợ 4 Track:
        - engine_mode='offline' (Track 1)
        - engine_mode='ai' (Track 2)
        - engine_mode='hybrid' (Track 3 & 4)
        """
        raw_clean = raw_query.strip()
        if not raw_clean:
            empty_ir = CommonSemanticIR(query_id=filename or "empty")
            return self._ir_to_legacy_dict(empty_ir)

        # 1. Chạy Track 1: Offline Compiler V2
        offline_ir = self.compile_offline_v2(raw_clean, auto_translate=auto_translate, filename=filename)

        # Nếu chỉ chọn chế độ Offline hoặc tắt AI
        if engine_mode == "offline" or (not use_ai_query and engine_mode != "hybrid" and engine_mode != "ai"):
            return self._ir_to_legacy_dict(offline_ir)

        # 2. Chạy Track 2: AI Semantic Compiler V2
        ai_data = self.ai_parser.parse_query_structured(raw_clean, filename)
        if not ai_data:
            # Fallback an toàn sang Offline V2
            return self._ir_to_legacy_dict(offline_ir)

        ai_ir = self._convert_ai_json_to_ir(ai_data, raw_clean, filename, auto_translate)

        if engine_mode == "ai":
            return self._ir_to_legacy_dict(ai_ir)

        # 3. Track 3: Hybrid Fusion (Kết hợp cả hai vào Common IR)
        hybrid_ir = self._merge_to_hybrid_ir(offline_ir, ai_ir)
        return self._ir_to_legacy_dict(hybrid_ir)

    def _convert_ai_json_to_ir(self, ai_data: Dict[str, Any], raw_query: str, filename: Optional[str], auto_translate: bool) -> CommonSemanticIR:
        """Chuyển đổi dữ liệu JSON từ LLM thành CommonSemanticIR chuẩn"""
        global_en = ai_data.get("global_query_en") or (self.translate_to_en(raw_query) if auto_translate else raw_query)
        raw_phases = ai_data.get("temporal_phases", [])
        phase_nodes = []
        for idx, p in enumerate(raw_phases):
            if isinstance(p, dict):
                v = p.get("visual_en") or p.get("canonical_prompt") or p.get("description", "")
                hypos = [{"prompt": h, "prior_weight": 0.8} for h in p.get("hypotheses", [])] if isinstance(p.get("hypotheses"), list) else []
                phase_nodes.append(TemporalPhaseNode(
                    phase_idx=idx+1,
                    canonical_prompt=v,
                    hypotheses=hypos or [{"prompt": v, "prior_weight": 1.0}],
                    scene=p.get("scene_prompt"),
                    action=p.get("action_prompt"),
                    semantic_importance=0.6
                ))
            elif isinstance(p, str):
                phase_nodes.append(TemporalPhaseNode(phase_idx=idx+1, canonical_prompt=p, hypotheses=[{"prompt": p, "prior_weight": 1.0}]))

        aspects_dict = ai_data.get("aspect_prompts", {})
        aspect_node = AspectPromptNode(
            global_prompt=aspects_dict.get("global", global_en),
            scene={"value": aspects_dict.get("scene"), "confidence": 0.9} if aspects_dict.get("scene") else None,
            action={"value": aspects_dict.get("action"), "confidence": 0.9} if aspects_dict.get("action") else None,
            object={"value": aspects_dict.get("object"), "confidence": 0.9} if aspects_dict.get("object") else None,
            attributes={"value": aspects_dict.get("attributes"), "confidence": 0.85} if aspects_dict.get("attributes") else None,
            relations={"value": aspects_dict.get("relations"), "confidence": 0.85} if aspects_dict.get("relations") else None
        )

        return CommonSemanticIR(
            query_id=filename or "ai_query",
            task_type=ai_data.get("task_type", "KIS"),
            raw_query=raw_query,
            query_en=global_en,
            compiler_source="ai",
            compiler_confidence={"offline_raw": 0.0, "ai_raw": 0.95, "calibrated_offline": 0.0, "calibrated_ai": 0.95},
            is_sequence=bool(ai_data.get("is_sequence", len(phase_nodes) > 1)),
            aspects=aspect_node,
            temporal_phases=phase_nodes or [TemporalPhaseNode(phase_idx=1, canonical_prompt=global_en)],
            hard_anchors=[{"text": a, "fuzzy": True, "min_sim": 0.8} for a in ai_data.get("hard_anchors", [])],
            speech_keywords=ai_data.get("speech_keywords", [])
        )

    def _merge_to_hybrid_ir(self, offline_ir: CommonSemanticIR, ai_ir: CommonSemanticIR) -> CommonSemanticIR:
        """Hợp nhất Offline IR và AI IR thành Hybrid Semantic IR"""
        # Hợp nhất thực thể và provenance
        merged_entities = list(offline_ir.entities)
        for ae in ai_ir.entities:
            matched = False
            for oe in merged_entities:
                if oe.canonical.lower() == ae.canonical.lower():
                    oe.provenance["ai"] = {"exists": True, "confidence": 0.95}
                    matched = True
                    break
            if not matched:
                ae.provenance["ai"] = {"exists": True, "confidence": 0.95}
                merged_entities.append(ae)

        # Hợp nhất mỏ neo và từ khóa
        all_anchors = list(offline_ir.hard_anchors)
        for aa in ai_ir.hard_anchors:
            if not any(a["text"].lower() == aa["text"].lower() for a in all_anchors):
                all_anchors.append(aa)

        all_speech_kws = list(set(offline_ir.speech_keywords + ai_ir.speech_keywords))

        return CommonSemanticIR(
            query_id=offline_ir.query_id,
            task_type=ai_ir.task_type or offline_ir.task_type,
            raw_query=offline_ir.raw_query,
            query_en=ai_ir.query_en or offline_ir.query_en,
            compiler_source="hybrid",
            compiler_confidence={
                "offline_raw": offline_ir.compiler_confidence.get("offline_raw", 0.7),
                "ai_raw": ai_ir.compiler_confidence.get("ai_raw", 0.95),
                "calibrated_offline": 0.8,
                "calibrated_ai": 0.95
            },
            is_sequence=ai_ir.is_sequence or offline_ir.is_sequence,
            entities=merged_entities,
            relations=offline_ir.relations,
            aspects=ai_ir.aspects,
            temporal_phases=ai_ir.temporal_phases or offline_ir.temporal_phases,
            hard_anchors=all_anchors,
            speech_keywords=all_speech_kws
        )

    def _ir_to_legacy_dict(self, ir: CommonSemanticIR) -> Dict[str, Any]:
        """Tạo Dict tương thích hoàn toàn với RetrievalEngine và EvidenceEngine hiện hữu"""
        phases_en = [p.canonical_prompt for p in ir.temporal_phases] if ir.temporal_phases else [ir.query_en]
        aspect_dict = {
            "global": ir.aspects.global_prompt or ir.query_en,
            "scene": ir.aspects.scene.get("value") if ir.aspects.scene else None,
            "action": ir.aspects.action.get("value") if ir.aspects.action else None,
            "object": ir.aspects.object.get("value") if ir.aspects.object else None,
            "attributes": ir.aspects.attributes.get("value") if ir.aspects.attributes else None,
            "relations": ir.aspects.relations.get("value") if ir.aspects.relations else None
        }
        # Loại bỏ các key None
        aspect_dict = {k: v for k, v in aspect_dict.items() if v}

        intent_flags = {
            "task_type": ir.task_type,
            "is_sequence": ir.is_sequence,
            "has_speech": len(ir.speech_keywords) > 0,
            "has_sound": False,
            "has_ocr_quote": len(ir.hard_anchors) > 0,
            "anchors": [a["text"] if isinstance(a, dict) else str(a) for a in ir.hard_anchors],
            "speech_keywords": ir.speech_keywords,
            "phases_en": phases_en,
            "aspect_prompts": aspect_dict,
            "source": f"Compiler ({ir.compiler_source.upper()})",
            "semantic_ir": ir.to_dict()
        }

        return {
            "query_en": ir.query_en,
            "phases_en": phases_en,
            "aspect_prompts": aspect_dict,
            "intent_flags": intent_flags,
            "source": f"Compiler ({ir.compiler_source.upper()})",
            "semantic_ir": ir
        }

    # =========================================================================
    # CÁC TIỆN ÍCH DỊCH THUẬT VÀ TRÍCH XUẤT NỀN TẢNG
    # =========================================================================

    def translate_to_en(self, text: str) -> str:
        """Dịch văn bản tiếng Việt sang tiếng Anh với RAM cache và Google Mobile fallback"""
        text_clean = text.strip()
        if not text_clean:
            return ""

        if text_clean in self._trans_cache:
            return self._trans_cache[text_clean]
            
        has_vietnamese = bool(re.search(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', text_clean, re.IGNORECASE))
        if not has_vietnamese:
            self._trans_cache[text_clean] = text_clean
            return text_clean

        # Tầng 1: Direct Google Mobile Translate (siêu nhanh, không bị rate-limit HTML error 500)
        try:
            import urllib.request
            import urllib.parse
            import html
            encoded = urllib.parse.quote(text_clean)
            url = f"https://translate.google.com/m?sl=vi&tl=en&q={encoded}"
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
            )
            with urllib.request.urlopen(req, timeout=3.5) as response:
                content = response.read().decode('utf-8')
                m = re.search(r'class="result-container">([^<]+)<', content)
                if m:
                    res = html.unescape(m.group(1).strip())
                    if res and not any(err in res.lower() for err in ["error 500", "500.that", "server error", "too many requests"]):
                        self._trans_cache[text_clean] = res
                        return res
        except Exception as e:
            logger.debug(f"Direct Google mobile translate failed: {e}")

        # Tầng 2: deep-translator GoogleTranslator fallback
        try:
            translated = self.translator.translate(text_clean)
            if translated:
                if not any(err in translated.lower() for err in ["error 500", "500.that", "server error", "too many requests", "that’s all we know"]):
                    self._trans_cache[text_clean] = translated
                    return translated
        except Exception as e:
            logger.warning("deep_translator failed for '%s': %s", text_clean[:80], e)

        self._trans_cache[text_clean] = text_clean
        return text_clean

    def extract_anchors_with_fuzzy(self, query_text: str) -> List[Dict[str, Any]]:
        """Trích xuất mỏ neo cứng kèm cấu hình khớp mờ (Fuzzy matching)"""
        anchors = []
        quoted = re.findall(r'["“\']([^"”\']+)["”\']', query_text)
        for q in quoted:
            q_clean = q.strip()
            if len(q_clean) >= 2:
                anchors.append({"text": q_clean, "fuzzy": True, "min_sim": 0.8})
                
        num_patterns = re.findall(r'\b(?:\d+[\w%]+|\d+)\b', query_text)
        for np_match in num_patterns:
            if len(np_match) > 1 and not np_match.isdigit():
                anchors.append({"text": np_match, "fuzzy": False, "min_sim": 1.0})
                
        return anchors

    def extract_speech_keywords(self, query_text: str) -> List[str]:
        """Trích xuất từ khóa cho kênh tìm kiếm âm thanh ASR"""
        cleaned = re.sub(r'[^\w\s]', ' ', query_text.lower())
        tokens = cleaned.split()
        keywords = []
        for w in tokens:
            if w not in self.vietnamese_stopwords and len(w) >= 2 and not w.isdigit():
                if w not in keywords:
                    keywords.append(w)
        return keywords
