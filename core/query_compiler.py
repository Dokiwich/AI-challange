"""
QueryCompiler V3.3 (Precision Natural Semantic Compiler)
Tích hợp:
1. Full Holistic Translation làm trục ngữ nghĩa cốt lõi (Bảo toàn 100% chi tiết bối cảnh)
2. Safe Sequence Segmentation: Chỉ tách pha khi có đánh số hoặc phân đoạn tường minh (Bước 1, 2 / Phân cảnh 1, 2 / Dòng riêng)
3. Multi-Aspect Enrichments (Action, Object, Scene) bổ trợ ngữ nghĩa
4. Bảo toàn nguyên vẹn ngữ pháp tiếng Anh tự nhiên cho CLIP
"""

import re
import logging
from typing import Dict, List, Any, Tuple, Optional
from deep_translator import GoogleTranslator

from core.ai_query_parser import AIQueryParser
from core.semantic_ir import (
    CommonSemanticIR, CoreEventNode, SupportFactNode,
    AspectPromptNode, TemporalPhaseNode, ConstraintNode,
    EntityNode, EventEdge, TemporalEdge
)

logger = logging.getLogger(__name__)

class QueryCompiler:
    """
    QueryCompiler V3.3:
    Biên dịch câu truy vấn tự nhiên bảo toàn trọn vẹn ngữ cảnh tổng thể và bóc tách các khía cạnh bổ trợ.
    """
    def __init__(self, translator: Optional[GoogleTranslator] = None, ai_parser: Optional[AIQueryParser] = None):
        self.translator = translator if translator else GoogleTranslator(source='auto', target='en')
        self.ai_parser = ai_parser if ai_parser else AIQueryParser()
        self._trans_cache: Dict[str, str] = {}

    def compile_offline_v2(
        self,
        raw_query: str,
        auto_translate: bool = True,
        filename: Optional[str] = None
    ) -> CommonSemanticIR:
        """
        Track 1: Biên dịch Offline bảo toàn ngữ nghĩa toàn vẹn (Holistic Semantic Parsing).
        """
        raw_clean = raw_query.strip()
        query_id = filename or "offline_query"
        if not raw_clean:
            return CommonSemanticIR(query_id=query_id, compiler_source="offline")

        # 1. Bóc tách Mỏ neo cứng (Hard Anchors) & Từ khóa ASR
        hard_anchors = self.extract_anchors_with_fuzzy(raw_clean)
        speech_kws = self.extract_speech_keywords(raw_clean)

        # 2. Dịch toàn bộ câu truy vấn tự nhiên (Bảo toàn 100% ngữ cảnh tổng thể)
        full_en = self.translate_to_en(raw_clean) if auto_translate else raw_clean

        # 3. Phân tích chuỗi thời gian an toàn (Safe Sequence Segmentation)
        phases_vi, is_sequence = self._safe_segment_phases_vi(raw_clean)

        phases_en = []
        if is_sequence and len(phases_vi) > 1:
            for p_vi in phases_vi:
                p_en = self.translate_to_en(p_vi) if auto_translate else p_vi
                if p_en and len(p_en.strip()) > 3:
                    phases_en.append(p_en.strip())
        
        if not phases_en:
            phases_en = [full_en]
            is_sequence = False

        # 4. Trích xuất Core Events & Support Facts
        core_events, support_facts = self._extract_core_vs_support(raw_clean, phases_en, full_en)

        # 5. Xây dựng các Phase Nodes
        temporal_phase_nodes = []
        for p_idx, p_text in enumerate(phases_en):
            temporal_phase_nodes.append(TemporalPhaseNode(
                phase_idx=p_idx + 1,
                canonical_prompt=p_text,
                hypotheses=[{"prompt": p_text, "prior_weight": 1.0}],
                action=p_text,
                semantic_importance=0.8,
                is_core_event=True
            ))

        # 6. Xây dựng Multi-Aspect Prompts
        aspect_node = self._build_aspect_prompts(full_en, core_events, support_facts)

        event_density = float(len(core_events) / max(1.0, float(len(phases_en))))
        task_type = "TRAKE" if (filename and "trake" in filename.lower()) else ("QA" if ("qa" in filename.lower() if filename else False) else "KIS")

        return CommonSemanticIR(
            query_id=query_id,
            task_type=task_type,
            raw_query=raw_clean,
            query_en=full_en,
            compiler_source="offline",
            compiler_confidence={
                "offline_raw": 0.95,
                "ai_raw": 0.0,
                "calibrated_offline": 0.92,
                "calibrated_ai": 0.0
            },
            is_sequence=is_sequence and len(phases_en) > 1,
            event_density=event_density,
            core_events=core_events,
            support_facts=support_facts,
            aspects=aspect_node,
            temporal_phases=temporal_phase_nodes,
            hard_anchors=hard_anchors,
            speech_keywords=speech_kws
        )

    def _safe_segment_phases_vi(self, text_vi: str) -> Tuple[List[str], bool]:
        """
        Chỉ phân đoạn khi có đánh dấu chuỗi thời gian tường minh (Explicit Steps/Phases).
        Tránh cắt gãy câu văn đơn lẻ.
        """
        lines = [l.strip() for l in text_vi.split('\n') if l.strip()]
        
        # 1. Nếu có đánh dấu danh sách gạch đầu dòng hoặc E1, E2, Step 1, 2
        if len(lines) > 1 and all(re.match(r'^(?:[-•*]|E\d+|bước\s*\d+|phân cảnh\s*\d+|\d+[.)])', l, re.IGNORECASE) for l in lines[1:]):
            cleaned_lines = [re.sub(r'^(?:[-•*]|E\d+[:.]?|bước\s*\d+[:.]?|phân cảnh\s*\d+[:.]?|\d+[.)])\s*', '', l, flags=re.IGNORECASE) for l in lines]
            return [l for l in cleaned_lines if len(l) > 5], True

        # 2. Tìm các điểm phân tách tường minh trong đoạn văn: "Phân cảnh bắt đầu...", "Phân cảnh tiếp theo...", "Bước đầu tiên...", "Sau đó..."
        explicit_split_pat = r'(?=(?:phân cảnh (?:bắt đầu|tiếp theo|sau|cuối)|bước (?:đầu tiên|thứ nhất|thứ hai|tiếp theo|1|2|3|4|sau|cuối)|sau đó,\s*|tiếp theo,\s*))'
        splits = re.split(explicit_split_pat, text_vi, flags=re.IGNORECASE)
        splits_clean = [s.strip() for s in splits if len(s.strip()) > 15]

        if len(splits_clean) >= 2:
            return splits_clean, True

        return [text_vi], False

    def _extract_core_vs_support(
        self,
        raw_vi: str,
        phases_en: List[str],
        global_en: str
    ) -> Tuple[List[CoreEventNode], List[SupportFactNode]]:
        """Trích xuất danh sách Core Events và Support Facts"""
        core_events = []
        support_facts = []

        for p_idx, p_text in enumerate(phases_en):
            core_events.append(CoreEventNode(
                event_id=f"EVT_{p_idx+1}",
                action_verb="action",
                canonical_prompt_en=p_text,
                importance_weight=0.85,
                is_core=True,
                phase_order=p_idx + 1
            ))

        return core_events, support_facts

    def _build_aspect_prompts(
        self,
        global_en: str,
        core_events: List[CoreEventNode],
        support_facts: List[SupportFactNode]
    ) -> AspectPromptNode:
        """Tạo các aspect prompts với global_prompt là câu hoàn chỉnh"""
        return AspectPromptNode(
            global_prompt=global_en.strip()
        )

    # =========================================================================
    # UNIVERSAL COMPILER INTERFACE (Tương thích 4 Track)
    # =========================================================================

    def compile_query(
        self,
        raw_query: str,
        auto_translate: bool = True,
        filename: Optional[str] = None,
        use_ai_query: bool = False,
        engine_mode: str = "hybrid"
    ) -> Dict[str, Any]:
        """Biên dịch truy vấn đa nhánh V3.3"""
        raw_clean = raw_query.strip()
        if not raw_clean:
            empty_ir = CommonSemanticIR(query_id=filename or "empty")
            return self._ir_to_legacy_dict(empty_ir)

        # 1. Chạy Track 1: Offline Holistic Compiler
        offline_ir = self.compile_offline_v2(raw_clean, auto_translate=auto_translate, filename=filename)

        if engine_mode == "offline" or not use_ai_query:
            return self._ir_to_legacy_dict(offline_ir)

        # 2. Chạy Track 2: AI Semantic Compiler
        ai_data = self.ai_parser.parse_query_structured(raw_clean, filename)
        if not ai_data:
            return self._ir_to_legacy_dict(offline_ir)

        ai_ir = self._convert_ai_json_to_ir(ai_data, raw_clean, filename, auto_translate)

        if engine_mode == "ai":
            return self._ir_to_legacy_dict(ai_ir)

        # 3. Track 3: Hybrid Fusion
        hybrid_ir = self._merge_to_hybrid_ir(offline_ir, ai_ir)
        return self._ir_to_legacy_dict(hybrid_ir)

    def _convert_ai_json_to_ir(self, ai_data: Dict[str, Any], raw_query: str, filename: Optional[str], auto_translate: bool) -> CommonSemanticIR:
        """Chuyển đổi dữ liệu JSON từ LLM thành IR chuẩn"""
        global_en = ai_data.get("global_query_en") or (self.translate_to_en(raw_query) if auto_translate else raw_query)

        raw_phases = ai_data.get("temporal_phases", [])
        phase_nodes = []
        core_events = []

        for idx, p in enumerate(raw_phases):
            if isinstance(p, dict):
                v = p.get("canonical_prompt") or p.get("visual_en") or p.get("description", "")
                imp = float(p.get("semantic_importance", 0.8))
                if v:
                    phase_nodes.append(TemporalPhaseNode(
                        phase_idx=idx+1,
                        canonical_prompt=v,
                        hypotheses=[{"prompt": v, "prior_weight": 1.0}],
                        action=p.get("action_prompt"),
                        semantic_importance=imp,
                        is_core_event=True
                    ))
                    core_events.append(CoreEventNode(
                        event_id=f"EVT_{idx+1}",
                        action_verb=p.get("action_prompt") or "action",
                        canonical_prompt_en=v,
                        importance_weight=imp,
                        is_core=True,
                        phase_order=idx+1
                    ))
            elif isinstance(p, str) and p.strip():
                phase_nodes.append(TemporalPhaseNode(phase_idx=idx+1, canonical_prompt=p.strip(), hypotheses=[{"prompt": p.strip(), "prior_weight": 1.0}]))

        aspects_dict = ai_data.get("aspect_prompts", {})
        aspect_node = AspectPromptNode(
            global_prompt=global_en,
            action_prompt=aspects_dict.get("action"),
            object_prompt=aspects_dict.get("object"),
            scene_prompt=aspects_dict.get("scene")
        )

        constraints = []
        for c in ai_data.get("constraints", []):
            if isinstance(c, dict):
                constraints.append(ConstraintNode(
                    constraint_type=c.get("constraint_type", "ATTRIBUTE"),
                    target_entity=c.get("target_entity", ""),
                    condition=c.get("condition", ""),
                    is_hard=c.get("is_hard", True)
                ))

        entities = []
        for e in ai_data.get("entities", []):
            if isinstance(e, dict):
                entities.append(EntityNode(
                    entity_id=e.get("entity_id", ""),
                    entity_type=e.get("type", "unknown"),
                    attributes=e.get("attributes", {}),
                    count=e.get("count", 1),
                    is_hard_constraint=e.get("is_hard_constraint", True)
                ))

        event_edges = []
        for ev in ai_data.get("events", []):
            if isinstance(ev, dict):
                event_edges.append(EventEdge(
                    event_id=ev.get("event_id", ""),
                    action=ev.get("action", ""),
                    subject_id=ev.get("subject", ""),
                    object_id=ev.get("object"),
                    phase_index=ev.get("phase_index", 1)
                ))

        temporal_edges = []
        for te in ai_data.get("temporal_edges", []):
            if isinstance(te, dict):
                temporal_edges.append(TemporalEdge(
                    from_event=te.get("from", ""),
                    to_event=te.get("to", ""),
                    relation=te.get("relation", "BEFORE")
                ))

        return CommonSemanticIR(
            query_id=filename or "ai_query",
            task_type=ai_data.get("task_type", "KIS"),
            raw_query=raw_query,
            query_en=global_en,
            compiler_source="ai",
            compiler_confidence={"offline_raw": 0.0, "ai_raw": 0.95, "calibrated_offline": 0.0, "calibrated_ai": 0.95},
            is_sequence=bool(ai_data.get("is_sequence", len(phase_nodes) > 1)),
            event_density=float(len(core_events) / max(1.0, float(len(phase_nodes)))),
            entities=entities,
            event_edges=event_edges,
            temporal_edges=temporal_edges,
            core_events=core_events,
            support_facts=[],
            aspects=aspect_node,
            temporal_phases=phase_nodes or [TemporalPhaseNode(phase_idx=1, canonical_prompt=global_en)],
            hard_anchors=[{"text": a, "fuzzy": True, "min_sim": 0.8} for a in ai_data.get("hard_anchors", [])],
            speech_keywords=ai_data.get("speech_keywords", []),
            constraints=constraints
        )

    def _merge_to_hybrid_ir(self, offline_ir: CommonSemanticIR, ai_ir: CommonSemanticIR) -> CommonSemanticIR:
        """Hợp nhất Event Graph giữa Offline và AI"""
        merged_phases = ai_ir.temporal_phases if ai_ir.temporal_phases else offline_ir.temporal_phases
        merged_core_events = ai_ir.core_events if ai_ir.core_events else offline_ir.core_events
        
        all_anchors = list(offline_ir.hard_anchors)
        for aa in ai_ir.hard_anchors:
            if not any(a["text"].lower() == aa["text"].lower() for a in all_anchors):
                all_anchors.append(aa)

        # 4. Speech Keywords (Trust AI over Offline)
        # Nếu AI track hoạt động, tin tưởng tuyệt đối vào khả năng trích xuất của LLM
        # Bỏ qua offline keywords để tránh bị nhiễu bởi stopword tĩnh
        all_speech_kws = ai_ir.speech_keywords

        return CommonSemanticIR(
            query_id=offline_ir.query_id,
            task_type=ai_ir.task_type or offline_ir.task_type,
            raw_query=offline_ir.raw_query,
            query_en=ai_ir.query_en or offline_ir.query_en,
            compiler_source="hybrid",
            compiler_confidence={
                "offline_raw": 0.92,
                "ai_raw": 0.95,
                "calibrated_offline": 0.92,
                "calibrated_ai": 0.95
            },
            is_sequence=ai_ir.is_sequence or offline_ir.is_sequence,
            event_density=ai_ir.event_density or offline_ir.event_density,
            entities=ai_ir.entities if ai_ir.entities else offline_ir.entities,
            event_edges=ai_ir.event_edges if ai_ir.event_edges else offline_ir.event_edges,
            temporal_edges=ai_ir.temporal_edges if ai_ir.temporal_edges else offline_ir.temporal_edges,
            core_events=merged_core_events,
            support_facts=[],
            aspects=ai_ir.aspects if ai_ir.aspects.global_prompt else offline_ir.aspects,
            temporal_phases=merged_phases,
            hard_anchors=all_anchors,
            speech_keywords=all_speech_kws,
            constraints=ai_ir.constraints if ai_ir.constraints else offline_ir.constraints
        )

    def _ir_to_legacy_dict(self, ir: CommonSemanticIR) -> Dict[str, Any]:
        """Tạo Dict tương thích hoàn toàn cho RetrievalEngine"""
        phases_en = [p.canonical_prompt for p in ir.temporal_phases] if ir.temporal_phases else [ir.query_en]
        aspect_dict = {
            "global": ir.aspects.global_prompt or ir.query_en,
            "action": ir.aspects.action_prompt,
            "object": ir.aspects.object_prompt,
            "scene": ir.aspects.scene_prompt
        }
        aspect_dict = {k: v for k, v in aspect_dict.items() if v}

        intent_flags = {
            "task_type": ir.task_type,
            "is_sequence": ir.is_sequence,
            "event_density": ir.event_density,
            "has_speech": len(ir.speech_keywords) > 0,
            "has_sound": False,
            "has_ocr_quote": len(ir.hard_anchors) > 0,
            "anchors": [a["text"] if isinstance(a, dict) else str(a) for a in ir.hard_anchors],
            "speech_keywords": ir.speech_keywords,
            "phases_en": phases_en,
            "aspect_prompts": aspect_dict,
            "entities": [
                {
                    "entity_id": e.entity_id,
                    "type": e.entity_type,
                    "attributes": e.attributes,
                    "count": e.count
                } for e in ir.entities
            ],
            "events": [
                {
                    "event_id": ev.event_id,
                    "action": ev.action,
                    "subject": ev.subject_id,
                    "object": ev.object_id,
                    "phase_index": ev.phase_index
                } for ev in ir.event_edges
            ],
            "temporal_edges": [
                {
                    "from": te.from_event,
                    "to": te.to_event,
                    "relation": te.relation
                } for te in ir.temporal_edges
            ],
            "constraints": [
                {
                    "type": c.constraint_type,
                    "target": c.target_entity,
                    "condition": c.condition,
                    "is_hard": c.is_hard
                } for c in ir.constraints
            ],
            "core_events": [e.canonical_prompt_en for e in ir.core_events],
            "support_facts": [f.description_en for f in ir.support_facts],
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
    # DỊCH THUẬT VÀ TRÍCH XUẤT CƠ SỞ
    # =========================================================================

    def translate_to_en(self, text: str) -> str:
        """Dịch văn bản tiếng Việt sang tiếng Anh với RAM cache và Google Mobile API"""
        text_clean = text.strip()
        if not text_clean:
            return ""

        if text_clean in self._trans_cache:
            return self._trans_cache[text_clean]
            
        has_vietnamese = bool(re.search(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', text_clean, re.IGNORECASE))
        if not has_vietnamese:
            self._trans_cache[text_clean] = text_clean
            return text_clean

        # Direct Google Mobile Translate
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
            with urllib.request.urlopen(req, timeout=4.0) as response:
                content = response.read().decode('utf-8')
                m = re.search(r'class="result-container">([^<]+)<', content)
                if m:
                    res = html.unescape(m.group(1).strip())
                    if res and not any(err in res.lower() for err in ["error 500", "500.that", "server error", "too many requests"]):
                        self._trans_cache[text_clean] = res
                        return res
        except Exception:
            pass

        # deep-translator fallback
        try:
            translated = self.translator.translate(text_clean)
            if translated:
                self._trans_cache[text_clean] = translated
                return translated
        except Exception:
            pass

        self._trans_cache[text_clean] = text_clean
        return text_clean

    def extract_anchors_with_fuzzy(self, query_text: str) -> List[Dict[str, Any]]:
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

    def extract_speech_keywords(self, query_text: str, intent_flags: dict = None) -> List[str]:
        # Hard block: AI parser đã xác nhận query thuần thị giác, không có lời thoại
        if intent_flags is not None and intent_flags.get("has_speech") is False:
            logger.info("[ASR] Blocked: has_speech=false from AI parser, returning empty keywords")
            return []

        cleaned = re.sub(r'[^\w\s]', ' ', query_text.lower())
        tokens = cleaned.split()
        base_stopwords = {
            # Tiếng Việt
            'là', 'của', 'và', 'có', 'với', 'trong', 'một', 'các', 'này', 'đã', 'được', 'cho', 'từ', 'đến', 'về', 'theo', 'để', 'bằng', 'không', 'những', 'hay', 'hoặc', 'nhưng', 'vì', 'do', 'nếu', 'thì', 'sẽ', 'đang', 'rất', 'cũng', 'đây', 'đó', 'ở', 'ra', 'lên', 'vào', 'khi', 'sau', 'trước', 'trên', 'dưới', 'ngoài', 'hình', 'ảnh', 'đoạn', 'clip', 'video',
            # Tiếng Anh
            'is', 'are', 'was', 'were', 'of', 'and', 'with', 'in', 'a', 'an', 'the', 'these', 'those', 'this', 'that', 'has', 'have', 'had', 'to', 'from', 'about', 'by', 'for', 'not', 'or', 'but', 'because', 'if', 'then', 'will', 'very', 'also', 'here', 'there', 'at', 'out', 'up', 'into', 'when', 'after', 'before', 'on', 'under', 'outside', 'image', 'picture', 'clip', 'video', 'scene', 'shows', 'showing'
        }
        # Từ mô tả thị giác: ngoại hình, tư thế, màu sắc, bộ phận cơ thể
        # Transcript gần như không bao giờ chứa — chỉ gây nhiễu BM25
        visual_stopwords = {
            # Tiếng Việt
            'cảnh', 'quay', 'mặc', 'đeo', 'đội', 'mũ', 'nón', 'kính',
            'áo', 'quần', 'váy', 'giày', 'dép', 'màu', 'đỏ', 'xanh',
            'vàng', 'trắng', 'đen', 'hồng', 'tím', 'nâu', 'cam',
            'người', 'nhóm', 'phụ', 'nữ', 'nam', 'trẻ', 'em', 'bé',
            'tay', 'chân', 'đầu', 'mặt', 'mắt', 'tóc', 'vai',
            'ngồi', 'đứng', 'nằm', 'cầm', 'giữ', 'bước', 'chạy',
            'xếp', 'hàng', 'thành', 'phía', 'bên', 'cạnh', 'giữa',
            'chiếc', 'cái', 'con', 'bộ', 'đôi', 'tấm',
            'thực', 'hiện', 'động', 'tác', 'chạm', 'mũi',
            # Tiếng Anh
            'wearing', 'wears', 'wore', 'hat', 'hats', 'cap', 'glasses', 'sunglasses',
            'shirt', 'pants', 'skirt', 'shoes', 'color', 'red', 'blue', 'green',
            'yellow', 'white', 'black', 'pink', 'purple', 'brown', 'orange',
            'person', 'people', 'group', 'woman', 'women', 'man', 'men', 'child', 'children', 'boy', 'girl',
            'hand', 'hands', 'foot', 'feet', 'head', 'face', 'eye', 'eyes', 'hair', 'shoulder', 'shoulders', 'toe', 'toes',
            'sit', 'sitting', 'stand', 'standing', 'lie', 'lying', 'hold', 'holding', 'walk', 'walking', 'run', 'running',
            'row', 'line', 'side', 'next', 'middle', 'front', 'back',
            'doing', 'movement', 'action', 'touch', 'touching', 'exercise', 'exercising', 'more', 'than', 'only', 'one', 'two', 'three', 'four', 'five', 'both', 'their'
        }
        stopwords = base_stopwords | visual_stopwords
        return [w for w in tokens if w not in stopwords and len(w) >= 2 and not w.isdigit()]
