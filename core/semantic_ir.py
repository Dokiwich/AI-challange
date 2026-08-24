"""
Common Semantic IR (Intermediate Representation)
Đặc tả cấu trúc dữ liệu trung gian chuẩn hóa dùng chung cho:
- Track 1: Offline Compiler V2
- Track 2: AI Semantic Compiler V2
- Track 3: Hybrid Fusion Engine
- Track 4: Meta-Router & Policy Engine
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class EntityNode:
    id: str
    canonical: str
    attributes: List[str] = field(default_factory=list)
    source_span: Optional[str] = None
    source_type: str = "explicit"  # "explicit" | "inferred"
    provenance: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "offline": {"exists": False, "confidence": 0.0},
        "ai": {"exists": False, "confidence": 0.0}
    })
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class RelationNode:
    type: str  # TEMPORAL | SIMULTANEOUS | ATTRIBUTE | SPATIAL | CAUSAL | CONDITIONAL | COREFERENCE | UNKNOWN
    subject: str
    action: Optional[str] = None
    relation: Optional[str] = None
    object: Optional[str] = None
    priority: str = "HARD_REQUIRED"  # HARD_REQUIRED | SOFT_PREFERRED | OPTIONAL_CONTEXT
    phase_idx: Optional[int] = None
    expected_gap: str = "UNKNOWN"  # IMMEDIATE | SHORT | LONG | UNKNOWN

@dataclass
class AspectPromptNode:
    global_prompt: str = ""
    scene: Optional[Dict[str, Any]] = None       # {"value": "...", "confidence": 0.8} or None
    action: Optional[Dict[str, Any]] = None      # {"value": "...", "confidence": 0.9} or None
    object: Optional[Dict[str, Any]] = None      # {"value": "...", "confidence": 0.85} or None
    attributes: Optional[Dict[str, Any]] = None  # {"value": "...", "confidence": 0.9} or None
    relations: Optional[Dict[str, Any]] = None   # {"value": "...", "confidence": 0.7} or None

@dataclass
class TemporalPhaseNode:
    phase_idx: int
    canonical_prompt: str
    hypotheses: List[Dict[str, Any]] = field(default_factory=list) # [{"prompt": "...", "prior_weight": 0.8}]
    scene: Optional[str] = None
    action: Optional[str] = None
    object: Optional[str] = None
    attributes: Optional[str] = None
    relations: Optional[str] = None
    semantic_importance: float = 0.5  # [0.0, 1.0]

@dataclass
class CommonSemanticIR:
    query_id: str
    task_type: str = "KIS"  # KIS | QA | TRAKE
    raw_query: str = ""
    query_en: str = ""
    compiler_source: str = "offline"  # offline | ai | hybrid
    compiler_confidence: Dict[str, float] = field(default_factory=lambda: {
        "offline_raw": 0.5,
        "ai_raw": 0.5,
        "calibrated_offline": 0.5,
        "calibrated_ai": 0.5
    })
    is_sequence: bool = False
    entities: List[EntityNode] = field(default_factory=list)
    relations: List[RelationNode] = field(default_factory=list)
    aspects: AspectPromptNode = field(default_factory=AspectPromptNode)
    temporal_phases: List[TemporalPhaseNode] = field(default_factory=list)
    hard_anchors: List[Dict[str, Any]] = field(default_factory=list) # [{"text": "BUS-01", "fuzzy": True, "min_sim": 0.8}]
    speech_keywords: List[str] = field(default_factory=list)
    qa_target: Optional[str] = None
    qa_expected_type: Optional[str] = None  # entity | number | color | yes_no

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành Dictionary chuẩn để serialize hoặc hiển thị giao diện"""
        return {
            "query_id": self.query_id,
            "task_type": self.task_type,
            "raw_query": self.raw_query,
            "query_en": self.query_en,
            "compiler_source": self.compiler_source,
            "compiler_confidence": self.compiler_confidence,
            "is_sequence": self.is_sequence,
            "entities": [
                {
                    "id": e.id,
                    "canonical": e.canonical,
                    "attributes": e.attributes,
                    "source_span": e.source_span,
                    "source_type": e.source_type,
                    "provenance": e.provenance,
                    "hypotheses": e.hypotheses
                }
                for e in self.entities
            ],
            "relations": [
                {
                    "type": r.type,
                    "subject": r.subject,
                    "action": r.action,
                    "relation": r.relation,
                    "object": r.object,
                    "priority": r.priority,
                    "phase_idx": r.phase_idx,
                    "expected_gap": r.expected_gap
                }
                for r in self.relations
            ],
            "aspects": {
                "global": self.aspects.global_prompt,
                "scene": self.aspects.scene,
                "action": self.aspects.action,
                "object": self.aspects.object,
                "attributes": self.aspects.attributes,
                "relations": self.aspects.relations
            },
            "temporal_phases": [
                {
                    "phase_idx": p.phase_idx,
                    "canonical_prompt": p.canonical_prompt,
                    "hypotheses": p.hypotheses,
                    "scene": p.scene,
                    "action": p.action,
                    "object": p.object,
                    "attributes": p.attributes,
                    "relations": p.relations,
                    "semantic_importance": p.semantic_importance
                }
                for p in self.temporal_phases
            ],
            "hard_anchors": self.hard_anchors,
            "speech_keywords": self.speech_keywords,
            "qa_target": self.qa_target,
            "qa_expected_type": self.qa_expected_type
        }
