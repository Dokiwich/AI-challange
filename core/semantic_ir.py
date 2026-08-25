"""
Common Semantic IR V3.2 (Event-Centric Intermediate Representation)
Đặc tả cấu trúc dữ liệu trung gian chuẩn hóa cho V3.2:
- Core Events (Hành động sống còn: w_core = 0.8..0.9)
- Support Facts (Đồ vật nền phụ trợ: w_supp = 0.1..0.2)
- Event Density (D_event) & Event Importance (I_event)
- Multi-View Prompt Budget (< 60 tokens)
- Provenance-Aware Node Tracking
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

@dataclass
class CoreEventNode:
    event_id: str
    action_verb: str
    subject: str = "agent"
    target_object: str = ""
    recipient_location: str = ""
    canonical_prompt_en: str = ""
    importance_weight: float = 0.8  # [0.5, 1.0]
    is_core: bool = True
    phase_order: int = 1
    expected_duration_sec: float = 5.0

@dataclass
class ConstraintNode:
    constraint_type: str  # COUNT | ATTRIBUTE | RELATION | SPATIAL
    target_entity: str
    condition: str
    is_hard: bool = True

@dataclass
class SupportFactNode:
    fact_id: str
    fact_type: str = "OBJECT"  # OBJECT | BACKGROUND | ATTRIBUTE | SPATIAL
    description_en: str = ""
    weight: float = 0.2  # [0.05, 0.3]
    source_span: Optional[str] = None

@dataclass
class AspectPromptNode:
    global_prompt: str = ""
    action_prompt: Optional[str] = None
    object_prompt: Optional[str] = None
    scene_prompt: Optional[str] = None

@dataclass
class TemporalPhaseNode:
    phase_idx: int
    canonical_prompt: str
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    action: Optional[str] = None
    object: Optional[str] = None
    scene: Optional[str] = None
    semantic_importance: float = 0.5  # [0.1, 1.0]
    is_core_event: bool = True

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
    event_density: float = 0.0  # D_event
    core_events: List[CoreEventNode] = field(default_factory=list)
    support_facts: List[SupportFactNode] = field(default_factory=list)
    aspects: AspectPromptNode = field(default_factory=AspectPromptNode)
    temporal_phases: List[TemporalPhaseNode] = field(default_factory=list)
    hard_anchors: List[Dict[str, Any]] = field(default_factory=list)
    speech_keywords: List[str] = field(default_factory=list)
    constraints: List[ConstraintNode] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize thành Dictionary chuẩn"""
        return {
            "query_id": self.query_id,
            "task_type": self.task_type,
            "raw_query": self.raw_query,
            "query_en": self.query_en,
            "compiler_source": self.compiler_source,
            "compiler_confidence": self.compiler_confidence,
            "is_sequence": self.is_sequence,
            "event_density": self.event_density,
            "core_events": [
                {
                    "event_id": e.event_id,
                    "action_verb": e.action_verb,
                    "target_object": e.target_object,
                    "recipient_location": e.recipient_location,
                    "canonical_prompt_en": e.canonical_prompt_en,
                    "importance_weight": e.importance_weight,
                    "is_core": e.is_core,
                    "phase_order": e.phase_order
                }
                for e in self.core_events
            ],
            "support_facts": [
                {
                    "fact_id": f.fact_id,
                    "fact_type": f.fact_type,
                    "description_en": f.description_en,
                    "weight": f.weight
                }
                for f in self.support_facts
            ],
            "temporal_phases": [
                {
                    "phase_idx": p.phase_idx,
                    "canonical_prompt": p.canonical_prompt,
                    "action": p.action,
                    "object": p.object,
                    "semantic_importance": p.semantic_importance,
                    "is_core_event": p.is_core_event
                }
                for p in self.temporal_phases
            ],
            "hard_anchors": self.hard_anchors,
            "speech_keywords": self.speech_keywords,
            "constraints": [
                {
                    "constraint_type": c.constraint_type,
                    "target_entity": c.target_entity,
                    "condition": c.condition,
                    "is_hard": c.is_hard
                }
                for c in self.constraints
            ]
        }
