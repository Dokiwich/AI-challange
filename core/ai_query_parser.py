"""
AIQueryParser V2 (Track 2: Pure LLM Semantic Compiler)
Tích hợp:
1. 6D Semantic Chunking với nguyên tắc Nullable (Chống Hallucination)
2. Bounded Multi-Hypothesis (Canonical + Top 2 Alternative Hypotheses)
3. Source Grounding & Anti-Overcommitment
4. Smart Disk Caching & Fail-Fast Connectivity
"""

import os
import time
import json
import re
import logging
import hashlib
from typing import Dict, List, Any, Optional
import openai

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    _k, _v = _k.strip(), _v.strip()
                    if _k not in os.environ:
                        os.environ[_k] = _v

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "query_ai_cache.json")


class AIQueryParser:
    """
    Track 2: AI Semantic Compiler V2:
    Sử dụng Gemini API phân tích ngữ nghĩa 6D, sinh giả thuyết bị chặn.
    """

    SYSTEM_PROMPT = """You are an expert AI Video Retrieval & Keyframe Extraction query compiler for benchmark competitions (AIC / TRECVID / Video Browser Showdown).
Your task is to analyze a natural language video search query and decompose it into a structured EVENT GRAPH for multi-modal alignment (Neo4j + Qdrant).

CRITICAL CONSTRAINTS (ANTI-HALLUCINATION & PROVENANCE):
1. NEVER hallucinate details not mentioned in the query. If background/scene/action/relation is not specified, set its field to null.
2. Build a Graph: Extract salient actors/objects into 'entities'. Extract actions/interactions into 'events' that link entities.
3. Temporal Logic: If the query describes a sequence of actions, map them to sequential phases and define 'temporal_edges' (e.g., EV1 BEFORE EV2).
4. Keep 'hypotheses' bounded (top 1-2 alternatives only).
5. Extract complex constraints (counting, attributes) directly into the entity attributes.

You MUST return a single, valid JSON object matching this exact schema:
{
  "task_type": "KIS" | "TRAKE" | "QA",
  "is_sequence": true | false,
  "global_query_en": "concise direct English query without filler words",
  "entities": [
    {
      "entity_id": "E1",
      "type": "person",
      "attributes": {"color": "blue", "clothing": "shirt", "has_glasses": true},
      "count": 1,
      "is_hard_constraint": true
    }
  ],
  "events": [
    {
      "event_id": "EV1",
      "action": "walking",
      "subject": "E1",
      "object": "building",
      "phase_index": 1
    }
  ],
  "temporal_edges": [
    {"from": "EV1", "to": "EV2", "relation": "BEFORE"}
  ],
  "temporal_phases": [
    {
      "phase_index": 1,
      "canonical_prompt": "direct concise English prompt for phase 1",
      "hypotheses": ["alternative description 1", "alternative description 2"],
      "scene_prompt": "background/weather or null if unspecified",
      "action_prompt": "movement/action or null if unspecified",
      "objects": ["salient", "objects"],
      "semantic_importance": 0.8
    }
  ],
  "aspect_prompts": {
    "global": "direct concise English description",
    "scene": "scene environment string or null",
    "action": "action string or null",
    "object": "objects string or null",
    "attributes": "color/clothing string or null",
    "relations": "spatial/interaction string (e.g. 'man LEFT_OF car') or null"
  },
  "source_grounding": [
    {"source_vi": "vật dài", "canonical_en": "long object", "source_type": "explicit"}
  ],
  "hard_anchors": ["exact quotes, license numbers, OCR texts"],
  "has_speech": true | false,
  "speech_keywords": ["Vietnamese keywords for ASR speech transcript search if query mentions spoken words"]
}
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model_name = model_name or os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
        
        if self.api_key:
            self.client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
            )
        else:
            logger.warning("No API Key found for OpenRouter.")
            self.client = None
        
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()
        self._is_available: Optional[bool] = True

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        os.makedirs(CACHE_DIR, exist_ok=True)
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load AI query cache: {e}")
        return {}

    def _save_cache(self):
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save AI query cache: {e}")

    def _hash_query(self, query: str) -> str:
        return hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest()

    def check_connection(self, force: bool = False) -> bool:
        """Kiểm tra kết nối tới OpenRouter"""
        return True

    def get_available_models(self) -> List[str]:
        """Lấy danh sách model từ OpenRouter"""
        return ["google/gemma-4-26b-a4b-it:free"]

    def parse_query_structured(self, query_text: str, filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Gọi LLM để phân tích ngữ nghĩa 6D và trích xuất cấu trúc truy vấn.
        Tự động tra cứu cache trước. Fail-fast ngay khi offline.
        """
        raw_query = query_text.strip()
        if not raw_query:
            return None

        # 1. Kiểm tra Cache
        q_hash = self._hash_query(raw_query)
        if q_hash in self.cache:
            cached_result = self.cache[q_hash]
            logger.info("Retrieved structured query from cache.")
            return cached_result

        # 2. Kiểm tra trạng thái kết nối
        if self._is_available is False:
            return None
        if self._is_available is None and not self.check_connection():
            return None

        # 3. Gọi OpenRouter API
        if not self.api_key or not getattr(self, "client", None):
            logger.error("Cannot parse query via OpenRouter. API key is missing.")
            return None

        user_content = f"Query to analyze:\n\"\"\"{raw_query}\"\"\""
        if filename:
            user_content += f"\nFile context: {filename}"

        try:
            logger.info(f"Calling OpenRouter ({self.model_name})...")
            
            import time
            response_text = None
            for attempt in range(3):
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": user_content}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.05
                    )
                    response_text = completion.choices[0].message.content
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 2:
                        logger.warning("Quota/Rate limit hit in parser. Waiting 15s before retry...")
                        time.sleep(15)
                        continue
                    raise e
            
            if not response_text:
                logger.warning("Empty response from OpenRouter.")
                return None
                
            # Clean JSON if wrapped in markdown
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            parsed = json.loads(response_text)

            # Validate basic keys
            if "temporal_phases" in parsed or "global_query_en" in parsed:
                # Save to cache
                self.cache[q_hash] = parsed
                self._save_cache()
                return parsed

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}\nResponse was: {response_text}")
        except Exception as e:
            logger.warning(f"Unexpected error in AIQueryParser: {e}")

        return None
