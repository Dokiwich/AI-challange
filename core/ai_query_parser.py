"""
AIQueryParser V2 (Track 2: Pure LLM Semantic Compiler)
Tích hợp:
1. 6D Semantic Chunking với nguyên tắc Nullable (Chống Hallucination)
2. Bounded Multi-Hypothesis (Canonical + Top 2 Alternative Hypotheses)
3. Source Grounding & Anti-Overcommitment
4. Smart Disk Caching & Fail-Fast Connectivity
"""

import os
import json
import re
import logging
import hashlib
from typing import Dict, List, Any, Optional
import requests

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
    Sử dụng LLM (9Router Proxy / OpenAI compatible) phân tích ngữ nghĩa 6D, sinh giả thuyết bị chặn.
    """

    SYSTEM_PROMPT = """You are an expert AI Video Retrieval & Keyframe Extraction query compiler for benchmark competitions (AIC / TRECVID / Video Browser Showdown).
Your task is to analyze a natural language video search query and decompose it into structured semantic components for CLIP embedding and Temporal Alignment.

CRITICAL CONSTRAINTS (ANTI-HALLUCINATION & PROVENANCE):
1. NEVER hallucinate details not mentioned in the query. If background/scene/action/relation is not specified, set its field to null. (e.g. "a red truck" -> scene: null, action: null, relations: null).
2. For ambiguous objects (e.g. "vật dài" / "long object"), DO NOT arbitrarily commit to a single object like "baseball bat". Instead, provide a canonical generic representation and at most 2 alternative hypotheses.
3. Keep 'hypotheses' bounded (top 1-2 alternatives only, DO NOT generate Cartesian product explosions).
4. Extract complex constraints (counting, attributes, relations) strictly into the 'constraints' array, DO NOT dilute them into the canonical prompt.
5. Strip all introductory conversational meta-text ("Đoạn phim bắt đầu bằng", "The video shows", "We see").

You MUST return a single, valid JSON object matching this exact schema:
{
  "task_type": "KIS" | "TRAKE" | "QA",
  "is_sequence": true | false,
  "global_query_en": "concise direct English query without filler words",
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
  "constraints": [
    {
      "constraint_type": "COUNT", 
      "target_entity": "e.g., people",
      "condition": "e.g., == 3",
      "is_hard": true
    }
  ],
  "source_grounding": [
    {"source_vi": "vật dài", "canonical_en": "long object", "source_type": "explicit"}
  ],
  "hard_anchors": ["exact quotes, license numbers, OCR texts"],
  "has_speech": true | false,
  "speech_keywords": ["Vietnamese keywords for ASR speech transcript search if query mentions spoken words"],
  "is_qa": true | false
}
"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:20128/v1").rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY", os.getenv("NINEROUTER_API_KEY", "default"))
        self.model_name = model_name or os.getenv("LLM_MODEL", "gh/gpt-4o-mini")
        
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()
        self._is_available: Optional[bool] = None

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
        """Kiểm tra kết nối tới 9Router/LLM endpoint"""
        import time
        now = time.time()
        if not force and hasattr(self, "_last_conn_check") and (now - self._last_conn_check < 30.0) and (self._is_available is not None):
            return self._is_available

        self._last_conn_check = now
        try:
            url = f"{self.base_url}/models"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.get(url, headers=headers, timeout=0.8)
            self._is_available = resp.status_code == 200
            return self._is_available
        except Exception:
            self._is_available = False
            return False

    def get_available_models(self) -> List[str]:
        """Lấy danh sách model từ 9Router"""
        try:
            url = f"{self.base_url}/models"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.get(url, headers=headers, timeout=0.8)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                models = [m["id"] for m in data if "id" in m]
                return models
        except Exception:
            pass
        return ["vip", "gh/gpt-4o-mini", "kr/auto"]

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

        # 3. Gọi LLM API với temperature=0.05
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        user_content = f"Query to analyze:\n\"\"\"{raw_query}\"\"\""
        if filename:
            user_content += f"\nFile context: {filename}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.05,
            "max_tokens": 1500
        }

        try:
            logger.info(f"Calling LLM ({self.model_name}) at {self.base_url}...")
            resp = requests.post(url, headers=headers, json=payload, timeout=8.0)
            if resp.status_code != 200:
                logger.warning(f"LLM API error ({resp.status_code}): {resp.text[:200]}")
                return None

            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            # Clean JSON markdown
            clean_json_str = re.sub(r"^```json\s*", "", content, flags=re.IGNORECASE)
            clean_json_str = re.sub(r"^```\s*", "", clean_json_str)
            clean_json_str = re.sub(r"\s*```$", "", clean_json_str).strip()

            parsed = json.loads(clean_json_str)

            # Validate basic keys
            if "temporal_phases" in parsed or "global_query_en" in parsed:
                # Save to cache
                self.cache[q_hash] = parsed
                self._save_cache()
                return parsed

        except requests.RequestException as e:
            self._is_available = False
            logger.debug(f"Network error calling LLM API: {e}")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error in AIQueryParser: {e}")

        return None
