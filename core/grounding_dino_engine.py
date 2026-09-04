import torch
import logging
from PIL import Image
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)

class GroundingDINOEngine:
    def __init__(self, device="cuda", model_id="IDEA-Research/grounding-dino-tiny"):
        """
        Khởi tạo Grounding DINO engine (Tải sẵn để sẵn sàng dùng ngay lập tức)
        """
        self.device = device
        self.model_id = model_id
        self.processor = None
        self.model = None
        self._lazy_load()

    def _lazy_load(self):
        if self.model is None:
            logger.info(f"[GroundingDINO] Đang nạp mô hình {self.model_id} lên {self.device}...")
            try:
                from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)
                self.model.eval()
                logger.info("[GroundingDINO] Nạp mô hình thành công.")
            except ImportError as e:
                logger.error("[GroundingDINO] Thiếu thư viện transformers phiên bản mới.")
                raise e
            except Exception as e:
                logger.error(f"[GroundingDINO] Lỗi khi nạp mô hình: {e}")
                raise e

    def unload(self):
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("[GroundingDINO] Đã giải phóng khỏi VRAM.")

    def count_objects(self, image: Image.Image, text_prompt: str, box_threshold=0.3, text_threshold=0.25) -> Tuple[int, Any, Any, Any]:
        """
        Đếm số lượng vật thể trong ảnh theo text_prompt.
        Grounding DINO yêu cầu text_prompt kết thúc bằng dấu chấm (VD: "fish.").
        """
        self._lazy_load()
        
        if not text_prompt.endswith("."):
            text_prompt += "."
            
        text_prompt = text_prompt.lower()
        
        try:
            inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # target_sizes là [height, width]
            target_sizes = torch.tensor([[image.height, image.width]], device=self.device)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes
            )
            
            # Kết quả cho ảnh đầu tiên
            result = results[0]
            boxes = result.get("boxes", [])
            scores = result.get("scores", [])
            labels = result.get("labels", [])
            
            return len(boxes), boxes, scores, labels
            
        except Exception as e:
            logger.error(f"[GroundingDINO] Lỗi khi đếm vật thể: {e}")
            return 0, [], [], []
