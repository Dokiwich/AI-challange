import os
import glob
import json
import re
import numpy as np
import torch
import clip
from PIL import Image
from deep_translator import GoogleTranslator

class RetrievalEngine:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RetrievalEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        features_path: str = "data/features/clip_features.npy",
        map_path: str = "data/mapping/map_keyframes.json",
        model_name: str = "ViT-B/32",
        device: str = None
    ):
        if getattr(self, "_initialized", False):
            return

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[RetrievalEngine] Khởi tạo trên thiết bị: {self.device}")

        # 1. Nạp Model CLIP
        print(f"[RetrievalEngine] Đang nạp mô hình CLIP {model_name}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        # 2. Nạp Ma trận đặc trưng
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Không tìm thấy file: {features_path}")
        
        print(f"[RetrievalEngine] Đang nạp ma trận đặc trưng từ '{features_path}'...")
        raw_features = np.load(features_path)
        self.image_features = torch.from_numpy(raw_features).to(self.device).float()
        # Chuẩn hóa L2 trên GPU/CPU
        self.image_features /= self.image_features.norm(dim=-1, keepdim=True)
        print(f"[RetrievalEngine] Đã nạp và chuẩn hóa {self.image_features.shape[0]} vector (dim={self.image_features.shape[1]}).")

        # 3. Nạp File Mapping
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Không tìm thấy file: {map_path}")
        
        print(f"[RetrievalEngine] Đang nạp map keyframes từ '{map_path}'...")
        with open(map_path, "r", encoding="utf-8") as f:
            self.keyframe_map = json.load(f)
        print(f"[RetrievalEngine] Đã nạp {len(self.keyframe_map)} bản ghi ánh xạ.")

        # Xây dựng cấu trúc tra cứu nhanh theo video_name -> list indices
        self.video_to_indices = {}
        for idx, item in enumerate(self.keyframe_map):
            vname = item.get("video") or item.get("video_name")
            if vname:
                if vname not in self.video_to_indices:
                    self.video_to_indices[vname] = []
                self.video_to_indices[vname].append(idx)

        # 4. Quét trước các thư mục Keyframes để tìm ảnh nhanh
        self.keyframe_roots = self._scan_keyframe_roots()
        print(f"[RetrievalEngine] Tìm thấy các thư mục keyframes: {self.keyframe_roots}")

        # Bộ dịch tự động
        self.translator = GoogleTranslator(source='auto', target='en')

        # Nạp ASR Transcripts
        self.has_bm25 = False
        self._load_transcripts()

        self._initialized = True

    def _load_transcripts(self, transcript_dir="data/transcripts", fps=25):
        self.keyframe_texts = ["" for _ in range(len(self.keyframe_map))]
        
        if not os.path.exists(transcript_dir):
            return
            
        print(f"[RetrievalEngine] Đang nạp transcripts (ASR) từ {transcript_dir}...")
        for json_path in glob.glob(os.path.join(transcript_dir, "*.json")):
            vname = os.path.splitext(os.path.basename(json_path))[0]
            v_indices = self.video_to_indices.get(vname)
            if not v_indices:
                continue
                
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    subs = json.load(f)
                except:
                    continue
                    
            kf_ids = []
            for idx in v_indices:
                kf_id = int(self.keyframe_map[idx].get("frame") or self.keyframe_map[idx].get("frame_idx") or 1)
                kf_ids.append(kf_id)
            kf_ids = np.array(kf_ids)
            
            for sub in subs:
                start_str = sub.get("start", "00:00:00.000")
                end_str = sub.get("end", "00:00:00.000")
                text = sub.get("text", "").strip()
                if not text:
                    continue
                    
                try:
                    h, m, s = start_str.split(':')
                    start_sec = int(h)*3600 + int(m)*60 + float(s)
                    h, m, s = end_str.split(':')
                    end_sec = int(h)*3600 + int(m)*60 + float(s)
                except:
                    continue
                    
                mid_sec = (start_sec + end_sec) / 2.0
                target_frame = mid_sec * fps
                
                diffs = np.abs(kf_ids - target_frame)
                best_local_idx = np.argmin(diffs)
                best_global_idx = v_indices[best_local_idx]
                
                self.keyframe_texts[best_global_idx] += " " + text
                
        try:
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [doc.lower().split() for doc in self.keyframe_texts]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.has_bm25 = True
            print("[RetrievalEngine] Đã khởi tạo BM25 thành công.")
        except ImportError:
            print("[RetrievalEngine] Thiếu thư viện rank_bm25. Không thể dùng tính năng tìm kiếm ASR.")
        except Exception as e:
            print(f"[RetrievalEngine] Lỗi khởi tạo BM25: {e}")

    def _scan_keyframe_roots(self):
        roots = []
        current_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        data_kf_dir = os.path.join(current_dir, "data", "keyframes")
        if os.path.isdir(data_kf_dir):
            for entry in os.listdir(data_kf_dir):
                full_path = os.path.join(data_kf_dir, entry)
                if os.path.isdir(full_path):
                    if "keyframe" in entry.lower():
                        sub_kf = os.path.join(full_path, "keyframes")
                        if os.path.isdir(sub_kf):
                            roots.append(sub_kf)
                        else:
                            roots.append(full_path)
        if not roots:
            roots.append(current_dir)
        return roots

    def get_extracted_videos(self) -> set[str]:
        """Lấy danh sách các video đã có thư mục ảnh thực tế trên ổ cứng"""
        extracted = set()
        for root in self.keyframe_roots:
            if os.path.isdir(root):
                for entry in os.listdir(root):
                    if os.path.isdir(os.path.join(root, entry)):
                        extracted.add(entry)
        return extracted

    def get_image_path(self, video_name: str, frame_id: int) -> str | None:
        """
        Tìm kiếm đường dẫn thực tế của ảnh keyframe theo video_name và frame_id
        """
        candidates = [
            f"{frame_id:03d}.jpg",
            f"{frame_id:04d}.jpg",
            f"{frame_id:05d}.jpg",
            f"{frame_id:06d}.jpg",
            f"{frame_id}.jpg",
            f"{frame_id:03d}.png",
            f"{frame_id:04d}.png",
            f"{frame_id}.png"
        ]

        for root in self.keyframe_roots:
            video_dir = os.path.join(root, video_name)
            if os.path.isdir(video_dir):
                for cand in candidates:
                    img_path = os.path.join(video_dir, cand)
                    if os.path.isfile(img_path):
                        return img_path
                    
            img_cand = os.path.join(root, f"{video_name}_{frame_id:03d}.jpg")
            if os.path.isfile(img_cand):
                return img_cand

        return None

    def _clean_vietnamese_query(self, text: str) -> str:
        """
        Bước 1: Loại bỏ các cụm từ rác tiếng Việt không mang giá trị hình ảnh.
        Ví dụ: "Đoạn clip cần tìm là cảnh..." → "cảnh..."
        """
        # Các pattern mở đầu phổ biến trong đề thi AIC — không liên quan tới hình ảnh
        noise_patterns = [
            r'đoạn (clip|video) (cần tìm|mô tả|về|trong|bắt đầu|là)',
            r'hãy tìm (chính xác )?',
            r'tìm (chính xác )?',
            r'phân cảnh (bắt đầu|tiếp theo|cuối cùng) (là|cho thấy|với)',
            r'(bước đầu tiên là việc|sau đó)',
            r'trong đoạn (clip|video) (có thể thấy|có là|có)',
            r'đây là (phần|loài|một|cảnh)',
            r'biết (rằng |sau đó )?',
            r'hỏi .+là gì\??',       # Câu hỏi Q&A → bỏ phần hỏi, giữ phần mô tả
            r'con số .+ là bao nhiêu\??',
            r'\(tại thời điểm đó\)',
        ]
        
        cleaned = text.strip()
        for pattern in noise_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Xóa khoảng trắng thừa
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned if cleaned else text.strip()

    def _extract_visual_subjects(self, text: str) -> list[str]:
        """
        Bước 2: Trích xuất các chủ thể trực quan chính từ câu tiếng Việt.
        Trả về danh sách các cụm từ mô tả hình ảnh quan trọng.
        """
        # Từ điển các đối tượng/chủ đề trực quan thường gặp trong video AIC
        visual_patterns = [
            # Động vật
            r'con (?:cá|mèo|chó|hổ|chim|bạch tuộc|mực|dê|lân|rồng|bọ)',
            # Người + mô tả
            r'(?:phi hành gia|đầu bếp|vận động viên|tay đua|cô bé|bà cụ|cô gái|chàng trai|người phụ nữ|người đàn ông)(?:\s+\w+){0,5}',
            # Quần áo + màu sắc  
            r'(?:mặc|đeo|đội|quàng|cầm)\s+[\wÀ-ỹ\s]+?(?:đen|trắng|đỏ|hồng|xanh|vàng|tím|nâu)',
            # Màu sắc + đối tượng
            r'(?:áo|quần|mũ|nón|khăn|váy|túi|đĩa|chén|ly|hộp)\s+[\wÀ-ỹ\s]*?(?:đen|trắng|đỏ|hồng|xanh|vàng|tím|nâu|sọc|kẻ)',
            # Đồ vật cụ thể
            r'(?:máy ảnh|ống kính|bánh rán|panna cotta|gỏi cuốn|xe đạp|tàu vũ trụ|cái cân)',
            # Số lượng + đối tượng
            r'\d+\s+(?:người|con|chiếc|cái|em|tay đua|phi hành gia|ly|bông hoa|cột khói)',
            # Bối cảnh
            r'(?:khu rừng|bệnh viện|biển|trại|lễ hội|bếp|sân khấu|đường phố)',
        ]
        
        subjects = []
        for pattern in visual_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            subjects.extend(matches)
        
        return subjects

    def translate_to_en(self, text: str) -> str:
        """Tự động dịch câu truy vấn tiếng Việt sang tiếng Anh cho CLIP"""
        try:
            # Kiểm tra xem có chứa ký tự tiếng Việt có dấu không
            if re.search(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', text, re.IGNORECASE):
                return self.translator.translate(text)
            return text
        except Exception:
            return text

    def preprocess_query(self, vietnamese_text: str, auto_translate: bool = True) -> str:
        """
        Pipeline tiền xử lý truy vấn tiếng Việt → tiếng Anh tối ưu cho CLIP.
        
        Quy trình:
        1. Loại bỏ rác tiếng Việt (cụm từ mở đầu không có giá trị hình ảnh)
        2. Trích xuất chủ thể trực quan (cá, cân, áo đen, phi hành gia...)
        3. Dịch sang tiếng Anh
        4. Bọc trong CLIP prompt template: "a photograph showing {X}"
        """
        # Bước 1: Làm sạch tiếng Việt
        cleaned = self._clean_vietnamese_query(vietnamese_text)
        
        if not auto_translate:
            return cleaned
        
        # Bước 2: Dịch sang tiếng Anh
        translated = self.translate_to_en(cleaned)
        
        # Bước 3: Bọc trong CLIP prompt template
        # Nghiên cứu CLIP cho thấy "a photograph of/showing {X}" cải thiện đáng kể độ chính xác
        if not translated.lower().startswith(('a photo', 'a photograph', 'an image')):
            translated = f"a photograph showing {translated}"
        
        return translated

    def encode_text(self, query_text: str) -> torch.Tensor:
        """Trích xuất vector văn bản 512 chiều từ câu truy vấn"""
        tokens = clip.tokenize([query_text], truncate=True).to(self.device)
        with torch.no_grad():
            text_feature = self.model.encode_text(tokens)
            text_feature /= text_feature.norm(dim=-1, keepdim=True)
        return text_feature

    def get_transcript_for_frame(self, global_idx: int) -> str:
        """Trả về lời thoại ASR gần keyframe này (nếu có)."""
        if hasattr(self, 'keyframe_texts') and global_idx < len(self.keyframe_texts):
            text = self.keyframe_texts[global_idx].strip()
            if text:
                return text
        return ""

    def search(
        self,
        query_text: str,
        top_k: int = 100,
        video_filter: str | list[str] | None = None,
        auto_translate: bool = True,
        audio_weight: float = 0.0
    ) -> tuple[list[dict], str]:
        """
        Tìm kiếm khung hình phù hợp nhất với câu truy vấn.
        Hỗ trợ tìm kiếm kết hợp Hình ảnh (CLIP) và Lời thoại (ASR BM25).
        
        Chiến lược Hybrid 2 tầng:
        - ASR (BM25) hoạt động ở CẤP ĐỘ VIDEO: Tìm đúng video nào có chứa lời thoại liên quan.
        - CLIP hoạt động ở CẤP ĐỘ KEYFRAME: Chọn đúng bức ảnh phù hợp nhất trong video đó.
        => ASR giúp đẩy đúng video lên top, CLIP giúp chọn đúng frame trong video đó.
        
        Trả về (results, translated_query)
        """
        processed_query = query_text.strip()
        if auto_translate:
            processed_query_en = self.translate_to_en(processed_query)
        else:
            processed_query_en = processed_query

        # === MULTI-SENTENCE DECOMPOSITION ===
        # Khi câu truy vấn có nhiều câu (chia bằng dấu chấm), tách ra tìm kiếm riêng từng câu.
        # Sau đó kết hợp bằng Trung bình nhân (Geometric Mean) để kết quả phải khớp TẤT CẢ các câu.
        # Ví dụ: "4 phi hành gia mặc áo đen" + "phóng tàu vũ trụ" + "nghiên cứu cực quang"
        # => Kết quả phải có cả: người mặc áo đen + liên quan vũ trụ + cực quang
        sentences = [s.strip() for s in re.split(r'[.。!?\n]+', processed_query_en) if len(s.strip()) > 5]
        
        if len(sentences) > 1:
            # Tìm kiếm từng câu riêng lẻ rồi kết hợp
            all_sim_scores = []
            for sent in sentences:
                feat = self.encode_text(sent)
                sim = (feat @ self.image_features.T).squeeze(0).cpu().numpy()
                # Shift về [0, 1] để chuẩn bị nhân geometric mean
                sim_min, sim_max = sim.min(), sim.max()
                if sim_max > sim_min:
                    sim_norm = (sim - sim_min) / (sim_max - sim_min)
                else:
                    sim_norm = np.zeros_like(sim)
                all_sim_scores.append(sim_norm + 1e-8)  # Tránh nhân với 0
            
            # Geometric Mean: Bắt buộc khớp TẤT CẢ câu, không cho phép chỉ khớp 1 câu
            combined = np.ones_like(all_sim_scores[0])
            for scores in all_sim_scores:
                combined *= scores
            visual_scores_raw = np.power(combined, 1.0 / len(all_sim_scores))
            
            # Áp dụng video filter
            if video_filter:
                if isinstance(video_filter, str):
                    video_filter = [video_filter]
                target_indices = []
                for vf in video_filter:
                    target_indices.extend(self.video_to_indices.get(vf, []))
                if not target_indices:
                    return [], processed_query_en
                mask_arr = np.zeros(len(visual_scores_raw), dtype=bool)
                for ti in target_indices:
                    mask_arr[ti] = True
                visual_scores_raw[~mask_arr] = 0.0
            
            visual_scores = visual_scores_raw
        else:
            # Câu đơn - tìm kiếm bình thường
            text_feature = self.encode_text(processed_query_en)
            similarity = (text_feature @ self.image_features.T).squeeze(0)

            if video_filter:
                if isinstance(video_filter, str):
                    video_filter = [video_filter]
                target_indices = []
                for vf in video_filter:
                    target_indices.extend(self.video_to_indices.get(vf, []))
                if not target_indices:
                    return [], processed_query_en
                mask = torch.zeros_like(similarity, dtype=torch.bool)
                mask[target_indices] = True
                similarity[~mask] = -100.0

            visual_scores = similarity.cpu().numpy()
        
        # Tính điểm Audio ở CẤP ĐỘ VIDEO - Chiến lược "Coverage Ratio"
        # Trích xuất nhiều từ khóa từ câu truy vấn, đếm xem video nào phủ được nhiều từ nhất.
        # Video phủ càng nhiều từ khóa → càng chắc chắn là đúng video → boost càng mạnh (lên đến 50%).
        # Video chỉ khớp 1-2 từ → không đủ tin cậy → không boost hoặc boost rất ít.
        audio_scores = np.zeros_like(visual_scores)
        if audio_weight > 0.0 and self.has_bm25:
            # Danh sách từ dừng tiếng Việt (không có giá trị phân biệt)
            stop_words = {
                'là', 'của', 'và', 'có', 'với', 'trong', 'một', 'các', 'này',
                'đã', 'được', 'cho', 'từ', 'đến', 'về', 'theo', 'để', 'bằng',
                'không', 'những', 'hay', 'hoặc', 'nhưng', 'vì', 'do', 'nếu',
                'thì', 'sẽ', 'đang', 'rất', 'cũng', 'đây', 'đó', 'ở', 'ra',
                'lên', 'vào', 'khi', 'sau', 'trước', 'trên', 'dưới', 'ngoài',
                'hình', 'ảnh', 'đoạn', 'clip', 'video', 'phần', 'bắt', 'đầu',
                'giới', 'thiệu', 'việc', 'người', 'nhiều', 'đội', 'nhóm'
            }
            
            # Trích xuất từ khóa có ý nghĩa từ câu truy vấn (bỏ từ dừng, bỏ từ quá ngắn)
            raw_tokens = query_text.lower().split()
            query_keywords = [w for w in raw_tokens if w not in stop_words and len(w) >= 2]
            
            if query_keywords:
                # Xây dựng văn bản đầy đủ cho TỪNG VIDEO (gộp tất cả lời thoại lại)
                video_full_text = {}
                for vname, v_indices in self.video_to_indices.items():
                    texts = []
                    for idx in v_indices:
                        t = self.keyframe_texts[idx].strip()
                        if t:
                            texts.append(t)
                    if texts:
                        video_full_text[vname] = " ".join(texts).lower()
                
                # Tính Coverage Ratio cho từng video
                MIN_COVERAGE = 0.25  # Phải khớp ít nhất 25% từ khóa mới được tính
                MAX_BOOST = 0.5      # Boost tối đa 50% khi khớp hoàn hảo
                
                video_boost = {}
                for vname, full_text in video_full_text.items():
                    if video_filter and vname not in video_filter:
                        continue
                    # Đếm số từ khóa xuất hiện trong toàn bộ lời thoại của video
                    matched = sum(1 for kw in query_keywords if kw in full_text)
                    coverage = matched / len(query_keywords)
                    
                    if coverage >= MIN_COVERAGE:
                        # Boost tỷ lệ thuận với coverage: 25% keywords → boost nhẹ, 100% → boost 50%
                        # Scale từ 0 đến MAX_BOOST khi coverage đi từ MIN_COVERAGE đến 1.0
                        boost = MAX_BOOST * ((coverage - MIN_COVERAGE) / (1.0 - MIN_COVERAGE))
                        boost = min(boost, MAX_BOOST)
                        video_boost[vname] = boost
                
                # Gán điểm boost cho tất cả keyframe trong video đó
                for vname, boost in video_boost.items():
                    v_indices = self.video_to_indices.get(vname, [])
                    for idx in v_indices:
                        audio_scores[idx] = boost

        # Normalize điểm về [0, 1] để Fusion
        def min_max_norm(arr):
            min_v, max_v = arr.min(), arr.max()
            if max_v > min_v:
                return (arr - min_v) / (max_v - min_v)
            return np.zeros_like(arr)

        if audio_weight > 0.0 and audio_scores.max() > 0:
            norm_visual = min_max_norm(visual_scores)
            norm_audio = min_max_norm(audio_scores)
            final_scores = (1.0 - audio_weight) * norm_visual + audio_weight * norm_audio
        else:
            final_scores = visual_scores

        final_scores_tensor = torch.from_numpy(final_scores).to(self.device)
        
        # Lấy nhiều ứng viên hơn cần thiết, sau đó lọc đa dạng hóa video
        # Tránh tình trạng 1 video chiếm hết top vì ASR boost quá mạnh
        fetch_k = min(top_k * 5, final_scores_tensor.shape[0])
        top_scores_all, top_pos_all = final_scores_tensor.topk(fetch_k)
        
        # Giới hạn tối đa số frame/video = top_k / 5 (ví dụ: top_k=50 → max 10 frame/video)
        max_per_video = max(3, top_k // 5)
        
        video_count = {}
        results = []
        for idx_t, score_t in zip(top_pos_all.tolist(), top_scores_all.tolist()):
            item = self.keyframe_map[idx_t]
            v_name = item.get("video") or item.get("video_name")
            
            # Đếm số frame đã lấy từ video này
            video_count[v_name] = video_count.get(v_name, 0) + 1
            if video_count[v_name] > max_per_video:
                continue  # Bỏ qua frame thừa, nhường chỗ cho video khác
            
            f_id = int(item.get("frame") or item.get("frame_idx") or item.get("frame_id") or 1)
            img_path = self.get_image_path(v_name, f_id)

            results.append({
                "index": idx_t,
                "video": v_name,
                "frame": f_id,
                "score": float(score_t),
                "image_path": img_path
            })
            
            if len(results) >= top_k:
                break

        return results, processed_query_en

    def search_trake(
        self,
        event_queries: list[str],
        top_k_videos: int = 10,
        video_filter: list[str] | None = None,
        auto_translate: bool = True
    ) -> tuple[list[dict], list[str]]:
        if not event_queries:
            return [], []

        processed_events = []
        for eq in event_queries:
            eq_proc = self.translate_to_en(eq.strip()) if auto_translate else eq.strip()
            processed_events.append(eq_proc)

        event_features = [self.encode_text(eq) for eq in processed_events]
        
        candidate_videos = set()
        video_scores = {}

        for feat in event_features:
            sim = (feat @ self.image_features.T).squeeze(0)
            top_val, top_idx = sim.topk(min(150, self.image_features.shape[0]))
            for v, idx in zip(top_val.tolist(), top_idx.tolist()):
                vname = self.keyframe_map[idx].get("video") or self.keyframe_map[idx].get("video_name")
                if video_filter and vname not in video_filter:
                    continue
                candidate_videos.add(vname)
                video_scores[vname] = video_scores.get(vname, 0.0) + v

        sorted_videos = sorted(candidate_videos, key=lambda x: video_scores[x], reverse=True)[:top_k_videos]

        trake_results = []
        for vname in sorted_videos:
            v_indices = self.video_to_indices.get(vname, [])
            if not v_indices:
                continue

            v_indices_tensor = torch.tensor(v_indices, device=self.device, dtype=torch.long)
            v_feats = self.image_features[v_indices_tensor]

            chosen_frames = []
            min_frame = 0
            valid_sequence = True

            for feat in event_features:
                sim = (feat @ v_feats.T).squeeze(0)
                sorted_score_pos = sim.argsort(descending=True)

                found = False
                for pos in sorted_score_pos.tolist():
                    global_idx = v_indices[pos]
                    f_id = int(self.keyframe_map[global_idx].get("frame") or self.keyframe_map[global_idx].get("frame_idx") or 1)
                    if f_id > min_frame:
                        chosen_frames.append({
                            "frame": f_id,
                            "score": float(sim[pos]),
                            "image_path": self.get_image_path(vname, f_id)
                        })
                        min_frame = f_id
                        found = True
                        break
                
                if not found:
                    valid_sequence = False
                    break

            if valid_sequence and len(chosen_frames) == len(processed_events):
                trake_results.append({
                    "video": vname,
                    "frames": chosen_frames,
                    "total_score": sum(cf["score"] for cf in chosen_frames)
                })

        trake_results.sort(key=lambda x: x["total_score"], reverse=True)
        return trake_results, processed_events
