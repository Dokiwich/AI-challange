import os
import csv
import zipfile
import glob
from typing import List, Tuple, Dict, Any

class SubmissionExporter:
    """
    Xử lý xuất dữ liệu dự đoán ra file CSV đúng chuẩn BTC ở Bước 8:
    - KIS: <video_name>, <frame_id>
    - QA: <video_name>, <frame_id>, "<answer>"
    - TRAKE: <video_name>, <frame_id_1>, <frame_id_2>, ..., <frame_id_N>
    """
    def __init__(self, submission_dir: str = "submission"):
        self.submission_dir = submission_dir
        os.makedirs(self.submission_dir, exist_ok=True)

    def export_kis(
        self,
        ai_results: List[Dict[str, Any]],
        manual_picks: Dict[int, Dict[str, Any]],
        output_filename: str,
        max_rows: int = 100
    ) -> str:
        """
        Xuất file CSV cho câu hỏi KIS chuẩn AIC:
        Kết hợp manual_picks (chốt tay theo hạng) và ai_results lấp vào chỗ trống.
        """
        if not output_filename.endswith(".csv"):
            output_filename += ".csv"
        filepath = os.path.join(self.submission_dir, output_filename)

        final_rows = [None] * max_rows
        used_frames = set()

        # 1. Gán các vị trí chốt tay
        for rank, item in manual_picks.items():
            idx = rank - 1
            if 0 <= idx < max_rows:
                video = item.get("video")
                frame = int(item.get("frame", 0))
                if video:
                    final_rows[idx] = [video, frame]
                    used_frames.add((video, frame))

        # 2. Lấp đầy khoảng trống bằng AI
        ai_idx = 0
        for i in range(max_rows):
            if final_rows[i] is None:
                # Tìm AI prediction tiếp theo chưa được dùng
                while ai_idx < len(ai_results):
                    a_item = ai_results[ai_idx]
                    a_vid = a_item.get("video")
                    a_frm = int(a_item.get("frame", 0))
                    ai_idx += 1
                    if a_vid and (a_vid, a_frm) not in used_frames:
                        final_rows[i] = [a_vid, a_frm]
                        used_frames.add((a_vid, a_frm))
                        break
                
                # Nếu hết AI predictions, đệm bằng Top 1
                if final_rows[i] is None:
                    if final_rows[0]:
                        final_rows[i] = final_rows[0]
                    else:
                        final_rows[i] = ["dummy_video.mp4", 0]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for r in final_rows:
                writer.writerow(r)

        print(f"[Exporter] Đã ghi {len(final_rows)} dòng vào '{filepath}'")
        return filepath

    def export_qa(
        self,
        ai_results: List[Dict[str, Any]],
        manual_picks: Dict[int, Dict[str, Any]],
        output_filename: str,
        max_rows: int = 100
    ) -> str:
        """
        Xuất file CSV cho câu hỏi Q&A chuẩn AIC.
        """
        if not output_filename.endswith(".csv"):
            output_filename += ".csv"
        filepath = os.path.join(self.submission_dir, output_filename)

        final_rows = [None] * max_rows
        used_frames = set()

        for rank, item in manual_picks.items():
            idx = rank - 1
            if 0 <= idx < max_rows:
                video = item.get("video")
                frame = int(item.get("frame", 0))
                ans = str(item.get("answer", "Có")).strip()
                if not ans or ans in ["0", "unknown"]: ans = "Có"
                if len(ans) > 100: ans = ans[:100]
                if video:
                    final_rows[idx] = [video, frame, ans]
                    used_frames.add((video, frame))

        ai_idx = 0
        for i in range(max_rows):
            if final_rows[i] is None:
                while ai_idx < len(ai_results):
                    a_item = ai_results[ai_idx]
                    a_vid = a_item.get("video")
                    a_frm = int(a_item.get("frame", 0))
                    a_ans = str(a_item.get("answer", "Có")).strip()
                    if not a_ans or a_ans in ["0", "unknown"]: a_ans = "Có"
                    if len(a_ans) > 100: a_ans = a_ans[:100]
                    ai_idx += 1
                    
                    if a_vid and (a_vid, a_frm) not in used_frames:
                        final_rows[i] = [a_vid, a_frm, a_ans]
                        used_frames.add((a_vid, a_frm))
                        break
                        
                if final_rows[i] is None:
                    if final_rows[0]:
                        final_rows[i] = final_rows[0]
                    else:
                        final_rows[i] = ["dummy_video.mp4", 0, "Có"]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for r in final_rows:
                writer.writerow(r)

        print(f"[Exporter] Đã ghi {len(final_rows)} dòng vào '{filepath}'")
        return filepath

    def export_trake(
        self,
        ai_results: List[Dict[str, Any]],
        manual_picks: Dict[int, Dict[str, Any]],
        output_filename: str,
        max_rows: int = 100
    ) -> str:
        """
        Xuất file CSV cho câu hỏi TRAKE chuẩn AIC.
        """
        if not output_filename.endswith(".csv"):
            output_filename += ".csv"
        filepath = os.path.join(self.submission_dir, output_filename)

        final_rows = [None] * max_rows
        used_videos = set()

        for rank, item in manual_picks.items():
            idx = rank - 1
            if 0 <= idx < max_rows:
                video = item.get("video")
                frames_dict = item.get("frames", {})
                if video and frames_dict:
                    # Sort by event_idx
                    frame_ids = [frames_dict[k] for k in sorted(frames_dict.keys())]
                    final_rows[idx] = [video] + frame_ids
                    used_videos.add(video)

        ai_idx = 0
        for i in range(max_rows):
            if final_rows[i] is None:
                while ai_idx < len(ai_results):
                    a_item = ai_results[ai_idx]
                    a_vid = a_item.get("video")
                    
                    frames = a_item.get("frames", [])
                    if not frames: frames = a_item.get("matched_timestamps", [])
                    
                    if isinstance(frames, list) and len(frames) > 0 and isinstance(frames[0], dict):
                        f_ids = [int(f.get("frame", 0)) for f in frames]
                    else:
                        f_ids = [int(f) for f in frames]
                        
                    ai_idx += 1
                    
                    if a_vid and f_ids and a_vid not in used_videos:
                        final_rows[i] = [a_vid] + f_ids
                        used_videos.add(a_vid)
                        break
                        
                if final_rows[i] is None:
                    if final_rows[0]:
                        final_rows[i] = final_rows[0]
                    else:
                        final_rows[i] = ["dummy_video.mp4", 0]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for r in final_rows:
                writer.writerow(r)

        print(f"[Exporter] Đã ghi {len(final_rows)} dòng vào '{filepath}'")
        return filepath

    def zip_submissions(
        self,
        zip_filepath: str = "submission.zip",
        csv_file_list: List[str] = None
    ) -> str:
        """
        Nén các file .csv vào file .zip nộp bài.
        Đúng chuẩn BTC: file ZIP PHẢI chứa thư mục submission/ bên trong.
        """
        if csv_file_list:
            csv_files = sorted(csv_file_list)
        else:
            csv_files = sorted(glob.glob(os.path.join(self.submission_dir, "*.csv")))

        if not csv_files:
            print(f"[Exporter] Cảnh báo: Không tìm thấy file .csv nào trong '{self.submission_dir}'")
            return ""

        with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in csv_files:
                # BTC yêu cầu bên trong zip có thư mục submission/<name>.csv
                arcname = f"submission/{os.path.basename(file_path)}"
                zipf.write(file_path, arcname)

        print(f"[Exporter] ✅ Đã nén thành công {len(csv_files)} file CSV vào '{zip_filepath}' (Cấu trúc: submission/)")
        return zip_filepath


