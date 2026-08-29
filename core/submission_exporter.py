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
        results: List[Tuple[str, int]] | List[Dict[str, Any]],
        output_filename: str,
        max_rows: int = 100
    ) -> str:
        """
        Xuất file CSV cho câu hỏi KIS:
        Format: <video_name>,<frame_idx>
        Đảm bảo đúng tối đa max_rows (100 dòng).
        """
        if not output_filename.endswith(".csv"):
            output_filename += ".csv"
        filepath = os.path.join(self.submission_dir, output_filename)

        rows = []
        for item in results:
            if isinstance(item, dict):
                video = item.get("video")
                frame = item.get("frame")
            elif isinstance(item, (list, tuple)):
                video = item[0]
                frame = item[1]
            else:
                continue
            if video and frame is not None:
                rows.append([video, int(frame)])

        # Nếu không đủ max_rows và có ít nhất 1 kết quả, pad thêm để đạt max_rows
        if 0 < len(rows) < max_rows:
            orig_len = len(rows)
            idx = 0
            while len(rows) < max_rows:
                rows.append(rows[idx % orig_len])
                idx += 1

        rows = rows[:max_rows]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for r in rows:
                writer.writerow(r)

        print(f"[Exporter] Đã ghi {len(rows)} dòng vào '{filepath}'")
        return filepath

    def export_qa(
        self,
        results: List[Dict[str, Any]] | List[Tuple[str, int, str]],
        output_filename: str,
        max_rows: int = 100
    ) -> str:
        """
        Xuất file CSV cho câu hỏi Q&A:
        Format: <video_name>,<frame_idx>,<answer>
        Đảm bảo đủ max_rows (100 dòng).
        """
        if not output_filename.endswith(".csv"):
            output_filename += ".csv"
        filepath = os.path.join(self.submission_dir, output_filename)

        rows = []
        for item in results:
            if isinstance(item, dict):
                video = item.get("video")
                frame = item.get("frame")
                ans = str(item.get("answer", "")).strip()
            elif isinstance(item, (list, tuple)):
                video = item[0]
                frame = item[1]
                ans = str(item[2]).strip() if len(item) > 2 else ""
            else:
                continue

            if not ans or ans in ["0", "unknown"]:
                ans = "Có"

            if len(ans) > 100:
                ans = ans[:100]

            if video and frame is not None:
                rows.append([video, int(frame), ans])

        # Nếu không đủ max_rows và có kết quả, pad thêm
        if 0 < len(rows) < max_rows:
            orig_len = len(rows)
            idx = 0
            while len(rows) < max_rows:
                rows.append(rows[idx % orig_len])
                idx += 1

        rows = rows[:max_rows]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for r in rows:
                writer.writerow(r)

        print(f"[Exporter] Đã ghi {len(rows)} dòng vào '{filepath}'")
        return filepath

    def export_trake(
        self,
        results: List[Dict[str, Any]] | List[Tuple[str, List[int]]],
        output_filename: str,
        max_rows: int = 100
    ) -> str:
        """
        Xuất file CSV cho câu hỏi TRAKE:
        Format: <video_name>,<frame_id_1>,<frame_id_2>,...,<frame_id_N>
        Đảm bảo đủ max_rows (100 dòng).
        """
        if not output_filename.endswith(".csv"):
            output_filename += ".csv"
        filepath = os.path.join(self.submission_dir, output_filename)

        rows = []
        for item in results:
            if isinstance(item, dict):
                video = item.get("video")
                frames = item.get("frames", [])
                if not frames:
                    frames = item.get("matched_timestamps", [])

                if isinstance(frames, list) and len(frames) > 0 and isinstance(frames[0], dict):
                    frame_ids = [int(f.get("frame", 0)) for f in frames]
                else:
                    frame_ids = [int(f) for f in frames]
            elif isinstance(item, (list, tuple)):
                video = item[0]
                frame_ids = [int(f) for f in item[1:]] if not isinstance(item[1], (list, tuple)) else [int(f) for f in item[1]]
            else:
                continue

            if video and frame_ids:
                row = [video] + frame_ids
                rows.append(row)

        # Nếu không đủ max_rows và có kết quả, pad thêm
        if 0 < len(rows) < max_rows:
            orig_len = len(rows)
            idx = 0
            while len(rows) < max_rows:
                rows.append(rows[idx % orig_len])
                idx += 1

        rows = rows[:max_rows]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for r in rows:
                writer.writerow(r)

        print(f"[Exporter] Đã ghi {len(rows)} dòng vào '{filepath}'")
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


