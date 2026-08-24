"""
AIC 2026 Batch Retrieval & Submission Exporter Pipeline
Hỗ trợ 4-Track System:
- engine_mode: 'meta' (Track 4 - Mặc định), 'hybrid' (Track 3), 'offline' (Track 1), 'ai' (Track 2)
"""

import os
import glob
import re
import argparse
from tqdm import tqdm
from core.retrieval_engine import RetrievalEngine
from core.submission_exporter import SubmissionExporter

def process_query_file(
    query_filepath: str,
    engine: RetrievalEngine,
    exporter: SubmissionExporter,
    top_k: int = 100,
    engine_mode: str = "meta",
    use_ai_query: bool = True,
    use_asr: bool = False,
    audio_weight: float = 0.3
):
    fname = os.path.basename(query_filepath)
    base_name = os.path.splitext(fname)[0]
    csv_filename = f"{base_name}.csv"

    with open(query_filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        print(f"[SKIP] File rỗng: {query_filepath}")
        return None

    # Tự động nhận diện loại đề bài từ tên file hoặc cấu trúc
    if "trake" in fname.lower():
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        event_queries = [
            re.sub(r"^E\d+[:.]\s*", "", l, flags=re.IGNORECASE)
            for l in lines if re.match(r"^E\d+[:.]", l, re.IGNORECASE)
        ]
        if not event_queries:
            event_queries = lines
        
        trake_res, _ = engine.search_trake(event_queries, top_k_videos=top_k, use_ai_query=use_ai_query)
        saved_path = exporter.export_trake(trake_res, csv_filename, max_rows=top_k)
        print(f"[TRAKE] Đã xuất: {saved_path} ({len(trake_res)} dòng)")
        return saved_path

    elif "qa" in fname.lower():
        res, _, _ = engine.search(
            content, top_k=top_k,
            engine_mode=engine_mode,
            use_ai_query=use_ai_query,
            use_asr=use_asr,
            audio_weight=audio_weight
        )
        qa_data = [{"video": r["video"], "frame": r["frame"], "answer": engine.answer_qa(content, r)} for r in res]
        saved_path = exporter.export_qa(qa_data, csv_filename, max_rows=top_k)
        print(f"[QA] Đã xuất: {saved_path} ({len(qa_data)} dòng)")
        return saved_path

    else:
        res, _, _ = engine.search(
            content, top_k=top_k,
            engine_mode=engine_mode,
            use_ai_query=use_ai_query,
            use_asr=use_asr,
            audio_weight=audio_weight
        )
        saved_path = exporter.export_kis(res, csv_filename, max_rows=top_k)
        print(f"[KIS] Đã xuất: {saved_path} ({len(res)} dòng)")
        return saved_path

def main():
    parser = argparse.ArgumentParser(description="AIC Video Retrieval Pipeline (AIC 2026 4-Track Engine)")
    parser.add_argument("--query", "-q", type=str, default=None, help="Đường dẫn file đề thi .txt cụ thể")
    parser.add_argument("--dir", "-d", type=str, default="de-thi", help="Thư mục chứa các file đề thi .txt")
    parser.add_argument("--top_k", "-k", type=int, default=100, help="Số lượng kết quả tối đa (mặc định: 100)")
    parser.add_argument("--engine_mode", "-m", type=str, default="meta", choices=["meta", "hybrid", "offline", "ai"], help="Track thực thi: 'meta' (T4), 'hybrid' (T3), 'offline' (T1), 'ai' (T2)")
    parser.add_argument("--use_ai", action="store_true", default=False, help="Bật phân tích AI (nếu dùng mode ai/hybrid)")
    parser.add_argument("--use_asr", action="store_true", default=False, help="Bật tìm kiếm lời thoại âm thanh ASR")
    parser.add_argument("--audio_weight", "-w", type=float, default=0.3, help="Trọng số ASR khi được bật (mặc định: 0.3)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"🚀 KHỞI ĐỘNG AIC 2026 RETRIEVAL PIPELINE (Track: {args.engine_mode.upper()})")
    print("=" * 60)

    engine = RetrievalEngine()
    exporter = SubmissionExporter()

    if args.query:
        if not os.path.exists(args.query):
            print(f"[ERROR] Không tìm thấy file: {args.query}")
            return
        process_query_file(
            args.query, engine, exporter,
            top_k=args.top_k,
            engine_mode=args.engine_mode,
            use_ai_query=args.use_ai or (args.engine_mode in ["meta", "hybrid", "ai"]),
            use_asr=args.use_asr,
            audio_weight=args.audio_weight
        )
    else:
        query_dir = args.dir
        if not os.path.exists(query_dir):
            if os.path.exists("THUNGHIEM-bo-de-thi"):
                query_dir = "THUNGHIEM-bo-de-thi"
            else:
                print(f"[ERROR] Không tìm thấy thư mục: {query_dir}")
                return

        query_files = sorted(glob.glob(os.path.join(query_dir, "*.txt")))
        query_files = [f for f in query_files if "requirement" not in f.lower()]

        print(f"[INFO] Tìm thấy {len(query_files)} file đề thi trong '{query_dir}'.")
        for qf in tqdm(query_files, desc="Đang xử lý đề thi"):
            process_query_file(
                qf, engine, exporter,
                top_k=args.top_k,
                engine_mode=args.engine_mode,
                use_ai_query=args.use_ai or (args.engine_mode in ["meta", "hybrid", "ai"]),
                use_asr=args.use_asr,
                audio_weight=args.audio_weight
            )

        zip_path = exporter.zip_submissions("submission.zip")
        print(f"\n✅ ĐÃ HOÀN TẤT VÀ ĐÓNG GÓI TẤT CẢ KẾT QUẢ VÀO FILE ZIP: {zip_path}")

if __name__ == "__main__":
    main()
