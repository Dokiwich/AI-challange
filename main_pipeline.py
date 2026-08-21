"""
Pipeline tìm kiếm và xuất kết quả nộp bài (Từ Bước 1 -> Bước 8)
Sử dụng trực tiếp các file dữ liệu đã setup:
- clip_features.npy
- map_keyframes.json
- Thư mục Keyframes_*
- Thư mục câu hỏi THUNGHIEM-bo-de-thi / file .txt
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
    top_k: int = 100
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
        
        trake_res, _ = engine.search_trake(event_queries, top_k_videos=top_k)
        saved_path = exporter.export_trake(trake_res, csv_filename, max_rows=top_k)
        print(f"[TRAKE] Đã xuất: {saved_path} ({len(trake_res)} dòng)")
        return saved_path

    elif "qa" in fname.lower():
        # Dạng Q&A
        res, _ = engine.search(content, top_k=top_k)
        qa_data = [{"video": r["video"], "frame": r["frame"], "answer": "yes"} for r in res]
        saved_path = exporter.export_qa(qa_data, csv_filename, max_rows=top_k)
        print(f"[QA] Đã xuất: {saved_path} ({len(qa_data)} dòng)")
        return saved_path

    else:
        # Dạng KIS
        res, _ = engine.search(content, top_k=top_k)
        saved_path = exporter.export_kis(res, csv_filename, max_rows=top_k)
        print(f"[KIS] Đã xuất: {saved_path} ({len(res)} dòng)")
        return saved_path

def main():
    parser = argparse.ArgumentParser(description="AIC Video Retrieval Pipeline (Step 1 -> Step 8)")
    parser.add_argument("--query", "-q", type=str, default=None, help="Đường dẫn file đề thi .txt cụ thể")
    parser.add_argument("--dir", "-d", type=str, default="THUNGHIEM-bo-de-thi", help="Thư mục chứa các file đề thi .txt")
    parser.add_argument("--top_k", "-k", type=int, default=100, help="Số lượng kết quả tối đa (mặc định: 100)")
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 KHỞI ĐỘNG AIC RETRIEVAL PIPELINE (BƯỚC 1 -> BƯỚC 8)")
    print("=" * 60)

    engine = RetrievalEngine()
    exporter = SubmissionExporter()

    if args.query:
        if not os.path.exists(args.query):
            print(f"[ERROR] Không tìm thấy file: {args.query}")
            return
        process_query_file(args.query, engine, exporter, top_k=args.top_k)
    else:
        if not os.path.exists(args.dir):
            print(f"[ERROR] Không tìm thấy thư mục: {args.dir}")
            return
        
        query_files = sorted(glob.glob(os.path.join(args.dir, "*.txt")))
        print(f"\n[INFO] Tìm thấy {len(query_files)} file đề thi trong '{args.dir}'")
        
        for qf in tqdm(query_files, desc="Đang xử lý đề thi"):
            process_query_file(qf, engine, exporter, top_k=args.top_k)

    print("\n" + "=" * 60)
    print("✅ ĐÃ HOÀN TẤT BƯỚC 8: TẤT CẢ FILE .CSV ĐÃ LƯU TRONG THƯ MỤC 'submission/'")
    print("=" * 60)

if __name__ == "__main__":
    main()
