"""
AIC 2026 Batch Retrieval & Submission Exporter Pipeline
Kiến trúc 2 Pha (Phase-Shifting GPU Allocation):
  Pha 1: CLIP search toàn bộ đề thi -> Xuất KIS/TRAKE ngay, xếp hàng QA
  Pha 2: Giải phóng CLIP -> Nạp VLM -> Chạy QA batch -> Xuất CSV
"""

import os
import glob
import re
import argparse
import logging
from tqdm import tqdm
from dotenv import load_dotenv

from core.retrieval_engine import RetrievalEngine
from core.submission_exporter import SubmissionExporter
from core.query_compiler import QueryCompiler

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _detect_query_type(fname: str) -> str:
    fl = fname.lower()
    if "trake" in fl:
        return "trake"
    if "qa" in fl:
        return "qa"
    return "kis"


def phase1_clip_search(
    query_files: list,
    engine: RetrievalEngine,
    exporter: SubmissionExporter,
    top_k: int,
    engine_mode: str,
    use_ai_query: bool,
    use_asr: bool,
    audio_weight: float
) -> tuple:
    """Pha 1: Dùng CLIP quét toàn bộ đề. KIS/TRAKE xuất ngay. QA lưu vào hàng đợi."""
    qa_queue = []
    saved_csv_paths = []

    for qf in tqdm(query_files, desc="Pha 1: CLIP Search"):
        fname = os.path.basename(qf)
        base_name = os.path.splitext(fname)[0]
        csv_filename = f"{base_name}.csv"
        csv_full_path = os.path.join(exporter.submission_dir, csv_filename)

        with open(qf, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            continue

        qtype = _detect_query_type(fname)

        if qtype == "trake":
            event_queries = QueryCompiler.parse_trake_events(content)
            trake_res, _ = engine.search_trake(event_queries, top_k_videos=top_k, use_ai_query=use_ai_query)
            saved = exporter.export_trake(trake_res, csv_filename, max_rows=top_k)
            saved_csv_paths.append(saved)
            logger.info(f"[TRAKE] Xuất: {saved} ({len(trake_res)} candidates, {len(event_queries)} events)")

        elif qtype == "qa":
            res, _, _ = engine.search(
                content, top_k=top_k,
                engine_mode=engine_mode,
                use_ai_query=use_ai_query,
                use_asr=use_asr,
                audio_weight=audio_weight
            )
            qa_queue.append({
                "csv_filename": csv_filename,
                "question": content,
                "results": res,
                "top_k": top_k
            })
            logger.info(f"[QA] Đã xếp hàng: {csv_filename} ({len(res)} candidates)")

        else:
            res, _, _ = engine.search(
                content, top_k=top_k,
                engine_mode=engine_mode,
                use_ai_query=use_ai_query,
                use_asr=use_asr,
                audio_weight=audio_weight
            )
            saved = exporter.export_kis(res, csv_filename, max_rows=top_k)
            saved_csv_paths.append(saved)
            logger.info(f"[KIS] Xuất: {saved} ({len(res)} dòng)")

    return qa_queue, saved_csv_paths


def phase2_vlm_qa(
    qa_queue: list,
    engine: RetrievalEngine,
    exporter: SubmissionExporter
) -> list:
    """Pha 2: Giải phóng CLIP, nạp VLM, chạy QA batch, xuất CSV."""
    if not qa_queue:
        logger.info("[Pha 2] Không có đề QA. Bỏ qua.")
        return []

    saved_qa_paths = []
    logger.info(f"[Pha 2] Bắt đầu xuất {len(qa_queue)} đề QA (Cần người duyệt)...")
    for qa_item in tqdm(qa_queue, desc="Pha 2: Export QA"):
        question = qa_item["question"]
        results = qa_item["results"]
        csv_filename = qa_item["csv_filename"]
        top_k = qa_item["top_k"]

        qa_data = []
        for r in results:
            answer = r.get("answer", "Có")
            qa_data.append({"video": r["video"], "frame": r["frame"], "answer": answer})

        saved = exporter.export_qa(qa_data, {}, csv_filename, max_rows=top_k)
        saved_qa_paths.append(saved)
        logger.info(f"[QA] Xuất: {saved} ({len(qa_data)} dòng)")

    return saved_qa_paths


def main():
    parser = argparse.ArgumentParser(description="AIC Video Retrieval Pipeline (2-Phase Architecture)")
    parser.add_argument("--query", "-q", type=str, default=None, help="File đề thi .txt cụ thể")
    parser.add_argument("--dir", "-d", type=str, default="SOTUYEN2-bo-de-thi", help="Thư mục chứa file đề thi .txt")
    parser.add_argument("--top_k", "-k", type=int, default=100)
    parser.add_argument("--engine_mode", "-m", type=str, default="meta", choices=["meta", "hybrid", "offline", "ai"])
    parser.add_argument("--use_ai", action="store_true", default=False)
    parser.add_argument("--use_asr", action="store_true", default=False)
    parser.add_argument("--audio_weight", "-w", type=float, default=0.3)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"🚀 AIC 2026 PIPELINE (2-Phase, Track: {args.engine_mode.upper()})")
    logger.info("=" * 60)

    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

    qclient = None
    try:
        from qdrant_client import QdrantClient
        qclient = QdrantClient(qdrant_host, port=qdrant_port)
        qclient.get_collections()
        logger.info(f"Connected to Qdrant at {qdrant_host}:{qdrant_port}")
    except Exception as e:
        logger.warning(f"Qdrant unavailable ({e}). Using numpy fallback.")

    ndriver = None
    try:
        from neo4j import GraphDatabase
        ndriver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        ndriver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {neo4j_uri}")
    except Exception as e:
        logger.warning(f"Neo4j unavailable ({e}). Using DP fallback.")

    engine = RetrievalEngine(qdrant_client=qclient, neo4j_driver=ndriver)
    exporter = SubmissionExporter()

    use_ai = args.use_ai or (args.engine_mode in ["meta", "hybrid", "ai"])

    if args.query:
        if not os.path.exists(args.query):
            logger.error(f"Không tìm thấy file: {args.query}")
            return
        query_files = [args.query]
    else:
        query_dir = args.dir
        if not os.path.exists(query_dir):
            if os.path.exists("SOTUYEN2-bo-de-thi"):
                query_dir = "SOTUYEN2-bo-de-thi"
            elif os.path.exists("de-thi"):
                query_dir = "de-thi"
            elif os.path.exists("THUNGHIEM-bo-de-thi"):
                query_dir = "THUNGHIEM-bo-de-thi"
            else:
                logger.error(f"Không tìm thấy thư mục: {query_dir}")
                return

        query_files = sorted(glob.glob(os.path.join(query_dir, "*.txt")))
        query_files = [f for f in query_files if "requirement" not in f.lower()]

    logger.info(f"Tìm thấy {len(query_files)} file đề thi trong '{args.dir}'.")

    # === PHA 1: CLIP SEARCH ===
    qa_queue, saved_paths = phase1_clip_search(
        query_files, engine, exporter,
        top_k=args.top_k, engine_mode=args.engine_mode,
        use_ai_query=use_ai, use_asr=args.use_asr,
        audio_weight=args.audio_weight
    )

    # === PHA 2: VLM QA ===
    saved_qa_paths = phase2_vlm_qa(qa_queue, engine, exporter)
    all_saved_paths = saved_paths + saved_qa_paths

    zip_path = exporter.zip_submissions("submission.zip", csv_file_list=all_saved_paths)
    logger.info(f"\n✅ HOÀN TẤT! Đã đóng gói chính xác {len(all_saved_paths)} file vào: {zip_path}")


if __name__ == "__main__":
    main()

