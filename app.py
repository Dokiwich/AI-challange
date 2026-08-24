import os
import glob
import re
import time
import streamlit as st
from PIL import Image

from core.retrieval_engine import RetrievalEngine
from core.submission_exporter import SubmissionExporter

st.set_page_config(page_title="AIC 2026 Video Retrieval", layout="wide", initial_sidebar_state="expanded")

_ENGINE_VERSION = 14
if st.session_state.get("_engine_ver") != _ENGINE_VERSION:
    for k in [k for k in st.session_state if k.startswith(("kis_", "qa_", "trake_"))]:
        del st.session_state[k]
    st.session_state["_engine_ver"] = _ENGINE_VERSION

@st.cache_resource(show_spinner="Loading CLIP model and 4-Track Vector Database...")
def get_engine(version=_ENGINE_VERSION):
    return RetrievalEngine()

engine = get_engine()
exporter = SubmissionExporter()
extracted_videos = engine.get_extracted_videos()

# =========================================================================
# SIDEBAR CONTROLLER
# =========================================================================
st.sidebar.markdown("## 🏆 AIC 2026 Settings")
query_mode = st.sidebar.radio("Tác vụ thi đấu (Task):", [
    "Textual KIS",
    "Visual QA",
    "TRAKE",
    "Batch Processing",
])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏎️ Kiến trúc 4 Hướng (4-Track Selector)")

track_choice = st.sidebar.selectbox(
    "Chọn Engine Mode:",
    [
        "🧠 Track 4: Adaptive Meta-Policy (Khuyên dùng)",
        "🌟 Track 3: Hybrid Fusion (Late Retrieval)",
        "⚡ Track 1: Offline V2 (Zero-LLM Cục bộ)",
        "🤖 Track 2: AI Semantic V2 (Thuần LLM)"
    ],
    index=0,
    help="Track 4 tự động định tuyến thông minh tối ưu tốc độ và độ chính xác cho từng loại câu hỏi."
)

track_mode_map = {
    "🧠 Track 4: Adaptive Meta-Policy (Khuyên dùng)": "meta",
    "🌟 Track 3: Hybrid Fusion (Late Retrieval)": "hybrid",
    "⚡ Track 1: Offline V2 (Zero-LLM Cục bộ)": "offline",
    "🤖 Track 2: AI Semantic V2 (Thuần LLM)": "ai"
}
engine_mode = track_mode_map[track_choice]

# Cấu hình LLM nếu dùng Track 2, 3 hoặc 4
if engine_mode in ["meta", "hybrid", "ai"]:
    st.sidebar.markdown("#### 🤖 Cấu hình AI Semantic Compiler")
    ai_connected = engine.compiler.ai_parser.check_connection()
    st.sidebar.caption(f"Trạng thái: {'🟢 Online (9Router)' if ai_connected else '🔴 Offline / Dự phòng cục bộ'}")
    if ai_connected:
        available_models = engine.compiler.ai_parser.get_available_models()
        curr_model = engine.compiler.ai_parser.model_name
        default_idx = available_models.index(curr_model) if curr_model in available_models else 0
        selected_model = st.sidebar.selectbox("AI Model:", available_models, index=default_idx)
        engine.compiler.ai_parser.model_name = selected_model
else:
    st.sidebar.caption("⚡ Track 1 đang hoạt động: 100% Cục bộ không phụ thuộc LLM.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎙️ Lời thoại ASR (Thủ công)")
use_asr = st.sidebar.checkbox("Bật tìm kiếm Lời thoại (ASR)", value=False, help="Chỉ kích hoạt kênh tìm kiếm ASR khi bạn bật tùy chọn này.")
if use_asr:
    audio_weight = st.sidebar.slider("ASR weight:", min_value=0.05, max_value=1.0, value=0.3, step=0.05)
    custom_asr_kws = st.sidebar.text_input("Từ khóa ASR tùy chọn:", value="")
else:
    audio_weight = 0.0
    custom_asr_kws = ""

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Cấu hình truy xuất")
auto_translate = st.sidebar.checkbox("Tự động dịch sang tiếng Anh", value=True)
filter_extracted = st.sidebar.checkbox(f"Chỉ tìm trong video đã trích xuất ({len(extracted_videos)})", value=False)
top_k = st.sidebar.slider("Top K kết quả:", min_value=10, max_value=100, value=50, step=10)
cols_count = st.sidebar.slider("Số cột hiển thị:", min_value=2, max_value=6, value=4, step=1)
diversity_top_2 = st.sidebar.checkbox("Lọc đa dạng hóa Top-2 (Diversity)", value=False)

st.sidebar.markdown("---")
st.sidebar.text(f"Device: {engine.device}")
st.sidebar.text(f"Keyframes: {len(engine.keyframe_map):,}")

# =========================================================================
# HEADER
# =========================================================================
st.markdown("# 🎬 AIC 2026 Video Retrieval System")
st.caption("Industrial-Grade Multimodal Video Retrieval with 4-Track Adaptive Meta-Policy & Constraint Graph Judge")

test_files = sorted(glob.glob(os.path.join("de-thi", "*.txt"))) + sorted(glob.glob(os.path.join("THUNGHIEM-bo-de-thi", "*.txt")))
test_files = [f for f in test_files if "requirement" not in f.lower()]
v_filter = list(extracted_videos) if filter_extracted else None

# =========================================================================
# KIS
# =========================================================================
if query_mode == "Textual KIS":
    st.subheader("Textual Known Item Search (KIS)")

    selected_file = st.selectbox("Tải file đề thi:", ["(Manual input)"] + [f for f in test_files if "kis" in f.lower() or not any(t in f.lower() for t in ["qa", "trake"])])

    default_query = ""
    default_sub_name = "query-1-kis.csv"
    if selected_file != "(Manual input)":
        try:
            with open(selected_file, "r", encoding="utf-8") as f:
                default_query = f.read().strip()
            default_sub_name = os.path.splitext(os.path.basename(selected_file))[0] + ".csv"
        except Exception as e:
            st.error(f"Error: {e}")

    query_text = st.text_area("Nội dung truy vấn (Query):", value=default_query, height=100)

    if st.button("🚀 Bắt đầu Tìm kiếm", type="primary", use_container_width=True) and query_text.strip():
        t0 = time.time()
        with st.spinner("Đang xử lý qua 4-Stage Precision Pipeline..."):
            results, trans_q, intent = engine.search(
                query_text.strip(),
                top_k=top_k,
                video_filter=v_filter,
                auto_translate=auto_translate,
                engine_mode=engine_mode,
                use_ai_query=(engine_mode in ["meta", "hybrid", "ai"]),
                use_asr=use_asr,
                audio_weight=audio_weight,
                asr_keywords=custom_asr_kws if custom_asr_kws.strip() else None,
                diversity_top_2=diversity_top_2
            )
            lat = time.time() - t0
            st.session_state["kis_results"] = results
            st.session_state["kis_trans_q"] = trans_q
            st.session_state["kis_intent"] = intent
            st.session_state["kis_sub_name"] = default_sub_name
            st.session_state["kis_lat"] = lat

    if "kis_results" in st.session_state and st.session_state["kis_results"]:
        results = st.session_state["kis_results"]
        trans_q = st.session_state.get("kis_trans_q", "")
        intent = st.session_state.get("kis_intent", {})
        lat = st.session_state.get("kis_lat", 0.0)

        active_t = intent.get("active_track", engine_mode).upper()
        if auto_translate and trans_q:
            st.caption(f"🌐 Query dịch: \"{trans_q}\" | ⏱️ Độ trễ: {lat:.3f}s | 📍 Track: {active_t}")

        # Show deep diagnostic info
        aspect_prompts = intent.get("aspect_prompts", {})
        eff_asr = intent.get("effective_audio_weight", 0.0)
        phases_list = intent.get("phases_en", [])
        top1_item = results[0] if results else {}

        with st.expander(f"🔍 Bảng Chẩn đoán Kỹ thuật (Diagnostics) | Track [{active_t}] | Top-1 Tier: {top1_item.get('tier', 'N/A')} | CSR: {top1_item.get('csr', 1.0):.2f}", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**1. Phân rã 6D Semantic Aspect Prompts:**")
                if isinstance(aspect_prompts, dict):
                    for k, v in aspect_prompts.items():
                        st.text(f"  [{k.upper()}] {v}")
                if intent.get("anchors"):
                    st.markdown(f"**2. Hard OCR Anchors:** `{', '.join(intent['anchors'])}`")
            with c2:
                if phases_list and len(phases_list) > 1:
                    st.markdown("**3. Chuỗi thời gian (Monotonic DP Phases):**")
                    for pi, p_txt in enumerate(phases_list):
                        st.text(f"  Pha {pi+1}: {p_txt}")
                st.markdown(f"**4. Lời thoại ASR:** `{'Bật (' + str(eff_asr) + ')' if eff_asr > 0 else 'Tắt'}`")

        with st.expander("📥 Xuất file nộp bài (Submission CSV)", expanded=False):
            c1, c2 = st.columns([3, 1])
            with c1:
                sub_fn = st.text_input("Tên file CSV:", value=st.session_state.get("kis_sub_name", default_sub_name))
            with c2:
                st.write("")
                st.write("")
                if st.button("Xuất CSV", use_container_width=True):
                    p = exporter.export_kis(results, sub_fn, max_rows=100)
                    st.success(f"Đã lưu: {p}")

        st.markdown(f"### Kết quả Top {len(results)} khung hình")
        grid = st.columns(cols_count)
        for i, item in enumerate(results):
            col = grid[i % cols_count]
            with col:
                st.markdown(f"**#{i+1}. {item['video']} F{item['frame']}**")
                if item["image_path"] and os.path.exists(item["image_path"]):
                    try:
                        st.image(Image.open(item["image_path"]), use_container_width=True)
                    except Exception as e:
                        st.text(f"(image error: {e})")
                else:
                    st.text(f"No image: {item['video']} F{item['frame']}")
                tier_info = f"[{item.get('tier', 'TIER_4')}] "
                st.caption(f"{tier_info}Score: {item.get('score', 0.0):.2f} (t={item.get('pts_time', 0.0):.1f}s)")

# =========================================================================
# QA
# =========================================================================
elif query_mode == "Visual QA":
    st.subheader("Visual Question Answering (QA)")

    selected_file = st.selectbox("Tải file đề thi:", ["(Manual input)"] + [f for f in test_files if "qa" in f.lower()])
    default_query = ""
    default_sub_name = "query-qa.csv"
    if selected_file != "(Manual input)":
        try:
            with open(selected_file, "r", encoding="utf-8") as f:
                default_query = f.read().strip()
            default_sub_name = os.path.splitext(os.path.basename(selected_file))[0] + ".csv"
        except Exception as e:
            st.error(f"Error: {e}")

    qa_text = st.text_area("Câu hỏi (Question):", value=default_query, height=100)

    if st.button("🚀 Bắt đầu Tìm kiếm & Trả lời", type="primary", use_container_width=True) and qa_text.strip():
        with st.spinner("Đang tìm kiếm bằng chứng và chuẩn hóa đáp án..."):
            results, trans_q, intent = engine.search(
                qa_text.strip(),
                top_k=top_k,
                video_filter=v_filter,
                auto_translate=auto_translate,
                engine_mode=engine_mode,
                use_ai_query=(engine_mode in ["meta", "hybrid", "ai"]),
                use_asr=use_asr,
                audio_weight=audio_weight,
                asr_keywords=custom_asr_kws if custom_asr_kws.strip() else None,
                diversity_top_2=diversity_top_2
            )
            st.session_state["qa_results"] = results
            st.session_state["qa_text"] = qa_text.strip()
            st.session_state["qa_sub_name"] = default_sub_name

    if "qa_results" in st.session_state and st.session_state["qa_results"]:
        results = st.session_state["qa_results"]
        qa_q = st.session_state.get("qa_text", "")

        top_ans = engine.answer_qa(qa_q, results[0])
        st.info(f"💡 Đáp án dự đoán chuẩn hóa (Top-1 Answer): **{top_ans}**")

        with st.expander("📥 Xuất file nộp bài (Submission CSV)", expanded=False):
            c1, c2 = st.columns([3, 1])
            with c1:
                sub_fn = st.text_input("Tên file CSV:", value=st.session_state.get("qa_sub_name", default_sub_name))
            with c2:
                st.write("")
                st.write("")
                if st.button("Xuất CSV", use_container_width=True):
                    qa_data = [{"video": r["video"], "frame": r["frame"], "answer": engine.answer_qa(qa_q, r)} for r in results]
                    p = exporter.export_qa(qa_data, sub_fn, max_rows=100)
                    st.success(f"Đã lưu: {p}")

        grid = st.columns(cols_count)
        for i, item in enumerate(results):
            col = grid[i % cols_count]
            with col:
                st.markdown(f"**#{i+1}. {item['video']} F{item['frame']}**")
                if item["image_path"] and os.path.exists(item["image_path"]):
                    try:
                        st.image(Image.open(item["image_path"]), use_container_width=True)
                    except Exception as e:
                        st.text(f"(image error: {e})")
                else:
                    st.text(f"No image")
                tier_info = f"[{item.get('tier', 'TIER_4')}] "
                st.caption(f"{tier_info}Score: {item.get('score', 0.0):.2f}")

# =========================================================================
# TRAKE
# =========================================================================
elif query_mode == "TRAKE":
    st.subheader("Temporal Action Keyframe Extraction (TRAKE)")

    selected_file = st.selectbox("Tải file đề thi:", ["(Manual input)"] + [f for f in test_files if "trake" in f.lower()])
    default_query = ""
    default_sub_name = "query-trake.csv"
    if selected_file != "(Manual input)":
        try:
            with open(selected_file, "r", encoding="utf-8") as f:
                default_query = f.read().strip()
            default_sub_name = os.path.splitext(os.path.basename(selected_file))[0] + ".csv"
        except Exception as e:
            st.error(f"Error: {e}")

    trake_text = st.text_area("Danh sách sự kiện (E1, E2, ...):", value=default_query, height=150)

    if st.button("🚀 Bắt đầu Định vị Sự kiện", type="primary", use_container_width=True) and trake_text.strip():
        lines = [l.strip() for l in trake_text.split("\n") if l.strip()]
        event_queries = [
            re.sub(r"^E\d+[:.]\s*", "", l, flags=re.IGNORECASE)
            for l in lines if re.match(r"^E\d+[:.]", l, re.IGNORECASE)
        ]
        if not event_queries:
            event_queries = lines[1:] if len(lines) > 1 else lines

        with st.spinner("Đang thực thi 3-Stage Temporal Refinement..."):
            trake_res, proc_events = engine.search_trake(
                event_queries,
                top_k_videos=10,
                video_filter=v_filter,
                auto_translate=auto_translate,
                use_ai_query=(engine_mode in ["meta", "hybrid", "ai"])
            )
            st.session_state["trake_results"] = trake_res
            st.session_state["trake_sub_name"] = default_sub_name

    if "trake_results" in st.session_state and st.session_state["trake_results"]:
        trake_res = st.session_state["trake_results"]

        with st.expander("📥 Xuất file nộp bài (Submission CSV)", expanded=False):
            c1, c2 = st.columns([3, 1])
            with c1:
                sub_fn = st.text_input("Tên file CSV:", value=st.session_state.get("trake_sub_name", default_sub_name))
            with c2:
                st.write("")
                st.write("")
                if st.button("Xuất CSV", use_container_width=True):
                    p = exporter.export_trake(trake_res, sub_fn, max_rows=100)
                    st.success(f"Đã lưu: {p}")

        for r_idx, v_item in enumerate(trake_res):
            if not v_item.get("frames"):
                continue
            st.markdown(f"#### #{r_idx+1}. {v_item['video']} (DP Score: {v_item['total_score']:.3f} | C_temp: {v_item.get('temporal_confidence', 0.9):.2f})")
            t_cols = st.columns(len(v_item["frames"]))
            for f_idx, f_data in enumerate(v_item["frames"]):
                with t_cols[f_idx]:
                    st.markdown(f"**E{f_idx+1}: F{f_data['frame']}** (t={f_data['pts_time']:.1f}s)")
                    if f_data["image_path"] and os.path.exists(f_data["image_path"]):
                        try:
                            st.image(Image.open(f_data["image_path"]), use_container_width=True)
                        except Exception as e:
                            st.text(f"(image error: {e})")
                    else:
                        st.text(f"F{f_data['frame']}")
            st.divider()

# =========================================================================
# BATCH
# =========================================================================
elif query_mode == "Batch Processing":
    st.subheader("Xử lý hàng loạt toàn bộ bộ đề thi (Batch Processing)")

    batch_dir = st.text_input("Thư mục đề thi:", value="de-thi" if os.path.exists("de-thi") else "THUNGHIEM-bo-de-thi")

    if st.button("🚀 Chạy Batch toàn bộ", type="primary", use_container_width=True):
        if not os.path.exists(batch_dir):
            st.error(f"Thư mục không tồn tại: {batch_dir}")
        else:
            q_files = sorted(glob.glob(os.path.join(batch_dir, "*.txt")))
            st.write(f"Tìm thấy {len(q_files)} file đề thi.")
            progress = st.progress(0.0)
            status = st.empty()

            from main_pipeline import process_query_file

            for i, qf in enumerate(q_files):
                status.text(f"Đang xử lý ({i+1}/{len(q_files)}): {os.path.basename(qf)}")
                process_query_file(
                    qf, engine, exporter,
                    top_k=top_k,
                    engine_mode=engine_mode,
                    use_ai_query=(engine_mode in ["meta", "hybrid", "ai"]),
                    use_asr=use_asr,
                    audio_weight=audio_weight
                )
                progress.progress((i + 1) / len(q_files))

            zip_path = exporter.zip_submissions("submission.zip")
            st.success(f"Hoàn tất! Đã lưu toàn bộ CSV và đóng gói ZIP: {zip_path}")
