import os
import glob
import re
import pandas as pd
import streamlit as st
from PIL import Image

from core.retrieval_engine import RetrievalEngine
from core.submission_exporter import SubmissionExporter

st.set_page_config(
    page_title="AIC Video Retrieval System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1E88E5 0%, #7E57C2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.2rem;
    }
    .no-img-box {
        height: 180px;
        background: #f8f9fa;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #6c757d;
        font-size: 0.85rem;
        border: 1px dashed #ced4da;
        text-align: center;
        padding: 8px;
    }
    .badge-gpu {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-cpu {
        background-color: #fff3e0;
        color: #e65100;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Đang nạp Model CLIP và Database Vector...")
def get_engine():
    return RetrievalEngine()

engine = get_engine()
exporter = SubmissionExporter()
extracted_videos = engine.get_extracted_videos()

# Sidebar
st.sidebar.markdown("## ⚙️ Bảng Điều Khiển")
query_mode = st.sidebar.radio(
    "Chọn chức năng:",
    [
        "🔍 Textual KIS (Tìm kiếm văn bản)",
        "❓ Visual Q&A (Hỏi - Đáp)",
        "⏱️ TRAKE (Chuỗi sự kiện video)",
        "⚡ Batch Processing (Xử lý hàng loạt)",
        "📥 Tải dữ liệu từ Google Drive (gdown)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Tùy chọn nâng cao")
auto_translate = st.sidebar.checkbox("🌐 Tự động dịch Tiếng Việt -> Tiếng Anh (Khuyên dùng)", value=True, help="OpenAI CLIP học trên tiếng Anh nên dịch sang tiếng Anh sẽ nâng độ chính xác lên gấp nhiều lần.")
filter_extracted = st.sidebar.checkbox(f"🖼️ Chỉ tìm video đã có ảnh trên máy ({len(extracted_videos)} video)", value=False, help="Chỉ tìm kiếm trong các video mà bạn đã tải/giải nén thư mục Keyframes về máy để luôn thấy ảnh.")

top_k = st.sidebar.slider("Số lượng kết quả (Top K):", min_value=10, max_value=100, value=50, step=10)
cols_count = st.sidebar.slider("Số cột hiển thị ảnh:", min_value=2, max_value=6, value=4, step=1)

st.sidebar.markdown("---")
if engine.device == "cuda":
    st.sidebar.markdown('**Thiết bị xử lý:** <span class="badge-gpu">GPU (CUDA)</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('**Thiết bị xử lý:** <span class="badge-cpu">CPU</span> (RTX 3050 có sẵn trên máy)', unsafe_allow_html=True)

st.sidebar.markdown(f"**Tổng số Keyframes:** `{len(engine.keyframe_map):,}`")
st.sidebar.markdown(f"**Video có sẵn ảnh:** `{len(extracted_videos)} video`")

# Header
st.markdown('<div class="main-header">AIC Video Retrieval & Submission Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hệ thống tìm kiếm khung hình bán tự động (Interactive / Semi-Auto) & Xuất kết quả nộp bài chuẩn BTC</div>', unsafe_allow_html=True)

test_files = sorted(glob.glob(os.path.join("THUNGHIEM-bo-de-thi", "*.txt"))) + sorted(glob.glob("*.txt"))
test_files = [f for f in test_files if "requirement" not in f.lower()]

v_filter = list(extracted_videos) if filter_extracted else None


# =========================================================================
# 1. TEXTUAL KIS
# =========================================================================
if query_mode == "🔍 Textual KIS (Tìm kiếm văn bản)":
    st.subheader("🎯 Textual Known Item Search (Textual KIS)")
    st.info("Tìm kiếm chính xác khung hình theo mô tả văn bản. Format nộp bài: `<video_name>, <frame_idx>`")

    col_select, col_custom = st.columns([1, 2])
    with col_select:
        selected_file = st.selectbox(
            "Chọn file đề thi mẫu (nếu có):",
            ["(Nhập thủ công)"] + [f for f in test_files if "kis" in f.lower() or "txt" in f.lower()]
        )

    default_query = ""
    default_sub_name = "query-1-kis.csv"
    if selected_file != "(Nhập thủ công)":
        try:
            with open(selected_file, "r", encoding="utf-8") as f:
                default_query = f.read().strip()
            base_name = os.path.splitext(os.path.basename(selected_file))[0]
            default_sub_name = f"{base_name}.csv"
        except Exception as e:
            st.error(f"Lỗi đọc file: {e}")

    query_text = st.text_area(
        "📝 Câu truy vấn mô tả khung hình cần tìm:",
        value=default_query,
        placeholder="Ví dụ: Đây là phần giới thiệu việc phóng tàu vũ trụ tư nhân...",
        height=100
    )

    btn_search = st.button("🚀 Bắt đầu tìm kiếm KIS", type="primary", use_container_width=True)

    if btn_search and query_text.strip():
        with st.spinner("Đang xử lý và tính Cosine Similarity trên toàn bộ vector database..."):
            results, translated_q = engine.search(
                query_text.strip(),
                top_k=top_k,
                video_filter=v_filter,
                auto_translate=auto_translate
            )
            st.session_state["kis_results"] = results
            st.session_state["kis_trans_q"] = translated_q
            st.session_state["kis_sub_name"] = default_sub_name

    if "kis_results" in st.session_state and st.session_state["kis_results"]:
        results = st.session_state["kis_results"]
        trans_q = st.session_state.get("kis_trans_q", "")

        if auto_translate and trans_q and trans_q != query_text.strip():
            st.success(f"🌐 **Câu truy vấn tiếng Anh dịch tự động:** *\"{trans_q}\"*")

        st.markdown(f"### 🖼️ Kết quả Top {len(results)} khung hình tương đồng nhất:")

        # Bảng điều khiển xuất file nộp
        with st.expander("📁 Tùy chọn xuất file nộp bài (Bước 8 - Submission CSV)", expanded=True):
            exp_col1, exp_col2 = st.columns([3, 1])
            with exp_col1:
                sub_filename = st.text_input("Tên file nộp bài:", value=st.session_state.get("kis_sub_name", default_sub_name))
            with exp_col2:
                st.write("")
                st.write("")
                if st.button("💾 Xuất toàn bộ Top vào CSV", type="primary", use_container_width=True):
                    saved_path = exporter.export_kis(results, sub_filename, max_rows=100)
                    st.success(f"✅ Đã tạo file nộp bài: `{saved_path}` (Tối đa 100 dòng)")

        # Grid ảnh
        grid_cols = st.columns(cols_count)
        for i, item in enumerate(results):
            col = grid_cols[i % cols_count]
            with col:
                st.markdown(f"**#{i+1}. 🎬 {item['video']} | Frame {item['frame']}**")

                if item["image_path"] and os.path.exists(item["image_path"]):
                    try:
                        img = Image.open(item["image_path"])
                        st.image(img, use_container_width=True)
                    except Exception:
                        st.markdown('<div class="no-img-box">Lỗi load ảnh</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="no-img-box">Ảnh chưa giải nén<br><b>{item["video"]}</b><br>Frame #{item["frame"]}</div>', unsafe_allow_html=True)

                score_pct = item["score"] * 100
                st.progress(max(0.0, min(1.0, float(item["score"]))))
                st.caption(f"Cosine Score: **{score_pct:.2f}%**")


# =========================================================================
# 2. VISUAL QUESTION ANSWERING (Q&A)
# =========================================================================
elif query_mode == "❓ Visual Q&A (Hỏi - Đáp)":
    st.subheader("❓ Visual Question Answering (Q&A)")
    st.info("Tìm khung hình liên quan và trả lời câu hỏi đề bài. Format: `<video_name>, <frame_idx>, <answer>` (answer <= 100 ký tự)")

    col_select, col_custom = st.columns([1, 2])
    with col_select:
        selected_file = st.selectbox(
            "Chọn file đề thi Q&A mẫu:",
            ["(Nhập thủ công)"] + [f for f in test_files if "qa" in f.lower() or "txt" in f.lower()]
        )

    default_query = ""
    default_sub_name = "query-3-qa.csv"
    if selected_file != "(Nhập thủ công)":
        try:
            with open(selected_file, "r", encoding="utf-8") as f:
                default_query = f.read().strip()
            base_name = os.path.splitext(os.path.basename(selected_file))[0]
            default_sub_name = f"{base_name}.csv"
        except Exception as e:
            st.error(f"Lỗi đọc file: {e}")

    qa_query_text = st.text_area(
        "📝 Câu hỏi & Mô tả ngữ cảnh Q&A:",
        value=default_query,
        placeholder="Ví dụ: Đoạn video về một chương trình từ thiện của một câu lạc bộ tên là FANA...",
        height=100
    )

    btn_search_qa = st.button("🚀 Tìm kiếm Keyframe cho câu hỏi Q&A", type="primary", use_container_width=True)

    if btn_search_qa and qa_query_text.strip():
        with st.spinner("Đang tìm kiếm keyframe..."):
            results, trans_q = engine.search(
                qa_query_text.strip(),
                top_k=top_k,
                video_filter=v_filter,
                auto_translate=auto_translate
            )
            st.session_state["qa_results"] = results
            st.session_state["qa_trans_q"] = trans_q
            st.session_state["qa_sub_name"] = default_sub_name

    if "qa_results" in st.session_state and st.session_state["qa_results"]:
        results = st.session_state["qa_results"]
        trans_q = st.session_state.get("qa_trans_q", "")

        if auto_translate and trans_q and trans_q != qa_query_text.strip():
            st.success(f"🌐 **Câu truy vấn tiếng Anh dịch tự động:** *\"{trans_q}\"*")

        st.markdown(f"### 🖼️ Các khung hình ứng viên hàng đầu:")

        # Khu vực nhập câu trả lời
        st.markdown("#### ✍️ Nhập câu trả lời (Answer):")
        ans_col1, ans_col2, ans_col3 = st.columns([2, 1, 1])
        with ans_col1:
            qa_answer = st.text_input("Nội dung câu trả lời (Tối đa 100 ký tự):", value="yes", placeholder="Nhập câu trả lời...")
            st.caption(f"Độ dài: {len(qa_answer)}/100 ký tự")
        with ans_col2:
            qa_sub_filename = st.text_input("Tên file nộp bài Q&A:", value=st.session_state.get("qa_sub_name", default_sub_name))
        with ans_col3:
            st.write("")
            st.write("")
            if st.button("💾 Xuất CSV Q&A", type="primary", use_container_width=True):
                qa_submission_data = []
                for item in results:
                    qa_submission_data.append({
                        "video": item["video"],
                        "frame": item["frame"],
                        "answer": qa_answer.strip()
                    })
                saved_path = exporter.export_qa(qa_submission_data, qa_sub_filename, max_rows=100)
                st.success(f"✅ Đã xuất file Q&A: `{saved_path}`")

        # Grid ảnh
        grid_cols = st.columns(cols_count)
        for i, item in enumerate(results):
            col = grid_cols[i % cols_count]
            with col:
                st.markdown(f"**#{i+1}. {item['video']} | Frame {item['frame']}**")
                if item["image_path"] and os.path.exists(item["image_path"]):
                    try:
                        img = Image.open(item["image_path"])
                        st.image(img, use_container_width=True)
                    except Exception:
                        st.markdown('<div class="no-img-box">Lỗi load ảnh</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="no-img-box">Ảnh chưa giải nén<br><b>{item["video"]}</b><br>Frame #{item["frame"]}</div>', unsafe_allow_html=True)

                st.caption(f"Score: **{item['score']*100:.2f}%**")


# =========================================================================
# 3. TRAKE (TEMPORAL RETRIEVAL & ALIGNMENT OF KEY EVENTS)
# =========================================================================
elif query_mode == "⏱️ TRAKE (Chuỗi sự kiện video)":
    st.subheader("⏱️ TRAKE: Chuỗi sự kiện video theo thời gian")
    st.info("Tìm chuỗi N keyframe trong cùng 1 video tương ứng với N events và tuân theo thứ tự thời gian: `Frame_1 < Frame_2 < ... < Frame_N`. Format: `<video_name>, <frame_1>, <frame_2>, ..., <frame_N>`")

    col_select, col_custom = st.columns([1, 2])
    with col_select:
        selected_file = st.selectbox(
            "Chọn file đề thi TRAKE mẫu:",
            ["(Nhập thủ công)"] + [f for f in test_files if "trake" in f.lower() or "txt" in f.lower()]
        )

    default_trake_text = ""
    default_sub_name = "query-4-trake.csv"
    if selected_file != "(Nhập thủ công)":
        try:
            with open(selected_file, "r", encoding="utf-8") as f:
                default_trake_text = f.read().strip()
            base_name = os.path.splitext(os.path.basename(selected_file))[0]
            default_sub_name = f"{base_name}.csv"
        except Exception as e:
            st.error(f"Lỗi đọc file: {e}")

    trake_input = st.text_area(
        "📝 Nội dung đề bài TRAKE (Gồm mô tả chung và các dòng E1, E2, E3...):",
        value=default_trake_text,
        placeholder="Đoạn video múa lân...\nE1: Lân quay vòng trên cột...\nE2: Khoảnh khắc 4 chân chạm đất...\nE3: 2 người biểu diễn cúi chào...",
        height=150
    )

    btn_search_trake = st.button("🚀 Bắt đầu căn chỉnh chuỗi sự kiện (TRAKE Search)", type="primary", use_container_width=True)

    if btn_search_trake and trake_input.strip():
        lines = [line.strip() for line in trake_input.strip().split("\n") if line.strip()]
        event_queries = []
        for line in lines:
            if re.match(r"^E\d+[:.]", line, re.IGNORECASE):
                content = re.sub(r"^E\d+[:.]\s*", "", line, flags=re.IGNORECASE)
                event_queries.append(content)
            elif len(lines) > 1 and line != lines[0]:
                event_queries.append(line)
        
        if not event_queries:
            event_queries = lines

        st.write(f"📌 Đã nhận diện **{len(event_queries)} events**:")
        for idx, eq in enumerate(event_queries):
            st.markdown(f"- **Event {idx+1}:** {eq}")

        with st.spinner("Đang tìm kiếm và căn chỉnh chuỗi thời gian..."):
            trake_results, trans_events = engine.search_trake(
                event_queries,
                top_k_videos=10,
                video_filter=v_filter,
                auto_translate=auto_translate
            )
            st.session_state["trake_results"] = trake_results
            st.session_state["trake_sub_name"] = default_sub_name

    if "trake_results" in st.session_state and st.session_state["trake_results"]:
        trake_results = st.session_state["trake_results"]

        st.markdown("### 🎬 Danh sách Video & Chuỗi Keyframe phù hợp:")

        with st.expander("📁 Xuất file nộp bài TRAKE", expanded=True):
            sub_col1, sub_col2 = st.columns([3, 1])
            with sub_col1:
                trake_sub_filename = st.text_input("Tên file CSV:", value=st.session_state.get("trake_sub_name", default_sub_name))
            with sub_col2:
                st.write("")
                st.write("")
                if st.button("💾 Xuất CSV TRAKE", type="primary", use_container_width=True):
                    saved_path = exporter.export_trake(trake_results, trake_sub_filename, max_rows=100)
                    st.success(f"✅ Đã tạo file TRAKE: `{saved_path}`")

        for r_idx, res in enumerate(trake_results):
            vname = res["video"]
            frames = res["frames"]
            st.markdown(f"#### #{r_idx+1}. Video: `{vname}` (Tổng điểm: {res['total_score']:.3f})")
            
            f_cols = st.columns(len(frames))
            for e_idx, f_info in enumerate(frames):
                with f_cols[e_idx]:
                    st.markdown(f"**Event {e_idx+1} | Frame {f_info['frame']}**")
                    if f_info["image_path"] and os.path.exists(f_info["image_path"]):
                        try:
                            img = Image.open(f_info["image_path"])
                            st.image(img, use_container_width=True)
                        except Exception:
                            st.markdown('<div class="no-img-box">Lỗi load ảnh</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="no-img-box">Ảnh chưa giải nén<br><b>{vname}</b><br>Frame #{f_info["frame"]}</div>', unsafe_allow_html=True)
                    st.caption(f"Score: {f_info['score']*100:.1f}%")
            st.divider()


# =========================================================================
# 4. BATCH PROCESSING / AUTO RUNNER
# =========================================================================
elif query_mode == "⚡ Batch Processing (Xử lý hàng loạt)":
    st.subheader("⚡ Xử lý tự động toàn bộ gói đề thi (Batch Runner)")
    st.info("Tự động duyệt tất cả các file .txt trong thư mục đề thi, phân loại (KIS / QA / TRAKE) và xuất toàn bộ file .csv vào thư mục `submission/` theo đúng quy chuẩn Bước 8.")

    dir_input = st.text_input("Thư mục chứa đề thi:", value="THUNGHIEM-bo-de-thi")

    if st.button("🔥 Chạy toàn bộ đề thi & Xuất file Submission", type="primary"):
        if not os.path.exists(dir_input):
            st.error(f"Không tìm thấy thư mục: {dir_input}")
        else:
            txt_files = sorted(glob.glob(os.path.join(dir_input, "*.txt")))
            st.write(f"Tìm thấy **{len(txt_files)} file** đề thi.")

            progress_bar = st.progress(0.0)
            summary_list = []

            for i, txt_path in enumerate(txt_files):
                fname = os.path.basename(txt_path)
                base_name = os.path.splitext(fname)[0]
                csv_name = f"{base_name}.csv"

                with open(txt_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                if "trake" in fname.lower():
                    lines = [l.strip() for l in content.split("\n") if l.strip()]
                    eqs = [re.sub(r"^E\d+[:.]\s*", "", l, flags=re.IGNORECASE) for l in lines if re.match(r"^E\d+[:.]", l, re.IGNORECASE)]
                    if not eqs:
                        eqs = lines
                    trake_res, _ = engine.search_trake(eqs, top_k_videos=100, auto_translate=auto_translate)
                    out_path = exporter.export_trake(trake_res, csv_name)
                    summary_list.append({"File đề": fname, "Loại": "TRAKE", "Số dòng": len(trake_res), "File CSV": csv_name})

                elif "qa" in fname.lower():
                    res, _ = engine.search(content, top_k=100, auto_translate=auto_translate)
                    qa_data = [{"video": r["video"], "frame": r["frame"], "answer": "yes"} for r in res]
                    out_path = exporter.export_qa(qa_data, csv_name)
                    summary_list.append({"File đề": fname, "Loại": "Q&A", "Số dòng": len(qa_data), "File CSV": csv_name})

                else:
                    res, _ = engine.search(content, top_k=100, auto_translate=auto_translate)
                    out_path = exporter.export_kis(res, csv_name)
                    summary_list.append({"File đề": fname, "Loại": "KIS", "Số dòng": len(res), "File CSV": csv_name})

                progress_bar.progress((i + 1) / len(txt_files))

            st.success(f"🎉 Hoàn thành xử lý {len(txt_files)} câu hỏi!")
            st.dataframe(pd.DataFrame(summary_list))


# =========================================================================
# 5. GOOGLE DRIVE DOWNLOADER
# =========================================================================
elif query_mode == "📥 Tải dữ liệu từ Google Drive (gdown)":
    st.subheader("📥 Tải dữ liệu & Keyframes từ Google Drive")
    st.info("Nhập link Google Drive chứa file Keyframes (.zip) hoặc file CLIP Features do BTC cung cấp. Hệ thống sẽ tự động tải về và giải nén trực tiếp vào thư mục dự án.")

    drive_url = st.text_input("🔗 Nhập Link chia sẻ hoặc File ID từ Google Drive:", placeholder="https://drive.google.com/file/d/...")
    
    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        custom_out = st.text_input("Tên file lưu lại (để trống nếu lấy tên gốc):", placeholder="Ví dụ: Keyframes_L22.zip")
    with col_d2:
        auto_unzip_opt = st.checkbox("📦 Tự động giải nén file .zip sau khi tải xong", value=True)

    if st.button("🚀 Bắt đầu tải từ Google Drive", type="primary"):
        if not drive_url.strip():
            st.warning("Vui lòng nhập link Google Drive hợp lệ!")
        else:
            try:
                from utils.download_from_drive import download_file_from_drive
                with st.spinner("Đang kết nối và tải file từ Google Drive (gdown)..."):
                    res_path = download_file_from_drive(
                        drive_url.strip(),
                        output_path=custom_out.strip() if custom_out.strip() else None,
                        auto_unzip=auto_unzip_opt
                    )
                st.success(f"✅ Tải thành công! Dữ liệu đã lưu tại: `{res_path}`")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Lỗi khi tải từ Google Drive: {e}")

