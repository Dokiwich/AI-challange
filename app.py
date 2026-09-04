import os
import glob
import re
import time
import streamlit as st
from PIL import Image

from core.retrieval_engine import RetrievalEngine
from core.submission_exporter import SubmissionExporter
from core.query_compiler import QueryCompiler
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AIC 2026 Video Retrieval", layout="wide", initial_sidebar_state="expanded")

_ENGINE_VERSION = 17
if st.session_state.get("_engine_ver") != _ENGINE_VERSION:
    for k in [k for k in st.session_state if k.startswith(("kis_", "qa_", "trake_"))]:
        del st.session_state[k]
    st.session_state["_engine_ver"] = _ENGINE_VERSION
    st.session_state["submission_cart"] = {}

if "submission_cart" not in st.session_state:
    st.session_state["submission_cart"] = {}

@st.cache_resource(show_spinner="Loading CLIP model and 4-Track Vector Database...")
def get_engine(version=_ENGINE_VERSION):
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

    try:
        qclient = QdrantClient(qdrant_host, port=qdrant_port)
        qclient.get_collections()
        logger.info(f"Connected to Qdrant at {qdrant_host}:{qdrant_port}")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant at {qdrant_host}:{qdrant_port}. Error: {e}")
        qclient = None
        
    try:
        ndriver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        ndriver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {neo4j_uri}")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j at {neo4j_uri}. Error: {e}")
        ndriver = None
        
    return RetrievalEngine(qdrant_client=qclient, neo4j_driver=ndriver)

engine = get_engine()
exporter = SubmissionExporter()
extracted_videos = engine.get_extracted_videos()

def get_current_rank(video, frame, event_idx=0, is_trake=False):
    cart = st.session_state.get("submission_cart", {})
    for r, item in cart.items():
        if is_trake:
            if item["video"] == video and item.get("frames", {}).get(event_idx) == frame:
                return r
        else:
            if item["video"] == video and item.get("frame") == frame:
                return r
    return 0

def update_cart(video, frame, target_rank: int, event_idx=0, is_trake=False):
    cart = st.session_state.get("submission_cart", {})
    curr_rank = get_current_rank(video, frame, event_idx, is_trake)
    
    if target_rank == 0:
        if curr_rank > 0:
            if is_trake:
                if event_idx in cart[curr_rank].get("frames", {}):
                    del cart[curr_rank]["frames"][event_idx]
                if not cart[curr_rank]["frames"]:
                    del cart[curr_rank]
            else:
                del cart[curr_rank]
            st.toast(f"🗑️ Đã xóa F{frame} khỏi giỏ.")
        else:
            next_r = 1
            if cart: next_r = max(cart.keys()) + 1
            if next_r > 100: next_r = 100
            
            if is_trake:
                cart[next_r] = {"video": video, "frames": {event_idx: frame}, "score": 1.0}
            else:
                cart[next_r] = {"video": video, "frame": frame, "score": 1.0}
            st.toast(f"✅ Đã thêm F{frame} vào Top {next_r}")
    else:
        if curr_rank > 0 and curr_rank != target_rank:
            if is_trake:
                if event_idx in cart[curr_rank].get("frames", {}):
                    del cart[curr_rank]["frames"][event_idx]
                if not cart[curr_rank]["frames"]:
                    del cart[curr_rank]
            else:
                del cart[curr_rank]
                
        if is_trake:
            if target_rank in cart:
                if cart[target_rank]["video"] != video:
                    st.toast(f"⚠️ Top {target_rank} bị đổi Video: {cart[target_rank]['video']} ➔ {video}!")
                    cart[target_rank] = {"video": video, "frames": {event_idx: frame}, "score": 1.0}
                else:
                    if event_idx in cart[target_rank].get("frames", {}) and cart[target_rank]["frames"][event_idx] != frame:
                        st.toast(f"⚠️ Event {event_idx+1} của Top {target_rank} bị ghi đè!")
                    if "frames" not in cart[target_rank]: cart[target_rank]["frames"] = {}
                    cart[target_rank]["frames"][event_idx] = frame
            else:
                cart[target_rank] = {"video": video, "frames": {event_idx: frame}, "score": 1.0}
        else:
            if target_rank in cart and (cart[target_rank]["video"] != video or cart[target_rank].get("frame") != frame):
                st.toast(f"⚠️ Vị trí Top {target_rank} đã bị ghi đè!")
            cart[target_rank] = {"video": video, "frame": frame, "score": 1.0}
        
    st.session_state["submission_cart"] = cart

def render_pick_ui(unique_key: str, video: str, frame: int, event_idx=0, is_trake=False):
    c1, c2 = st.columns([2, 1])
    curr_r = get_current_rank(video, frame, event_idx, is_trake)
    
    with c1:
        rank_val = st.number_input("Hạng", min_value=0, max_value=100, value=curr_r, label_visibility="collapsed", key=f"num_{unique_key}")
    with c2:
        btn_label = "OK" if curr_r == 0 else "Sửa/Xóa"
        if st.button(btn_label, key=f"btn_{unique_key}", use_container_width=True):
            update_cart(video, frame, rank_val, event_idx, is_trake)
            st.rerun()

import io

@st.cache_data(show_spinner=False, max_entries=2000)
def load_cached_thumbnail(image_path: str, size: int = 600) -> bytes:
    try:
        img = Image.open(image_path)
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    except Exception:
        return b""

def render_temporal_browsing(unique_key: str, video_name: str, center_frame: int):
    if st.button("[🔎] Duyệt lân cận", key=f"btn_browse_{unique_key}", use_container_width=True):
        st.session_state["browsing_video"] = video_name
        st.session_state["browsing_center_frame"] = center_frame
        st.rerun()

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
st.sidebar.markdown("### 🤖 Chế độ AI (AI Mode)")

# NÚT BẬT / TẮT MODE AI CHÍNH
enable_ai_mode = st.sidebar.toggle(
    "⚡ Kích hoạt Chế độ AI (OpenRouter / LLM)",
    value=st.session_state.get("enable_ai_mode", True),
    key="enable_ai_mode_toggle",
    help="Bật để AI phân tích cấu trúc 6D ngữ nghĩa, bóc tách hành động cốt lõi, sinh giả thuyết và định tuyến thông minh."
)
st.session_state["enable_ai_mode"] = enable_ai_mode

if enable_ai_mode:
    st.sidebar.markdown("#### 🏎️ Chọn AI Engine Policy:")
    track_choice = st.sidebar.selectbox(
        "Engine Track:",
        [
            "🧠 Track 4: Adaptive Meta-Policy (Khuyên dùng)",
            "🌟 Track 3: Hybrid Fusion (Late Retrieval)",
            "🤖 Track 2: AI Semantic V2 (Thuần LLM)"
        ],
        index=0,
        help="Track 4 tự động định tuyến thông minh tối ưu tốc độ và độ chính xác cho từng loại câu hỏi."
    )
    track_mode_map = {
        "🧠 Track 4: Adaptive Meta-Policy (Khuyên dùng)": "meta",
        "🌟 Track 3: Hybrid Fusion (Late Retrieval)": "hybrid",
        "🤖 Track 2: AI Semantic V2 (Thuần LLM)": "ai"
    }
    engine_mode = track_mode_map[track_choice]

    col_status, col_btn = st.sidebar.columns([3, 1])
    force_check = False
    with col_btn:
        if st.button("🔄", help="Kiểm tra lại kết nối OpenRouter / LLM"):
            force_check = True

    ai_connected = engine.compiler.ai_parser.check_connection(force=force_check)
    with col_status:
        st.caption(f"Trạng thái: {'🟢 Online (OpenRouter)' if ai_connected else '🔴 Offline / Fallback Cục bộ'}")

    if ai_connected:
        available_models = engine.compiler.ai_parser.get_available_models()
        curr_model = engine.compiler.ai_parser.model_name
        default_idx = available_models.index(curr_model) if curr_model in available_models else 0
        selected_model = st.sidebar.selectbox("AI Model:", available_models, index=default_idx)
        engine.compiler.ai_parser.model_name = selected_model
    else:
        st.sidebar.caption("⚠️ Không kết nối được OpenRouter. Hệ thống sẽ tự động dùng bộ dịch Google và phân tích cục bộ.")
else:
    engine_mode = "offline"
    st.sidebar.info("⚡ **Track 1: Offline V2 (Cục bộ 100% - Zero LLM)** đang bật.\nTốc độ phản hồi cực nhanh, không phụ thuộc vào internet hoặc mô hình LLM.")

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
top_k = st.sidebar.slider("Top K kết quả:", min_value=10, max_value=200, value=100, step=10)
cols_count = st.sidebar.slider("Số cột hiển thị:", min_value=2, max_value=6, value=4, step=1)
use_dino_counting = st.sidebar.checkbox("🦖 Bật Grounding DINO Đếm số lượng", value=True, help="Tự động re-rank ảnh nếu truy vấn có yêu cầu đếm (vd: 4 con cá)")
diversity_top_2 = st.sidebar.checkbox("Lọc đa dạng hóa Top-2 (Diversity)", value=False)

st.sidebar.markdown("---")
st.sidebar.text(f"Device: {engine.device}")
st.sidebar.text(f"Keyframes: {len(engine.keyframe_map):,}")

# --- GIỎ HÀNG ĐÁP ÁN (SUBMISSION CART) ---
st.sidebar.markdown("---")
st.sidebar.markdown("### [Cart] Giỏ hàng (Chốt Hạng)")
cart_items = st.session_state.get("submission_cart", {})
if not cart_items:
    st.sidebar.caption("Chưa chốt hạng nào.")
else:
    for rank in sorted(cart_items.keys()):
        item = cart_items[rank]
        if "frames" in item:
            f_str = ", ".join([f"E{e+1}:F{f}" for e, f in sorted(item["frames"].items())])
            st.sidebar.text(f"🔒 Top {rank}: {item['video']}\n    ↳ {f_str}")
        else:
            st.sidebar.text(f"🔒 Top {rank}: {item['video']} - F{item.get('frame', 0)}")
    
    col_c1, col_c2 = st.sidebar.columns(2)
    with col_c1:
        if st.button("[X] Xóa hết", use_container_width=True):
            st.session_state["submission_cart"] = {}
            st.rerun()
    with col_c2:
        cart_sub_fn = st.text_input("Tên file CSV:", value="my_submission.csv", label_visibility="collapsed")
        if st.button("[Export] Xuất CSV", type="primary", use_container_width=True):
            ai_res = st.session_state.get("kis_results", [])
            p = exporter.export_kis(ai_res, cart_items, cart_sub_fn, max_rows=100)
            st.sidebar.success(f"Đã lưu: {p}")

# =========================================================================
# HEADER
# =========================================================================
st.markdown("# 🎬 AIC 2026 Video Retrieval System")
st.caption("Industrial-Grade Multimodal Video Retrieval with 4-Track Adaptive Meta-Policy & Constraint Graph Judge")

test_files = sorted(glob.glob(os.path.join("de-thi", "*.txt"))) + sorted(glob.glob(os.path.join("THUNGHIEM-bo-de-thi", "*.txt")))
test_files = [f for f in test_files if "requirement" not in f.lower()]
v_filter = list(extracted_videos) if filter_extracted else None

# =========================================================================
# TAB DUYỆT LÂN CẬN (TEMPORAL BROWSING VIEW)
# =========================================================================
if st.session_state.get("browsing_video"):
    v_name = st.session_state["browsing_video"]
    c_frame = st.session_state["browsing_center_frame"]
    
    st.markdown(f"## [🔎] Duyệt lân cận: `{v_name}`")
    st.caption(f"Đang hiển thị các khung hình xung quanh tâm **F{c_frame}**.")
    
    if st.button("🔙 Quay lại kết quả tìm kiếm", type="primary"):
        st.session_state["browsing_video"] = None
        st.session_state["browsing_center_frame"] = None
        st.rerun()
        
    st.divider()
    
    # Cho phép người dùng tùy chỉnh số lượng ảnh muốn duyệt sang 2 bên
    br_radius = st.slider(
        "Phạm vi duyệt (số khung hình sang mỗi bên):", 
        min_value=10, 
        max_value=500, 
        value=120, 
        step=10,
        help="120 khung hình tương đương hiển thị tổng cộng 241 ảnh xung quanh tâm."
    )
    
    adj_items = engine.get_adjacent_keyframes(v_name, c_frame, radius=br_radius)
    if not adj_items:
        st.warning("Không tìm thấy khung hình.")
    else:
        # Render ảnh theo cột (grid 3 cột kéo dọc)
        b_cols = st.columns(3)
        for i, item in enumerate(adj_items):
            with b_cols[i % 3]:
                is_center = (item["frame"] == c_frame)
                mark = "⭐ " if is_center else ""
                border_color = "red" if is_center else "gray"
                st.markdown(f"{mark}**F{item['frame']}** (t={item['pts_time']:.1f}s)")
                
                if item["image_path"] and os.path.exists(item["image_path"]):
                    img_bytes = load_cached_thumbnail(item["image_path"])
                    if img_bytes:
                        st.image(img_bytes, use_container_width=True)
                    else:
                        st.text("Image load error")
                else:
                    st.text("No image")
                    
                render_pick_ui(f"pick_br_{item['frame']}", v_name, item['frame'], is_trake=(query_mode == "TRAKE"))
            
            # Cứ 3 ảnh thì vẽ 1 dải phân cách cho dễ nhìn
            if (i + 1) % 3 == 0:
                st.markdown("---")
                
    st.stop()


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

    # KHU VỰC ĐIỀU KHIỂN MODE AI & TÌM KIẾM
    c_toggle, c_mode_badge = st.columns([2, 2])
    with c_toggle:
        kis_use_ai = st.toggle("🤖 Bật AI Semantic Mode cho truy vấn này", value=enable_ai_mode, key="kis_use_ai_toggle")
    with c_mode_badge:
        if kis_use_ai:
            st.caption(f"Trạng thái: 🤖 **AI Mode ({engine_mode.upper()})** | Model: `{engine.compiler.ai_parser.model_name}`")
        else:
            st.caption("Trạng thái: ⚡ **Offline V2 (Zero-LLM Cục bộ)**")

    c_b1, c_b2, c_b3 = st.columns([3, 3, 2])
    btn_ai_search = c_b1.button("🤖 Tìm kiếm với AI (Deep)", type="primary" if kis_use_ai else "secondary", use_container_width=True)
    btn_fast_search = c_b2.button("⚡ Tìm kiếm Nhanh (Offline)", type="secondary" if kis_use_ai else "primary", use_container_width=True)
    btn_inspect = c_b3.button("🧠 Phân tích AI", use_container_width=True, help="Xem cấu trúc 6D & Event Graph do LLM trích xuất mà không cần chạy tìm kiếm toàn bộ video")

    # Xử lý xem trước phân tích AI
    if btn_inspect and query_text.strip():
        with st.spinner("Đang gọi AI Semantic Compiler để trích xuất Event Graph..."):
            ai_decomp = engine.compiler.compile_query(
                query_text.strip(),
                auto_translate=auto_translate,
                use_ai_query=True,
                engine_mode="ai"
            )
            st.session_state["kis_inspect_data"] = ai_decomp

    if "kis_inspect_data" in st.session_state:
        insp = st.session_state["kis_inspect_data"]
        with st.expander("🧠 Bảng cấu trúc Ngữ nghĩa AI (Semantic Event Graph Preview)", expanded=True):
            st.markdown(f"**English Translated Query:** `{insp.get('query_en')}`")
            col_i1, col_i2 = st.columns(2)
            with col_i1:
                st.markdown("**Core Events (Hành động cốt lõi):**")
                ce_list = insp.get("intent_flags", {}).get("core_events", [])
                if ce_list:
                    for ce in ce_list:
                        st.markdown(f"- 🎯 `{ce}`")
                else:
                    st.caption("Không phát hiện action cụ thể.")

                st.markdown("**Support Facts (Đồ vật & Bối cảnh nền):**")
                sf_list = insp.get("intent_flags", {}).get("support_facts", [])
                if sf_list:
                    for sf in sf_list:
                        st.markdown(f"- 📦 `{sf}`")
                else:
                    st.caption("Không có support facts.")

            with col_i2:
                st.markdown("**Temporal Phases (Chuỗi thời gian):**")
                ph_list = insp.get("phases_en", [])
                for pi, ph in enumerate(ph_list):
                    st.markdown(f"- **Pha {pi+1}:** `{ph}`")

                anchors = insp.get("intent_flags", {}).get("anchors", [])
                if anchors:
                    st.markdown(f"**Hard OCR Anchors:** `{', '.join(anchors)}`")

            st.json(insp.get("aspect_prompts", {}))

    # Thực thi tìm kiếm KIS
    do_search = False
    search_with_ai = kis_use_ai
    if btn_ai_search:
        do_search = True
        search_with_ai = True
    elif btn_fast_search:
        do_search = True
        search_with_ai = False

    if do_search and query_text.strip():
        t0 = time.time()
        curr_active_mode = engine_mode if search_with_ai else "offline"
        with st.spinner(f"Đang xử lý ({'AI Mode: ' + curr_active_mode.upper() if search_with_ai else 'Offline Zero-LLM'})..."):
            results, trans_q, intent = engine.search(
                query_text.strip(),
                top_k=top_k,
                video_filter=v_filter,
                auto_translate=auto_translate,
                engine_mode=curr_active_mode,
                use_ai_query=search_with_ai,
                use_asr=use_asr,
                audio_weight=audio_weight,
                asr_keywords=custom_asr_kws if custom_asr_kws.strip() else None,
                diversity_top_2=diversity_top_2,
                use_dino_counting=use_dino_counting
            )
            lat = time.time() - t0
            st.session_state["kis_results"] = results
            st.session_state["kis_trans_q"] = trans_q
            st.session_state["kis_intent"] = intent
            st.session_state["kis_sub_name"] = default_sub_name
            st.session_state["kis_lat"] = lat
            st.session_state["kis_used_ai"] = search_with_ai

    if "kis_results" in st.session_state and st.session_state["kis_results"]:
        results = st.session_state["kis_results"]
        trans_q = st.session_state.get("kis_trans_q", "")
        intent = st.session_state.get("kis_intent", {})
        lat = st.session_state.get("kis_lat", 0.0)
        was_ai = st.session_state.get("kis_used_ai", True)

        active_t = intent.get("active_track", engine_mode).upper()
        mode_badge = f"🤖 AI Semantic ({active_t})" if was_ai else "⚡ Offline V2 (Zero-LLM)"
        if auto_translate and trans_q:
            st.caption(f"🌐 Query dịch: \"{trans_q}\" | ⏱️ Độ trễ: {lat:.3f}s | 📍 Chế độ: **{mode_badge}**")

        # Nút chuyển đổi nhanh nếu vừa tìm Offline
        if not was_ai:
            col_re1, col_re2 = st.columns([3, 1])
            with col_re1:
                st.info("💡 Bạn vừa tìm kiếm bằng chế độ Offline Cục bộ. Để có độ chính xác cao hơn với phân tích chuỗi thời gian & hành động, bạn có thể thử lại với AI.")
            with col_re2:
                if st.button("🤖 Thử lại với AI Mode", type="primary", use_container_width=True):
                    with st.spinner("Đang tìm kiếm lại với AI Semantic Compiler..."):
                        t0 = time.time()
                        results, trans_q, intent = engine.search(
                            query_text.strip(),
                            top_k=top_k,
                            video_filter=v_filter,
                            auto_translate=auto_translate,
                            engine_mode=engine_mode,
                            use_ai_query=True,
                            use_asr=use_asr,
                            audio_weight=audio_weight,
                            asr_keywords=custom_asr_kws if custom_asr_kws.strip() else None,
                            diversity_top_2=diversity_top_2,
                            use_dino_counting=use_dino_counting
                        )
                        st.session_state["kis_results"] = results
                        st.session_state["kis_trans_q"] = trans_q
                        st.session_state["kis_intent"] = intent
                        st.session_state["kis_lat"] = time.time() - t0
                        st.session_state["kis_used_ai"] = True
                        st.rerun()

        # Show deep diagnostic info V3.2
        aspect_prompts = intent.get("aspect_prompts", {})
        eff_asr = intent.get("effective_audio_weight", 0.0)
        phases_list = intent.get("phases_en", [])
        core_events = intent.get("core_events", [])
        support_facts = intent.get("support_facts", [])
        top1_item = results[0] if results else {}

        top1_tier = top1_item.get('tier', 'N/A')
        top1_cec = top1_item.get('cec', 0.0)
        top1_ambig = top1_item.get('is_ambiguous', False)
        ambig_str = "⚠️ AMBIGUOUS" if top1_ambig else "🎯 CONFIDENT"

        with st.expander(f"🔍 Bảng Chẩn đoán Chứng cứ (Evidence Judge V3.2) | Track [{active_t}] | {top1_tier} | CEC: {top1_cec:.2f} | {ambig_str}", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**1. Core Events (Hành động sống còn):**")
                if core_events:
                    for ce in core_events:
                        st.text(f"  • [ACTION] {ce}")
                else:
                    st.text("  • General scene action")

                st.markdown("**2. Support Facts (Bối cảnh nền):**")
                if support_facts:
                    for sf in support_facts:
                        st.text(f"  • [FACT] {sf}")
                else:
                    st.text("  • No explicit support facts")

                if intent.get("anchors"):
                    st.markdown(f"**3. Hard OCR Anchors:** `{', '.join(intent['anchors'])}`")
            with c2:
                if phases_list and len(phases_list) > 1:
                    st.markdown("**4. Chuỗi pha thời gian (Skip-Aware DP):**")
                    for pi, p_txt in enumerate(phases_list):
                        st.text(f"  Pha {pi+1}: {p_txt}")
                st.markdown(f"**5. Lời thoại ASR:** `{'Bật (' + str(eff_asr) + ')' if eff_asr > 0 else 'Tắt'}`")
                
                trace = top1_item.get("evidence_trace", {})
                if trace:
                    st.markdown("**6. Query-to-Evidence Trace (Top-1):**")
                    st.json(trace)

        with st.expander("📥 Xuất file nộp bài (Submission CSV)", expanded=False):
            c1, c2 = st.columns([3, 1])
            with c1:
                sub_fn = st.text_input("Tên file CSV:", value=st.session_state.get("kis_sub_name", default_sub_name))
            with c2:
                st.write("")
                st.write("")
                if st.button("Xuất CSV", use_container_width=True):
                    cart = st.session_state.get("submission_cart", {})
                    p = exporter.export_kis(results, cart, sub_fn, max_rows=100)
                    st.success(f"Đã lưu: {p}")

        st.markdown(f"### Kết quả Top {len(results)} khung hình")
        grid = st.columns(cols_count)
        for i, item in enumerate(results):
            col = grid[i % cols_count]
            with col:
                st.markdown(f"**#{i+1}. {item['video']} F{item['frame']}**")
                if item["image_path"] and os.path.exists(item["image_path"]):
                    try:
                        img_bytes = load_cached_thumbnail(item["image_path"])
                        if img_bytes:
                            st.image(img_bytes, use_container_width=True)
                        else:
                            st.text("Image load error")
                    except Exception as e:
                        st.text(f"(image error: {e})")
                else:
                    st.text(f"No image: {item['video']} F{item['frame']}")
                
                tier_str = item.get("tier", "TIER_4")
                tier_badge = "✅ " if tier_str == "TIER_0" else ("🔒 " if "TIER" in str(tier_str) else "")
                ambig_mark = " ⚠️" if item.get("is_ambiguous") else ""
                st.caption(f"{tier_badge}[{tier_str}{ambig_mark}] CEC: {item.get('cec', 0.0):.2f} | Score: {item.get('score', 0.0):.1f} (t={item.get('pts_time', 0.0):.1f}s)")
                
                render_pick_ui(f"kis_add_{i}", item['video'], item['frame'])
                    
                render_temporal_browsing(f"kis_{i}", item['video'], item['frame'])

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

    c_toggle, c_mode_badge = st.columns([2, 2])
    with c_toggle:
        qa_use_ai = st.toggle("🤖 Bật AI Mode cho câu hỏi này", value=enable_ai_mode, key="qa_use_ai_toggle")
    with c_mode_badge:
        if qa_use_ai:
            st.caption(f"Trạng thái: 🤖 **AI Mode ({engine_mode.upper()})**")
        else:
            st.caption("Trạng thái: ⚡ **Offline V2 (Zero-LLM Cục bộ)**")

    c_b1, c_b2 = st.columns(2)
    btn_ai_qa = c_b1.button("🤖 Tìm kiếm & Trả lời với AI", type="primary" if qa_use_ai else "secondary", use_container_width=True)
    btn_fast_qa = c_b2.button("⚡ Tìm kiếm & Trả lời Nhanh (Offline)", type="secondary" if qa_use_ai else "primary", use_container_width=True)

    do_qa = False
    qa_search_ai = qa_use_ai
    if btn_ai_qa:
        do_qa = True
        qa_search_ai = True
    elif btn_fast_qa:
        do_qa = True
        qa_search_ai = False

    if do_qa and qa_text.strip():
        with st.spinner("Đang tìm kiếm bằng chứng và chuẩn hóa đáp án..."):
            curr_active_mode = engine_mode if qa_search_ai else "offline"
            results, trans_q, intent = engine.search(
                qa_text.strip(),
                top_k=top_k,
                video_filter=v_filter,
                auto_translate=auto_translate,
                engine_mode=curr_active_mode,
                use_ai_query=qa_search_ai,
                use_asr=use_asr,
                audio_weight=audio_weight,
                asr_keywords=custom_asr_kws if custom_asr_kws.strip() else None,
                diversity_top_2=diversity_top_2
            )
            st.session_state["qa_results"] = results
            st.session_state["qa_text"] = qa_text.strip()
            st.session_state["qa_sub_name"] = default_sub_name
            st.session_state["qa_used_ai"] = qa_search_ai

    if "qa_results" in st.session_state and st.session_state["qa_results"]:
        results = st.session_state["qa_results"]
        qa_q = st.session_state.get("qa_text", "")

        st.info("💡 **Gợi ý:** Hãy quan sát các khung hình được tìm thấy bên dưới, chọn khung hình đúng nhất và tự nhập đáp án vào ô dưới đây để xuất CSV.")

        with st.expander("📥 Xuất file nộp bài (Submission CSV)", expanded=False):
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                sub_fn = st.text_input("Tên file CSV:", value=st.session_state.get("qa_sub_name", default_sub_name))
            with c2:
                user_ans = st.text_input("📝 Nhập đáp án thủ công:", value="Có")
            with c3:
                st.write("")
                st.write("")
                if st.button("Xuất CSV", use_container_width=True):
                    # Populate answer for all results
                    qa_data = [{"video": r["video"], "frame": r["frame"], "answer": user_ans} for r in results]
                    # Also update cart items if any
                    cart = st.session_state.get("submission_cart", {})
                    for rank, c_item in cart.items():
                        c_item["answer"] = user_ans
                        
                    p = exporter.export_qa(qa_data, cart, sub_fn, max_rows=100)
                    st.success(f"Đã lưu: {p}")

        grid = st.columns(cols_count)
        for i, item in enumerate(results):
            col = grid[i % cols_count]
            with col:
                st.markdown(f"**#{i+1}. {item['video']} F{item['frame']}**")
                if item["image_path"] and os.path.exists(item["image_path"]):
                    try:
                        img_bytes = load_cached_thumbnail(item["image_path"])
                        if img_bytes:
                            st.image(img_bytes, use_container_width=True)
                        else:
                            st.text("Image load error")
                    except Exception as e:
                        st.text(f"(image error: {e})")
                else:
                    st.text(f"No image")
                tier_info = f"[{item.get('tier', 'TIER_4')}] "
                st.caption(f"{tier_info}Score: {item.get('score', 0.0):.2f}")
                
                render_pick_ui(f"qa_add_{i}", item['video'], item['frame'])
                    
                render_temporal_browsing(f"qa_{i}", item['video'], item['frame'])

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

    c_toggle, c_mode_badge = st.columns([2, 2])
    with c_toggle:
        trake_use_ai = st.toggle("🤖 Bật AI Mode cho chuỗi sự kiện", value=enable_ai_mode, key="trake_use_ai_toggle")
    with c_mode_badge:
        if trake_use_ai:
            st.caption(f"Trạng thái: 🤖 **AI Mode ({engine_mode.upper()})**")
        else:
            st.caption("Trạng thái: ⚡ **Offline V2 (Zero-LLM Cục bộ)**")

    c_b1, c_b2 = st.columns(2)
    btn_ai_trake = c_b1.button("🤖 Định vị Sự kiện với AI", type="primary" if trake_use_ai else "secondary", use_container_width=True)
    btn_fast_trake = c_b2.button("⚡ Định vị Nhanh (Offline)", type="secondary" if trake_use_ai else "primary", use_container_width=True)

    do_trake = False
    trake_search_ai = trake_use_ai
    if btn_ai_trake:
        do_trake = True
        trake_search_ai = True
    elif btn_fast_trake:
        do_trake = True
        trake_search_ai = False

    if do_trake and trake_text.strip():
        event_queries = QueryCompiler.parse_trake_events(trake_text)

        with st.spinner(f"Đang thực thi 3-Stage Temporal Refinement cho {len(event_queries)} sự kiện..."):
            trake_res, proc_events = engine.search_trake(
                event_queries,
                top_k_videos=100,
                video_filter=v_filter,
                auto_translate=auto_translate,
                use_ai_query=trake_search_ai
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
                    cart = st.session_state.get("submission_cart", {})
                    p = exporter.export_trake(trake_res, cart, sub_fn, max_rows=100)
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
                            img_bytes = load_cached_thumbnail(f_data["image_path"])
                            if img_bytes:
                                st.image(img_bytes, use_container_width=True)
                            else:
                                st.text("Image load error")
                        except Exception as e:
                            st.text(f"(image error: {e})")
                    else:
                        st.text(f"F{f_data['frame']}")
                        
                    render_pick_ui(f"trake_add_{r_idx}_{f_idx}", v_item['video'], f_data['frame'], f_idx, is_trake=True)
                        
                    render_temporal_browsing(f"trake_{r_idx}_{f_idx}", v_item['video'], f_data['frame'])
            st.divider()

# =========================================================================
# BATCH
# =========================================================================
elif query_mode == "Batch Processing":
    st.subheader("Xử lý hàng loạt toàn bộ bộ đề thi (Batch Processing)")

    batch_dir = st.text_input("Thư mục đề thi:", value="de-thi" if os.path.exists("de-thi") else "THUNGHIEM-bo-de-thi")

    batch_use_ai = st.toggle("🤖 Kích hoạt AI Semantic Mode cho toàn bộ Batch", value=enable_ai_mode)
    curr_batch_mode = engine_mode if batch_use_ai else "offline"
    st.caption(f"Cấu hình Batch hiện tại: **{'🤖 AI Mode (' + curr_batch_mode.upper() + ')' if batch_use_ai else '⚡ Offline Zero-LLM'}**")

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
                    engine_mode=curr_batch_mode,
                    use_ai_query=batch_use_ai,
                    use_asr=use_asr,
                    audio_weight=audio_weight
                )
                progress.progress((i + 1) / len(q_files))

            zip_path = exporter.zip_submissions("submission.zip")
            st.success(f"Hoàn tất! Đã lưu toàn bộ CSV và đóng gói ZIP: {zip_path}")
