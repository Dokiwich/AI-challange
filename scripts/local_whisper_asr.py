import os
import json
import glob
import subprocess
try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Vui lòng cài đặt: pip install faster-whisper")
    exit(1)

# Cấu hình thư mục
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE_DIR, "data", "audio")
VIDEOS_DIR = os.path.join(BASE_DIR, "data", "videos") # Thư mục chứa video tải thủ công (nếu có)
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "data", "transcripts")

os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

# Lấy tất cả file âm thanh/video cần dịch
media_files = glob.glob(os.path.join(AUDIO_DIR, "*.*")) + glob.glob(os.path.join(VIDEOS_DIR, "*.*"))

# Hàm định dạng thời gian chuẩn JSON
def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def main():
    if not media_files:
        print(f"Không tìm thấy file nào trong {AUDIO_DIR} hoặc {VIDEOS_DIR}!")
        return

    print("Đang khởi tạo AI Whisper (Sử dụng GPU RTX 3050)...")
    print("Mô hình: Medium (Cân bằng tốt giữa VRAM và Độ chính xác)")
    # Dùng compute_type="float16" để tối ưu VRAM cho card RTX
    model = WhisperModel("medium", device="cuda", compute_type="float16")

    for i, file_path in enumerate(media_files):
        vid_name = os.path.splitext(os.path.basename(file_path))[0]
        out_json = os.path.join(TRANSCRIPT_DIR, f"{vid_name}.json")
        
        # Bỏ qua nếu đã dịch
        if os.path.exists(out_json):
            print(f"[{i+1}/{len(media_files)}] {vid_name} - Đã có sẵn Transcript. Bỏ qua.")
            continue
            
        print(f"[{i+1}/{len(media_files)}] Đang dịch âm thanh cho: {vid_name} ...", end=" ", flush=True)
        
        try:
            # Whisper có thể đọc trực tiếp mp4, mkv, m4a, wav (yêu cầu máy có cài FFmpeg)
            segments, info = model.transcribe(file_path, language="vi", beam_size=5)
            
            subs = []
            for segment in segments:
                subs.append({
                    "start": format_time(segment.start),
                    "end": format_time(segment.end),
                    "text": segment.text.strip()
                })
                
            if subs:
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(subs, f, ensure_ascii=False, indent=2)
                print(f"-> THÀNH CÔNG! ({len(subs)} câu)")
            else:
                print("-> THÀNH CÔNG! (Nhưng không nghe thấy tiếng/lời thoại nào)")
                
        except Exception as e:
            print(f"\n-> LỖI khi xử lý {vid_name}: {e}")
            if "ffmpeg" in str(e).lower() or "ffprobe" in str(e).lower():
                print("   !!! GỢI Ý: Lỗi này xảy ra do máy của bạn chưa cài phần mềm lõi FFmpeg.")
                print("   !!! Hãy cài FFmpeg vào máy tính để phần mềm đọc được file video/audio nhé.")

if __name__ == "__main__":
    main()
