import os
import glob
import json
import subprocess
import sys

# This script will be run from the scripts directory, so paths are relative to scripts/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPPING_DIR = os.path.join(BASE_DIR, "data", "mapping", "media-info-aic25-b1", "media-info")
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "data", "transcripts")
AUDIO_DIR = os.path.join(BASE_DIR, "data", "audio")
TEMP_DIR = os.path.join(BASE_DIR, "data", "temp_vtt")

os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

try:
    import webvtt
except ImportError:
    print("Vui lòng cài đặt: pip install webvtt-py yt-dlp")
    exit(1)

def get_yt_subtitles(video_id, url):
    """
    Dùng yt-dlp tải phụ đề tự động tiếng Việt.
    Lưu vào thư mục TEMP_DIR
    """
    output_template = os.path.join(TEMP_DIR, f"{video_id}.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--write-auto-subs",
        "--write-subs",
        "--sub-lang", "vi",
        "--skip-download",
        "--output", output_template,
        url
    ]
    try:
        # Chạy ẩn
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Lỗi chạy yt-dlp cho {video_id}: {e}")
    
    # Tìm file vtt vừa tải (có thể là .vi.vtt)
    vtt_files = glob.glob(os.path.join(TEMP_DIR, f"{video_id}*.vtt"))
    if vtt_files:
        return vtt_files[0]
    return None

def download_audio(video_id, url):
    """
    Tải audio (m4a) nếu không có phụ đề.
    """
    output_path = os.path.join(AUDIO_DIR, f"{video_id}.m4a")
    if os.path.exists(output_path):
        return True # Đã tải
    
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestaudio[ext=m4a]/bestaudio",
        "--output", output_path,
        url
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        return os.path.exists(output_path)
    except subprocess.TimeoutExpired:
        print(" (Quá 60s) ", end="", flush=True)
        return False
    except:
        return False

def parse_vtt_to_json(vtt_path, json_path):
    """
    Đọc VTT và xuất JSON
    """
    subs = []
    try:
        for caption in webvtt.read(vtt_path):
            subs.append({
                "start": caption.start,
                "end": caption.end,
                "text": caption.text.strip().replace('\n', ' ')
            })
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(subs, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Lỗi parse VTT {vtt_path}: {e}")
        return False

def main():
    json_files = sorted(glob.glob(os.path.join(MAPPING_DIR, "*.json")))
    if not json_files:
        print(f"Không tìm thấy file metadata nào trong {MAPPING_DIR}")
        return
    
    print(f"Tìm thấy {len(json_files)} videos. Bắt đầu quá trình quét...")
    
    success_sub = 0
    success_audio = 0
    failed = 0
    
    for i, meta_file in enumerate(json_files):
        video_id = os.path.splitext(os.path.basename(meta_file))[0]
        
        # Check if transcript exists
        final_json_path = os.path.join(TRANSCRIPT_DIR, f"{video_id}.json")
        audio_path = os.path.join(AUDIO_DIR, f"{video_id}.m4a")
        
        if os.path.exists(final_json_path):
            print(f"[{i+1}/{len(json_files)}] {video_id} - Đã có Subtitles (Bỏ qua)")
            success_sub += 1
            continue
        elif os.path.exists(audio_path):
            print(f"[{i+1}/{len(json_files)}] {video_id} - Đã có Audio (Bỏ qua)")
            success_audio += 1
            continue
            
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        url = meta.get("watch_url")
        if not url:
            print(f"[{i+1}/{len(json_files)}] {video_id} - KHÔNG CÓ URL!")
            failed += 1
            continue
            
        print(f"[{i+1}/{len(json_files)}] {video_id} - Đang cào phụ đề...", end=" ", flush=True)
        vtt_path = get_yt_subtitles(video_id, url)
        
        if vtt_path:
            if parse_vtt_to_json(vtt_path, final_json_path):
                print(f"-> THÀNH CÔNG (Subtitles)")
                success_sub += 1
            else:
                print(f"-> LỖI parse")
                failed += 1
            # Clean up VTT
            try:
                os.remove(vtt_path)
            except:
                pass
        else:
            print(f"-> KHÔNG CÓ Subtitles. Đang tải Audio...", end=" ", flush=True)
            if download_audio(video_id, url):
                print(f"-> THÀNH CÔNG (Audio)")
                success_audio += 1
            else:
                print(f"-> LỖI TẢI AUDIO")
                failed += 1
                
        # Xóa sạch temp VTT thư mục để không bị rác
        for f in glob.glob(os.path.join(TEMP_DIR, "*")):
            try:
                os.remove(f)
            except:
                pass
                
    print("\n" + "="*40)
    print("HOÀN TẤT!")
    print(f"Tổng cộng: {len(json_files)} videos")
    print(f"- Đã lấy Subtitles: {success_sub}")
    print(f"- Đã lấy Audio: {success_audio}")
    print(f"- Thất bại: {failed}")
    print("="*40)

if __name__ == "__main__":
    main()
