"""
Công cụ tự động tải file/thư mục từ Google Drive và giải nén Keyframes
Sử dụng thư viện gdown
"""

import os
import zipfile
import gdown
from typing import Optional

def download_file_from_drive(
    url_or_id: str,
    output_path: Optional[str] = None,
    auto_unzip: bool = True
) -> str:
    """
    Tải file từ Google Drive link và tự động giải nén nếu là file .zip
    - url_or_id: Link chia sẻ Google Drive hoặc ID file
    - output_path: Tên file lưu lại (nếu None sẽ tự lấy tên gốc trên Drive)
    """
    print(f"\n[Drive Downloader] Bắt đầu tải từ: {url_or_id}")
    
    # gdown hỗ trợ link chia sẻ trực tiếp
    downloaded_file = gdown.download(
        url=url_or_id if "http" in url_or_id else None,
        id=url_or_id if "http" not in url_or_id else None,
        output=output_path,
        quiet=False,
        fuzzy=True
    )

    if not downloaded_file or not os.path.exists(downloaded_file):
        raise RuntimeError(f"Tải thất bại từ Google Drive: {url_or_id}")

    print(f"[Drive Downloader] Tải thành công: {downloaded_file} ({os.path.getsize(downloaded_file):,} bytes)")

    # Tự động giải nén nếu là file zip
    if auto_unzip and downloaded_file.endswith(".zip"):
        extract_dir = os.path.splitext(downloaded_file)[0]
        print(f"[Drive Downloader] Đang giải nén '{downloaded_file}' vào '{extract_dir}'...")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(downloaded_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"[Drive Downloader] ✅ Đã giải nén xong vào: {extract_dir}")
        return extract_dir

    return downloaded_file

def download_folder_from_drive(
    folder_url_or_id: str,
    output_dir: str = "downloaded_drive_folder"
) -> str:
    """
    Tải toàn bộ thư mục từ Google Drive
    """
    print(f"\n[Drive Downloader] Bắt đầu tải toàn bộ thư mục từ: {folder_url_or_id}")
    os.makedirs(output_dir, exist_ok=True)
    gdown.download_folder(
        url=folder_url_or_id if "http" in folder_url_or_id else None,
        id=folder_url_or_id if "http" not in folder_url_or_id else None,
        output=output_dir,
        quiet=False
    )
    print(f"[Drive Downloader] ✅ Đã tải thư mục vào: {output_dir}")
    return output_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tải dữ liệu từ Google Drive")
    parser.add_argument("url", type=str, help="Link Google Drive hoặc File ID")
    parser.add_argument("--out", "-o", type=str, default=None, help="Đường dẫn file đầu ra")
    parser.add_argument("--folder", action="store_true", help="Bật cờ này nếu tải toàn bộ thư mục Drive")
    args = parser.parse_args()

    if args.folder:
        download_folder_from_drive(args.url, args.out or "downloaded_folder")
    else:
        download_file_from_drive(args.url, args.out)
