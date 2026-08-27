# Danh sách Tài khoản & Cấu hình Hệ thống (Setup Accounts)

Tài liệu này lưu trữ các thông tin đăng nhập và cấu hình quan trọng của dự án HCMC AI Challenge 2026.

## 1. Tài khoản Cuộc thi (HCMC AI Challenge 2026)
- **Tên đăng nhập (Team Account):** `aic2026-180`
- **Mật khẩu (Team Password):** `jtt7ikKvj8lx`

## 2. Neo4j Graph Database (Truy xuất Đồ thị)
Thông tin đăng nhập mặc định khi khởi chạy Neo4j qua Docker (`docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest`):
- **Giao diện quản lý (Browser):** [http://localhost:7474](http://localhost:7474)
- **Bolt URI (Connection String):** `bolt://localhost:7687`
- **Username:** `neo4j`
- **Password:** `password`

## 3. 9Router Proxy & LLM API (Truy vấn AI)
Thông tin cấu hình LLM (được sử dụng bởi `core/ai_query_parser.py` thông qua file `.env`):
- **LLM Base URL:** `http://localhost:20128/v1`
- **API Key:** `sk-7bc925a89b2ce4c9-m7as6w-555d8f4b`
- **Model Name:** `vip`
- **Provider:** `openai_compatible`

## 4. Qdrant Vector Database (Lưu trữ Embedding) - Sắp triển khai
- *(Dự kiến)* Giao diện quản lý: `http://localhost:6333/dashboard`
- *(Dự kiến)* GRPC Port: `6334`

---
> **Lưu ý bảo mật:** File này không nên được đẩy lên các kho lưu trữ mã nguồn mở công khai (Public Repository) để tránh rò rỉ API Key và tài khoản cuộc thi. Hãy cân nhắc thêm `setup_accounts.md` vào file `.gitignore` nếu bạn sử dụng Git.
