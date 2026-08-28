import os
import json
from dotenv import load_dotenv

load_dotenv()

def test_openrouter():
    from core.ai_query_parser import AIQueryParser
    parser = AIQueryParser()
    
    if not parser.api_key:
        print("LỖI: Chưa có OPENROUTER_API_KEY trong file .env")
        return
        
    print(f"Đang gọi OpenRouter với model: {parser.model_name}")
    result = parser.parse_query_structured("một người đàn ông mặc áo đỏ đang đi xe đạp")
    
    if result:
        print("\n✅ Phân tích THÀNH CÔNG!")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("\n❌ Phân tích THẤT BẠI. Hãy kiểm tra lại API Key hoặc mạng.")

if __name__ == "__main__":
    test_openrouter()
