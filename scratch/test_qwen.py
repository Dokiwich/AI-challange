import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image

def test_qwen_local():
    print("Loading Qwen2-VL-2B-Instruct...")
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="cuda"
    )

    print("Model loaded successfully. Running inference...")
    
    # Tạo một ảnh dummy
    img = Image.new('RGB', (224, 224), color='red')
    
    prompt = "Màu của bức ảnh là gì?"
    messages = [
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to("cuda")
    
    generated_ids = model.generate(**inputs, max_new_tokens=15)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    raw_ans = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    print(f"Question: {prompt}")
    print(f"Answer: {raw_ans}")

if __name__ == "__main__":
    test_qwen_local()
