import torch
from PIL import Image
import requests

def chat(model, processor):
    device = model.device

    is_vl_model = hasattr(processor, "image_processor")

    messages_text = [{"role": "user", "content": "你好，请自我介绍一下。"}]
    text_prompt = processor.apply_chat_template(messages_text, tokenize=False, add_generation_prompt=True)
    inputs_text = processor(text=[text_prompt], return_tensors="pt").to(device)

    with torch.no_grad():
        output_text = model.generate(**inputs_text, max_new_tokens=1024, do_sample=False)
    
    print("Output:", processor.decode(output_text[0], skip_special_tokens=True))

    if is_vl_model:
        try:
            url = "http://images.cocodataset.org/val2017/000000000632.jpg"
            image = Image.open(requests.get(url, stream=True).raw)

            messages_vl = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "请你分析一下这张图片"}
                    ],
                }
            ]

            vl_prompt = processor.apply_chat_template(messages_vl, tokenize=False, add_generation_prompt=True)
            inputs_vl = processor(text=[vl_prompt], images=[image], return_tensors="pt").to(device)

            with torch.no_grad():
                output_vl = model.generate(**inputs_vl, max_new_tokens=1024, do_sample=False)
            
            print("Output:", processor.decode(output_vl[0], skip_special_tokens=True))
        except Exception as e:
            print(f"多模态测试失败: {e} (可能需要根据特定模型的 Template 格式微调 messages 结构)")
    else:
        print("\n[跳过] 该模型不支持视觉输入。")