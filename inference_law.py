import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download
import os

def predict(messages, model, tokenizer):
    if torch.backends.mps.is_available():
        print(f"当前使用 MPS 进行推理，速度较快。")
        device = "mps"
    elif torch.cuda.is_available():
        print(f"当前使用 GPU 进行推理，速度较快。")
        device = "cuda"
    else:
        print(f"当前使用 CPU 进行推理，速度较慢。")
        device = "cpu"

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=2048)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# 定义模型名称
model_name = "Qwen/Qwen3-0.6B"

# 获取脚本所在目录，并创建模型缓存路径
script_path = os.path.dirname(os.path.abspath(__file__))
cache_path = os.path.join(script_path, "../models")

# 在modelscope上下载Qwen模型到本地目录下
model_dir = snapshot_download(model_name, cache_dir=cache_path, revision="master")

# 使用实际下载的目录名，并根据设备选择合适的dtype
if torch.cuda.is_available():
    load_dtype = torch.float16
else:
    load_dtype = torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=load_dtype)

test_texts = {
    'instruction': "你是一个法律专家，你需要根据用户的问题，给出专业法律回答。",
    'input': "如果被告人不服判决，有什么权利？"
}

instruction = test_texts['instruction']
input_value = test_texts['input']

messages = [
    {"role": "system", "content": f"{instruction}"},
    {"role": "user", "content": f"{input_value}"}
]

print(f"推理问题：{input_value}")
response = predict(messages, model, tokenizer)
print(response)
