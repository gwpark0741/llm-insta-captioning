import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()

prompt = "Write a short Instagram caption about a cozy coffee shop. Use 2 emojis."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
    )

print(tokenizer.decode(out[0], skip_special_tokens=True))
