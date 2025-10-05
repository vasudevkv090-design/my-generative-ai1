from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

MODEL_DIR = "./gpt2-lora"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
base_model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, MODEL_DIR)

prompt = "Write a two-line poem about friendship:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=60, do_sample=True, top_p=0.95, top_k=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
