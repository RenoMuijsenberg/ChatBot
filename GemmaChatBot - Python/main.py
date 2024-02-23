import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

start = datetime.now()

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    torch_dtype=torch.float16,
).to("cuda")

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=4000)
print(tokenizer.decode(outputs[0]))
print(datetime.now() - start)
