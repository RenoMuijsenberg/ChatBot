import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

start = datetime.now()

print("Device name:", torch.cuda.get_device_properties('cuda').name)
print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())
print(f'torch version: {torch.version}')

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    torch_dtype=torch.float16,
    device_map="auto"
)

conversation_history = []

while True:
    history_string = "\n".join(conversation_history)

    input_text = input("> ")

    input_ids = tokenizer.encode_plus(history_string, input_text, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids, max_new_tokens=1000, pad_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(outputs[0])
    print(response)
