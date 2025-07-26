import sentencepiece as spm
import torch
import re
from model import GPT, GPTConfig

# โหลด tokenizer
sp = spm.SentencePieceProcessor(model_file="thai_health_tokenizer.model")

# โหลดโมเดล checkpoint
checkpoint = torch.load("out-shakespeare-char/ckpt.pt", map_location="cpu", weights_only=True)
model_args = checkpoint["model_args"]
model = GPT(GPTConfig(**model_args))
model.load_state_dict(checkpoint["model"])
model.eval()

# รับ input
user_input = input("พิมพ์คำถาม: ")
prompt = f"<|question|>{user_input}<|answer|>"

# encode prompt
input_ids = sp.encode(prompt, out_type=int)
x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

# generate
y = model.generate(x, max_new_tokens=64)[0].tolist()

# decode เป็นข้อความ
output = sp.decode(y)

# [วางตรงนี้!] ลบ pattern ขยะ และตัดเฉพาะคำตอบ
output = re.sub(r'^[\"\' ]*t o[\"\' ]*', '', output)  # ตัด " ” t o ”" หน้า string
if "<|answer|>" in output:
    output = output.split("<|answer|>", 1)[1]

# print คำตอบสุดท้าย
print("คำตอบ:", output.strip())
