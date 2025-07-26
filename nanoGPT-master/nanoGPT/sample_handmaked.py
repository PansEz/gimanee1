import sentencepiece as spm
import torch
from model import GPT, GPTConfig

# 1. โหลด tokenizer
sp = spm.SentencePieceProcessor(model_file="thai_health_tokenizer.model")

# 2. โหลดโมเดล checkpoint
checkpoint = torch.load("out-shakespeare-char/ckpt.pt", map_location="cpu", weights_only=True)
model_args = checkpoint["model_args"]
model = GPT(GPTConfig(**model_args))
model.load_state_dict(checkpoint["model"])
model.eval()

# 3. encode prompt
prompt = input("input : ")
# prompt = "หมอค่ะ ดิฉันมีอาการไอแห้ง"
input_ids = sp.encode(prompt, out_type=int)
x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)


# 4. generate
y = model.generate(x, max_new_tokens=50)[0].tolist()

# 5. decode ผลลัพธ์
response_ids = y[len(input_ids):]   # ตัด prompt ออก เหลือแต่ response

output = sp.decode(response_ids)
print("ㅤ", output)


# import torch
# from model import GPT, GPTConfig
# import sentencepiece as spm

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # โหลด tokenizer
# sp = spm.SentencePieceProcessor(model_file='thai_health_tokenizer.model')

# # โหลดโมเดล
# checkpoint = torch.load('out/ckpt.pt', map_location=device)
# model_args = checkpoint['model_args']
# model = GPT(GPTConfig(**model_args))
# model.load_state_dict(checkpoint['model'])
# model.to(device)
# model.eval()

# # รับ prompt จาก user
# prompt = input("input : ")
# prompt_ids = sp.encode("<|question|>" + prompt + "<|answer|>", out_type=int)
# idx = torch.tensor([prompt_ids], dtype=torch.long).to(device)

# y = model.generate(idx, max_new_tokens=50)[0].tolist()
# output = sp.decode(y)
# print("output:", output)



