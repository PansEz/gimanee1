# my_tokenizer.py
import sentencepiece as spm

# โหลด tokenizer จากไฟล์
sp = spm.SentencePieceProcessor()
sp.load("thai_health_tokenizer.model")

# ฟังก์ชันสำหรับ encode และ decode
def encode(text):
    return sp.encode(text, out_type=int)

def decode(token_ids):
    return sp.decode(token_ids)
