import sentencepiece as spm

# โหลด tokenizer
sp = spm.SentencePieceProcessor(model_file='thai_health_tokenizer.model')

# แปลงข้อความเป็น token ids
text = "ฉันอยากรู้เกี่ยวกับสุขภาพ"
ids = sp.encode(text, out_type=int)
print(ids)

# แปลงกลับจาก token ids เป็นข้อความ
decoded = sp.decode(ids)
print(decoded)
