import sentencepiece as spm

# โหลด tokenizer ที่ train เอง
sp = spm.SentencePieceProcessor(model_file='thai_health_tokenizer.model')

# ตัวอย่างข้อความ
texts = ["ice", "lens", "abc"]

for text in texts:
    ids = sp.encode(text, out_type=int)
    decoded = sp.decode(ids)
    print(f"ต้นฉบับ: {text}")
    print(f"token ids: {ids}")
    print(f"decode กลับ: {decoded}")
    print("===")
