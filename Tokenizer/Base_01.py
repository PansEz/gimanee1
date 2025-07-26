import sentencepiece as spm

# กำหนดชื่อไฟล์ input (ต้องตรงกับที่อัปโหลดไว้)
input_file = r'C:\Users\soroj\Desktop\GIMANEE - HEALTH CARE CHATBOT\pretrain corpus\pretrain_corpus_305.txt'
model_prefix = 'thai_health_tokenizer'
vocab_size = 1200      # จำนวน subwords ที่ต้องการ (เช่น 3000, 5000, 8000 หรือ 16000)

# train sentencepiece (unigram เหมาะกับภาษาไทย, สามารถใช้ 'bpe' ก็ได้)
spm.SentencePieceTrainer.train(
    input=input_file,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    character_coverage=0.9995,    # ครอบคลุมตัวอักษรภาษาไทย-อังกฤษ-เลข
    model_type='unigram',         # หรือ 'bpe'
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
)
