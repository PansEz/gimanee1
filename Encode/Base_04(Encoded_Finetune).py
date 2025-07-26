import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file='thai_health_tokenizer.model')

import json

with open("health_finetune_35000_sep.txt", encoding="utf-8") as fin, \
     open("health_finetune_35000_sep_encoded.txt", "w", encoding="utf-8") as fout:
    for line in fin:
        item = json.loads(line)
        prompt_ids = sp.encode(item["prompt"], out_type=int)
        response_ids = sp.encode(item["response"], out_type=int)
        # สมมุติว่า token id = 3 คือ <eos> (ตามที่คุณ train)
        ids = prompt_ids + [3] + response_ids + [3]
        # เขียนเป็นข้อความตัวเลขต่อกันในแต่ละบรรทัด
        fout.write(' '.join(map(str, ids)) + '\n')
