import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file='thai_health_tokenizer.model')

with open('health_finetune_35000_sep.txt', encoding='utf-8') as fin, \
     open('health_finetune_35000_sep_encoded.txt', 'w', encoding='utf-8') as fout:
    for line in fin:
        line = line.strip()
        if not line: continue
        ids = sp.encode(line, out_type=int)
        fout.write(' '.join(map(str, ids)) + '\n')
