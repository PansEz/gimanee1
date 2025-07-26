import random

input_file = 'health_finetune_35000_sep_encoded.txt'
train_file = 'train.txt'
val_file = 'val.txt'

with open(input_file, 'r', encoding='utf-8') as fin:
    lines = fin.readlines()

random.shuffle(lines)  # สุ่มลำดับก่อนแบ่ง

split_idx = int(len(lines) * 0.9)  # 90% สำหรับ train

with open(train_file, 'w', encoding='utf-8') as ftrain, \
     open(val_file, 'w', encoding='utf-8') as fval:
    ftrain.writelines(lines[:split_idx])
    fval.writelines(lines[split_idx:])
