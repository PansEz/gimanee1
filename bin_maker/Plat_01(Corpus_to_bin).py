import numpy as np

with open("val.txt", encoding="utf-8") as f:
    data = []
    for line in f:
        tokens = list(map(int, line.strip().split()))
        data.extend(tokens)
data = np.array(data, dtype=np.uint16)
data.tofile("val.bin")
